"""Adapter for the configured Telly model, preserving the shared transcript."""
import json
import os
import urllib.request
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class OllamaAnalysisModel:
    """Use Ollama's real tool-call protocol instead of legacy JSON chat prompts."""
    def __init__(self, model, base_url="http://localhost:11434"):
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"

    def __call__(self, messages, tools):
        converted = []
        for message in messages:
            item = {"role": message["role"], "content": message.get("content") or ""}
            if message.get("tool_calls"):
                item["tool_calls"] = [{"type": "function", "function": {
                    "name": c["name"], "arguments": c["arguments"]}} for c in message["tool_calls"]]
            if message.get("_thinking"):
                item["thinking"] = message["_thinking"]
            if message["role"] == "tool":
                item["tool_name"] = message["name"]
            converted.append(item)
        payload = {"model": self.model, "messages": converted,
                   "tools": [{"type": "function", "function": t} for t in tools],
                   "stream": False, "think": True,
                   "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 4096}}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        if result.get("done_reason") == "length":
            raise ValueError("모델 응답 한도에 도달했습니다. 요청 범위를 좁혀주세요.")
        message = result["message"]
        return {"role": "assistant", "content": message.get("content", ""),
                "_thinking": message.get("thinking", ""),
                "tool_calls": [{"id": str(uuid4()), "name": c["function"]["name"],
                                "arguments": c["function"]["arguments"]}
                               for c in message.get("tool_calls", [])]}


def build_analysis_model():
    from dotenv import load_dotenv
    load_dotenv()
    if os.environ.get("LLM_PROVIDER", "google").lower() == "ollama":
        return OllamaAnalysisModel(os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
                                   os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    from core.llm import load_llm
    return AnalysisModel(load_llm(max_tokens=4096))


class AnalysisModel:
    def __init__(self, llm):
        self.llm = llm
        self.native = True

    def __call__(self, messages, tools):
        converted = []
        for message in messages:
            role, content = message["role"], message.get("content") or ""
            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "tool":
                converted.append(ToolMessage(content=content, tool_call_id=message["tool_call_id"]))
            else:
                converted.append(AIMessage(content=content, tool_calls=[
                    {"id": c["id"], "name": c["name"], "args": c["arguments"]}
                    for c in message.get("tool_calls", [])]))
        if self.native:
            try:
                bound = self.llm.bind_tools(tools)
            except (NotImplementedError, AttributeError):
                self.native = False
            else:
                response = bound.invoke(converted)
                if getattr(response, "invalid_tool_calls", []):
                    raise ValueError("모델이 도구 입력을 잘못 생성했습니다. 요청을 다시 시도해주세요.")
                content = response.content
                if isinstance(content, list):
                    content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                return {"role": "assistant", "content": content, "tool_calls": [
                    {"id": c["id"], "name": c["name"], "arguments": c["args"]}
                    for c in response.tool_calls]}
        # Legacy local models don't support bind_tools. A single JSON action
        # protocol uses the same transcript and executor, not a separate router.
        instruction = ('Return one JSON object only: {"text":"Korean response",'
                       '"tool":null,"arguments":{}} or {"text":"",'
                       '"tool":"tool_name","arguments":{...}}. Tools:\n' + json.dumps(tools, ensure_ascii=False))
        response = self.llm.invoke([SystemMessage(content=messages[0]["content"] + "\n" + instruction),
            HumanMessage(content=json.dumps(messages[1:], ensure_ascii=False))])
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        payload = json.loads(text)
        name = payload.get("tool")
        return {"role": "assistant", "content": payload.get("text", ""),
                "tool_calls": [{"id": str(uuid4()), "name": name,
                                "arguments": payload.get("arguments", {})}] if name else []}
