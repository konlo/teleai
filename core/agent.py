from typing import Sequence

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool


class SimpleCollectCallback(BaseCallbackHandler):
    """
    Collect intermediate events from the agent for UI surfacing.
    """

    def __init__(self) -> None:
        self.events = []

    def on_tool_error(self, error, **kwargs):
        self.events.append({"type": "tool_error", "error": str(error)})

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.events.append({"type": "llm_start", "prompts": prompts})

    def on_llm_end(self, response, **kwargs):
        self.events.append({"type": "llm_end"})


_PARSING_ERROR_GUIDANCE = (
    "올바른 형식으로 다시 응답해주세요. 반드시 아래 JSON 형식을 사용하세요:\n"
    "```\n"
    '{"action": "도구이름", "action_input": "입력값"}\n'
    "```\n"
    "또는 최종 답변이라면:\n"
    "```\n"
    '{"action": "Final Answer", "action_input": "최종 응답 내용"}\n'
    "```"
)


def build_agent(
    llm,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    history_getter,
    *,
    max_iterations: int = 10,
    max_execution_time: int = 120,
):
    """
    Build an AgentExecutor with message history support.
    Uses create_structured_chat_agent (JSON-based) for better
    compatibility with local LLMs that struggle with text-based ReAct format.
    """
    structured_runnable = create_structured_chat_agent(llm, tools, prompt=prompt)
    agent = AgentExecutor(
        agent=structured_runnable,
        tools=list(tools),
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        handle_parsing_errors=_PARSING_ERROR_GUIDANCE,
    )

    agent_with_history = RunnableWithMessageHistory(
        agent,
        history_getter,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent, agent_with_history


__all__ = ["SimpleCollectCallback", "build_agent", "StdOutCallbackHandler"]
