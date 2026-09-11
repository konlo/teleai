"""Model-independent conversation/tool loop; no UI or provider calls here.

Adapters supply complete assistant messages using role/content/tool_calls.
The transcript, including observations, stays in this object between requests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from core.analysis_approval import ApprovalQueue


from core.analysis_tool_contract import ToolDefinition


@dataclass
class AnalysisSession:
    id: str
    instructions: str
    tools: list[ToolDefinition]
    max_steps: int = 12
    history: list[dict] = field(default_factory=list)
    state: str = "idle"
    last_goal: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict, repr=False)
    reference_context: list[dict] = field(default_factory=list)
    context_provider: Callable[[], dict] | None = field(default=None, repr=False)
    approvals: ApprovalQueue = field(init=False)

    def __post_init__(self):
        self.approvals = ApprovalQueue(self.id)
        if len({t.name for t in self.tools}) != len(self.tools):
            raise ValueError("도구 이름이 중복되었습니다.")

    def submit(self, text: str, model: Callable[[list[dict], list[dict]], dict]) -> dict:
        if self.state == "running":
            raise RuntimeError("현재 분석이 진행 중입니다.")
        if not text.strip():
            raise ValueError("요청을 입력해주세요.")
        self.approvals.advance()
        self.last_goal = text
        self.history.append({"role": "user", "content": text})
        return self._drive(model)

    def resume_after_approval(self, request_id: str, *, execute: Callable,
                              model: Callable, approved: bool) -> dict:
        if self.state != "awaiting_approval":
            raise ValueError("현재 승인 대기 중인 분석이 아닙니다.")
        if approved:
            self.approvals.approve(request_id)
            try:
                result = self.approvals.execute(request_id, execute)
            except Exception:
                result = {"status": "failed", "message": self.approvals.get(request_id).error}
        else:
            self.approvals.decline(request_id)
            result = {"status": "declined", "message": "사용자가 조회를 거절했습니다. 기존 데이터만 사용하세요."}
        # Preserve user decision and actual execution observation, not just 'yes'.
        self.history.append({"role": "user", "content": json.dumps(
            {"approval_request_id": request_id, "approved": approved, "result": result},
            ensure_ascii=False, default=str)})
        return self._drive(model)

    def _drive(self, model: Callable) -> dict:
        self.state = "running"
        tools = {tool.name: tool for tool in self.tools}
        try:
            for _ in range(self.max_steps):
                instructions = self.instructions
                if self.context_provider:
                    instructions += "\n현재 분석 환경:\n" + json.dumps(self.context_provider(), ensure_ascii=False, default=str)
                response = model(
                    [{"role": "system", "content": instructions}, *self.history],
                    [tool.schema() for tool in self.tools])
                if response.get("role") != "assistant":
                    raise ValueError("모델 어댑터가 올바른 assistant 메시지를 반환하지 않았습니다.")
                calls = response.get("tool_calls") or []
                ids = [call.get("id") for call in calls]
                if any(not i for i in ids) or len(set(ids)) != len(ids):
                    raise ValueError("도구 호출 ID가 없거나 중복되었습니다.")
                self.history.append(response)
                if not calls:
                    self.state = "idle"
                    return {"status": "answered", "text": response.get("content", "")}
                pending = False
                for call in calls:
                    if pending:
                        result = {"status": "deferred", "message": "조회 승인 후 다시 판단하세요."}
                    elif call.get("name") not in tools:
                        result = {"status": "error", "message": "사용할 수 없는 도구입니다."}
                    else:
                        try:
                            result = tools[call["name"]].run(**call.get("arguments", {}))
                        except (ValueError, KeyError, TypeError) as exc:
                            result = {"status": "error", "message": str(exc)[:500]}
                        except Exception:
                            result = {"status": "error", "message": "도구 실행이 실패했습니다."}
                    self.history.append({"role": "tool", "tool_call_id": call["id"],
                        "name": call["name"], "content": json.dumps(result, ensure_ascii=False, default=str)})
                    pending |= result.get("status") == "awaiting_approval"
                if pending:
                    self.state = "awaiting_approval"
                    return {"status": self.state,
                            "requests": [r.id for r in self.approvals.pending()]}
            self.state = "limit_reached"
            return {"status": self.state, "text": "분석 실행 한도에 도달했습니다. 완료된 결과를 확인하고 범위를 좁혀주세요."}
        except Exception:
            self.state = "failed"
            raise
