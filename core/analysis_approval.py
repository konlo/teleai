"""One-shot, explicit approval for a concrete remote data operation.

Only UI/controller code may call approve/decline. Never expose these methods
as agent tools. The executor receives the immutable approved request, rather
than a replacement query supplied after approval.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, Callable
from uuid import uuid4


@dataclass(frozen=True)
class DataRequest:
    id: str
    session_id: str
    revision: int
    source: str
    query: str
    reason: str
    goal: str
    status: str = "proposed"
    error: str = ""


class ApprovalQueue:
    """Session-owned requests. Claiming an approval is atomic across reruns."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.revision = 0
        self._requests: dict[str, DataRequest] = {}
        self._lock = RLock()

    def advance(self) -> None:
        """A new user instruction invalidates unexecuted proposals/approvals."""
        with self._lock:
            self.revision += 1
            for key, request in self._requests.items():
                if request.status in {"proposed", "approved"}:
                    self._requests[key] = replace(request, status="invalidated")

    def propose(self, *, source: str, query: str, reason: str, goal: str) -> DataRequest:
        if not source.strip() or not query.strip() or not reason.strip():
            raise ValueError("조회 대상, SQL, 추가 조회 이유가 필요합니다.")
        with self._lock:
            # A model retry must not produce duplicate approval cards.
            for request in self._requests.values():
                if (request.revision == self.revision and request.status == "proposed"
                        and request.source == source and request.query == query):
                    return request
            request = DataRequest(str(uuid4()), self.session_id, self.revision,
                                  source, query, reason, goal)
            self._requests[request.id] = request
            return request

    def get(self, request_id: str) -> DataRequest:
        with self._lock:
            return self._requests[request_id]

    def pending(self) -> list[DataRequest]:
        with self._lock:
            return [r for r in self._requests.values() if r.status == "proposed"]

    def approve(self, request_id: str) -> None:
        with self._lock:
            request = self._requests[request_id]
            if request.status != "proposed" or request.revision != self.revision:
                raise ValueError("이미 처리되었거나 변경된 조회 요청입니다.")
            self._requests[request_id] = replace(request, status="approved")

    def decline(self, request_id: str) -> None:
        with self._lock:
            request = self._requests[request_id]
            if request.status not in {"proposed", "approved"}:
                raise ValueError("이 요청은 취소 가능한 대기 상태가 아닙니다.")
            self._requests[request_id] = replace(request, status="declined")

    def execute(self, request_id: str, executor: Callable[[DataRequest], Any]) -> Any:
        with self._lock:
            request = self._requests[request_id]
            if request.status != "approved" or request.revision != self.revision:
                raise PermissionError("이 조회에 대한 사용자 승인이 필요합니다.")
            running = replace(request, status="executing")
            self._requests[request_id] = running
        try:
            result = executor(running)
        except Exception:
            with self._lock:
                # Backend exception strings may contain credentials or row values.
                self._requests[request_id] = replace(running, status="failed",
                    error="조회가 완료되지 않았습니다. 실행 상태를 확인한 뒤 재승인해주세요.")
            raise
        with self._lock:
            self._requests[request_id] = replace(running, status="completed")
        return result
