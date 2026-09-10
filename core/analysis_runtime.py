"""UI-facing runtime boundary; the current loop is one implementation.

Persistence and LangGraph will implement this interface in a separate stage.
The UI never approves, executes, or mutates conversation state directly.
"""
from copy import deepcopy
from dataclasses import asdict
from typing import Protocol

from core.analysis_sql import validate_query


class AnalysisRuntime(Protocol):
    def submit(self, text, model): ...
    def inspect(self) -> dict: ...
    def respond(self, request_id, *, approved, execute, model): ...
    def cancel(self, request_id): ...
    def propose_table(self, table): ...
    def select_chart(self, card_id): ...
    def events(self) -> list[dict]: ...


class CurrentAnalysisRuntime:
    version = "analysis-session-v1"

    def __init__(self, session):
        self.session = session

    def submit(self, text, model):
        return self.session.submit(text, model)

    def inspect(self):
        return {"id": self.session.id, "runtime_version": self.version,
                "state": self.session.state,
                "requests": [asdict(r) for r in self.session.approvals.pending()]}

    def events(self):
        # Renderers cannot modify the authoritative transcript through this snapshot.
        return deepcopy(self.session.history)

    def respond(self, request_id, *, approved, execute, model):
        return self.session.resume_after_approval(request_id, approved=approved,
                                                 execute=execute, model=model)

    def cancel(self, request_id):
        if self.session.state == "running":
            raise ValueError("실행 중인 작업은 대기 취소로 처리할 수 없습니다.")
        self.session.approvals.decline(request_id)
        self.session.state = "awaiting_approval" if self.session.approvals.pending() else "idle"
        self.session.history.append({"role": "user", "content":
            "추가 데이터 조회를 취소했어요. 기존 데이터만 사용해주세요."})

    def propose_table(self, table):
        if self.session.state == "running":
            raise ValueError("현재 분석이 진행 중입니다.")
        parts = table.strip().split(".")
        if not 1 <= len(parts) <= 3 or any(not p.strip() for p in parts):
            raise ValueError("catalog.schema.table 형태를 확인해주세요.")
        identifier = ".".join("`" + p.replace("`", "``") + "`" for p in parts)
        query = f"SELECT * FROM {identifier} LIMIT 10000"
        validate_query(query)
        # Validate before invalidating a previous, valid proposal.
        self.session.approvals.advance()
        self.session.last_goal = f"{table} 데이터 살펴보기"
        request = self.session.approvals.propose(source=table, query=query,
            reason="데이터 구조와 예시를 살펴볼 최대 10,000행을 가져옵니다. 전체 통계용 데이터가 아닙니다.",
            goal=self.session.last_goal)
        self.session.state = "awaiting_approval"
        return request.id

    def select_chart(self, card_id):
        if self.session.state in {"running", "awaiting_approval"}:
            raise ValueError("현재 작업이 끝난 뒤 차트를 선택해주세요.")
        card = self.session.artifacts[card_id]
        self.session.history.extend([
            {"role": "user", "content": f"이 차트를 선택했어요: {card.title}. 결과 ID={card.dataset_id}, 차트 ID={card.id}"},
            {"role": "assistant", "content": "선택한 차트를 크게 표시했습니다. 조건이나 비교 방법을 이어서 말씀해주세요."}])
        return card.id
