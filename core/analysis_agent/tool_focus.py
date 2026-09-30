"""Conservatively focus model-visible tools for one grounded local scalar."""
from __future__ import annotations

from collections.abc import Mapping

from langchain.agents.middleware import AgentMiddleware


SCALAR_OPERATIONS = frozenset({"AVG", "SUM", "MEDIAN", "MIN", "MAX"})


class ExplanationToolsMiddleware(AgentMiddleware):
    """Explicit explanation-only turns cannot execute analysis tools."""
    def wrap_model_call(self, request, handler):
        if not (request.state.get('recovery') or {}).get('explanation_only'):
            return handler(request)
        from langchain_core.messages import SystemMessage
        content = (str(request.system_message.content)+'\n' if request.system_message else '')
        content += '이번 요청은 설명만 합니다. 도구 실행·데이터 계산·차트 생성 없이 질문의 개념을 설명하세요. 실제 분석을 수행했다고 말하지 마세요.'
        return handler(request.override(tools=[],system_message=SystemMessage(content=content)))
SCALAR_TOOLS = frozenset({
    "list_analysis_context", "inspect_table_context", "inspect_dataset",
    "profile_dataset", "use_dataset", "aggregate_dataset", "local_analysis_sql",
    "prepare_numeric_dataset",
    "search_analysis_tools", "read_analysis_skill",
})


class FocusedRemoteCatalogToolsMiddleware(AgentMiddleware):
    """Expose discovery/query tools for a known catalog read, not EDA tools."""

    def __init__(self, context, diagnostics=None):
        self.context, self.diagnostics = context, diagnostics

    def wrap_model_call(self, request, handler):
        from core.analysis_agent.remote_completion import catalog_read_requested
        from core.analysis_catalog import _source_key
        from langchain_core.messages import SystemMessage
        current = request.state.get('recovery') or {}
        if not catalog_read_requested(current):
            return handler(request)
        wanted = {_source_key(s) for s in current.get('required_sources', [])}
        known = {_source_key(c.get('table', '')) for c in self.context.reference_context}
        loaded = {_source_key(info.source) for info in self.context.datasets.metadata.values()}
        if not wanted.issubset(known) or wanted & loaded:
            return handler(request)
        names = {'query_databricks', 'inspect_table_context', 'list_analysis_context', 'plan_source_discovery'}
        selected = [t for t in request.tools if getattr(t, 'name', '') in names]
        if not any(t.name == 'query_databricks' for t in selected):
            return handler(request)
        instruction = ('현재 요청은 확인된 catalog의 실제 목록 조회입니다. 로컬 분석 도구는 필요하지 않습니다. '
            '확인된 출처와 컬럼으로 query_databricks를 호출하여 결과를 얻으세요. '
            '현재 원격 실행 정책을 따르세요. SQL을 문장으로만 제시하고 종료하지 마세요.')
        content = (str(request.system_message.content) + '\n' if request.system_message else '') + instruction
        if self.diagnostics:
            self.diagnostics.emit('model_tools_focused', mode='remote_catalog', tool_count=len(selected))
        return handler(request.override(tools=selected, system_message=SystemMessage(content=content)))

REMOTE_JOIN_TOOLS = frozenset({
    'list_analysis_context', 'inspect_table_context', 'inspect_table_relationships',
    'inspect_column_definitions', 'query_databricks', 'search_analysis_tools',
    'read_analysis_skill',
})


class FocusedRemoteJoinToolsMiddleware(AgentMiddleware):
    """Focus remote scalar joins only while a required raw source is absent.

    Discovery, SQL approval and error-driven SQL repair remain available. Once
    both sources are loaded, local alternatives become visible again. This
    changes the menu, never execution permissions or SQL correctness checks.
    """

    def __init__(self, context, diagnostics=None):
        self.context = context
        self.diagnostics = diagnostics

    def wrap_model_call(self, request, handler):
        from core.analysis_catalog import _source_key, table_context_freshness
        current = request.state.get('recovery') or {}
        tools = request.tools or ()
        sources = current.get('required_sources') or []
        scope = current.get('scope') or {}
        if (not current.get('requested_join') or not current.get('calculation')
                or current.get('chart') or current.get('data_load')
                or current.get('evidence_ids') or scope.get('unresolved')
                or not 2 <= len(sources) <= 4
                or not any(getattr(t, 'name', '') == 'query_databricks' for t in tools)):
            return handler(request)
        wanted = {_source_key(s) for s in sources}
        known = {_source_key(item.get('table', '')) for item in self.context.reference_context
                 if table_context_freshness(item) == 'fresh'}
        loaded = {_source_key(info.source) for info in self.context.datasets.metadata.values()
                  if info.grain == 'raw' and info.coverage == 'complete' and info.predicate_known}
        if not wanted.issubset(known) or wanted.issubset(loaded):
            return handler(request)
        selected = [t for t in tools if getattr(t, 'name', '') in REMOTE_JOIN_TOOLS]
        if self.diagnostics is not None:
            self.diagnostics.emit('model_tools_focused', mode='remote_join', tool_count=len(selected))
        from langchain_core.messages import SystemMessage
        instruction = ('현재 요청은 여러 테이블의 통계이며 필요한 원본 일부가 로컬에 없습니다. '
            '현재 제공된 도구만 사용하세요. 최신 스키마와 선언된 키 관계, 요청 조건을 확인한 다음 '
            'query_databricks로 JOIN 집계 SELECT를 실행하세요. 이 도구는 현재 연결된 SQL 엔진에 실행됩니다. '
            '이미 확인한 스키마를 반복 탐색하거나 없는 로컬 dataset ID를 만들지 마세요. '
            '키나 업무 역할이 모호하면 해당 정보만 확인하세요.')
        content = (str(request.system_message.content) + '\n' if request.system_message else '') + instruction
        return handler(request.override(tools=selected, system_message=SystemMessage(content=content)))


class FocusedScalarToolsMiddleware(AgentMiddleware):
    """Keep a small tool menu only when the stored local scope is unambiguous.

    All registered tools remain available to the runtime. For ambiguous,
    filtered, remote, or multi-operation requests the original menu is used.
    """

    def __init__(self, context, diagnostics=None):
        self.context = context
        self.diagnostics = diagnostics

    def wrap_model_call(self, request, handler):
        state = request.state if isinstance(request.state, Mapping) else {}
        current = state.get("recovery") or {}
        if not isinstance(current, Mapping) or not self._safe_local_scalar(current):
            return handler(request)
        selected = [tool for tool in (request.tools or ())
                    if getattr(tool, "name", None) in SCALAR_TOOLS]
        if len(selected) != len(SCALAR_TOOLS):
            return handler(request)
        if self.diagnostics is not None:
            self.diagnostics.emit("model_tools_focused", tool_count=len(selected),
                                  operation=current["operations"][0])
        return handler(request.override(tools=selected))

    def _safe_local_scalar(self, current: Mapping) -> bool:
        if (not current.get("calculation") or current.get("chart")
                or current.get("data_load") or current.get("fresh_source_required")
                or current.get("current_result_only") or current.get("join")
                or current.get("pivot_requested") or current.get("statistical_kind")
                or current.get("group_summary_requested") or current.get("outlier_spec")
                or current.get("winsor_spec") or current.get("time_series_frequency")
                or current.get("evidence_ids")):
            return False
        operations = current.get("operations") or []
        if len(operations) != 1 or operations[0] not in SCALAR_OPERATIONS:
            return False
        scope = current.get("scope") or {}
        if any(scope.get(key) for key in (
                "conditions", "any_conditions", "measure_conditions", "unresolved")):
            return False
        columns = current.get("required_columns") or []
        if len(columns) != 1 or not isinstance(columns[0], str):
            return False
        candidates = [info for info in self.context.datasets.metadata.values()
                      if info.grain == "raw" and info.coverage == "complete"
                      and info.predicate_known and columns[0] in info.columns]
        required_sources = current.get("required_sources") or []
        if required_sources:
            wanted = {str(source).casefold() for source in required_sources}
            candidates = [info for info in candidates if info.source.casefold() in wanted]
        return len(candidates) == 1
