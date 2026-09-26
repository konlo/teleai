"""Conservatively focus model-visible tools for one grounded local scalar."""
from __future__ import annotations

from collections.abc import Mapping

from langchain.agents.middleware import AgentMiddleware


SCALAR_OPERATIONS = frozenset({"AVG", "SUM", "MEDIAN", "MIN", "MAX"})
SCALAR_TOOLS = frozenset({
    "list_analysis_context", "inspect_table_context", "inspect_dataset",
    "profile_dataset", "use_dataset", "aggregate_dataset", "local_analysis_sql",
    "search_analysis_tools", "read_analysis_skill",
})


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
