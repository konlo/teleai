"""Provider-independent tool contracts. No conversation runtime dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


TOOL_RESULT_STATUSES = (
    "ready",
    "planned",
    "awaiting_approval",
    "needs_context",
    "needs_refresh",
    "needs_data",
    "no_valid_chart",
    "unavailable",
    "rejected",
    "error",
)

COMMON_TOOL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(TOOL_RESULT_STATUSES)},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "error_code": {"type": ["string", "null"]},
        "retryable": {"type": "boolean"},
        "user_action": {"type": "string"},
        "scope": {"type": ["string", "object"]},
    },
    "required": ["status", "evidence_ids", "error_code", "retryable", "user_action", "scope"],
    "additionalProperties": True,
}


def _evidence_ids(result: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    dataset = result.get("dataset")
    if isinstance(dataset, dict) and dataset.get("id"):
        ids.append(str(dataset["id"]))
    for key in ("dataset_id", "loaded_dataset", "schema_fingerprint", "version"):
        if result.get(key):
            ids.append(str(result[key]))
    for item in result.get("cards", []) if isinstance(result.get("cards"), list) else []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return list(dict.fromkeys(ids))


def normalize_tool_result(result: dict[str, Any], *, default_status: str = "ready") -> dict[str, Any]:
    """Add the common envelope without hiding a tool's domain fields."""
    if not isinstance(result, dict):
        raise TypeError("분석 도구는 dict 결과를 반환해야 합니다.")
    normalized = dict(result)
    status = normalized.setdefault("status", default_status)
    if status not in TOOL_RESULT_STATUSES:
        raise ValueError(f"지원하지 않는 분석 도구 상태입니다: {status}")
    normalized.setdefault("evidence_ids", _evidence_ids(normalized))
    normalized.setdefault("error_code", None)
    normalized.setdefault("retryable", status in {"unavailable", "error"})
    normalized.setdefault(
        "user_action",
        "Databricks 조회 승인 여부를 결정해주세요." if status == "awaiting_approval" else "",
    )
    normalized.setdefault("scope", "도구가 반환한 구조화 결과 범위입니다.")
    return normalized


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    run: Callable[..., dict]
    output_schema: dict | None = None
    statuses: tuple[str, ...] = TOOL_RESULT_STATUSES

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "output_schema": self.output_schema or COMMON_TOOL_RESULT_SCHEMA,
            "statuses": list(self.statuses),
        }


@dataclass
class AnalysisToolContext:
    """Explicit services supplied by a runtime; never checkpoint this object."""
    datasets: object
    artifacts: dict
    reference_context: list
    propose_query: Callable[..., dict]
    max_join_rows: int = 100_000
    max_join_expansion_ratio: float = 5.0
    selected_dataset_id: str = ""
    semantic_resolver: object = None
