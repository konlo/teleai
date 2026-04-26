from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd


class DataReadinessDecision(str, Enum):
    USE_CURRENT = "USE_CURRENT"
    RELOAD_REQUIRED = "RELOAD_REQUIRED"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DataFrameState:
    role: str = ""
    source_table: str = ""
    query: str = ""
    columns: Tuple[str, ...] = ()
    row_count: int = 0
    is_preview: bool = False
    parent_query: str = ""
    created_by: str = ""


@dataclass(frozen=True)
class DataRequirement:
    columns: Tuple[str, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)
    task: str = ""
    source_table: str = ""
    min_rows: int = 1


@dataclass(frozen=True)
class DataReadinessResult:
    decision: DataReadinessDecision
    reason: str
    missing_columns: Tuple[str, ...] = ()
    state: Optional[DataFrameState] = None
    requirement: Optional[DataRequirement] = None


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


def normalize_columns(columns: Iterable[Any]) -> Tuple[str, ...]:
    seen: dict[str, None] = {}
    iterable = [] if columns is None else columns
    for column in iterable:
        name = str(column).strip()
        if name:
            seen.setdefault(name, None)
    return tuple(seen.keys())


def make_dataframe_state(
    df: Any,
    *,
    role: str,
    source_table: str = "",
    query: str = "",
    parent_query: str = "",
    created_by: str = "",
    is_preview: bool = False,
) -> DataFrameState:
    columns: Sequence[Any] = getattr(df, "columns", ())
    row_count = len(df) if isinstance(df, pd.DataFrame) else 0
    return DataFrameState(
        role=(role or "").strip(),
        source_table=(source_table or "").strip(),
        query=(query or "").strip(),
        columns=normalize_columns(columns),
        row_count=row_count,
        is_preview=bool(is_preview),
        parent_query=(parent_query or "").strip(),
        created_by=(created_by or "").strip(),
    )


def coerce_dataframe_state(value: Any) -> Optional[DataFrameState]:
    if isinstance(value, DataFrameState):
        return value
    if isinstance(value, dict):
        return DataFrameState(
            role=str(value.get("role", "") or ""),
            source_table=str(value.get("source_table", "") or ""),
            query=str(value.get("query", "") or ""),
            columns=normalize_columns(value.get("columns", ())),
            row_count=int(value.get("row_count", 0) or 0),
            is_preview=bool(value.get("is_preview", False)),
            parent_query=str(value.get("parent_query", "") or ""),
            created_by=str(value.get("created_by", "") or ""),
        )
    return None


def requirement_from_controlled_plan(plan: Any, *, min_rows: int = 1) -> DataRequirement:
    target_column = str(getattr(plan, "target_column", "") or "").strip()
    filters = getattr(plan, "filters", {}) or {}
    filter_columns = [str(column).strip() for column in filters.keys()]
    condition_columns = [
        str(getattr(condition, "column", "") or "").strip()
        for condition in (getattr(plan, "filter_conditions", ()) or ())
    ]
    return DataRequirement(
        columns=normalize_columns([target_column, *filter_columns, *condition_columns]),
        filters=filters,
        task=str(getattr(plan, "task", "") or ""),
        source_table=str(getattr(plan, "table", "") or ""),
        min_rows=min_rows,
    )


def evaluate_data_readiness(
    state: Optional[Any],
    requirement: DataRequirement,
) -> DataReadinessResult:
    coerced_state = coerce_dataframe_state(state)
    required_columns = normalize_columns(requirement.columns)
    requirement = DataRequirement(
        columns=required_columns,
        filters=requirement.filters,
        task=requirement.task,
        source_table=(requirement.source_table or "").strip(),
        min_rows=max(0, int(requirement.min_rows or 0)),
    )

    available_source = requirement.source_table or (
        coerced_state.source_table if coerced_state else ""
    )
    if requirement.task == "ranked_distribution":
        if available_source:
            return DataReadinessResult(
                decision=DataReadinessDecision.RELOAD_REQUIRED,
                reason="Ranked distribution requests must be recomputed from the source table.",
                state=coerced_state,
                requirement=requirement,
            )
        return DataReadinessResult(
            decision=DataReadinessDecision.FAIL,
            reason="Ranked distribution request needs a source table.",
            missing_columns=required_columns,
            state=coerced_state,
            requirement=requirement,
        )

    if coerced_state is None:
        if available_source:
            return DataReadinessResult(
                decision=DataReadinessDecision.RELOAD_REQUIRED,
                reason="No current dataframe state; reload from known source table.",
                missing_columns=required_columns,
                state=coerced_state,
                requirement=requirement,
            )
        return DataReadinessResult(
            decision=DataReadinessDecision.FAIL,
            reason="No current dataframe state and no source table is available.",
            missing_columns=required_columns,
            state=coerced_state,
            requirement=requirement,
        )

    if (
        requirement.source_table
        and coerced_state.source_table
        and requirement.source_table != coerced_state.source_table
    ):
        return DataReadinessResult(
            decision=DataReadinessDecision.RELOAD_REQUIRED,
            reason="Current dataframe source table differs from request source table.",
            missing_columns=required_columns,
            state=coerced_state,
            requirement=requirement,
        )

    if coerced_state.is_preview:
        if available_source:
            return DataReadinessResult(
                decision=DataReadinessDecision.RELOAD_REQUIRED,
                reason="Current dataframe is only a preview; reload full request data from source table.",
                state=coerced_state,
                requirement=requirement,
            )
        return DataReadinessResult(
            decision=DataReadinessDecision.FAIL,
            reason="Current dataframe is only a preview and source table is unknown.",
            state=coerced_state,
            requirement=requirement,
        )

    state_columns = set(coerced_state.columns)
    missing_columns = tuple(column for column in required_columns if column not in state_columns)
    if missing_columns:
        if available_source:
            return DataReadinessResult(
                decision=DataReadinessDecision.RELOAD_REQUIRED,
                reason="Current dataframe is missing request-required columns.",
                missing_columns=missing_columns,
                state=coerced_state,
                requirement=requirement,
            )
        return DataReadinessResult(
            decision=DataReadinessDecision.FAIL,
            reason="Current dataframe is missing required columns and source table is unknown.",
            missing_columns=missing_columns,
            state=coerced_state,
            requirement=requirement,
        )

    if coerced_state.row_count < requirement.min_rows:
        if available_source:
            return DataReadinessResult(
                decision=DataReadinessDecision.RELOAD_REQUIRED,
                reason="Current dataframe has fewer rows than required.",
                state=coerced_state,
                requirement=requirement,
            )
        return DataReadinessResult(
            decision=DataReadinessDecision.FAIL,
            reason="Current dataframe has too few rows and source table is unknown.",
            state=coerced_state,
            requirement=requirement,
        )

    return DataReadinessResult(
        decision=DataReadinessDecision.USE_CURRENT,
        reason="Current dataframe satisfies request-required data.",
        state=coerced_state,
        requirement=requirement,
    )


def resolve_source_table(
    state: Optional[Any],
    *,
    requirement_source: str = "",
    last_sql_table: str = "",
    selected_table: str = "",
) -> str:
    coerced_state = coerce_dataframe_state(state)
    return (
        (coerced_state.source_table if coerced_state else "")
        or (last_sql_table or "").strip()
        or (selected_table or "").strip()
        or (requirement_source or "").strip()
    )


def _format_sql_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _where_clause_from_filters(filters: Mapping[str, Any]) -> str:
    clauses = []
    for column, value in (filters or {}).items():
        column_name = str(column).strip()
        if not column_name or not _IDENTIFIER_RE.match(column_name):
            raise ValueError(f"Unsupported filter column: {column_name}")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            clauses.append(
                f"{column_name} BETWEEN {_format_sql_value(value[0])} AND {_format_sql_value(value[1])}"
            )
        else:
            clauses.append(f"{column_name} = {_format_sql_value(value)}")
    return f" WHERE {' AND '.join(clauses)}" if clauses else ""


def build_reload_sql_for_requirement(
    requirement: DataRequirement,
    source_table: str,
    *,
    limit: int = 2000,
) -> str:
    table = (source_table or requirement.source_table or "").strip()
    if not table or not _TABLE_RE.match(table):
        raise ValueError("Source table is not available or has an unsupported format.")
    columns = normalize_columns(requirement.columns)
    if not columns:
        raise ValueError("At least one required column is needed to build reload SQL.")
    for column in columns:
        if not _IDENTIFIER_RE.match(column):
            raise ValueError(f"Unsupported required column: {column}")
    where_sql = _where_clause_from_filters(requirement.filters)
    limit_value = max(1, int(limit))
    return f"SELECT {', '.join(columns)} FROM {table}{where_sql} LIMIT {limit_value}"


def format_dataframe_state_for_log(state: Optional[Any]) -> str:
    coerced_state = coerce_dataframe_state(state)
    if coerced_state is None:
        return "현재 데이터: none"
    source = coerced_state.source_table or "unknown"
    role = coerced_state.role or "unknown"
    preview = "preview" if coerced_state.is_preview else "loaded"
    return (
        f"현재 데이터: {role} | columns={list(coerced_state.columns)} | "
        f"source={source} | rows={coerced_state.row_count} | {preview}"
    )


__all__ = [
    "DataFrameState",
    "DataReadinessDecision",
    "DataReadinessResult",
    "DataRequirement",
    "build_reload_sql_for_requirement",
    "coerce_dataframe_state",
    "evaluate_data_readiness",
    "format_dataframe_state_for_log",
    "make_dataframe_state",
    "normalize_columns",
    "requirement_from_controlled_plan",
    "resolve_source_table",
]
