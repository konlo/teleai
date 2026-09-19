"""Dataset provenance and conservative reuse checks for the analysis agent.

The model describes its request; this module checks what the cached data can
actually answer. Unsupported predicate implication requires a source query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd


@dataclass(frozen=True)
class Condition:
    column: str
    op: str
    value: Any

    def __post_init__(self):
        if self.op not in {"eq", "ne", "gt", "ge", "lt", "le", "in"}:
            raise ValueError("지원하지 않는 조건 연산자입니다.")
        if self.op == "in":
            if not isinstance(self.value, (list, tuple)):
                raise ValueError("in 조건에는 값 목록이 필요합니다.")
            object.__setattr__(self, "value", tuple(self.value))


def _implies(requested: Condition, stored: Condition) -> bool:
    if requested.column != stored.column:
        return False
    if requested == stored:
        return True
    try:
        if requested.op == "eq":
            checks = {"eq": lambda: requested.value == stored.value,
                      "ne": lambda: requested.value != stored.value,
                      "in": lambda: requested.value in stored.value,
                      "gt": lambda: requested.value > stored.value,
                      "ge": lambda: requested.value >= stored.value,
                      "lt": lambda: requested.value < stored.value,
                      "le": lambda: requested.value <= stored.value}
            return bool(checks[stored.op]())
        if requested.op == "in" and requested.value:
            return all(_implies(Condition(requested.column, "eq", v), stored)
                       for v in requested.value)
        if requested.op in {"gt", "ge"} and stored.op in {"gt", "ge"}:
            return (requested.value > stored.value or
                    (requested.value == stored.value and
                     (requested.op == "gt" or stored.op == "ge")))
        if requested.op in {"lt", "le"} and stored.op in {"lt", "le"}:
            return (requested.value < stored.value or
                    (requested.value == stored.value and
                     (requested.op == "lt" or stored.op == "le")))
    except (TypeError, ValueError):
        return False
    return False


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    source: str
    columns: tuple[str, ...]
    rows: int
    coverage: str = "unknown"  # complete for predicate, sampled, truncated, unknown
    conditions: tuple[Condition, ...] = ()  # conjunction only
    predicate_known: bool = False
    grain: str = "raw"
    aggregation: str = ""
    snapshot: str = ""
    query: str = ""
    parent_id: str = ""
    parent_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisNeed:
    source: str
    columns: tuple[str, ...]
    conditions: tuple[Condition, ...] = ()
    grain: str = "raw"
    aggregation: str = ""
    snapshot: str = ""
    current_result_only: bool = False


@dataclass(frozen=True)
class ReuseDecision:
    action: str
    reason: str


def assess_reuse(info: DatasetInfo, need: AnalysisNeed) -> ReuseDecision:
    def query(reason):
        return ReuseDecision("query_source", reason)
    if info.source != need.source:
        return query("요청한 데이터 출처가 현재 결과와 다릅니다.")
    if need.snapshot and need.snapshot != info.snapshot:
        return query("요청한 데이터 버전이 현재 결과와 다릅니다.")
    if not set(need.columns).issubset(info.columns):
        return query("현재 결과에 필요한 컬럼이 없습니다.")
    if info.grain != need.grain or info.aggregation != need.aggregation:
        return query("현재 집계 수준으로 요청한 분석을 정확하게 계산할 수 없습니다.")
    if not need.current_result_only:
        if info.coverage != "complete" or not info.predicate_known:
            return query("현재 결과가 요청 범위 전체를 포함하는지 확인할 수 없습니다.")
        if not all(any(_implies(wanted, old) for wanted in need.conditions)
                   for old in info.conditions):
            return query("요청 범위가 현재 데이터 범위 안에 포함된다고 확인할 수 없습니다.")
    residual = tuple(c for c in need.conditions if c not in info.conditions)
    if any(c.column not in info.columns for c in residual):
        return query("추가 필터를 계산할 컬럼이 현재 결과에 없습니다.")
    if residual:
        if info.grain != "raw":
            return query("집계 결과에 새로운 원본 필터를 적용할 수 없습니다.")
        return ReuseDecision("derive_local", "현재 데이터 안에서 조건을 좁힐 수 있습니다.")
    return ReuseDecision("reuse", "현재 결과를 그대로 사용할 수 있습니다.")


class InvalidConditionValue(ValueError):
    """A type mismatch must not silently become a zero-row population."""
    def __init__(self, column, dtype, examples):
        super().__init__('Filter literal type is incompatible with observed column type')
        self.column, self.dtype, self.examples = column, str(dtype), examples


def filter_frame(frame: pd.DataFrame, conditions: tuple[Condition, ...]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        series = frame[condition.column]
        value = condition.value
        observed = series.head(256).dropna().tolist()
        requested = list(value) if condition.op == 'in' else [value]
        if observed and all(isinstance(item, str) for item in observed) and any(not isinstance(item, str) for item in requested):
            examples = list(dict.fromkeys(item for item in observed if len(item) <= 128))[:10]
            raise InvalidConditionValue(condition.column, series.dtype, examples)
        ops = {"eq": series.eq, "ne": series.ne, "gt": series.gt,
               "ge": series.ge, "lt": series.lt, "le": series.le,
               "in": series.isin}
        # SQL-style comparisons never include nulls, including !=.
        mask &= ops[condition.op](value).fillna(False) & series.notna()
    return frame.loc[mask].copy()


@dataclass
class DatasetStore:
    """References to owned immutable inputs; callers must not mutate frames."""
    frames: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    metadata: dict[str, DatasetInfo] = field(default_factory=dict)

    def register(self, frame: pd.DataFrame, *, source: str, **provenance) -> DatasetInfo:
        if any(not isinstance(c, str) for c in frame.columns) or not frame.columns.is_unique:
            raise ValueError("데이터 컬럼 이름은 중복 없는 문자열이어야 합니다.")
        coverage = provenance.get("coverage", "unknown")
        if coverage not in {"complete", "unknown", "sampled", "truncated"}:
            raise ValueError("지원하지 않는 coverage입니다.")
        info = DatasetInfo(str(uuid4()), source, tuple(frame.columns), len(frame), **provenance)
        self.frames[info.id] = frame
        self.metadata[info.id] = info
        return info

    def derive(self, dataset_id: str, need: AnalysisNeed) -> DatasetInfo:
        info = self.metadata[dataset_id]
        decision = assess_reuse(info, need)
        if decision.action == "query_source":
            raise ValueError(decision.reason)
        if decision.action == "reuse":
            return info
        residual = tuple(c for c in need.conditions if c not in info.conditions)
        frame = filter_frame(self.frames[dataset_id], residual)
        return self.register(frame, source=info.source, coverage=info.coverage,
            predicate_known=info.predicate_known,
            conditions=tuple(dict.fromkeys((*info.conditions, *need.conditions))),
            grain=info.grain, aggregation=info.aggregation, snapshot=info.snapshot,
            parent_id=dataset_id)
