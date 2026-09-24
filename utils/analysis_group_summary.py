"""Bounded, declarative multi-metric summaries over loaded raw datasets."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd

from utils.analysis_datasets import Condition, DatasetStore, filter_frame


AGGREGATIONS = frozenset({
    "count", "mean", "sum", "median", "min", "max",
    "conditional_count", "conditional_percent", "conditional_mean",
})


def dataset_digest(frame: pd.DataFrame) -> str:
    normalized = frame.reset_index(drop=True)
    columns = json.dumps(list(normalized.columns), ensure_ascii=False).encode()
    values = pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    return sha256(columns + values).hexdigest()


def _validated_metrics(source: pd.DataFrame, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not 1 <= len(metrics) <= 8:
        raise ValueError("집계 지표는 1~8개여야 합니다.")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in metrics:
        if not isinstance(raw, dict):
            raise ValueError("각 집계 지표는 객체여야 합니다.")
        unexpected = set(raw) - {"name", "aggregation", "value_column", "condition", "empty_value"}
        if unexpected:
            raise ValueError("집계 지표에 지원하지 않는 필드가 있습니다.")
        name = raw.get("name")
        aggregation = raw.get("aggregation")
        value_column = raw.get("value_column", "")
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or name in names:
            raise ValueError("지표 이름은 중복 없는 1~80자 문자열이어야 합니다.")
        if aggregation not in AGGREGATIONS:
            raise ValueError("지원하지 않는 그룹 집계 방식입니다.")
        if value_column and value_column not in source.columns:
            raise ValueError("지표 값 컬럼은 현재 dataset의 실제 컬럼이어야 합니다.")
        numeric = {"mean", "sum", "median", "min", "max", "conditional_mean"}
        if aggregation in numeric:
            if (not value_column or not pd.api.types.is_numeric_dtype(source[value_column])
                    or pd.api.types.is_bool_dtype(source[value_column])):
                raise ValueError("수치 집계에는 bool이 아닌 수치형 value_column이 필요합니다.")
        elif aggregation in {"count", "conditional_count", "conditional_percent"} and value_column:
            raise ValueError(f"{aggregation}에는 value_column을 지정하지 않습니다.")

        condition = raw.get("condition")
        conditional = aggregation.startswith("conditional_")
        if conditional != (condition is not None):
            raise ValueError("조건부 집계에는 condition이 필요하고 일반 집계에는 사용할 수 없습니다.")
        condition_object = None
        if condition is not None:
            if not isinstance(condition, dict) or set(condition) != {"column", "op", "value"}:
                raise ValueError("condition은 column, op, value만 포함해야 합니다.")
            condition_object = Condition(**condition)
            if condition_object.column not in source.columns:
                raise ValueError("조건 컬럼은 현재 dataset의 실제 컬럼이어야 합니다.")
        empty_value = raw.get("empty_value")
        if empty_value is not None:
            if aggregation != "conditional_mean" or isinstance(empty_value, bool) or not isinstance(
                    empty_value, (int, float)) or not np.isfinite(empty_value):
                raise ValueError("empty_value는 conditional_mean의 유한 숫자로만 지정합니다.")
        names.add(name)
        normalized.append({
            "name": name,
            "aggregation": aggregation,
            "value_column": value_column,
            "condition": asdict(condition_object) if condition_object else None,
            "empty_value": empty_value,
        })
    return normalized


def summarize_groups(
    store: DatasetStore,
    dataset_id: str,
    *,
    group_columns: list[str],
    metrics: list[dict[str, Any]],
    conditions: list[dict[str, Any]] | None = None,
    sort: str = "group_ascending",
    max_groups: int = 1_000,
    max_output_rows: int = 1_000,
) -> dict[str, Any]:
    """Compute multiple bounded metrics without arbitrary generated code."""
    group_columns = list(group_columns)
    if not 1 <= len(group_columns) <= 3 or len(group_columns) != len(set(group_columns)):
        raise ValueError("그룹 컬럼은 중복 없이 1~3개여야 합니다.")
    if sort not in {"group_ascending", "group_descending", "none"}:
        raise ValueError("sort는 group_ascending, group_descending, none 중 하나여야 합니다.")
    if not 1 <= int(max_groups) <= 1_000 or not 1 <= int(max_output_rows) <= 1_000:
        raise ValueError("그룹과 출력 행 한도가 허용 범위를 벗어났습니다.")

    info = store.metadata[dataset_id]
    source = store.frames[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("그룹 요약은 집계되지 않은 raw dataset에서만 실행합니다.")
    if any(column not in source.columns for column in group_columns):
        raise ValueError("그룹 컬럼은 현재 dataset의 실제 컬럼이어야 합니다.")
    normalized_metrics = _validated_metrics(source, list(metrics))
    if set(group_columns) & {metric["name"] for metric in normalized_metrics}:
        raise ValueError("지표 이름은 그룹 컬럼명과 달라야 합니다.")

    condition_objects = tuple(Condition(**item) for item in (conditions or []))
    if any(condition.column not in source.columns for condition in condition_objects):
        raise ValueError("전체 필터 컬럼은 현재 dataset의 실제 컬럼이어야 합니다.")
    filtered = filter_frame(source, condition_objects) if condition_objects else source.copy()
    grouped_input = filtered.dropna(subset=group_columns)
    if grouped_input.empty:
        raise ValueError("조건 적용 후 그룹 요약할 행이 없습니다.")
    group_count = int(grouped_input.groupby(group_columns, observed=True, dropna=True).ngroups)
    if not 1 <= group_count <= int(max_groups):
        raise ValueError("그룹 수가 허용 범위를 벗어났습니다.")

    # Start from every observed group so conditional metrics cannot silently
    # remove groups whose numerator is zero.
    result = grouped_input[group_columns].drop_duplicates().reset_index(drop=True)
    metric_counts: dict[str, dict[str, int]] = {}
    for metric in normalized_metrics:
        name, aggregation = metric["name"], metric["aggregation"]
        value_column = metric["value_column"]
        condition = metric["condition"]
        if condition is not None:
            condition_object = Condition(**condition)
            selected = filter_frame(grouped_input, (condition_object,))
            metric_counts[name] = {
                "denominator_rows": len(grouped_input),
                "selected_rows": len(selected),
            }
            if aggregation == "conditional_percent":
                denominator = grouped_input.groupby(
                    group_columns, observed=True, dropna=True).size().rename("__denominator")
                numerator = selected.groupby(
                    group_columns, observed=True, dropna=True).size().rename("__numerator")
                values = (100.0 * numerator / denominator).fillna(0.0).rename(name).reset_index()
            elif aggregation == "conditional_count":
                values = selected.groupby(
                    group_columns, observed=True, dropna=True).size().rename(name).reset_index()
            else:
                values = selected.groupby(group_columns, observed=True, dropna=True)[
                    value_column].mean().rename(name).reset_index()
        elif aggregation == "count":
            values = grouped_input.groupby(
                group_columns, observed=True, dropna=True).size().rename(name).reset_index()
            metric_counts[name] = {"input_rows": len(grouped_input), "non_null_rows": len(grouped_input)}
        else:
            values = grouped_input.groupby(group_columns, observed=True, dropna=True)[
                value_column].agg(aggregation).rename(name).reset_index()
            metric_counts[name] = {
                "input_rows": len(grouped_input),
                "non_null_rows": int(grouped_input[value_column].notna().sum()),
            }
        result = result.merge(values, on=group_columns, how="left", validate="one_to_one")
        if aggregation in {"conditional_count", "conditional_percent"}:
            result[name] = result[name].fillna(0)
        elif aggregation == "conditional_mean" and metric["empty_value"] is not None:
            result[name] = result[name].fillna(metric["empty_value"])

    if sort != "none":
        result = result.sort_values(
            group_columns, ascending=sort == "group_ascending", kind="mergesort")
    result = result.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    if len(result) > int(max_output_rows):
        raise ValueError("그룹 요약 결과가 출력 행 한도를 넘습니다.")

    all_conditions = tuple(dict.fromkeys((*info.conditions, *condition_objects)))
    specification = {
        "group_columns": group_columns,
        "metrics": normalized_metrics,
        "conditions": [asdict(condition) for condition in condition_objects],
        "sort": sort,
    }
    derived = store.register(
        result,
        source=info.source,
        coverage=info.coverage,
        predicate_known=info.predicate_known,
        conditions=all_conditions,
        grain="aggregate",
        aggregation="group_summary:" + json.dumps(
            specification, ensure_ascii=False, sort_keys=True),
        snapshot=info.snapshot,
        query="GROUP SUMMARY " + json.dumps(
            specification, ensure_ascii=False, sort_keys=True),
        parent_id=dataset_id,
    )
    summary = {
        "kind": "dataset_group_summary",
        "parent_dataset_id": dataset_id,
        **specification,
        "source_rows": len(source),
        "filtered_rows": len(filtered),
        "group_input_rows": len(grouped_input),
        "dropped_group_rows": len(filtered) - len(grouped_input),
        "group_count": group_count,
        "output_rows": len(result),
        "output_columns": len(result.columns),
        "metric_counts": metric_counts,
        "data_sha256": dataset_digest(result),
    }
    return {
        "status": "ready",
        "dataset": asdict(derived),
        "group_summary_result": summary,
        "preview": json.loads(result.head(15).to_json(orient="records")),
        "scope": (
            f"{info.source}의 보유 raw dataset {len(source):,}행에서 조건 적용 {len(filtered):,}행, "
            f"그룹 키가 있는 {len(grouped_input):,}행을 {len(normalized_metrics):,}개 지표로 요약한 "
            f"{len(result):,}행 · coverage={info.coverage} · snapshot={info.snapshot or 'unknown'}"
        ),
    }
