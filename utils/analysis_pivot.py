"""Bounded, schema-neutral pivot tables for loaded raw datasets."""
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
    "success_rate", "overall_percent",
})
MULTI_AGGREGATIONS = frozenset({"count", "mean", "sum", "median", "min", "max"})
MONTH_ORDER = {
    name: index for index, names in enumerate((
        ("jan", "january"), ("feb", "february"), ("mar", "march"),
        ("apr", "april"), ("may",), ("jun", "june"),
        ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
        ("oct", "october"), ("nov", "november"), ("dec", "december"),
    )) for name in names
}


def dataset_digest(frame: pd.DataFrame) -> str:
    normalized = frame.reset_index(drop=True)
    columns = json.dumps(list(normalized.columns), ensure_ascii=False).encode()
    values = pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    return sha256(columns + values).hexdigest()


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flattened = frame.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            " | ".join(str(part) for part in column if str(part) not in {"", "None"})
            for column in flattened.columns.to_flat_index()
        ]
    else:
        flattened.columns = [str(column) for column in flattened.columns]
    if len(flattened.columns) != len(set(flattened.columns)):
        raise ValueError("피벗 결과의 평탄화된 컬럼명이 중복됩니다.")
    return flattened


def _calendar_sort(frame: pd.DataFrame, index_column: str) -> pd.DataFrame:
    tokens = frame[index_column].astype(str).str.strip().str.casefold()
    if tokens.empty or not set(tokens).issubset(MONTH_ORDER):
        raise ValueError("calendar_month 정렬은 영문 월 값에만 사용할 수 있습니다.")
    order = tokens.map(MONTH_ORDER)
    return frame.assign(__month_order=order).sort_values(
        "__month_order", kind="mergesort").drop(columns="__month_order").reset_index(drop=True)


def _normalize_aggregation(aggregation: str | list[str]) -> str | list[str]:
    if isinstance(aggregation, str):
        if aggregation not in AGGREGATIONS:
            raise ValueError("지원하지 않는 피벗 집계 방식입니다.")
        return aggregation
    if (not isinstance(aggregation, list) or not 2 <= len(aggregation) <= 6
            or len(aggregation) != len(set(aggregation))
            or any(item not in MULTI_AGGREGATIONS for item in aggregation)):
        raise ValueError("복수 피벗 집계는 중복 없는 count, mean, sum, median, min, max 2~6개여야 합니다.")
    return list(aggregation)


def _apply_derived_bins(
    frame: pd.DataFrame, derived_bins: list[dict[str, Any]]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if len(derived_bins) > 2:
        raise ValueError("파생 구간 축은 최대 2개까지 허용합니다.")
    working = frame.copy()
    normalized: list[dict[str, Any]] = []
    outputs: set[str] = set()
    for raw in derived_bins:
        if not isinstance(raw, dict) or set(raw) != {
                "source_column", "output_column", "cut_points", "labels"}:
            raise ValueError("파생 구간은 source_column, output_column, cut_points, labels만 포함해야 합니다.")
        source_column, output_column = raw["source_column"], raw["output_column"]
        cut_points, labels = raw["cut_points"], raw["labels"]
        if source_column not in working.columns:
            raise ValueError("파생 구간의 원본 컬럼은 현재 dataset에 있어야 합니다.")
        if (not isinstance(output_column, str) or not output_column or len(output_column) > 80
                or output_column in working.columns or output_column in outputs):
            raise ValueError("파생 구간 출력 컬럼은 새롭고 중복 없는 1~80자 이름이어야 합니다.")
        if (not pd.api.types.is_numeric_dtype(working[source_column])
                or pd.api.types.is_bool_dtype(working[source_column])):
            raise ValueError("파생 구간의 원본 컬럼은 bool이 아닌 수치형이어야 합니다.")
        if (not isinstance(cut_points, list) or not 1 <= len(cut_points) <= 20
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) for value in cut_points)
                or cut_points != sorted(set(cut_points))):
            raise ValueError("cut_points는 오름차순의 중복 없는 유한 숫자 1~20개여야 합니다.")
        if (not isinstance(labels, list) or len(labels) != len(cut_points) + 1
                or len(labels) != len(set(labels)) or any(
                    not isinstance(label, str) or not label or len(label) > 40 for label in labels)):
            raise ValueError("labels는 구간 수와 맞는 중복 없는 1~40자 문자열이어야 합니다.")
        working[output_column] = pd.cut(
            pd.to_numeric(working[source_column], errors="coerce"),
            bins=[-np.inf, *cut_points, np.inf],
            labels=labels,
            right=False,
            ordered=True,
        )
        outputs.add(output_column)
        normalized.append({
            "source_column": source_column,
            "output_column": output_column,
            "cut_points": list(cut_points),
            "labels": list(labels),
        })
    return working, normalized


def pivot_dataset(
    store: DatasetStore,
    dataset_id: str,
    *,
    index_columns: list[str],
    column_columns: list[str],
    aggregation: str | list[str],
    value_column: str = "",
    success_value: Any = None,
    conditions: list[dict[str, Any]] | None = None,
    derived_bins: list[dict[str, Any]] | None = None,
    margins: bool = False,
    margins_name: str = "전체",
    sort: str = "ascending",
    max_axis_values: int = 100,
    max_output_cells: int = 1_000,
) -> dict[str, Any]:
    """Create a bounded pivot result without arbitrary generated code."""
    index_columns = list(index_columns)
    column_columns = list(column_columns)
    axes = index_columns + column_columns
    if not 1 <= len(index_columns) <= 3 or not 1 <= len(column_columns) <= 2:
        raise ValueError("피벗은 행 축 1~3개와 열 축 1~2개를 요구합니다.")
    if len(axes) != len(set(axes)) or len(axes) > 4:
        raise ValueError("피벗 축은 중복 없이 최대 4개까지만 허용합니다.")
    aggregation = _normalize_aggregation(aggregation)
    if sort not in {"ascending", "descending", "calendar_month"}:
        raise ValueError("sort는 ascending, descending, calendar_month 중 하나여야 합니다.")
    if not 1 <= int(max_axis_values) <= 100 or not 1 <= int(max_output_cells) <= 10_000:
        raise ValueError("피벗 축 또는 출력 셀 한도가 허용 범위를 벗어났습니다.")
    if not isinstance(margins_name, str) or not margins_name.strip() or len(margins_name) > 40:
        raise ValueError("margins_name은 1~40자의 문자열이어야 합니다.")

    info = store.metadata[dataset_id]
    source = store.frames[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("피벗은 집계되지 않은 raw dataset에서만 실행합니다.")
    condition_objects = tuple(Condition(**item) for item in (conditions or []))
    condition_columns = [condition.column for condition in condition_objects]
    raw_derived_bins = list(derived_bins or [])
    derived_outputs = {item.get("output_column") for item in raw_derived_bins if isinstance(item, dict)}
    derived_sources = {item.get("source_column") for item in raw_derived_bins if isinstance(item, dict)}
    requested = [column for column in axes if column not in derived_outputs]
    requested += ([value_column] if value_column else []) + condition_columns + list(derived_sources)
    if any(column not in source.columns for column in requested):
        raise ValueError("피벗 축·값·조건 컬럼은 현재 dataset의 실제 컬럼이어야 합니다.")
    if any(output not in axes for output in derived_outputs):
        raise ValueError("파생 구간 출력 컬럼은 피벗 축으로 사용해야 합니다.")
    aggregations = [aggregation] if isinstance(aggregation, str) else aggregation
    if any(item in {"mean", "sum", "median", "min", "max", "success_rate"}
           for item in aggregations) and not value_column:
        raise ValueError("요청한 피벗 집계에는 value_column이 필요합니다.")
    if any(item in {"mean", "sum", "median", "min", "max"} for item in aggregations) and (
        not pd.api.types.is_numeric_dtype(source[value_column])
        or pd.api.types.is_bool_dtype(source[value_column])
    ):
        raise ValueError("수치 피벗의 value_column은 bool이 아닌 수치형이어야 합니다.")
    if aggregation == "success_rate" and success_value is None:
        raise ValueError("success_rate에는 명시적인 success_value가 필요합니다.")
    if aggregation != "success_rate" and success_value is not None:
        raise ValueError("success_value는 success_rate에서만 지정합니다.")

    filtered = filter_frame(source, condition_objects) if condition_objects else source.copy()
    filtered, normalized_derived_bins = _apply_derived_bins(filtered, raw_derived_bins)
    if filtered.empty:
        raise ValueError("조건 적용 후 피벗할 행이 없습니다.")
    for column in axes:
        cardinality = int(filtered[column].nunique(dropna=True))
        if not 1 <= cardinality <= int(max_axis_values):
            raise ValueError(f"피벗 축 {column}의 고유값 수가 허용 범위를 벗어났습니다.")

    needed = axes + ([value_column] if value_column else [])
    complete = filtered.dropna(subset=needed)
    if complete.empty:
        raise ValueError("피벗에 사용할 완전한 관측값이 없습니다.")
    if aggregation == "count":
        matrix = pd.crosstab(
            [complete[column] for column in index_columns],
            [complete[column] for column in column_columns],
            margins=bool(margins), margins_name=margins_name, dropna=True,
        )
    elif aggregation == "overall_percent":
        matrix = pd.crosstab(
            [complete[column] for column in index_columns],
            [complete[column] for column in column_columns],
            normalize="all", margins=bool(margins), margins_name=margins_name,
            dropna=True,
        ) * 100.0
    else:
        working = complete.copy()
        pivot_value = value_column
        aggfunc = aggregation
        if aggregation == "success_rate":
            pivot_value = "__success_rate_value"
            while pivot_value in working.columns:
                pivot_value = "_" + pivot_value
            working[pivot_value] = (working[value_column] == success_value).astype(float)
            aggfunc = "mean"
        matrix = working.pivot_table(
            index=index_columns,
            columns=column_columns,
            values=pivot_value,
            aggfunc=aggfunc,
            margins=bool(margins),
            margins_name=margins_name,
            observed=True,
            dropna=True,
        )
        if aggregation == "success_rate":
            matrix = matrix * 100.0

    result = _flatten_columns(matrix.reset_index())
    if sort == "calendar_month":
        if len(index_columns) != 1:
            raise ValueError("calendar_month 정렬은 단일 행 축에서만 사용할 수 있습니다.")
        if margins:
            margin_mask = result[index_columns[0]].astype(str).eq(margins_name)
            body = _calendar_sort(result.loc[~margin_mask], index_columns[0])
            result = pd.concat([body, result.loc[margin_mask]], ignore_index=True)
        else:
            result = _calendar_sort(result, index_columns[0])
    elif sort in {"ascending", "descending"}:
        if margins and len(index_columns) == 1:
            margin_mask = result[index_columns[0]].astype(str).eq(margins_name)
            body = result.loc[~margin_mask].sort_values(
                index_columns, ascending=sort == "ascending", kind="mergesort")
            result = pd.concat([body, result.loc[margin_mask]], ignore_index=True)
        elif not margins:
            result = result.sort_values(
                index_columns, ascending=sort == "ascending", kind="mergesort"
            ).reset_index(drop=True)
    cells = int(result.shape[0] * max(0, result.shape[1] - len(index_columns)))
    if cells > int(max_output_cells):
        raise ValueError("피벗 결과가 출력 셀 한도를 넘습니다.")
    result = result.replace([np.inf, -np.inf], np.nan)
    # A margin label can share an index column with numeric categories. Arrow
    # cannot persist mixed integer/string object columns, so normalize only
    # those mixed index labels to their lossless display representation.
    for column in index_columns:
        observed_types = {type(value) for value in result[column].dropna().tolist()}
        if len(observed_types) > 1:
            result[column] = result[column].astype(str)

    all_conditions = tuple(dict.fromkeys((*info.conditions, *condition_objects)))
    specification = {
        "index_columns": index_columns,
        "column_columns": column_columns,
        "aggregation": aggregation,
        "value_column": value_column,
        "success_value": success_value,
        "conditions": [asdict(condition) for condition in condition_objects],
        "derived_bins": normalized_derived_bins,
        "margins": bool(margins),
        "margins_name": margins_name,
        "sort": sort,
    }
    derived = store.register(
        result,
        source=info.source,
        coverage=info.coverage,
        predicate_known=info.predicate_known,
        conditions=all_conditions,
        grain="aggregate",
        aggregation="pivot:" + json.dumps(specification, ensure_ascii=False, sort_keys=True),
        snapshot=info.snapshot,
        query="PIVOT " + json.dumps(specification, ensure_ascii=False, sort_keys=True),
        parent_id=dataset_id,
    )
    summary = {
        "kind": "dataset_pivot",
        "parent_dataset_id": dataset_id,
        **specification,
        "source_rows": len(source),
        "filtered_rows": len(filtered),
        "complete_rows": len(complete),
        "dropped_rows": len(filtered) - len(complete),
        "output_rows": len(result),
        "output_columns": len(result.columns),
        "output_cells": cells,
        "data_sha256": dataset_digest(result),
    }
    return {
        "status": "ready",
        "dataset": asdict(derived),
        "pivot_result": summary,
        "preview": json.loads(result.head(15).to_json(orient="records")),
        "scope": (
            f"{info.source}의 보유 raw dataset {len(source):,}행에서 조건 적용 {len(filtered):,}행, "
            f"완전한 관측값 {len(complete):,}행을 {aggregation} 피벗한 "
            f"{len(result):,}행 × {len(result.columns):,}열 · coverage={info.coverage} · "
            f"snapshot={info.snapshot or 'unknown'}"
        ),
    }
