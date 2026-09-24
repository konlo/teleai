"""Bounded local joins with explicit cardinality and lineage evidence."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from threading import Timer

import duckdb
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from utils.analysis_datasets import DatasetStore, full_read_preflight, project_dataset


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _dtype_family(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "bool"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_string_dtype(series) or is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return "text"
    return str(series.dtype)


def _key_counts(frame: pd.DataFrame, keys: list[str]):
    valid = frame[keys].notna().all(axis=1)
    counts = frame.loc[valid, keys].groupby(keys, sort=False, observed=True).size()
    if len(keys) == 1:
        counts.index = pd.Index(counts.index, name="__key_0")
    else:
        counts.index = counts.index.set_names([f"__key_{index}" for index in range(len(keys))])
    return counts.astype("int64"), int((~valid).sum())


def _projection(left_columns, right_columns, left_keys, right_keys):
    same_keys = {right for left, right in zip(left_keys, right_keys) if left == right}
    overlap = set(left_columns) & set(right_columns) - same_keys
    projections, output = [], []
    for column in left_columns:
        alias = column + "_left" if column in overlap else column
        if column in same_keys:
            # A right/full join can contain rows with no left match. Keeping
            # only l.key would erase the actual right-side key for those rows.
            projections.append(
                f"COALESCE(l.{_quote(column)}, r.{_quote(column)}) AS {_quote(alias)}"
            )
        else:
            projections.append(f"l.{_quote(column)} AS {_quote(alias)}")
        output.append(alias)
    for column in right_columns:
        if column in same_keys:
            continue
        alias = column + "_right" if column in overlap else column
        projections.append(f"r.{_quote(column)} AS {_quote(alias)}")
        output.append(alias)
    if len(output) != len(set(output)):
        raise ValueError("조인 suffix를 적용해도 출력 컬럼명이 충돌합니다. 먼저 컬럼명을 명시적으로 변경해주세요.")
    return projections, output


def join_datasets(store: DatasetStore, left_dataset_id: str, right_dataset_id: str, *,
                  left_on: list[str], right_on: list[str], how: str,
                  max_rows: int = 100_000, max_expansion_ratio: float = 5.0):
    """Join two loaded frames only after an exact cardinality preflight.

    Many-to-many matches are rejected even when small. The agent must first
    aggregate or deduplicate one side, which makes the intended grain explicit.
    """
    if left_dataset_id == right_dataset_id:
        raise ValueError("self join은 별도 복제·역할 정의 없이 실행하지 않습니다.")
    if how not in {"inner", "left", "right", "outer"}:
        raise ValueError("how는 inner, left, right, outer 중 하나여야 합니다.")
    if not isinstance(left_on, list) or not isinstance(right_on, list):
        raise ValueError("left_on과 right_on은 컬럼 배열이어야 합니다.")
    if not 1 <= len(left_on) <= 4 or len(left_on) != len(right_on):
        raise ValueError("조인 key는 양쪽에 같은 개수로 1~4개 지정해야 합니다.")
    if len(set(left_on)) != len(left_on) or len(set(right_on)) != len(right_on):
        raise ValueError("조인 key 컬럼은 중복될 수 없습니다.")
    if max_rows <= 0 or max_expansion_ratio <= 0:
        raise ValueError("join 운영 한도는 0보다 커야 합니다.")

    left_info, right_info = store.metadata[left_dataset_id], store.metadata[right_dataset_id]
    for label, info in (("left", left_info), ("right", right_info)):
        if info.grain not in {"raw", "aggregate"}:
            raise ValueError(f"{label} dataset의 grain은 raw 또는 명시적 aggregate여야 합니다.")
        if (info.grain == "aggregate") != bool(info.aggregation):
            raise ValueError(f"{label} dataset의 grain과 aggregation provenance가 일치하지 않습니다.")
    if any(column not in left_info.columns for column in left_on):
        raise ValueError("left_on에 왼쪽 dataset에 없는 컬럼이 있습니다.")
    if any(column not in right_info.columns for column in right_on):
        raise ValueError("right_on에 오른쪽 dataset에 없는 컬럼이 있습니다.")
    left_keys = project_dataset(store, left_dataset_id, left_on)
    right_keys = project_dataset(store, right_dataset_id, right_on)
    incompatible = [
        {"left": lkey, "right": rkey,
         "left_dtype": str(left_keys[lkey].dtype), "right_dtype": str(right_keys[rkey].dtype)}
        for lkey, rkey in zip(left_on, right_on)
        if _dtype_family(left_keys[lkey]) != _dtype_family(right_keys[rkey])
    ]
    if incompatible:
        raise ValueError("조인 key의 자료형 계열이 서로 다릅니다. 명시적으로 정규화한 뒤 다시 시도하세요.")

    left_counts, left_null_rows = _key_counts(left_keys, left_on)
    right_counts, right_null_rows = _key_counts(right_keys, right_on)
    overlap = left_counts.rename("left").to_frame().join(
        right_counts.rename("right"), how="inner")
    inner_rows = int((overlap["left"] * overlap["right"]).sum()) if not overlap.empty else 0
    matched_left = int(overlap["left"].sum()) if not overlap.empty else 0
    matched_right = int(overlap["right"].sum()) if not overlap.empty else 0
    unmatched_left = left_info.rows - matched_left
    unmatched_right = right_info.rows - matched_right
    expected_rows = {
        "inner": inner_rows,
        "left": inner_rows + unmatched_left,
        "right": inner_rows + unmatched_right,
        "outer": inner_rows + unmatched_left + unmatched_right,
    }[how]
    left_many = bool(not overlap.empty and (overlap["left"] > 1).any())
    right_many = bool(not overlap.empty and (overlap["right"] > 1).any())
    exact_many_to_many = bool(not overlap.empty and ((overlap["left"] > 1) & (overlap["right"] > 1)).any())
    relationship = (
        "many_to_many" if exact_many_to_many else
        "mixed_to_one_or_many" if left_many and right_many else
        "many_to_one" if left_many else
        "one_to_many" if right_many else
        "one_to_one" if not overlap.empty else "no_matches"
    )
    expansion_ratio = expected_rows / max(left_info.rows, right_info.rows, 1)
    summary = {
        "how": how,
        "left_on": left_on,
        "right_on": right_on,
        "relationship": relationship,
        "left_rows": left_info.rows,
        "right_rows": right_info.rows,
        "left_coverage": left_info.coverage,
        "right_coverage": right_info.coverage,
        "left_conditions": [asdict(condition) for condition in left_info.conditions],
        "right_conditions": [asdict(condition) for condition in right_info.conditions],
        "left_distinct_non_null_keys": len(left_counts),
        "right_distinct_non_null_keys": len(right_counts),
        "matched_distinct_keys": len(overlap),
        "left_null_key_rows": left_null_rows,
        "right_null_key_rows": right_null_rows,
        "unmatched_left_rows": unmatched_left,
        "unmatched_right_rows": unmatched_right,
        "expected_rows": expected_rows,
        "expansion_ratio": round(expansion_ratio, 6),
        "max_rows": int(max_rows),
        "max_expansion_ratio": float(max_expansion_ratio),
    }
    if relationship == "many_to_many":
        return {
            "status": "rejected",
            "error_code": "many_to_many_join",
            "retryable": False,
            "user_action": "한쪽 dataset을 key별로 집계하거나 중복 제거해 grain을 명확히 한 뒤 다시 조인하세요.",
            "join_summary": summary,
            "scope": "실행 전 cardinality 검사에서 중단했으며 새 dataset을 만들지 않았습니다.",
        }
    if expected_rows > max_rows or expansion_ratio > max_expansion_ratio:
        return {
            "status": "rejected",
            "error_code": "join_output_limit",
            "retryable": False,
            "user_action": "조인 전에 컬럼·행을 줄이거나 key별 집계로 출력 규모를 축소하세요.",
            "join_summary": summary,
            "scope": "예상 출력 규모가 운영 한도를 넘어 실행하지 않았습니다.",
        }

    projections, output_columns = _projection(
        list(left_info.columns), list(right_info.columns), left_on, right_on)
    rejected = full_read_preflight(store, [left_dataset_id, right_dataset_id],
                                   output_rows=expected_rows, output_columns=len(output_columns))
    if rejected:
        return {**rejected, 'join_summary': summary}
    left, right = store.frames[left_dataset_id], store.frames[right_dataset_id]
    conditions = " AND ".join(
        f"l.{_quote(lkey)} = r.{_quote(rkey)}"
        for lkey, rkey in zip(left_on, right_on)
    )
    join_sql = {"inner": "INNER", "left": "LEFT", "right": "RIGHT", "outer": "FULL OUTER"}[how]
    query = (f"SELECT {', '.join(projections)} FROM left_data AS l "
             f"{join_sql} JOIN right_data AS r ON {conditions}")
    with duckdb.connect(config={"enable_external_access": False, "memory_limit": "512MB", "threads": 2}) as conn:
        conn.register("left_data", left)
        conn.register("right_data", right)
        timer = Timer(15, conn.interrupt)
        timer.start()
        try:
            result = conn.execute(query).fetchdf()
        finally:
            timer.cancel()
            timer.join()
    if len(result) != expected_rows or list(result.columns) != output_columns:
        raise RuntimeError("join preflight와 실제 결과가 일치하지 않습니다.")

    summary["actual_rows"] = len(result)
    lineage = json.dumps({
        "parents": [left_info.id, right_info.id],
        "snapshots": [left_info.snapshot, right_info.snapshot],
        "how": how,
        "left_on": left_on,
        "right_on": right_on,
    }, ensure_ascii=False, sort_keys=True)
    if "truncated" in {left_info.coverage, right_info.coverage}:
        coverage = "truncated"
    elif "sampled" in {left_info.coverage, right_info.coverage}:
        coverage = "sampled"
    elif left_info.coverage == right_info.coverage == "complete":
        coverage = "complete"
    else:
        coverage = "unknown"
    info = store.register(
        result,
        source=f"{left_info.source} JOIN {right_info.source}",
        coverage=coverage,
        predicate_known=left_info.predicate_known and right_info.predicate_known,
        grain="joined",
        aggregation="",
        snapshot="join:" + sha256(lineage.encode()).hexdigest(),
        query=query,
        parent_id=left_info.id,
        parent_ids=(left_info.id, right_info.id),
    )
    return {
        "status": "ready",
        "dataset": asdict(info),
        "join_summary": summary,
        "preview": result.head(15).to_dict(orient="records"),
        "scope": (f"왼쪽 {len(left):,}행과 오른쪽 {len(right):,}행의 {relationship} {how} join 결과 "
                  f"{len(result):,}행입니다. coverage={coverage}; 부모 dataset 2개의 snapshot을 lineage로 보존했습니다."),
    }
