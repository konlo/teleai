"""Bounded, schema-neutral aggregation for loaded analysis datasets."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from utils.analysis_datasets import DatasetStore, project_dataset


AGGREGATIONS = frozenset({"count", "mean", "sum", "median", "min", "max"})
SORT_ORDERS = frozenset({"ascending", "descending"})


def dataset_digest(frame: pd.DataFrame) -> str:
    normalized = frame.reset_index(drop=True)
    columns = json.dumps(list(normalized.columns), ensure_ascii=False).encode()
    values = pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    return sha256(columns + values).hexdigest()


def aggregate_dataset(
    store: DatasetStore,
    dataset_id: str,
    *,
    aggregation: str,
    value_column: str = "",
    group_column: str = "",
    sort: str = "descending",
    top_n: int = 0,
    max_groups: int = 1_000,
    max_output_rows: int = 1_000,
) -> dict:
    """Aggregate a raw dataset without executing arbitrary model-generated code."""
    if aggregation not in AGGREGATIONS:
        raise ValueError("지원하지 않는 집계 방식입니다.")
    if sort not in SORT_ORDERS:
        raise ValueError("sort는 ascending 또는 descending이어야 합니다.")
    if not 0 <= int(top_n) <= 50:
        raise ValueError("top_n은 0~50이어야 합니다.")
    if not 1 <= int(max_groups) <= 1_000 or not 1 <= int(max_output_rows) <= 1_000:
        raise ValueError("그룹과 출력 행 한도가 허용 범위를 벗어났습니다.")

    info = store.metadata[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("집계는 raw grain dataset에서만 수행합니다.")
    requested = [column for column in (value_column, group_column) if column]
    if len(requested) != len(set(requested)) or any(column not in info.columns for column in requested):
        raise ValueError("값·그룹 컬럼은 현재 dataset의 중복 없는 실제 컬럼이어야 합니다.")
    if aggregation == "count" and value_column:
        raise ValueError("count 집계에는 value_column을 지정하지 않습니다.")
    if aggregation != "count" and not value_column:
        raise ValueError("수치 집계에는 value_column이 필요합니다.")
    source = project_dataset(store, dataset_id, requested)
    if value_column and (
        not pd.api.types.is_numeric_dtype(source[value_column])
        or pd.api.types.is_bool_dtype(source[value_column])
    ):
        raise ValueError("value_column은 bool이 아닌 수치형 컬럼이어야 합니다.")

    required = ([group_column] if group_column else []) + ([value_column] if value_column else [])
    working = source.loc[:, required].copy() if required else source.iloc[:, :0].copy()
    if value_column:
        working[value_column] = pd.to_numeric(working[value_column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    complete = working.dropna(subset=required) if required else working
    if len(complete) == 0:
        raise ValueError("집계할 완전한 관측값이 없습니다.")

    group_count = 0
    if group_column:
        group_count = int(complete[group_column].nunique(dropna=True))
        if not 1 <= group_count <= int(max_groups):
            raise ValueError("그룹 고유값 수가 허용 범위를 벗어났습니다.")

    result_column = "count" if aggregation == "count" else f"{aggregation}_{value_column}"
    if group_column:
        grouped = complete.groupby(group_column, sort=False, observed=True, dropna=True)
        if aggregation == "count":
            result = grouped.size().rename(result_column).reset_index()
        else:
            result = grouped[value_column].agg(aggregation).rename(result_column).reset_index()
        result = result.sort_values(
            result_column, ascending=sort == "ascending", kind="mergesort"
        )
        if top_n:
            result = result.head(int(top_n))
    else:
        if aggregation == "count":
            value = len(complete)
        else:
            value = getattr(complete[value_column], aggregation)()
        result = pd.DataFrame({result_column: [value]})
    result = result.reset_index(drop=True)
    if len(result) > int(max_output_rows):
        raise ValueError("집계 결과가 출력 행 한도를 넘습니다.")

    quoted_group = f'"{group_column.replace(chr(34), chr(34) * 2)}"' if group_column else ""
    group_sql = f" GROUP BY {quoted_group}" if group_column else ""
    value_sql = "*" if aggregation == "count" else f'"{value_column.replace(chr(34), chr(34) * 2)}"'
    projection = f"{quoted_group}, " if group_column else ""
    sql_function = {"mean": "AVG"}.get(aggregation, aggregation.upper())
    quoted_result = result_column.replace('"', '""')
    query = f"SELECT {projection}{sql_function}({value_sql}) AS \"{quoted_result}\" FROM data{group_sql}"
    if group_column:
        query += f' ORDER BY "{quoted_result}" {"ASC" if sort == "ascending" else "DESC"}'
        if top_n:
            query += f" LIMIT {int(top_n)}"
    derived = store.register(
        result,
        source=info.source,
        coverage=info.coverage,
        predicate_known=info.predicate_known,
        conditions=info.conditions,
        grain="aggregate",
        aggregation=(f"{aggregation}({value_column or '*'})"
                     + (f" by {group_column}" if group_column else "")
                     + (f" top {top_n}" if top_n else "")),
        snapshot=info.snapshot,
        query=query,
        parent_id=dataset_id,
    )
    summary = {
        "kind": "dataset_aggregation",
        "parent_dataset_id": dataset_id,
        "aggregation": aggregation,
        "value_column": value_column,
        "group_column": group_column,
        "result_column": result_column,
        "sort": sort,
        "top_n": int(top_n),
        "source_rows": len(source),
        "complete_rows": len(complete),
        "dropped_rows": len(source) - len(complete),
        "group_count": group_count,
        "output_rows": len(result),
        "data_sha256": dataset_digest(result),
    }
    return {
        "status": "ready",
        "dataset": asdict(derived),
        "aggregation_result": summary,
        "preview": json.loads(result.head(15).to_json(orient="records")),
        "scope": (
            f"{info.source}의 보유 raw dataset {len(source):,}행에서 완전한 관측값 "
            f"{len(complete):,}행을 {aggregation} 집계한 {len(result):,}행 · {info.coverage} · "
            f"snapshot {info.snapshot or 'unknown'}"
        ),
    }
