"""Bounded group aggregate comparison for lineage-related raw datasets."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from utils.analysis_datasets import DatasetStore


AGGREGATIONS = frozenset({"count", "mean", "sum", "median", "min", "max"})
SORT_ORDERS = frozenset({"group_ascending", "difference_descending", "none"})


def dataset_digest(frame: pd.DataFrame) -> str:
    normalized = frame.reset_index(drop=True)
    columns = json.dumps(list(normalized.columns), ensure_ascii=False).encode()
    values = pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    return sha256(columns + values).hexdigest()


def _is_descendant(store: DatasetStore, child_id: str, ancestor_id: str) -> bool:
    pending = [child_id]
    seen = set()
    while pending:
        dataset_id = pending.pop()
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        info = store.metadata.get(dataset_id)
        if info is None:
            continue
        parents = tuple(info.parent_ids) or ((info.parent_id,) if info.parent_id else ())
        if ancestor_id in parents:
            return True
        pending.extend(parents)
    return False


def compare_group_aggregates(
    store: DatasetStore,
    baseline_dataset_id: str,
    cohort_dataset_id: str,
    *,
    aggregation: str,
    group_column: str,
    value_column: str = "",
    sort: str = "group_ascending",
    max_groups: int = 1_000,
    max_output_rows: int = 1_000,
) -> dict:
    """Compare the same grouped aggregate across an ancestor and its cohort."""
    if baseline_dataset_id == cohort_dataset_id:
        raise ValueError("기준 dataset과 cohort dataset은 달라야 합니다.")
    if aggregation not in AGGREGATIONS:
        raise ValueError("지원하지 않는 집계 방식입니다.")
    if sort not in SORT_ORDERS:
        raise ValueError("지원하지 않는 정렬 방식입니다.")
    if not 1 <= int(max_groups) <= 1_000 or not 1 <= int(max_output_rows) <= 1_000:
        raise ValueError("그룹과 출력 행 한도가 허용 범위를 벗어났습니다.")

    baseline_info = store.metadata[baseline_dataset_id]
    cohort_info = store.metadata[cohort_dataset_id]
    baseline = store.frames[baseline_dataset_id]
    cohort = store.frames[cohort_dataset_id]
    if any(info.grain != "raw" or info.aggregation for info in (baseline_info, cohort_info)):
        raise ValueError("비교는 raw grain dataset 두 개에서만 수행합니다.")
    if baseline_info.coverage != "complete" or not baseline_info.predicate_known:
        raise ValueError("기준 dataset은 범위가 확인된 complete 원본이어야 합니다.")
    if baseline_info.source != cohort_info.source or baseline_info.snapshot != cohort_info.snapshot:
        raise ValueError("같은 source와 snapshot의 dataset만 비교할 수 있습니다.")
    if not _is_descendant(store, cohort_dataset_id, baseline_dataset_id):
        raise ValueError("cohort dataset은 기준 dataset의 lineage 후손이어야 합니다.")
    required = [group_column] + ([value_column] if value_column else [])
    if not group_column or len(required) != len(set(required)):
        raise ValueError("그룹 컬럼과 값 컬럼은 중복 없이 지정해야 합니다.")
    if any(column not in frame.columns for frame in (baseline, cohort) for column in required):
        raise ValueError("두 dataset 모두에 존재하는 실제 컬럼만 사용할 수 있습니다.")
    if aggregation == "count" and value_column:
        raise ValueError("count 집계에는 value_column을 지정하지 않습니다.")
    if aggregation != "count" and not value_column:
        raise ValueError("수치 집계에는 value_column이 필요합니다.")
    if value_column and any(
        not pd.api.types.is_numeric_dtype(frame[value_column])
        or pd.api.types.is_bool_dtype(frame[value_column])
        for frame in (baseline, cohort)
    ):
        raise ValueError("value_column은 두 dataset 모두에서 bool이 아닌 수치형이어야 합니다.")

    prepared = []
    dropped = []
    for frame in (baseline, cohort):
        work = frame.loc[:, required].copy()
        if value_column:
            work[value_column] = pd.to_numeric(work[value_column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan)
        complete = work.dropna(subset=required)
        if complete.empty:
            raise ValueError("비교할 완전한 관측값이 없습니다.")
        prepared.append(complete)
        dropped.append(len(frame) - len(complete))

    groups = set(prepared[0][group_column].tolist()) | set(prepared[1][group_column].tolist())
    if not 1 <= len(groups) <= int(max_groups):
        raise ValueError("그룹 고유값 수가 허용 범위를 벗어났습니다.")

    metric_name = "count" if aggregation == "count" else f"{aggregation}_{value_column}"
    baseline_column = f"baseline_{metric_name}"
    cohort_column = f"cohort_{metric_name}"

    def grouped(frame: pd.DataFrame, result_column: str) -> pd.DataFrame:
        group = frame.groupby(group_column, sort=False, observed=True, dropna=True)
        values = group.size() if aggregation == "count" else group[value_column].agg(aggregation)
        return values.rename(result_column).reset_index()

    result = grouped(prepared[0], baseline_column).merge(
        grouped(prepared[1], cohort_column), on=group_column, how="outer", validate="one_to_one")
    result["difference"] = result[cohort_column] - result[baseline_column]
    comparable = result[baseline_column].notna() & (result[baseline_column] != 0)
    result["percent_change"] = np.nan
    result.loc[comparable, "percent_change"] = (
        result.loc[comparable, "difference"]
        / result.loc[comparable, baseline_column].abs() * 100)
    if sort == "group_ascending":
        result = (result.assign(_group_sort=result[group_column].astype(str))
                  .sort_values("_group_sort", kind="mergesort").drop(columns="_group_sort"))
    elif sort == "difference_descending":
        result = result.sort_values("difference", ascending=False, kind="mergesort")
    result = result.reset_index(drop=True)
    if len(result) > int(max_output_rows):
        raise ValueError("비교 결과가 출력 행 한도를 넘습니다.")

    escaped_group = group_column.replace('"', '""')
    escaped_value = value_column.replace('"', '""')
    escaped_baseline = baseline_column.replace('"', '""')
    escaped_cohort = cohort_column.replace('"', '""')
    sql_function = {"mean": "AVG"}.get(aggregation, aggregation.upper())
    expression = "*" if aggregation == "count" else f'"{escaped_value}"'
    group_expression = f'COALESCE(b."{escaped_group}", c."{escaped_group}")'
    query = (
        f'WITH baseline_agg AS (SELECT "{escaped_group}", {sql_function}({expression}) '
        f'AS "{escaped_baseline}" FROM baseline_data GROUP BY "{escaped_group}"), '
        f'cohort_agg AS (SELECT "{escaped_group}", {sql_function}({expression}) '
        f'AS "{escaped_cohort}" FROM cohort_data GROUP BY "{escaped_group}") '
        f'SELECT {group_expression} AS "{escaped_group}", '
        f'b."{escaped_baseline}", c."{escaped_cohort}", '
        f'c."{escaped_cohort}" - b."{escaped_baseline}" AS difference, '
        f'CASE WHEN b."{escaped_baseline}" IS NOT NULL AND b."{escaped_baseline}" <> 0 '
        f'THEN 100.0 * (c."{escaped_cohort}" - b."{escaped_baseline}") '
        f'/ ABS(b."{escaped_baseline}") ELSE NULL END AS percent_change '
        f'FROM baseline_agg AS b FULL OUTER JOIN cohort_agg AS c '
        f'ON b."{escaped_group}" = c."{escaped_group}"'
    )
    if sort == "group_ascending":
        query += f' ORDER BY CAST({group_expression} AS VARCHAR) ASC'
    elif sort == "difference_descending":
        query += ' ORDER BY difference DESC'
    derived = store.register(
        result,
        source=baseline_info.source,
        coverage=baseline_info.coverage,
        predicate_known=False,
        conditions=baseline_info.conditions,
        grain="aggregate",
        aggregation=f"compare {aggregation}({value_column or '*'}) by {group_column}",
        snapshot=baseline_info.snapshot,
        query=query,
        parent_ids=(baseline_dataset_id, cohort_dataset_id),
    )
    summary = {
        "kind": "group_aggregate_comparison",
        "baseline_dataset_id": baseline_dataset_id,
        "cohort_dataset_id": cohort_dataset_id,
        "aggregation": aggregation,
        "value_column": value_column,
        "group_column": group_column,
        "baseline_result_column": baseline_column,
        "cohort_result_column": cohort_column,
        "sort": sort,
        "baseline_rows": len(baseline),
        "cohort_rows": len(cohort),
        "baseline_complete_rows": len(prepared[0]),
        "cohort_complete_rows": len(prepared[1]),
        "baseline_dropped_rows": dropped[0],
        "cohort_dropped_rows": dropped[1],
        "group_count": len(groups),
        "output_rows": len(result),
        "data_sha256": dataset_digest(result),
    }
    return {
        "status": "ready",
        "dataset": asdict(derived),
        "comparison_result": summary,
        "preview": json.loads(result.head(15).to_json(orient="records")),
        "scope": (
            f"{baseline_info.source}의 같은 snapshot에서 기준 {len(baseline):,}행과 "
            f"cohort {len(cohort):,}행의 {aggregation} 그룹 집계를 비교한 {len(result):,}행"
        ),
    }
