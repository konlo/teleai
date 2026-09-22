"""Bounded, schema-neutral preparation of time-series datasets."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from utils.analysis_datasets import DatasetStore


FREQUENCIES = {
    "hour": "h",
    "day": "D",
    "week": "7D",
    "month": "MS",
}
AGGREGATIONS = {"count", "sum", "mean", "median", "min", "max"}
GAP_POLICIES = {"omit", "zero", "nan"}


def dataset_digest(frame: pd.DataFrame) -> str:
    columns = json.dumps(list(frame.columns), ensure_ascii=False).encode()
    values = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return sha256(columns + values).hexdigest()


def _parse_time(series: pd.Series, timezone: str) -> tuple[pd.Series, str, int]:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError("수치형 시간 컬럼의 epoch 단위를 추측하지 않습니다.")
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    source_non_null = int(series.notna().sum())
    parsed_non_null = int(parsed.notna().sum())
    if parsed_non_null < 2:
        raise ValueError("시간 컬럼에는 파싱 가능한 값이 2개 이상 필요합니다.")
    if source_non_null and parsed_non_null / source_non_null < 0.95:
        raise ValueError("시간 컬럼 파싱 성공률이 95% 미만입니다.")
    try:
        current_timezone = parsed.dt.tz
    except AttributeError as exc:
        raise ValueError("혼합된 시간대나 일관되지 않은 시간값은 지원하지 않습니다.") from exc
    if timezone:
        if len(timezone) > 64:
            raise ValueError("timezone 이름이 너무 깁니다.")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("유효한 IANA timezone 이름이 필요합니다.") from exc
        if current_timezone is None:
            try:
                parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError("DST 중복 또는 존재하지 않는 현지 시각을 자동 보정하지 않습니다.") from exc
        else:
            parsed = parsed.dt.tz_convert(timezone)
        timezone_label = timezone
    else:
        timezone_label = str(current_timezone) if current_timezone is not None else "naive"
    return parsed, timezone_label, source_non_null - parsed_non_null


def _bucket_time(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "hour":
        return series.dt.floor("h")
    if frequency == "day":
        return series.dt.floor("D")
    timezone = series.dt.tz
    local = series.dt.tz_localize(None) if timezone is not None else series
    if frequency == "week":
        bucketed = local.dt.floor("D") - pd.to_timedelta(local.dt.weekday, unit="D")
    else:
        bucketed = local.dt.to_period("M").dt.to_timestamp()
    if timezone is not None:
        try:
            bucketed = bucketed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("시간 bucket 경계의 DST 충돌을 자동 보정하지 않습니다.") from exc
    return bucketed


def prepare_time_series(store: DatasetStore, dataset_id: str, *, time_column: str,
                        value_column: str = "", group_column: str = "",
                        frequency: str, aggregation: str,
                        gap_policy: str = "omit", timezone: str = "",
                        max_output_rows: int = 5_000) -> dict:
    """Create a bounded aggregate dataset for charting or later analysis."""
    if frequency not in FREQUENCIES or aggregation not in AGGREGATIONS:
        raise ValueError("지원하는 시간 빈도와 집계 방식을 사용해주세요.")
    if gap_policy not in GAP_POLICIES:
        raise ValueError("gap_policy는 omit, zero, nan 중 하나여야 합니다.")
    if not 2 <= int(max_output_rows) <= 5_000:
        raise ValueError("max_output_rows는 2~5,000이어야 합니다.")
    info = store.metadata[dataset_id]
    source = store.frames[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("시간 재집계는 완전한 raw grain dataset에서만 수행합니다.")
    requested = [column for column in (time_column, value_column, group_column) if column]
    if (not time_column or len(requested) != len(set(requested))
            or any(column not in source.columns for column in requested)):
        raise ValueError("시간·값·그룹 컬럼은 현재 dataset의 중복 없는 실제 컬럼이어야 합니다.")
    if aggregation == "count" and value_column:
        raise ValueError("count 집계에는 value_column을 지정하지 않습니다.")
    if aggregation != "count" and not value_column:
        raise ValueError("수치 집계에는 value_column이 필요합니다.")
    if value_column and (not pd.api.types.is_numeric_dtype(source[value_column])
                         or pd.api.types.is_bool_dtype(source[value_column])):
        raise ValueError("value_column은 수치형이어야 합니다.")

    parsed, timezone_label, parse_failures = _parse_time(source[time_column], timezone)
    working = pd.DataFrame({time_column: parsed})
    if value_column:
        working[value_column] = pd.to_numeric(source[value_column], errors="coerce").replace(
            [float("inf"), -float("inf")], float("nan"))
    if group_column:
        working[group_column] = source[group_column]
    required = [time_column] + ([value_column] if value_column else []) + ([group_column] if group_column else [])
    working = working.dropna(subset=required)
    if len(working) < 2:
        raise ValueError("완전한 시계열 관측값이 2개 이상 필요합니다.")
    if group_column:
        group_count = int(working[group_column].nunique(dropna=True))
        if not 1 <= group_count <= 20:
            raise ValueError("다중 시계열의 group 고유값은 1~20개여야 합니다.")
    else:
        group_count = 0

    duplicate_keys = [time_column] + ([group_column] if group_column else [])
    duplicate_rows = int(working.duplicated(duplicate_keys, keep=False).sum())
    working[time_column] = _bucket_time(working[time_column], frequency)
    keys = [time_column] + ([group_column] if group_column else [])
    if aggregation == "count":
        value_name = "count"
        aggregated = working.groupby(keys, sort=True, observed=True).size().rename(value_name).reset_index()
    else:
        value_name = value_column
        aggregated = (working.groupby(keys, sort=True, observed=True)[value_column]
                      .agg(aggregation).reset_index())
    observed_buckets = len(aggregated)

    if gap_policy != "omit":
        ranges = []
        group_values = [None] if not group_column else sorted(
            aggregated[group_column].unique(), key=lambda value: str(value))
        for group_value in group_values:
            subset = aggregated if group_value is None else aggregated.loc[
                aggregated[group_column] == group_value]
            index = pd.date_range(subset[time_column].min(), subset[time_column].max(),
                                  freq=FREQUENCIES[frequency])
            piece = pd.DataFrame({time_column: index})
            if group_column:
                piece[group_column] = group_value
            ranges.append(piece)
        expected = pd.concat(ranges, ignore_index=True)
        if len(expected) > int(max_output_rows):
            raise ValueError("빈 구간을 포함한 시계열이 출력 행 한도를 넘습니다.")
        aggregated = expected.merge(aggregated, on=keys, how="left", validate="one_to_one")
        if gap_policy == "zero":
            aggregated[value_name] = aggregated[value_name].fillna(0)
    if len(aggregated) > int(max_output_rows):
        raise ValueError("시계열 결과가 출력 행 한도를 넘습니다.")
    aggregated = aggregated.sort_values(keys).reset_index(drop=True)
    gap_rows = int(len(aggregated) - observed_buckets)
    result = store.register(
        aggregated, source=info.source, coverage=info.coverage,
        predicate_known=info.predicate_known, conditions=info.conditions,
        grain="aggregate",
        aggregation=(f"{frequency}:{aggregation}({value_column or '*'})"
                     + (f" by {group_column}" if group_column else "")
                     + f" gaps={gap_policy} timezone={timezone_label}"),
        snapshot=info.snapshot, parent_id=dataset_id,
    )
    summary = {
        "kind": "time_series_preparation",
        "parent_dataset_id": dataset_id,
        "time_column": time_column,
        "value_column": value_name,
        "group_column": group_column,
        "frequency": frequency,
        "aggregation": aggregation,
        "gap_policy": gap_policy,
        "timezone": timezone_label,
        "source_rows": len(source),
        "complete_rows": len(working),
        "dropped_rows": len(source) - len(working),
        "parse_failures": parse_failures,
        "duplicate_time_rows": duplicate_rows,
        "observed_buckets": observed_buckets,
        "gap_rows_added": gap_rows,
        "output_rows": len(aggregated),
        "group_count": group_count,
        "start": aggregated[time_column].min().isoformat(),
        "end": aggregated[time_column].max().isoformat(),
        "data_sha256": dataset_digest(aggregated),
    }
    return {
        "status": "ready",
        "dataset": asdict(result),
        "time_series_result": summary,
        "preview": json.loads(aggregated.head(15).to_json(
            orient="records", date_format="iso")),
        "scope": (f"{info.source}의 보유 {len(source):,}행을 {frequency} 단위 {aggregation}로 "
                  f"집계한 {len(aggregated):,}행 · {info.coverage} · timezone {timezone_label}"),
    }
