"""Bounded, deterministic dataset profiles for the analysis agent."""
from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


def _safe_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, str):
        return value[:128]
    return value


def _column_profile(series: pd.Series) -> dict[str, Any]:
    rows = len(series)
    non_null = series.dropna()
    result: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "non_null_count": int(non_null.shape[0]),
        "null_count": int(series.isna().sum()),
        "null_ratio_pct": round(float(series.isna().mean() * 100), 4) if rows else 0.0,
        "distinct_count": int(non_null.nunique(dropna=True)),
    }
    if is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            result["numeric_summary"] = {
                "min": _safe_scalar(numeric.min()),
                "q1": _safe_scalar(numeric.quantile(0.25)),
                "median": _safe_scalar(numeric.median()),
                "mean": _safe_scalar(numeric.mean()),
                "q3": _safe_scalar(numeric.quantile(0.75)),
                "max": _safe_scalar(numeric.max()),
                "std": _safe_scalar(numeric.std()) if len(numeric) > 1 else None,
            }
    elif is_datetime64_any_dtype(series.dtype):
        if not non_null.empty:
            result["datetime_summary"] = {
                "min": _safe_scalar(non_null.min()),
                "max": _safe_scalar(non_null.max()),
            }
    # High-cardinality text can contain identifiers or private values. Return
    # frequencies only for genuinely categorical columns and cap all content.
    if result["distinct_count"] <= 50:
        counts = non_null.value_counts(dropna=True).head(10)
        result["top_values"] = [
            {"value": _safe_scalar(value), "count": int(count)}
            for value, count in counts.items()
        ]
        result["top_values_truncated"] = result["distinct_count"] > len(counts)
    else:
        result["top_values_omitted"] = "high_cardinality"
    return result


def profile_dataset(datasets, dataset_id: str, columns=None, offset: int = 0, limit: int = 50) -> dict:
    info = datasets.metadata[dataset_id]
    if offset < 0 or not 1 <= limit <= 64:
        raise ValueError("offset은 0 이상, limit은 1~64여야 합니다.")
    selected = list(columns) if columns else list(info.columns)
    if len(selected) != len(set(selected)) or any(column not in info.columns for column in selected):
        raise ValueError("프로파일 컬럼은 로딩된 dataset의 중복 없는 실제 컬럼이어야 합니다.")
    page = selected[offset:offset + limit]
    frame = (datasets.frames.project(dataset_id, page)
             if hasattr(datasets.frames, 'project') else datasets.frames[dataset_id])
    return {
        "status": "ready",
        "dataset_id": dataset_id,
        "profile": {
            "source": info.source,
            "rows": int(info.rows),
            "column_count": len(info.columns),
            "coverage": info.coverage,
            "predicate_known": info.predicate_known,
            "grain": info.grain,
            "aggregation": info.aggregation,
            "snapshot": info.snapshot,
            "columns": [_column_profile(frame[column]) for column in page],
            "column_page": {
                "offset": offset,
                "limit": limit,
                "returned": len(page),
                "total_selected": len(selected),
                "has_more": offset + len(page) < len(selected),
            },
        },
        "scope": (
            f"dataset {dataset_id}의 {info.rows}행을 프로파일링했습니다. "
            f"coverage={info.coverage}, grain={info.grain}, snapshot={info.snapshot or 'unspecified'}"
        ),
    }
