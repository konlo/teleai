"""Bounded, table-neutral outlier detection for loaded analysis datasets."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.analysis_datasets import DatasetStore


SUPPORTED_METHODS = frozenset({"iqr", "zscore", "mad", "quantile"})
SUPPORTED_TAILS = frozenset({"both", "upper", "lower"})


def _number(value: Any) -> int | float | None:
    result = float(value)
    if not np.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


def detect_outliers(
    store: DatasetStore,
    dataset_id: str,
    *,
    column: str,
    method: str,
    tail: str = "both",
    threshold: float = 1.5,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, Any]:
    """Return aggregate outlier evidence without exposing or materializing rows."""
    if method not in SUPPORTED_METHODS:
        raise ValueError("지원하지 않는 이상치 탐지 방법입니다.")
    if tail not in SUPPORTED_TAILS:
        raise ValueError("tail은 both, upper, lower 중 하나여야 합니다.")
    info = store.metadata[dataset_id]
    frame = store.frames[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("이상치 탐지는 집계되지 않은 raw dataset에서만 실행합니다.")
    if column not in frame.columns or not is_numeric_dtype(frame[column]):
        raise ValueError("column은 dataset에 존재하는 수치형 컬럼이어야 합니다.")

    clean = frame[column].dropna().astype(float)
    if len(clean) < 4:
        raise ValueError("이상치 탐지에는 결측 제외 관측치가 최소 4개 필요합니다.")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("수치 컬럼에 무한대가 있어 이상치 기준을 계산할 수 없습니다.")
    if clean.nunique() < 2:
        raise ValueError("값의 변이가 없어 이상치 기준을 계산할 수 없습니다.")

    center = scale = lower = upper = None
    parameters: dict[str, Any]
    if method == "iqr":
        if not 0.1 <= float(threshold) <= 10:
            raise ValueError("IQR threshold는 0.1~10 범위여야 합니다.")
        q1, q3 = clean.quantile([0.25, 0.75]).tolist()
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            raise ValueError("IQR이 0이어서 이상치 기준을 계산할 수 없습니다.")
        lower, upper = q1 - float(threshold) * iqr, q3 + float(threshold) * iqr
        center, scale = clean.median(), iqr
        parameters = {"multiplier": float(threshold), "q1": _number(q1),
                      "q3": _number(q3), "iqr": _number(iqr)}
    elif method == "zscore":
        if not 0.5 <= float(threshold) <= 10:
            raise ValueError("Z-score threshold는 0.5~10 범위여야 합니다.")
        center, scale = clean.mean(), clean.std(ddof=1)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("표본 표준편차가 0이어서 Z-score를 계산할 수 없습니다.")
        lower, upper = center - float(threshold) * scale, center + float(threshold) * scale
        parameters = {"z_threshold": float(threshold), "mean": _number(center),
                      "sample_std": _number(scale), "ddof": 1}
    elif method == "mad":
        if not 0.5 <= float(threshold) <= 10:
            raise ValueError("MAD threshold는 0.5~10 범위여야 합니다.")
        center = clean.median()
        scale = (clean - center).abs().median()
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("MAD가 0이어서 수정 Z-score를 계산할 수 없습니다.")
        distance = float(threshold) * scale / 0.6744897501960817
        lower, upper = center - distance, center + distance
        parameters = {"modified_z_threshold": float(threshold),
                      "median": _number(center), "mad": _number(scale),
                      "consistency_constant": 0.6744897501960817}
    else:
        if not (0 <= float(lower_quantile) < float(upper_quantile) <= 1):
            raise ValueError("quantile 경계는 0 <= lower < upper <= 1이어야 합니다.")
        if tail in {"both", "lower"} and float(lower_quantile) <= 0:
            raise ValueError("lower tail에는 0보다 큰 lower_quantile이 필요합니다.")
        if tail in {"both", "upper"} and float(upper_quantile) >= 1:
            raise ValueError("upper tail에는 1보다 작은 upper_quantile이 필요합니다.")
        lower = clean.quantile(float(lower_quantile))
        upper = clean.quantile(float(upper_quantile))
        center = clean.median()
        parameters = {"lower_quantile": float(lower_quantile),
                      "upper_quantile": float(upper_quantile)}

    lower_mask = clean < float(lower)
    upper_mask = clean > float(upper)
    selected = lower_mask | upper_mask if tail == "both" else upper_mask if tail == "upper" else lower_mask
    outliers = clean[selected]
    valid_rows = len(clean)
    result = {
        "kind": "outlier_detection",
        "method": method,
        "column": column,
        "tail": tail,
        "parameters": parameters,
        "thresholds": {"lower": _number(lower), "upper": _number(upper)},
        "sample": {
            "input_rows": len(frame),
            "valid_rows": valid_rows,
            "missing_rows": len(frame) - valid_rows,
        },
        "counts": {
            "selected": int(selected.sum()),
            "lower": int(lower_mask.sum()),
            "upper": int(upper_mask.sum()),
            "selected_percent": float(selected.mean() * 100),
        },
        "distribution": {
            "minimum": _number(clean.min()),
            "maximum": _number(clean.max()),
            "center": _number(center),
            "scale": _number(scale) if scale is not None else None,
        },
        "selected_extrema": {
            "minimum": _number(outliers.min()) if len(outliers) else None,
            "maximum": _number(outliers.max()) if len(outliers) else None,
        },
        "missing_policy": "대상 컬럼의 결측 행을 제외",
        "boundary_policy": "경계와 같은 값은 이상치로 분류하지 않고 경계를 초과한 값만 분류",
        "source": info.source,
        "coverage": info.coverage,
        "grain": info.grain,
        "snapshot": info.snapshot,
        "warnings": [],
    }
    if info.coverage != "complete":
        result["warnings"].append("현재 dataset이 전체 원본을 포함하지 않아 임계값과 비율은 보유 범위에만 적용됩니다.")
    return {
        "status": "ready",
        "dataset_id": dataset_id,
        "outlier_result": result,
        "scope": (
            f"{info.source}의 로딩된 raw dataset {len(frame):,}행 중 유효값 {valid_rows:,}개에 "
            f"{method}/{tail} 기준을 적용했습니다. coverage={info.coverage}; "
            f"snapshot={info.snapshot or 'unknown'}."
        ),
    }
