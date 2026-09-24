"""Bounded statistical tests over one loaded, row-level dataset."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats

from utils.analysis_datasets import DatasetStore, project_dataset


SUPPORTED_TESTS = {
    "independent_t",
    "paired_t",
    "chi_square",
    "one_way_anova",
    "mann_whitney",
    "mean_ci",
}


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return int(result) if result.is_integer() else result


def _scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normality(values: np.ndarray) -> dict[str, Any]:
    """Return a bounded diagnostic; it is evidence, not an automatic veto."""
    n = len(values)
    if n < 3:
        return {"method": "shapiro", "n": n, "status": "insufficient", "p_value": None}
    if np.ptp(values) == 0:
        return {"method": "shapiro", "n": n, "status": "constant", "p_value": None}
    if n <= 5_000:
        statistic, p_value = stats.shapiro(values)
        method, tested_n = "shapiro", n
    else:
        statistic, p_value = stats.normaltest(values)
        method, tested_n = "dagostino_k2", n
    return {
        "method": method,
        "n": tested_n,
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "normal_at_0_05": bool(p_value >= 0.05),
    }


def _mean_summary(values: np.ndarray, confidence: float) -> dict[str, Any]:
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    sem = std / math.sqrt(n) if n > 1 else float("nan")
    critical = stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": _number(mean),
        "std": _number(std),
        "confidence_interval": {
            "level": confidence,
            "lower": _number(mean - critical * sem),
            "upper": _number(mean + critical * sem),
        },
    }


def _grouped_numeric(frame: pd.DataFrame, value_column: str, group_column: str,
                     *, exactly_two: bool) -> tuple[pd.DataFrame, list[tuple[Any, np.ndarray]]]:
    if not is_numeric_dtype(frame[value_column]):
        raise ValueError("value_column은 수치형이어야 합니다.")
    complete = frame[[value_column, group_column]].dropna()
    levels = list(pd.unique(complete[group_column]))
    if exactly_two and len(levels) != 2:
        raise ValueError("이 검정은 결측을 제외한 group 값이 정확히 2개여야 합니다.")
    if not exactly_two and not 2 <= len(levels) <= 20:
        raise ValueError("그룹 수는 2~20개여야 합니다.")
    groups = [
        (level, complete.loc[complete[group_column] == level, value_column].to_numpy(dtype=float))
        for level in levels
    ]
    if any(len(values) < 2 for _, values in groups):
        raise ValueError("각 그룹에 결측 제외 관측치가 최소 2개 필요합니다.")
    return complete, groups


def _base(info, kind: str, columns: list[str], alpha: float, input_rows: int,
          complete_rows: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "columns": columns,
        "alpha": alpha,
        "alternative": "two-sided",
        "missing_policy": "검정에 필요한 컬럼 중 결측이 있는 행을 제외",
        "sample": {
            "input_rows": input_rows,
            "complete_rows": complete_rows,
            "dropped_rows": input_rows - complete_rows,
        },
        "source": info.source,
        "coverage": info.coverage,
        "grain": info.grain,
        "snapshot": info.snapshot,
        "significant": None,
        "confidence_intervals": [],
        "warnings": [],
    }


def _independent_t(frame, info, value_column, group_column, alpha):
    complete, groups = _grouped_numeric(frame, value_column, group_column, exactly_two=True)
    (level_a, a), (level_b, b) = groups
    variance_a, variance_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se2 = variance_a / len(a) + variance_b / len(b)
    if se2 <= 0 or not np.isfinite(se2):
        raise ValueError("두 그룹의 분산으로 표준오차를 계산할 수 없습니다.")
    statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("유한한 t 통계량과 p-value를 계산할 수 없습니다.")
    df = se2 ** 2 / (
        (variance_a / len(a)) ** 2 / (len(a) - 1) +
        (variance_b / len(b)) ** 2 / (len(b) - 1)
    )
    difference = float(np.mean(a) - np.mean(b))
    margin = stats.t.ppf(1 - alpha / 2, df) * math.sqrt(se2)
    pooled_df = len(a) + len(b) - 2
    pooled_sd = math.sqrt(((len(a) - 1) * variance_a + (len(b) - 1) * variance_b) / pooled_df)
    cohen_d = difference / pooled_sd if pooled_sd else None
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    result = _base(info, "independent_t", [value_column, group_column], alpha,
                   len(frame), len(complete))
    result.update({
        "method": "Welch two-sample t-test",
        "null_hypothesis": "두 그룹의 모집단 평균이 같다.",
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "degrees_of_freedom": _number(df),
        "significant": bool(p_value < alpha),
        "groups": [
            {"value": _scalar(level_a), **_mean_summary(a, 1 - alpha)},
            {"value": _scalar(level_b), **_mean_summary(b, 1 - alpha)},
        ],
        "estimate": {"name": "mean_difference_first_minus_second", "value": _number(difference)},
        "effect_size": {"name": "hedges_g", "value": _number(cohen_d * correction) if cohen_d is not None else None},
        "confidence_intervals": [{
            "parameter": "mean_difference_first_minus_second",
            "level": 1 - alpha,
            "lower": _number(difference - margin),
            "upper": _number(difference + margin),
        }],
        "assumptions": {
            "normality": [
                {"group": _scalar(level_a), **_normality(a)},
                {"group": _scalar(level_b), **_normality(b)},
            ],
            "equal_variance": {
                "required": False,
                "diagnostic": "levene",
                "p_value": _number(stats.levene(a, b).pvalue),
            },
            "independent_observations": "데이터만으로 검증할 수 없으므로 사용자/수집 설계 확인 필요",
        },
    })
    if any(item.get("p_value") is not None and item["p_value"] < 0.05
           for item in result["assumptions"]["normality"]):
        result["warnings"].append("그룹 정규성 진단이 0.05 기준을 충족하지 않습니다. 큰 표본 여부와 분포를 함께 확인하세요.")
    return result


def _paired_t(frame, info, value_column, paired_column, alpha):
    if not is_numeric_dtype(frame[value_column]) or not is_numeric_dtype(frame[paired_column]):
        raise ValueError("paired_t의 두 컬럼은 수치형이어야 합니다.")
    complete = frame[[value_column, paired_column]].dropna()
    if len(complete) < 2:
        raise ValueError("결측 제외 쌍이 최소 2개 필요합니다.")
    first = complete[value_column].to_numpy(dtype=float)
    second = complete[paired_column].to_numpy(dtype=float)
    difference = first - second
    std = float(np.std(difference, ddof=1))
    if not np.isfinite(std) or std == 0:
        raise ValueError("쌍별 차이의 분산이 없어 paired t-test를 계산할 수 없습니다.")
    statistic, p_value = stats.ttest_rel(first, second)
    mean_difference = float(np.mean(difference))
    df = len(difference) - 1
    margin = stats.t.ppf(1 - alpha / 2, df) * std / math.sqrt(len(difference))
    result = _base(info, "paired_t", [value_column, paired_column], alpha,
                   len(frame), len(complete))
    result.update({
        "method": "paired t-test",
        "null_hypothesis": "쌍별 평균 차이가 0이다.",
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "degrees_of_freedom": df,
        "significant": bool(p_value < alpha),
        "estimate": {"name": "paired_mean_difference_first_minus_second", "value": _number(mean_difference)},
        "effect_size": {"name": "cohen_dz", "value": _number(mean_difference / std)},
        "confidence_intervals": [{
            "parameter": "paired_mean_difference_first_minus_second",
            "level": 1 - alpha,
            "lower": _number(mean_difference - margin),
            "upper": _number(mean_difference + margin),
        }],
        "assumptions": {
            "difference_normality": _normality(difference),
            "pairing_validity": "행 단위 쌍이 실제 동일 관측 단위인지 사용자/수집 설계 확인 필요",
        },
    })
    if result["assumptions"]["difference_normality"].get("p_value") is not None and \
            result["assumptions"]["difference_normality"]["p_value"] < 0.05:
        result["warnings"].append("쌍별 차이의 정규성 진단이 0.05 기준을 충족하지 않습니다.")
    return result


def _anova(frame, info, value_column, group_column, alpha):
    complete, groups = _grouped_numeric(frame, value_column, group_column, exactly_two=False)
    arrays = [values for _, values in groups]
    within_variation = sum(float(np.var(values, ddof=1)) * (len(values) - 1) for values in arrays)
    if within_variation <= 0 or not np.isfinite(within_variation):
        raise ValueError("그룹 내 분산이 없어 ANOVA를 계산할 수 없습니다.")
    statistic, p_value = stats.f_oneway(*arrays)
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("유한한 F 통계량과 p-value를 계산할 수 없습니다.")
    grand_mean = float(complete[value_column].mean())
    ss_between = sum(len(values) * (float(np.mean(values)) - grand_mean) ** 2 for _, values in groups)
    ss_total = float(((complete[value_column] - grand_mean) ** 2).sum())
    result = _base(info, "one_way_anova", [value_column, group_column], alpha,
                   len(frame), len(complete))
    result.update({
        "method": "one-way ANOVA",
        "null_hypothesis": "모든 그룹의 모집단 평균이 같다.",
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "degrees_of_freedom": {"between": len(groups) - 1, "within": len(complete) - len(groups)},
        "significant": bool(p_value < alpha),
        "groups": [
            {"value": _scalar(level), **_mean_summary(values, 1 - alpha)}
            for level, values in groups
        ],
        "effect_size": {"name": "eta_squared", "value": _number(ss_between / ss_total) if ss_total else None},
        "confidence_intervals": [
            {"parameter": "group_mean", "group": _scalar(level), **summary["confidence_interval"]}
            for level, values in groups
            for summary in [_mean_summary(values, 1 - alpha)]
        ],
        "assumptions": {
            "normality": [
                {"group": _scalar(level), **_normality(values)} for level, values in groups
            ],
            "equal_variance": {
                "diagnostic": "levene",
                "p_value": _number(stats.levene(*arrays).pvalue),
            },
            "independent_observations": "데이터만으로 검증할 수 없으므로 사용자/수집 설계 확인 필요",
        },
    })
    if any(item.get("p_value") is not None and item["p_value"] < 0.05
           for item in result["assumptions"]["normality"]):
        result["warnings"].append("하나 이상의 그룹이 정규성 진단의 0.05 기준을 충족하지 않습니다.")
    if result["assumptions"]["equal_variance"]["p_value"] is not None and \
            result["assumptions"]["equal_variance"]["p_value"] < alpha:
        result["warnings"].append("Levene 등분산 진단이 alpha 기준을 충족하지 않습니다.")
    return result


def _chi_square(frame, info, value_column, group_column, alpha):
    complete = frame[[value_column, group_column]].dropna()
    table = pd.crosstab(complete[value_column], complete[group_column], dropna=True)
    if not 2 <= table.shape[0] <= 20 or not 2 <= table.shape[1] <= 20:
        raise ValueError("카이제곱 검정의 각 범주 컬럼은 결측 제외 2~20개 수준이어야 합니다.")
    statistic, p_value, df, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = math.sqrt(float(statistic) / (n * denominator)) if n and denominator else None
    confidence_intervals = []
    effect_size = {"name": "cramers_v", "value": _number(cramers_v)}
    if table.shape == (2, 2):
        cells = table.to_numpy(dtype=float)
        corrected = cells + 0.5 if (cells == 0).any() else cells
        odds_ratio = corrected[0, 0] * corrected[1, 1] / (corrected[0, 1] * corrected[1, 0])
        se_log = math.sqrt(float((1 / corrected).sum()))
        critical = stats.norm.ppf(1 - alpha / 2)
        confidence_intervals.append({
            "parameter": "odds_ratio",
            "level": 1 - alpha,
            "lower": _number(math.exp(math.log(odds_ratio) - critical * se_log)),
            "upper": _number(math.exp(math.log(odds_ratio) + critical * se_log)),
        })
        effect_size["odds_ratio"] = _number(odds_ratio)
    expected_array = np.asarray(expected)
    result = _base(info, "chi_square", [value_column, group_column], alpha,
                   len(frame), len(complete))
    result.update({
        "method": "Pearson chi-square test of independence",
        "null_hypothesis": "두 범주형 변수는 독립이다.",
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "degrees_of_freedom": int(df),
        "significant": bool(p_value < alpha),
        "effect_size": effect_size,
        "confidence_intervals": confidence_intervals,
        "contingency": {
            "row_levels": [_scalar(value) for value in table.index],
            "column_levels": [_scalar(value) for value in table.columns],
            "observed": table.to_numpy(dtype=int).tolist(),
            "expected": [[_number(value) for value in row] for row in expected_array],
        },
        "assumptions": {
            "minimum_expected_count": _number(expected_array.min()),
            "cells_expected_below_5": int((expected_array < 5).sum()),
            "cells_expected_below_5_ratio": _number(float((expected_array < 5).mean())),
            "rule_of_thumb_met": bool(expected_array.min() >= 1 and (expected_array < 5).mean() <= 0.2),
            "independent_observations": "데이터만으로 검증할 수 없으므로 사용자/수집 설계 확인 필요",
        },
    })
    if not result["assumptions"]["rule_of_thumb_met"]:
        result["warnings"].append("기대빈도 기준을 충족하지 않아 p-value 해석에 주의가 필요합니다.")
    if table.shape != (2, 2):
        result["warnings"].append("일반 RxC 표의 Cramer's V 신뢰구간은 이 제한된 도구에서 계산하지 않습니다.")
    return result


def _mann_whitney(frame, info, value_column, group_column, alpha):
    complete, groups = _grouped_numeric(frame, value_column, group_column, exactly_two=True)
    (level_a, a), (level_b, b) = groups
    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        raise ValueError("유한한 U 통계량과 p-value를 계산할 수 없습니다.")
    rank_biserial = 2 * float(statistic) / (len(a) * len(b)) - 1
    result = _base(info, "mann_whitney", [value_column, group_column], alpha,
                   len(frame), len(complete))
    result.update({
        "method": "Mann-Whitney U test",
        "null_hypothesis": "두 그룹의 분포가 같다.",
        "statistic": _number(statistic),
        "p_value": _number(p_value),
        "degrees_of_freedom": None,
        "significant": bool(p_value < alpha),
        "groups": [
            {"value": _scalar(level_a), "n": len(a), "median": _number(np.median(a))},
            {"value": _scalar(level_b), "n": len(b), "median": _number(np.median(b))},
        ],
        "effect_size": {"name": "rank_biserial_correlation", "value": _number(rank_biserial)},
        "assumptions": {
            "independent_observations": "데이터만으로 검증할 수 없으므로 사용자/수집 설계 확인 필요",
            "similar_distribution_shape": "데이터와 도메인 기준으로 별도 확인 필요",
        },
    })
    result["warnings"].append("이 제한된 구현은 Mann-Whitney 효과크기의 신뢰구간을 계산하지 않습니다.")
    return result


def _mean_ci(frame, info, value_column, alpha):
    if not is_numeric_dtype(frame[value_column]):
        raise ValueError("mean_ci의 value_column은 수치형이어야 합니다.")
    values = frame[value_column].dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("평균 신뢰구간에는 결측 제외 관측치가 최소 2개 필요합니다.")
    summary = _mean_summary(values, 1 - alpha)
    result = _base(info, "mean_ci", [value_column], alpha, len(frame), len(values))
    result.update({
        "method": "one-sample t confidence interval for the mean",
        "null_hypothesis": None,
        "statistic": None,
        "p_value": None,
        "degrees_of_freedom": len(values) - 1,
        "estimate": {"name": "sample_mean", "value": summary["mean"]},
        "effect_size": None,
        "confidence_intervals": [{
            "parameter": "population_mean",
            **summary["confidence_interval"],
        }],
        "assumptions": {
            "normality": _normality(values),
            "independent_observations": "데이터만으로 검증할 수 없으므로 사용자/수집 설계 확인 필요",
        },
    })
    if result["assumptions"]["normality"].get("p_value") is not None and \
            result["assumptions"]["normality"]["p_value"] < 0.05:
        result["warnings"].append("평균 신뢰구간의 정규성 진단이 0.05 기준을 충족하지 않습니다. 표본 크기와 분포를 함께 확인하세요.")
    return result


def statistical_test(store: DatasetStore, dataset_id: str, *, test: str,
                     value_column: str, group_column: str = "",
                     paired_column: str = "", alpha: float = 0.05) -> dict[str, Any]:
    """Run one declared statistical method and return structured evidence."""
    if test not in SUPPORTED_TESTS:
        raise ValueError("지원하지 않는 통계 검정입니다.")
    if not 0.001 <= float(alpha) <= 0.2:
        raise ValueError("alpha는 0.001~0.2 범위여야 합니다.")
    info = store.metadata[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("통계 검정은 집계되지 않은 raw dataset에서만 실행합니다.")
    required = [value_column]
    if test in {"independent_t", "chi_square", "one_way_anova", "mann_whitney"}:
        if not group_column:
            raise ValueError("이 검정에는 group_column이 필요합니다.")
        required.append(group_column)
    if test == "paired_t":
        if not paired_column:
            raise ValueError("paired_t에는 paired_column이 필요합니다.")
        required.append(paired_column)
    if len(set(required)) != len(required) or any(column not in info.columns for column in required):
        raise ValueError("검정 컬럼은 서로 달라야 하며 모두 dataset에 존재해야 합니다.")
    frame = project_dataset(store, dataset_id, required)

    runners = {
        "independent_t": lambda: _independent_t(frame, info, value_column, group_column, float(alpha)),
        "paired_t": lambda: _paired_t(frame, info, value_column, paired_column, float(alpha)),
        "chi_square": lambda: _chi_square(frame, info, value_column, group_column, float(alpha)),
        "one_way_anova": lambda: _anova(frame, info, value_column, group_column, float(alpha)),
        "mann_whitney": lambda: _mann_whitney(frame, info, value_column, group_column, float(alpha)),
        "mean_ci": lambda: _mean_ci(frame, info, value_column, float(alpha)),
    }
    evidence = runners[test]()
    return {
        "status": "ready",
        "dataset_id": dataset_id,
        "test_result": evidence,
        "scope": (
            f"{info.source}의 로딩된 raw dataset {len(frame):,}행에서 {evidence['sample']['complete_rows']:,}행을 "
            f"사용한 {evidence['method']} 결과입니다. coverage={info.coverage}; snapshot={info.snapshot or 'unknown'}."
        ),
    }
