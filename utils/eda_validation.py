from dataclasses import dataclass
import re
from typing import Any, Optional, Sequence

import pandas as pd

from utils.table_context import resolve_column_from_prompt


INTERNAL_CONTEXT_MARKER = "\n\n[중요 컨텍스트]"


@dataclass(frozen=True)
class EDAValidationResult:
    ok: bool
    column: Optional[Any] = None
    dtype: str = ""
    chart_type: str = ""
    reason: str = ""


@dataclass(frozen=True)
class DataSufficiencyResult:
    sufficient: bool
    missing_columns: tuple[str, ...] = ()
    reason: str = ""


def strip_internal_eda_context(prompt: str) -> str:
    """Return only the user-authored part of an EDA prompt."""

    return (prompt or "").split(INTERNAL_CONTEXT_MARKER, 1)[0]


def find_exact_prompt_column(df: pd.DataFrame, prompt: str, *, table_context=None) -> Optional[Any]:
    context_column = resolve_column_from_prompt(prompt, table_context)
    if context_column is not None:
        return context_column

    user_prompt = strip_internal_eda_context(prompt)
    lowered = user_prompt.lower()
    matches = [
        column
        for column in df.columns
        if str(column) in user_prompt or str(column).lower() in lowered
    ]
    return matches[0] if len(matches) == 1 else None


def _requested_identifier_tokens(prompt: str) -> list[str]:
    ignore = {
        "df",
        "df_a",
        "data",
        "chart",
        "plot",
        "hist",
        "histogram",
        "box",
        "boxplot",
        "kde",
        "sql",
    }
    user_prompt = strip_internal_eda_context(prompt)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", user_prompt)
    return [token for token in tokens if token.lower() not in ignore]


def choose_distribution_chart(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return ""
    if pd.api.types.is_numeric_dtype(clean):
        return "hist" if len(clean) > 100 else "boxplot"
    return "bar"


def validate_eda_visualization_request(df: Any, prompt: str, *, table_context=None) -> EDAValidationResult:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return EDAValidationResult(ok=False, reason="df_A 데이터가 비어 있어 시각화할 수 없습니다.")

    column = find_exact_prompt_column(df, prompt, table_context=table_context)
    if column is None:
        requested_tokens = _requested_identifier_tokens(prompt)
        if requested_tokens:
            available = {str(column).lower() for column in df.columns}
            missing = [token for token in requested_tokens if token.lower() not in available]
            if missing:
                return EDAValidationResult(
                    ok=False,
                    reason=f"요청한 컬럼을 df_A에서 찾을 수 없습니다: {missing}. 사용 가능한 컬럼: {list(df.columns)}",
                )
        return EDAValidationResult(
            ok=True,
            reason="프롬프트에서 단일 컬럼명을 확정하지 못해 EDA Agent가 직접 판단합니다.",
        )

    if column not in df.columns:
        return EDAValidationResult(ok=False, column=column, reason=f"`{column}` 컬럼이 df_A에 없습니다.")

    series = df[column].dropna()
    dtype = str(df[column].dtype)
    if series.empty:
        return EDAValidationResult(
            ok=False,
            column=column,
            dtype=dtype,
            reason=f"`{column}` 컬럼에 시각화 가능한 값이 없습니다.",
        )

    chart_type = choose_distribution_chart(series)
    if not chart_type:
        return EDAValidationResult(
            ok=False,
            column=column,
            dtype=dtype,
            reason=f"`{column}` 컬럼의 분포 시각화 유형을 결정할 수 없습니다.",
        )

    return EDAValidationResult(
        ok=True,
        column=column,
        dtype=dtype,
        chart_type=chart_type,
        reason="시각화 전 검증을 통과했습니다.",
    )


def validate_data_sufficiency(df: Any, required_columns: Sequence[str]) -> DataSufficiencyResult:
    """Validate whether the current df_A is sufficient for the requested task."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return DataSufficiencyResult(
            sufficient=False,
            missing_columns=tuple(required_columns),
            reason="df_A가 비어 있어 요청을 수행하기에 충분하지 않습니다.",
        )

    available = {str(column) for column in df.columns}
    missing = tuple(column for column in required_columns if str(column) not in available)
    if missing:
        return DataSufficiencyResult(
            sufficient=False,
            missing_columns=missing,
            reason=f"현재 df_A는 요청 대비 충분하지 않습니다. 누락 컬럼: {list(missing)}. 사용 가능한 컬럼: {list(df.columns)}",
        )
    return DataSufficiencyResult(
        sufficient=True,
        reason="현재 df_A가 요청 수행에 필요한 컬럼을 포함합니다.",
    )


__all__ = [
    "EDAValidationResult",
    "DataSufficiencyResult",
    "choose_distribution_chart",
    "find_exact_prompt_column",
    "strip_internal_eda_context",
    "validate_data_sufficiency",
    "validate_eda_visualization_request",
]
