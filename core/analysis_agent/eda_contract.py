"""Explicit EDA options grounded in the request, independent of model output."""
import re


def histogram_bins(text):
    patterns = (
        r'구간\s*(?:수|개수)\s*(?:는|를|을|만|:|=)?\s*(\d+)',
        r'(\d+)\s*(?:개(?:의)?\s*)?(?:구간|bins?\b)',
        r'\bbins?\s*(?:count\s*)?(?:=|:|of)?\s*(\d+)',
    )
    values = {int(m.group(1)) for pattern in patterns for m in re.finditer(pattern, text, re.I)}
    return next(iter(values)) if len(values) == 1 else None


def profile_request_text(text):
    # Remove only an explicit quoted-token conversion clause, not requests
    # for missing counts/ratios alongside a calculation.
    return re.sub(r'''["'`][^"'`\n]{1,64}["'`]\s*(?:문자열)?(?:만|을|를)?\s*
        (?:결측값?|NULL|NaN)(?:으로|로)?\s*(?:처리|변환|간주)(?:해줘|해|하세요|한다)?''',
        '', text, flags=re.I | re.X)


def group_columns(text, columns):
    return [column for column in columns if re.search(
        r'(?<![A-Za-z0-9_])[`"\']?' + re.escape(column)
        + r'[`"\']?\s*(?:별|마다|기준으로)', text, re.I)]
