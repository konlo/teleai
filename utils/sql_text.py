from __future__ import annotations

import re


SQL_CODE_FENCE_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
SQL_START_PATTERN = re.compile(r"(?is)^\s*(select|with)\b")
SQL_LINE_START_PATTERN = re.compile(r"(?im)^\s*(select|with)\b")
SQL_STOP_MARKERS = (
    "\n\nExplanation:",
    "\nExplanation:",
    "\n\nExecution:",
    "\nExecution:",
    "\n\n설명:",
    "\n설명:",
)


def strip_sql_code_fence(text: str) -> str:
    value = (text or "").strip()
    match = SQL_CODE_FENCE_PATTERN.search(value)
    if match:
        return match.group(1).strip()
    return value


def extract_sql_from_text(text: str) -> str:
    """Extract a SQL statement from agent text, including plain SELECT/WITH output."""

    value = (text or "").strip()
    if not value:
        return ""

    fenced = SQL_CODE_FENCE_PATTERN.search(value)
    if fenced:
        candidate = fenced.group(1).strip()
    elif "SQL:" in value:
        candidate = value.split("SQL:", 1)[1].strip()
    else:
        line_match = SQL_LINE_START_PATTERN.search(value)
        if not line_match:
            return ""
        candidate = value[line_match.start() :].strip()

    for marker in SQL_STOP_MARKERS:
        if marker in candidate:
            candidate = candidate.split(marker, 1)[0].strip()

    candidate = strip_sql_code_fence(candidate)
    if candidate.count("`") % 2:
        while candidate.endswith("`"):
            candidate = candidate[:-1].rstrip()

    if not SQL_START_PATTERN.match(candidate):
        return ""
    if not re.search(r"(?is)\bfrom\b", candidate):
        return ""
    return candidate


__all__ = [
    "SQL_CODE_FENCE_PATTERN",
    "extract_sql_from_text",
    "strip_sql_code_fence",
]
