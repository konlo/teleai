import datetime
import json
import os
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import streamlit as st

from modules.dataload.databricks_sql_loader import run_sql
from utils.session import databricks_connector_available, get_databricks_credentials


LOG_TABLE = "workspace.telly_log.chatbot_turn_logs"


def _escape(value: str) -> str:
    """Escape single quotes for safe SQL string literals."""
    return value.replace("'", "''")


def _literal(value: Any) -> str:
    """Best-effort conversion to a SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{_escape(str(value))}'"


def _array_literal(values: Iterable[str]) -> Optional[str]:
    items = list(values or [])
    if not items:
        return None
    escaped = [f"'{_escape(str(v))}'" for v in items]
    return f"ARRAY({', '.join(escaped)})"


def _default_user_id() -> str:
    """Return a stable default user identifier."""
    return st.session_state.get("user_id_hash") or os.environ.get("USER_ID_HASH", "konlo.na")


def _ensure_log_table() -> None:
    """Placeholder to keep compatibility; table creation is handled externally."""


def build_turn_payload(
    *,
    llm,
    conversation_id: str,
    turn_id: int,
    user_message: str,
    assistant_message: str,
    intent: str,
    tools_used,
    generated_sql: str,
    sql_execution_status: Optional[str],
    sql_error_message: str,
    result_row_count: Optional[int],
    result_schema_json: Optional[str],
    result_sample_json: Optional[Any],
    latency_ms: int,
    df_latest: Optional[pd.DataFrame],
    generated_python: str,
    python_execution_status: Optional[str],
    python_error_message: str,
    python_output_summary: str,
) -> Dict[str, Any]:
    """Construct a logging payload for a single turn."""
    model_id = getattr(llm, "model", None) or getattr(llm, "model_name", None) or ""
    payload: Dict[str, Any] = {
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "event_ts": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "event_date": datetime.datetime.utcnow().date().isoformat(),
        "system_prompt_version": st.session_state.get("system_prompt_version", "v1"),
        "model_id": model_id,
        "temperature": getattr(llm, "temperature", None),
        "max_tokens": getattr(llm, "max_tokens", None),
        "user_message": user_message,
        "assistant_message": assistant_message,
        "intent": intent,
        "tools_used": tools_used,
        "generated_sql": generated_sql,
        "sql_execution_status": sql_execution_status,
        "sql_error_message": sql_error_message,
        "result_row_count": result_row_count,
        "result_schema_json": result_schema_json,
        "result_sample_json": result_sample_json,
        "latency_ms": latency_ms,
        "generated_python": generated_python,
        "python_execution_status": python_execution_status,
        "python_error_message": python_error_message,
        "python_output_summary": python_output_summary,
    }

    if isinstance(df_latest, pd.DataFrame) and not df_latest.empty:
        payload.setdefault("result_row_count", len(df_latest))
        payload.setdefault(
            "result_schema_json",
            json.dumps({c: str(t) for c, t in df_latest.dtypes.items()}, ensure_ascii=False),
        )
        payload.setdefault("result_sample_json", df_latest.head(5).to_dict(orient="records"))

    return payload


def log_turn(payload: Dict[str, Any]) -> None:
    """
    Persist a single turn to Databricks.

    Only a lightweight subset of fields is required; missing fields will be NULL.
    """
    if not payload.get("user_id_hash"):
        payload["user_id_hash"] = _default_user_id()
    for key in (
        "generated_python",
        "python_execution_status",
        "python_error_message",
        "python_output_summary",
        "python_artifact_paths",
    ):
        payload.setdefault(key, None)
    if not databricks_connector_available():
        return
    try:
        _ensure_log_table()
    except Exception:
        return

    columns = []
    values = []
    for key, value in payload.items():
        literal = None
        if key in {"tools_used", "issue_types", "python_artifact_paths"}:
            literal = _array_literal(value) if value else None
        elif key in {"result_schema_json", "result_sample_json"} and not isinstance(
            value, str
        ):
            literal = _literal(json.dumps(value, ensure_ascii=False))
        else:
            literal = _literal(value)
        if literal is None:
            continue
        columns.append(key)
        values.append(literal)

    if not columns:
        return

    cols_sql = ", ".join(columns)
    vals_sql = ", ".join(values)
    statement = f"INSERT INTO {LOG_TABLE} ({cols_sql}) VALUES ({vals_sql})"

    try:
        creds = get_databricks_credentials()
        run_sql(statement, creds)
    except Exception as exc:  # pragma: no cover - network/SQL errors
        st.warning(f"로그 저장 실패: {exc}")
