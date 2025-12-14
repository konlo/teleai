import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.session import (
    get_default_sql_limit,
    load_preview_from_databricks_query,
)


AssistantAppender = Callable[[str, str, str], None]
FigureAttacher = Callable[[str, List[Dict[str, Any]]], None]

CODE_FENCE_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def infer_table_from_sql(sql: str) -> str:
    """Extract table name from a SQL string's FROM clause if possible."""

    text = (sql or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    marker = " from "
    idx = lowered.find(marker)
    if idx == -1:
        if lowered.startswith("from "):
            idx = 0
        else:
            return ""
    idx += len(marker)
    remainder = text[idx:].strip()
    if not remainder:
        return ""
    candidate = remainder.split()[0]
    candidate = candidate.rstrip(";,)")
    return candidate.strip()


def sanitize_sql_text(sql: str) -> str:
    """Return SQL stripped of code fences and stray trailing backticks."""

    text = (sql or "").strip()
    if not text:
        return ""

    match = CODE_FENCE_PATTERN.search(text)
    if match:
        text = match.group(1).strip()

    if text.count("`") % 2:
        while text.endswith("`"):
            text = text[:-1].rstrip()

    return text


def ensure_limit_clause(sql: str, limit: Optional[int] = None) -> str:
    """Inject or normalize a LIMIT clause at the end of SQL."""

    limit_value = limit if limit is not None else get_default_sql_limit()
    text = sanitize_sql_text(sql)
    if not text:
        return text

    semicolon = "" if not text.endswith(";") else ";"
    body = text[:-1].rstrip() if semicolon else text

    pattern = re.compile(r"(?is)\blimit\s+\d+(\s+offset\s+\d+)?\s*$")
    match = pattern.search(body)
    if match:
        prefix = body[: match.start()].rstrip()
        offset_part = (match.group(1) or "").upper()
        body = f"{prefix} LIMIT {limit_value}{offset_part}"
    else:
        body = f"{body.rstrip()} LIMIT {limit_value}"

    return f"{body}{semicolon}"


def execute_sql_preview(
    *,
    run_id: str,
    sql_text: str,
    log_container,
    append_assistant_message: AssistantAppender,
    attach_figures_to_run: FigureAttacher,
    show_logs: bool = True,
    auto_trigger: bool = False,
) -> bool:
    """
    Execute SQL against Databricks, attach preview payloads, and log UI messages.
    Returns True on success.
    """

    sql_to_run = sanitize_sql_text(sql_text)
    if not sql_to_run:
        warning_msg = "실행할 SQL이 없습니다. 먼저 SQL Builder로 쿼리를 생성해주세요."
        st.warning(warning_msg)
        append_assistant_message(run_id, warning_msg, "SQL Execution")
        st.session_state["active_run_id"] = None
        return False

    st.session_state["last_sql_statement"] = sql_to_run
    st.session_state["last_agent_mode"] = "SQL Builder"
    if show_logs and log_container is not None:
        st.session_state["log_has_content"] = True
        log_container.empty()
        with log_container.container():
            st.subheader("실시간 실행 로그")
            status_msg = (
                "SQL Builder가 생성한 쿼리를 Databricks에서 실행합니다."
                if auto_trigger
                else "SQL Builder의 마지막 쿼리를 Databricks에서 실행합니다."
            )
            st.write(status_msg)
    else:
        st.session_state["log_has_content"] = False

    cfg = st.session_state.get("databricks_config", {})
    catalog = cfg.get("catalog") or "hive_metastore"
    schema = cfg.get("schema") or "default"
    cfg["catalog"] = catalog
    cfg["schema"] = schema
    st.session_state["databricks_config"] = cfg
    st.session_state.setdefault("databricks_selected_catalog", catalog)
    st.session_state.setdefault("databricks_selected_schema", schema)

    table_name_input = st.session_state.get("databricks_table_input", "").strip()
    selected_table = st.session_state.get("databricks_selected_table", "").strip()
    table_name_inferred = infer_table_from_sql(sql_to_run)
    table_name = (
        table_name_input
        or selected_table
        or table_name_inferred
        or st.session_state.get("last_sql_table", "")
    )
    if not table_name:
        warning_msg = (
            "실행할 테이블을 결정할 수 없습니다. SQL Builder에서 사용할 테이블을 지정하거나 Sidebar에서 테이블을 선택해주세요."
        )
        st.warning(warning_msg)
        append_assistant_message(run_id, warning_msg, "SQL Execution")
        st.session_state["active_run_id"] = None
        return False

    answer_container = st.container()
    with st.spinner("Databricks SQL 실행 중..."):
        success, message = load_preview_from_databricks_query(
            table_name,
            query=sql_to_run,
            target="A",
            limit=10,
        )
    preview_payloads: List[Dict[str, Any]] = []
    with answer_container:
        st.subheader("Answer")
        if success:
            st.success(message)
        else:
            st.error(message)

    if success:
        st.session_state["last_agent_mode"] = "EDA Analyst"
        st.session_state["last_sql_table"] = table_name
        st.session_state["databricks_table_input"] = table_name
        st.session_state["databricks_selected_table"] = table_name
        st.session_state["skip_next_df_a_preview"] = True
        df_latest = st.session_state.get("df_A_data")
        if isinstance(df_latest, pd.DataFrame) and not df_latest.empty:
            preview_payloads.append(
                {
                    "kind": "dataframe",
                    "title": f"df_A Preview — {st.session_state.get('df_A_name', 'df_A')}",
                    "data": df_latest.head(10),
                }
            )

    mode_label = "SQL Execution" if auto_trigger else "SQL Builder"
    append_assistant_message(run_id, message, mode_label)
    if preview_payloads:
        attach_figures_to_run(run_id, preview_payloads)
    st.session_state["active_run_id"] = None
    st.session_state["last_sql_status"] = "success" if success else "fail"
    st.session_state["last_sql_error"] = "" if success else message
    if success and auto_trigger:
        st.session_state["pending_rerun"] = True
    return success


__all__ = [
    "CODE_FENCE_PATTERN",
    "ensure_limit_clause",
    "execute_sql_preview",
    "infer_table_from_sql",
    "sanitize_sql_text",
]
