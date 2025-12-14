from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st
from langchain_core.tools import BaseTool, tool

from core.sql_utils import infer_table_from_sql
from utils.session import (
    list_databricks_catalogs_in_session,
    list_databricks_schemas_in_session,
    list_databricks_tables_in_session,
    load_preview_from_databricks_query,
)


def _df_preview_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return ""
    head = df.head(max_rows)
    try:
        return head.to_markdown(index=False)
    except Exception:  # pragma: no cover - tabulate dependency missing
        return head.to_string(index=False)


@tool
def databricks_list_catalogs() -> str:
    """List available Databricks catalogs using current credentials."""
    ok, df, message = list_databricks_catalogs_in_session()
    if not ok:
        return f"[databricks_list_catalogs:error] {message}"
    body = _df_preview_markdown(df) if df is not None else ""
    return message + ("\n" + body if body else "")


@tool
def databricks_list_schemas(catalog: str = "") -> str:
    """List schemas for a given catalog (uses selected catalog by default)."""
    catalog_clean = catalog.strip() if isinstance(catalog, str) else ""
    if not catalog_clean:
        catalog_clean = st.session_state.get("databricks_selected_catalog", "")
    if not catalog_clean:
        return "Catalog name required."
    ok, df, message = list_databricks_schemas_in_session(catalog_clean)
    if not ok:
        return f"[databricks_list_schemas:error] {message}"
    body = _df_preview_markdown(df) if df is not None else ""
    return message + ("\n" + body if body else "")


@tool
def databricks_list_tables(pattern: str = "") -> str:
    """List tables for the selected catalog/schema (optional LIKE pattern)."""
    ok, df, message = list_databricks_tables_in_session(pattern)
    if not ok:
        return f"[databricks_list_tables:error] {message}"
    body = _df_preview_markdown(df) if df is not None else ""
    return message + ("\n" + body if body else "")


@tool
def databricks_preview_sql(
    query: str,
    target: str = "df_A",
    label: str = "",
) -> str:
    """Execute a Databricks SQL query and load the result into df_A or df_B."""
    target_raw = str(target).strip().upper()
    if target_raw in {"B", "DF_B", "DFB"}:
        target_key = "B"
    else:
        target_key = "A"

    cfg = st.session_state.get("databricks_config", {})
    catalog = cfg.get("catalog") or "hive_metastore"
    schema = cfg.get("schema") or "default"
    cfg["catalog"] = catalog
    cfg["schema"] = schema
    st.session_state["databricks_config"] = cfg
    if not st.session_state.get("databricks_selected_catalog"):
        st.session_state["databricks_selected_catalog"] = catalog
    if not st.session_state.get("databricks_selected_schema"):
        st.session_state["databricks_selected_schema"] = schema

    table_name_input = st.session_state.get("databricks_table_input", "").strip()
    selected_table = st.session_state.get("databricks_selected_table", "").strip()
    table_name = (
        table_name_input
        or selected_table
        or infer_table_from_sql(query)
        or st.session_state.get("last_sql_table", "")
    )
    if not table_name:
        return (
            "[databricks_preview_sql:error] 실행할 테이블을 결정할 수 없습니다. "
            "Sidebar에서 테이블을 선택하거나 SQL에 FROM 절을 명확히 지정해주세요."
        )

    success, message = load_preview_from_databricks_query(
        table_name,
        query=query,
        target=target_key,
        limit=10,
    )
    if success:
        st.session_state["last_sql_table"] = table_name
        st.session_state["databricks_table_input"] = table_name
        st.session_state["databricks_selected_table"] = table_name
        return message

    lowered = message.lower()
    if "uc_hive_metastore_disabled_exception" in lowered:
        hint = (
            "Unity Catalog 기본 카탈로그/스키마가 지정되지 않아 실행에 실패했습니다. "
            "Databricks Loader에서 catalog/schema를 먼저 설정한 뒤 다시 시도해주세요."
        )
        return f"[databricks_preview_sql:error] {hint}"

    return f"[databricks_preview_sql:error] {message}"


def build_sql_tools() -> Sequence[BaseTool]:
    """Return the toolset for the Databricks SQL agent."""
    return [
        databricks_list_catalogs,
        databricks_list_schemas,
        databricks_list_tables,
        databricks_preview_sql,
    ]


__all__ = [
    "build_sql_tools",
]
