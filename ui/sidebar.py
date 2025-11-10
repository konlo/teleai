from typing import List

import pandas as pd
import streamlit as st

from utils.session import (
    ensure_session_state,
    databricks_connector_available,
    list_databricks_tables_in_session,
    load_df_from_databricks,
    update_databricks_namespace_from_table,
)


def render_sidebar(show_debug: bool = True) -> None:
    """Render the Streamlit sidebar controls for Databricks access."""
    ensure_session_state()

    cfg = st.session_state.get("databricks_config", {})
    catalog = cfg.get("catalog", "") or st.session_state.get(
        "databricks_selected_catalog", ""
    )
    schema = cfg.get("schema", "") or st.session_state.get(
        "databricks_selected_schema", ""
    )

    st.session_state.setdefault("databricks_selected_catalog", catalog or "hive_metastore")
    st.session_state.setdefault("databricks_selected_schema", schema or "default")

    with st.sidebar:
        st.markdown("### 고객사 Telemetry Database")
        if not databricks_connector_available():
            st.info(
                "databricks-sql-connector가 설치되어 있지 않습니다. "
                "`pip install databricks-sql-connector` 후 다시 시도해주세요."
            )
            return

        server_hostname = cfg.get("server_hostname", "")
        http_path = cfg.get("http_path", "")
        access_token = cfg.get("access_token", "")

        if not (server_hostname and http_path and access_token):
            st.error(
                "환경 변수 DATABRICKS_HOST / DATABRICKS_HTTP_PATH / "
                "DATABRICKS_TOKEN 값을 .env에 설정 후 앱을 재시작하세요."
            )
            return

        if show_debug:
            st.caption("Databricks Connection (.env)")
            st.write(f"• Server Hostname: `{server_hostname}`")
            st.write(f"• HTTP Path: `{http_path}`")
            st.write(f"• Catalog: `{st.session_state['databricks_selected_catalog']}`")
            st.write(f"• Schema: `{st.session_state['databricks_selected_schema']}`")
            st.markdown("---")
        st.markdown("#### 현재 선택된 Data")
        preview_status = st.empty()

        table_options: List[str] = st.session_state.get("databricks_table_options", [])
        list_refreshed = False
        if not table_options:
            with st.spinner("Databricks 테이블 목록을 불러오는 중..."):
                ok, _, message = list_databricks_tables_in_session()
                list_refreshed = True
            if ok and message:
                st.caption(message)
            if not ok:
                st.error(message)
            st.session_state["databricks_last_preview_table"] = ""

        table_options = st.session_state.get("databricks_table_options", [])
        selected_table = st.session_state.get("databricks_selected_table", "").strip()

        if not table_options:
            st.info(
                "접근 가능한 Databricks 테이블을 찾을 수 없습니다. "
                "환경 설정과 권한을 다시 확인해주세요."
            )
            return

        if list_refreshed or selected_table not in table_options:
            selected_table = table_options[0]
            st.session_state["databricks_selected_table"] = selected_table
            st.session_state["databricks_table_input"] = selected_table
            update_databricks_namespace_from_table(selected_table)

        default_index = table_options.index(st.session_state["databricks_selected_table"])
        def _table_display_name(table_ref: str) -> str:
            """Return the table name portion for display in the select box."""

            return table_ref.split(".")[-1] if table_ref else ""

        current_choice = st.selectbox(
            "테이블 선택",
            options=table_options,
            index=default_index,
            format_func=_table_display_name,
        )
        if current_choice != st.session_state["databricks_selected_table"]:
            st.session_state["databricks_selected_table"] = current_choice
            st.session_state["databricks_table_input"] = current_choice
            update_databricks_namespace_from_table(current_choice)
            st.session_state["databricks_last_preview_table"] = ""

        column_select_container = st.container()

        st.caption(
            f"현재 선택된 테이블: `{st.session_state['databricks_selected_table']}` "
            "— 프롬프트에서 자동으로 사용됩니다."
        )

        final_selection = st.session_state["databricks_selected_table"].strip()
        last_preview_table = st.session_state.get("databricks_last_preview_table", "").strip()
        df_a_data = st.session_state.get("df_A_data")
        needs_preview = bool(final_selection) and (
            final_selection != last_preview_table or df_a_data is None
        )

        if needs_preview:
            with st.spinner("선택한 테이블 미리보기를 불러오는 중..."):
                ok, message = load_df_from_databricks(final_selection, limit=10)
            if ok:
                df_init = st.session_state.get("df_A_data")
                if isinstance(df_init, pd.DataFrame):
                    st.session_state["df_init_data"] = df_init.copy(deep=True)
                preview_status.success(message)
            else:
                preview_status.error(message)
        else:
            last_message = st.session_state.get("databricks_last_preview_message", "")
            if final_selection and last_message and final_selection == last_preview_table:
                preview_status.caption(last_message)

        with column_select_container:
            column_key = "databricks_selected_column"
            df_a_data = st.session_state.get("df_A_data")
            selected_column_value = ""

            column_source_table = st.session_state.get("databricks_column_source_table", "")
            stored_columns = st.session_state.get("databricks_column_options", []) or []
            column_options: List[str] = []

            st.markdown("**Telemetry 항목**")

            if final_selection and final_selection == column_source_table:
                column_options = list(stored_columns)
            elif not final_selection and isinstance(df_a_data, pd.DataFrame) and not df_a_data.empty:
                column_options = list(df_a_data.columns)

            if column_options:
                selected_column = st.session_state.get(column_key, "")
                if selected_column not in column_options:
                    selected_column = column_options[0]
                    st.session_state[column_key] = selected_column
                placeholder_key = f"{column_key}_placeholder"
                if placeholder_key in st.session_state:
                    del st.session_state[placeholder_key]
                selected_column_value = st.selectbox(
                    "Telemetry 항목",
                    options=column_options,
                    key=column_key,
                    help="불러온 테이블의 컬럼을 확인하세요.",
                    label_visibility="collapsed",
                )

                if selected_column_value:
                    st.code(selected_column_value, language="text")
            else:
                st.session_state[column_key] = ""
                placeholder_options = ["컬럼 정보를 불러오는 중..."] if final_selection else ["불러온 데이터가 없습니다."]
                st.selectbox(
                    "Telemetry 항목",
                    options=placeholder_options,
                    index=0,
                    disabled=True,
                    key=f"{column_key}_placeholder",
                    label_visibility="collapsed",
                )
