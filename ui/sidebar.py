from typing import List

import streamlit as st

from utils.session import (
    ensure_session_state,
    databricks_connector_available,
    list_databricks_tables_in_session,
    update_databricks_namespace_from_table,
)


def render_sidebar() -> None:
    """Render the Streamlit sidebar controls for language and Databricks access."""
    ensure_session_state()

    with st.sidebar:
        st.markdown("### 💬 EDA 설명 언어")
        lang_options = ["English", "한국어"]
        current_lang = st.session_state.get("explanation_lang", "English")
        selected_idx = (
            lang_options.index(current_lang) if current_lang in lang_options else 0
        )
        st.session_state["explanation_lang"] = st.selectbox(
            "Agent 요약 언어",
            options=lang_options,
            index=selected_idx,
        )

        st.markdown("---")
        st.markdown("### 🧱 Databricks 테이블")
        if not databricks_connector_available():
            st.info(
                "databricks-sql-connector가 설치되어 있지 않습니다. "
                "`pip install databricks-sql-connector` 후 다시 시도해주세요."
            )
            return

        cfg = st.session_state.get("databricks_config", {})
        server_hostname = cfg.get("server_hostname", "")
        http_path = cfg.get("http_path", "")
        access_token = cfg.get("access_token", "")
        catalog = cfg.get("catalog", "") or st.session_state.get(
            "databricks_selected_catalog", ""
        )
        schema = cfg.get("schema", "") or st.session_state.get(
            "databricks_selected_schema", ""
        )

        if not (server_hostname and http_path and access_token):
            st.error(
                "환경 변수 DATABRICKS_HOST / DATABRICKS_HTTP_PATH / "
                "DATABRICKS_TOKEN 값을 .env에 설정 후 앱을 재시작하세요."
            )
            return

        st.session_state.setdefault("databricks_selected_catalog", catalog or "hive_metastore")
        st.session_state.setdefault("databricks_selected_schema", schema or "default")

        st.caption("Databricks Connection (.env)")
        st.write(f"• Server Hostname: `{server_hostname}`")
        st.write(f"• HTTP Path: `{http_path}`")
        st.write(f"• Catalog: `{st.session_state['databricks_selected_catalog']}`")
        st.write(f"• Schema: `{st.session_state['databricks_selected_schema']}`")

        st.markdown("---")
        st.markdown("#### 사용 가능한 테이블")
        refresh_clicked = st.button("🔄 테이블 새로고침", use_container_width=True)

        table_options: List[str] = st.session_state.get("databricks_table_options", [])
        if refresh_clicked or not table_options:
            with st.spinner("Databricks 테이블 목록을 불러오는 중..."):
                ok, _, message = list_databricks_tables_in_session()
            if ok and message:
                st.caption(message)
            if not ok:
                st.error(message)

        table_options = st.session_state.get("databricks_table_options", [])
        selected_table = st.session_state.get("databricks_selected_table", "").strip()

        if not table_options:
            st.info(
                "접근 가능한 Databricks 테이블을 찾을 수 없습니다. "
                "권한을 확인한 뒤 새로고침해주세요."
            )
            return

        if selected_table not in table_options:
            selected_table = table_options[0]
            st.session_state["databricks_selected_table"] = selected_table
            st.session_state["databricks_table_input"] = selected_table
            update_databricks_namespace_from_table(selected_table)

        default_index = table_options.index(st.session_state["databricks_selected_table"])
        current_choice = st.selectbox(
            "테이블 선택",
            options=table_options,
            index=default_index,
        )
        if current_choice != st.session_state["databricks_selected_table"]:
            st.session_state["databricks_selected_table"] = current_choice
            st.session_state["databricks_table_input"] = current_choice
            update_databricks_namespace_from_table(current_choice)

        st.caption(
            f"현재 선택된 테이블: `{st.session_state['databricks_selected_table']}` "
            "— 프롬프트에서 자동으로 사용됩니다."
        )
