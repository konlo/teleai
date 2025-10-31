from typing import List

import streamlit as st

from utils.session import (
    databricks_connector_available,
    list_databricks_catalogs_in_session,
    list_databricks_schemas_in_session,
    list_databricks_tables_in_session,
    load_df_from_databricks,
    load_preview_from_databricks_query,
    DEFAULT_TABLE_SUGGESTIONS,
    generate_select_all_query,
    update_databricks_namespace_from_table,
)


def render_sidebar() -> None:
    """Render the Streamlit sidebar controls for data selection and language."""
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
        st.markdown("### 🧱 Databricks Loader")
        if not databricks_connector_available():
            st.info(
                "databricks-sql-connector가 설치되어 있지 않습니다. "
                "`pip install databricks-sql-connector` 후 다시 시도해주세요."
            )
        else:
            cfg = st.session_state.get("databricks_config", {})
            server_hostname = cfg.get("server_hostname", "")
            http_path = cfg.get("http_path", "")
            access_token = cfg.get("access_token", "")
            catalog = cfg.get("catalog", "")
            schema = cfg.get("schema", "")

            if not (server_hostname and http_path and access_token):
                st.error(
                    "환경 변수 DATABRICKS_HOST / DATABRICKS_HTTP_PATH / "
                    "DATABRICKS_TOKEN 값을 .env에 설정 후 앱을 재시작하세요."
                )
                return

            st.caption("Databricks Connection (환경 변수에서 자동 설정)")
            st.write(f"• Server Hostname: `{server_hostname}`")
            st.write(f"• HTTP Path: `{http_path}`")
            if catalog:
                st.write(f"• Catalog: `{catalog}`")
            if schema:
                st.write(f"• Schema: `{schema}`")

            st.markdown("---")
            st.markdown("#### 🔍 테이블 선택")

            prev_table_value = st.session_state.get("databricks_table_input", "")
            table_input = st.text_input(
                "조회할 테이블 이름을 입력하세요",
                value=prev_table_value,
                key="databricks_table_input_widget",
                placeholder="catalog.schema.table",
                help="예: samples.bakehouse.sales_franchises",
            )
            table_clean = table_input.strip()
            table_changed = table_input != prev_table_value
            st.session_state["databricks_table_input"] = table_input

            if table_changed and table_clean:
                update_databricks_namespace_from_table(table_clean)
                try:
                    st.session_state["databricks_sql_query"] = generate_select_all_query(table_clean)
                except ValueError:
                    st.session_state["databricks_sql_query"] = ""
            elif table_changed and not table_clean:
                st.session_state["databricks_sql_query"] = ""

            if not table_clean:
                st.caption("입력하지 않았을 경우 선택 가능한 테이블 예시입니다.")
                st.write(", ".join(f"`{name}`" for name in DEFAULT_TABLE_SUGGESTIONS))

            default_query = st.session_state.get("databricks_sql_query", "")
            if table_clean and not default_query:
                try:
                    default_query = generate_select_all_query(table_clean)
                except ValueError:
                    default_query = ""
            query_input = st.text_area(
                "생성된 SQL (수정 가능)",
                value=default_query,
                height=140,
                help="기본으로 SELECT * FROM {table} 형식으로 생성됩니다.",
            )
            st.session_state["databricks_sql_query"] = query_input
            st.caption("Loading을 누르면 전체 데이터를 불러오고 화면에는 처음 10개의 행만 보여줍니다.")

            preview_target = st.radio(
                "Load 대상",
                options=("df_A", "df_B"),
                horizontal=True,
                key="databricks_preview_target",
            )

            if st.button("Loading", type="primary", disabled=not table_clean):
                success, message = load_preview_from_databricks_query(
                    table_clean,
                    query_input,
                    target="A" if preview_target == "df_A" else "B",
                    limit=10,
                )
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

            st.markdown("---")
            st.markdown("#### 🧭 Catalog / Schema 탐색")

            if st.button("List Catalogs"):
                ok, catalogs_df, message = list_databricks_catalogs_in_session()
                if ok:
                    st.success(message)
                    if catalogs_df is not None and not catalogs_df.empty:
                        display_cols = catalogs_df.columns.tolist()
                        st.dataframe(catalogs_df[display_cols], use_container_width=True)
                        st.session_state["databricks_catalog_options"] = catalogs_df["name"].tolist() if "name" in catalogs_df.columns else catalogs_df.iloc[:, 0].tolist()
                else:
                    st.error(message)

            catalog_options = st.session_state.get("databricks_catalog_options", [])
            current_catalog = st.session_state.get("databricks_selected_catalog", catalog or "")
            catalog_choices = ["--- Select a catalog ---"] + catalog_options
            default_catalog_idx = catalog_choices.index(current_catalog) if current_catalog in catalog_options else 0
            selected_catalog = st.selectbox(
                "Catalog 선택",
                options=catalog_choices,
                index=default_catalog_idx,
            )
            if selected_catalog != current_catalog:
                st.session_state["databricks_selected_catalog"] = selected_catalog if selected_catalog != "--- Select a catalog ---" else ""
                st.session_state["databricks_selected_schema"] = ""
                st.session_state["databricks_schema_options"] = []

            if st.button("List Schemas", disabled=selected_catalog == "--- Select a catalog ---" and not catalog):
                selected_catalog_value = selected_catalog if selected_catalog != "--- Select a catalog ---" else catalog or ""
                if not selected_catalog_value:
                    st.error("Catalog를 먼저 선택해주세요.")
                else:
                    st.session_state["databricks_selected_catalog"] = selected_catalog_value
                    ok, schemas_df, message = list_databricks_schemas_in_session(selected_catalog_value)
                    if ok:
                        st.success(message)
                        if schemas_df is not None and not schemas_df.empty:
                            display_cols = schemas_df.columns.tolist()
                            st.dataframe(schemas_df[display_cols], use_container_width=True)
                    else:
                        st.error(message)

            schema_options = st.session_state.get("databricks_schema_options", [])
            current_schema = st.session_state.get("databricks_selected_schema", schema or "")
            schema_choices = ["--- Select a schema ---"] + schema_options
            default_schema_idx = schema_choices.index(current_schema) if current_schema in schema_options else 0
            selected_schema = st.selectbox(
                "Schema 선택",
                options=schema_choices,
                index=default_schema_idx,
            )
            if selected_schema != current_schema:
                st.session_state["databricks_selected_schema"] = selected_schema if selected_schema != "--- Select a schema ---" else ""

            pattern = st.text_input(
                "Table filter (LIKE pattern)",
                value=st.session_state.get("databricks_table_filter", ""),
            )

            tables_container = st.container()
            if st.button("List Databricks Tables"):
                selected_catalog_value = selected_catalog if selected_catalog != "--- Select a catalog ---" else catalog or ""
                selected_schema_value = selected_schema if selected_schema != "--- Select a schema ---" else ""
                if not selected_catalog_value:
                    st.error("Catalog를 선택해주세요.")
                elif not (schema_options or selected_schema_value):
                    st.error("Schema 목록을 먼저 불러오고 선택해주세요.")
                elif not selected_schema_value:
                    st.error("Schema를 선택해주세요.")
                else:
                    st.session_state["databricks_table_filter"] = pattern
                    st.session_state["databricks_selected_catalog"] = selected_catalog_value
                    st.session_state["databricks_selected_schema"] = selected_schema_value
                    ok, table_df, message = list_databricks_tables_in_session(pattern.strip())
                    if ok:
                        st.success(message)
                        if table_df is not None and not table_df.empty:
                            display_cols = [col for col in ["full_name", "schema", "table"] if col in table_df.columns]
                            tables_container.dataframe(
                                table_df[display_cols],
                                use_container_width=True,
                            )
                    else:
                        st.error(message)

            table_options: List[str] = st.session_state.get("databricks_table_options", [])
            selected_table = st.selectbox(
                "Select Databricks table",
                options=["--- Select a table ---"] + table_options,
                index=0,
            )
            col_target, col_limit = st.columns([1, 1])
            target = col_target.radio(
                "Load into",
                options=("df_A", "df_B"),
                horizontal=True,
            )
            limit_value = col_limit.number_input(
                "Row limit (0 = full table)",
                min_value=0,
                step=1000,
                value=int(st.session_state.get("databricks_limit", 0) or 0),
            )
            st.session_state["databricks_limit"] = limit_value

            if st.button("Load Databricks Table"):
                table_name = selected_table if selected_table != "--- Select a table ---" else ""
                if not table_name:
                    st.error("Load할 테이블을 선택해주세요.")
                else:
                    selected_catalog_value = st.session_state.get("databricks_selected_catalog", catalog or "")
                    selected_schema_value = st.session_state.get("databricks_selected_schema", "")
                    if not selected_catalog_value:
                        st.error("Catalog를 먼저 선택해주세요.")
                    elif not selected_schema_value:
                        st.error("Schema를 먼저 선택해주세요.")
                    else:
                        limit_arg = None if not limit_value else int(limit_value)
                        ok, message = load_df_from_databricks(
                            table=table_name,
                            target="A" if target == "df_A" else "B",
                            limit=limit_arg,
                        )
                        if ok:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
