import os
from typing import List

import streamlit as st

from utils.session import (
    DEFAULT_DATA_DIR,
    DFB_DEFAULT_NAME,
    SUPPORTED_EXTENSIONS,
    load_df_a,
    load_df_b,
    databricks_connector_available,
    list_databricks_catalogs_in_session,
    list_databricks_schemas_in_session,
    list_databricks_tables_in_session,
    load_df_from_databricks,
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

        st.markdown("### 🗂️ 1. 데이터 폴더 설정")
        new_data_dir = st.text_input(
            "Enter Data Directory Path",
            value=st.session_state["DATA_DIR"],
            key="data_dir_input",
        )
        if st.button("Set Directory"):
            if os.path.isdir(new_data_dir):
                st.session_state["DATA_DIR"] = new_data_dir
                st.session_state["df_A_data"] = None
                st.session_state["df_A_name"] = "No Data"
                st.session_state["csv_path"] = ""
                st.session_state["df_B_data"] = None
                st.session_state["df_B_name"] = "No Data"
                st.session_state["csv_b_path"] = ""
                st.success(f"Directory set to: `{new_data_dir}`")
                st.rerun()
            else:
                st.error(f"Invalid directory path: `{new_data_dir}`")

        data_dir = st.session_state["DATA_DIR"]
        dfb_default = os.path.join(data_dir, DFB_DEFAULT_NAME)

        st.markdown("---")
        st.markdown("### 📄 2. df_A 파일 선택")
        st.caption(f"Search directory: `{data_dir}`")
        data_files: List[str] = []
        try:
            if os.path.isdir(data_dir):
                for fname in os.listdir(data_dir):
                    if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                        data_files.append(fname)
                data_files.sort()
            else:
                st.warning("유효한 데이터 디렉토리를 설정해주세요.")
        except Exception as exc:
            st.error(f"폴더 접근 오류: {exc}")

        selected_file = st.selectbox(
            "Select data file for df_A",
            options=["--- Select a file ---"] + data_files,
            key="file_selector",
        )
        if st.button("Load Selected File (df_A)"):
            if selected_file and selected_file != "--- Select a file ---":
                file_path = os.path.join(data_dir, selected_file)
                success, message = load_df_a(file_path, selected_file)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("df_A 파일을 선택해주세요.")

        st.markdown("---")
        st.markdown("### 📄 3. df_B 파일 선택 (비교용)")
        selected_file_b = st.selectbox(
            "Select data file for df_B",
            options=["--- Select a file ---"] + data_files,
            key="file_selector_b",
        )
        if st.button("Load Selected File (df_B)"):
            if selected_file_b and selected_file_b != "--- Select a file ---":
                file_path_b = os.path.join(data_dir, selected_file_b)
                success, message = load_df_b(file_path_b, selected_file_b)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("df_B 파일을 선택해주세요.")

        st.markdown("---")
        st.caption(
            f"**현재 로드 파일 경로(df_A):** `{st.session_state.get('csv_path', 'Not loaded')}`"
        )
        st.caption(
            f"**현재 로드 파일 경로(df_B):** `{st.session_state.get('csv_b_path', 'Not loaded')}`"
        )
        st.caption(f"df_B 기본 가정 파일: `{os.path.basename(dfb_default)}`")

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
                    "환경 변수 DATABRICKS_SERVER_HOSTNAME / DATABRICKS_HTTP_PATH / "
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
