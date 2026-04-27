import pandas as pd
import streamlit as st


def _render_csv_download_button(label: str, df: pd.DataFrame, dataset_name: str) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    file_name = f"{dataset_name or label}.csv"
    st.download_button(
        label=f"Download {label} CSV",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


def render_data_preview_section(df_a_ready: bool, df_A: pd.DataFrame, df_B: pd.DataFrame) -> None:
    table_sample = st.session_state.get("df_table_sample")
    table_sample_table = st.session_state.get("df_table_sample_table", "")

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) {
            gap: 3px !important;
            justify-content: flex-start !important;
        }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPopover"]) > div {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: fit-content !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sample_col, preview_col, _ = st.columns([1, 1, 12], gap=None)
    with sample_col:
        with st.popover("🔎 Sample Data"):
            if isinstance(table_sample, pd.DataFrame) and not table_sample.empty:
                st.write(
                    f"**Selected table sample:** `{table_sample_table or 'Unknown Table'}` "
                    f"(Shape: {table_sample.shape})"
                )
                st.dataframe(table_sample.head(10), width="stretch")
                _render_csv_download_button(
                    "Table Sample",
                    table_sample,
                    table_sample_table or "table_sample",
                )
            else:
                st.info("선택된 테이블 sample이 아직 로드되지 않았습니다.")
    with preview_col:
        with st.popover("📊 Preview Data"):
            if df_a_ready:
                st.write(
                    f"**Current working df_A:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})"
                )
                st.dataframe(df_A.head(10), width="stretch")
                _render_csv_download_button("df_A", df_A, st.session_state.get("df_A_name", "df_A"))
                if isinstance(df_B, pd.DataFrame):
                    st.markdown(
                        f"**df_B Preview —** `{st.session_state['df_B_name']}` (Shape: {df_B.shape})"
                    )
                    st.dataframe(df_B.head(10), width="stretch")
                    _render_csv_download_button(
                        "df_B", df_B, st.session_state.get("df_B_name", "df_B")
                    )
            else:
                st.info(
                    "현재 작업 데이터 df_A가 아직 로드되지 않았습니다. SQL Builder 또는 Controlled Executor로 데이터를 불러오면 여기에 표시됩니다."
                )


__all__ = ["render_data_preview_section"]
