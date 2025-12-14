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
    if df_a_ready:
        with st.popover("📊 Data Preview"):
            st.write(
                f"**Loaded file for df_A:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})"
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
            "df_A 데이터가 아직 로드되지 않았습니다. 왼쪽 Databricks Loader 또는 SQL Builder 에이전트를 사용해 데이터를 불러오세요."
        )


__all__ = ["render_data_preview_section"]
