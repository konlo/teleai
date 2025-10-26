from typing import Any

import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None


def render_visualizations(pytool: Any) -> None:
    """Render charts and tables based on intermediate DataFrames."""
    st.markdown("---")
    st.subheader("EDA Visualizations")
    visuals_rendered = False

    outlier_rank_df = pytool.globals.get("df_outlier_rank")
    if isinstance(outlier_rank_df, pd.DataFrame) and not outlier_rank_df.empty:
        if {"column", "outlier_rate_%"} <= set(outlier_rank_df.columns):
            visuals_rendered = True
            top_outliers = (
                outlier_rank_df.head(15)
                .set_index("column")["outlier_rate_%"]
            )
            st.markdown("**Top Outlier Columns (IQR %)**")
            st.bar_chart(top_outliers)

    stl_df = pytool.globals.get("df_A_stl")
    if isinstance(stl_df, pd.DataFrame) and not stl_df.empty:
        stl_chart = stl_df.copy()
        time_col = stl_chart.columns[0]
        if time_col in stl_chart.columns:
            stl_chart[time_col] = pd.to_datetime(
                stl_chart[time_col], errors="coerce"
            )
            stl_chart = (
                stl_chart.dropna(subset=[time_col])
                .set_index(time_col)
                .sort_index()
            )
            numeric_cols = [
                c
                for c in stl_chart.columns
                if pd.api.types.is_numeric_dtype(stl_chart[c])
            ]
            if numeric_cols:
                visuals_rendered = True
                st.markdown("**STL Decomposition Components**")
                st.line_chart(stl_chart[numeric_cols])

    rolling_df = pytool.globals.get("df_A_rolling")
    if isinstance(rolling_df, pd.DataFrame) and not rolling_df.empty:
        roll_chart = rolling_df.copy()
        time_col = roll_chart.columns[0]
        if time_col in roll_chart.columns:
            roll_chart[time_col] = pd.to_datetime(
                roll_chart[time_col], errors="coerce"
            )
            roll_chart = (
                roll_chart.dropna(subset=[time_col])
                .set_index(time_col)
                .sort_index()
            )
            metric_cols = [
                c
                for c in roll_chart.columns
                if pd.api.types.is_numeric_dtype(roll_chart[c])
            ]
            if metric_cols:
                visuals_rendered = True
                st.markdown("**Rolling Statistics**")
                st.line_chart(roll_chart[metric_cols])

    topn_df = pytool.globals.get("df_topN")
    if isinstance(topn_df, pd.DataFrame) and not topn_df.empty:
        visuals_rendered = True
        st.markdown("**Top-N Entities**")
        st.dataframe(topn_df, use_container_width=True)

    current_df_a = pytool.globals.get("df_A")
    if isinstance(current_df_a, pd.DataFrame) and not current_df_a.empty:
        outlier_cols = [
            c for c in current_df_a.columns if c.endswith("_is_outlier_iqr")
        ]
        if "isoforest_outlier" in current_df_a.columns or outlier_cols:
            visuals_rendered = True
            st.markdown("**Outlier Flags Overview**")
            with st.container():
                if "isoforest_outlier" in current_df_a.columns:
                    iso_counts = current_df_a["isoforest_outlier"].value_counts(
                        dropna=False
                    )
                    st.write(
                        {
                            "isoforest_outlier=True": int(
                                iso_counts.get(True, 0)
                            ),
                            "isoforest_outlier=False": int(
                                iso_counts.get(False, 0)
                            ),
                        }
                    )
                if outlier_cols:
                    counts = current_df_a[outlier_cols].sum().to_dict()
                    st.write({"iqr_outlier_counts": counts})

    if plt is not None:
        fig_ids = plt.get_fignums()
        if fig_ids:
            visuals_rendered = True
            st.markdown("**Matplotlib Figures**")
            for fig_id in fig_ids:
                fig = plt.figure(fig_id)
                st.pyplot(fig)
            plt.close("all")

    if not visuals_rendered:
        st.info("Run 분석 도구를 통해 시각화 가능한 결과를 생성하세요.")
