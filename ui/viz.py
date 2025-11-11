import base64
import io
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None


def render_visualizations(pytool: Any) -> List[Dict[str, Any]]:
    """Render charts and tables based on intermediate DataFrames."""
    # st.markdown("---")s
    # st.subheader("EDA Visualizations")
    visuals_rendered = False
    figure_payloads: List[Dict[str, Any]] = []

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
            figure_payloads.append(
                {
                    "kind": "bar_chart",
                    "title": "Top Outlier Columns (IQR %)",
                    "data": top_outliers.to_frame(name="outlier_rate_%"),
                }
            )

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
                chart_df = stl_chart[numeric_cols]
                st.line_chart(chart_df)
                figure_payloads.append(
                    {
                        "kind": "line_chart",
                        "title": "STL Decomposition Components",
                        "data": chart_df,
                    }
                )

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
                chart_df = roll_chart[metric_cols]
                st.line_chart(chart_df)
                figure_payloads.append(
                    {
                        "kind": "line_chart",
                        "title": "Rolling Statistics",
                        "data": chart_df,
                    }
                )

    topn_df = pytool.globals.get("df_topN")
    if isinstance(topn_df, pd.DataFrame) and not topn_df.empty:
        visuals_rendered = True
        st.markdown("**Top-N Entities**")
        st.dataframe(topn_df)
        figure_payloads.append(
            {
                "kind": "dataframe",
                "title": "Top-N Entities",
                "data": topn_df,
            }
        )

    current_df_a = pytool.globals.get("df_A")
    if isinstance(current_df_a, pd.DataFrame) and not current_df_a.empty:
        outlier_cols = [
            c for c in current_df_a.columns if c.endswith("_is_outlier_iqr")
        ]
        if "isoforest_outlier" in current_df_a.columns or outlier_cols:
            visuals_rendered = True
            st.markdown("**Outlier Flags Overview**")
            with st.container():
                iso_counts = None
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
                figure_payloads.append(
                    {
                        "kind": "json",
                        "title": "Outlier Flags Overview",
                        "data": {
                            "isoforest_counts": {
                                "true": int(iso_counts.get(True, 0)) if iso_counts is not None else 0,
                                "false": int(iso_counts.get(False, 0)) if iso_counts is not None else 0,
                            },
                            "iqr_counts": counts if outlier_cols else {},
                        },
                    }
                )

    if plt is not None:
        fig_ids = plt.get_fignums()
        if fig_ids:
            visuals_rendered = True
            # st.markdown("**Matplotlib Figures**")
            for fig_id in fig_ids:
                fig = plt.figure(fig_id)
                fig.set_size_inches(12, 6, forward=True)
                # st.pyplot(fig, use_container_width=True)
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode("utf-8")
                suptitle = getattr(fig, "_suptitle", None)
                if suptitle is not None:
                    title_text = suptitle.get_text()
                elif fig.axes:
                    title_text = fig.axes[0].get_title()
                else:
                    title_text = "Matplotlib Figure"
                figure_payloads.append(
                    {
                        "kind": "matplotlib",
                        "title": title_text,
                        "image": encoded,
                    }
                )
            plt.close("all")

    if not visuals_rendered:
        st.info("Run 분석 도구를 통해 시각화 가능한 결과를 생성하세요.")
    return figure_payloads
