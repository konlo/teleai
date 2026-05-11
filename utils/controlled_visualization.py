from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

import pandas as pd


MAX_POINT_PLOT_ROWS = 5000
MAX_PAIRPLOT_ROWS = 1200


def _sample_for_plot(frame: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, int]:
    total_rows = len(frame)
    if total_rows > max_rows:
        return frame.sample(n=max_rows, random_state=42), total_rows
    return frame, total_rows


def _stat_column(df: pd.DataFrame) -> str:
    if "stat_value" in df.columns and pd.api.types.is_numeric_dtype(df["stat_value"]):
        return "stat_value"
    if "stat_count" in df.columns and pd.api.types.is_numeric_dtype(df["stat_count"]):
        return "stat_count"
    return ""


def _top_n_series(series: pd.Series, top_n: Any) -> pd.Series:
    if not top_n:
        return series
    try:
        limit = int(top_n)
    except (TypeError, ValueError):
        return series
    return series.head(limit) if limit > 0 else series


def plot_controlled_visualization(df: pd.DataFrame, config) -> str:
    import matplotlib.pyplot as plt

    column = config.column
    if config.plot_type in {"grouped_bar", "stacked_bar"}:
        group_column = getattr(config, "group_column", "") or ""
        if not group_column or group_column not in df.columns:
            raise ValueError(f"`{group_column}` 그룹 컬럼을 찾을 수 없습니다.")
        if column not in df.columns:
            raise ValueError(f"`{column}` 컬럼을 찾을 수 없습니다.")

        stat_column = _stat_column(df)
        if stat_column:
            plot_rows = (
                df[[group_column, column, stat_column]]
                .dropna()
                .assign(
                    **{
                        group_column: lambda frame: frame[group_column].astype(str),
                        column: lambda frame: frame[column].astype(str),
                    }
                )
                .groupby([group_column, column], sort=False)[stat_column]
                .sum()
                .reset_index(name="count")
            )
        else:
            plot_rows = (
                df[[group_column, column]]
                .dropna()
                .assign(
                    **{
                        group_column: lambda frame: frame[group_column].astype(str),
                        column: lambda frame: frame[column].astype(str),
                    }
                )
                .groupby([group_column, column], sort=False)
                .size()
                .reset_index(name="count")
            )

        group_values = tuple(getattr(config, "group_values", ()) or ())
        if group_values:
            allowed = {str(value) for value in group_values}
            plot_rows = plot_rows[plot_rows[group_column].astype(str).isin(allowed)]
        if plot_rows.empty:
            raise ValueError(f"`{column}` 컬럼에 그룹별 시각화 가능한 값이 없습니다.")

        if getattr(config, "top_n", None):
            top_categories = (
                plot_rows.groupby(column)["count"]
                .sum()
                .sort_values(ascending=False)
                .head(int(config.top_n))
                .index
            )
            plot_rows = plot_rows[plot_rows[column].isin(top_categories)]

        pivot = plot_rows.pivot_table(
            index=column,
            columns=group_column,
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        if group_values:
            ordered_columns = [
                actual
                for expected in group_values
                for actual in pivot.columns
                if str(actual) == str(expected)
            ]
            if ordered_columns:
                pivot = pivot[ordered_columns]

        if config.plot_type == "stacked_bar":
            plt.figure(figsize=(10, 5))
            pivot.head(30).plot(kind="bar", stacked=True)
            plt.ylabel("Value")
            plt.title(f"{column} stacked by {group_column}")
            plt.xlabel(column)
        elif getattr(config, "group_mode", "") == "separate" and len(pivot.columns) > 1:
            fig, axes = plt.subplots(
                1,
                len(pivot.columns),
                figsize=(max(10, 4 * len(pivot.columns)), 5),
                sharey=True,
            )
            if len(pivot.columns) == 1:
                axes = [axes]
            for axis, group_value in zip(axes, pivot.columns):
                pivot[group_value].sort_values(ascending=False).head(30).plot(kind="bar", ax=axis)
                axis.set_title(f"{group_column}={group_value}")
                axis.set_xlabel(column)
                axis.set_ylabel("Count")
        else:
            plt.figure(figsize=(10, 5))
            pivot.head(30).plot(kind="bar")
            plt.ylabel("Count")
            plt.title(f"{column} distribution by {group_column}")
            plt.xlabel(column)
        plt.tight_layout()
        return (
            f"plot_type={config.plot_type}, column={column}, group_column={group_column}, "
            f"groups={list(pivot.columns)}, rows={len(df)}"
        )

    if config.plot_type == "scatter":
        x_column = getattr(config, "x_column", "") or ""
        y_column = getattr(config, "y_column", "") or ""
        color_column = getattr(config, "color_column", "") or ""
        required = [x_column, y_column] + ([color_column] if color_column else [])
        missing = [item for item in required if item not in df.columns]
        if missing:
            raise ValueError(f"산점도 컬럼을 찾을 수 없습니다: {missing}")
        plot_df = df[required].dropna()
        if plot_df.empty:
            raise ValueError("산점도에 사용할 값이 없습니다.")
        sampled, total_rows = _sample_for_plot(plot_df, MAX_POINT_PLOT_ROWS)
        plt.figure(figsize=(10, 6))
        if color_column and not pd.api.types.is_numeric_dtype(sampled[color_column]):
            for label, group in sampled.groupby(color_column, sort=False):
                plt.scatter(group[x_column], group[y_column], s=14, alpha=0.65, label=str(label))
            plt.legend(title=color_column, loc="best")
        elif color_column:
            points = plt.scatter(sampled[x_column], sampled[y_column], c=sampled[color_column], s=14, alpha=0.65)
            plt.colorbar(points, label=color_column)
        else:
            plt.scatter(sampled[x_column], sampled[y_column], s=14, alpha=0.65)
        plt.title(f"{y_column} by {x_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout()
        return (
            f"plot_type=scatter, x_column={x_column}, y_column={y_column}, "
            f"rows={total_rows}, plotted_rows={len(sampled)}"
        )

    if config.plot_type == "line":
        x_column = getattr(config, "x_column", "") or ""
        y_column = getattr(config, "y_column", "") or ""
        missing = [item for item in (x_column, y_column) if item not in df.columns]
        if missing:
            raise ValueError(f"라인 차트 컬럼을 찾을 수 없습니다: {missing}")
        plot_df = df[[x_column, y_column]].dropna().sort_values(x_column)
        if plot_df.empty:
            raise ValueError("라인 차트에 사용할 값이 없습니다.")
        plt.figure(figsize=(10, 5))
        plt.plot(plot_df[x_column], plot_df[y_column], marker="o", linewidth=1.5, markersize=3)
        plt.title(f"{y_column} trend by {x_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return f"plot_type=line, x_column={x_column}, y_column={y_column}, rows={len(plot_df)}"

    if config.plot_type == "heatmap":
        columns = tuple(getattr(config, "columns", ()) or ())
        if columns:
            plot_df = df[list(columns)].dropna()
            if plot_df.empty:
                raise ValueError("상관 히트맵에 사용할 값이 없습니다.")
            corr = plot_df.corr(numeric_only=True)
            if corr.empty:
                raise ValueError("상관 히트맵을 만들 numeric 컬럼이 없습니다.")
            plt.figure(figsize=(max(6, len(corr.columns) * 1.1), max(5, len(corr.index) * 0.9)))
            image = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(image, label="Correlation")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title("Correlation heatmap")
            plt.tight_layout()
            return f"plot_type=heatmap, aggregation=correlation, columns={list(corr.columns)}, rows={len(plot_df)}"

        x_column = getattr(config, "x_column", "") or ""
        y_column = getattr(config, "y_column", "") or ""
        value_column = getattr(config, "value_column", "") or ""
        stat_column = _stat_column(df)
        required = [x_column, y_column] + ([value_column] if value_column and not stat_column else [])
        missing = [item for item in required if item not in df.columns]
        if missing:
            raise ValueError(f"히트맵 컬럼을 찾을 수 없습니다: {missing}")
        if stat_column:
            plot_rows = df[[x_column, y_column, stat_column]].dropna()
            pivot = plot_rows.pivot_table(index=y_column, columns=x_column, values=stat_column, aggfunc="sum", fill_value=0)
        elif value_column:
            plot_rows = df[[x_column, y_column, value_column]].dropna()
            pivot = plot_rows.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean", fill_value=0)
        else:
            plot_rows = df[[x_column, y_column]].dropna()
            pivot = plot_rows.assign(_count=1).pivot_table(index=y_column, columns=x_column, values="_count", aggfunc="sum", fill_value=0)
        if pivot.empty:
            raise ValueError("히트맵에 사용할 값이 없습니다.")
        plt.figure(figsize=(max(7, len(pivot.columns) * 0.8), max(5, len(pivot.index) * 0.45)))
        image = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(image, label=value_column or stat_column or "Count")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title(f"{y_column} by {x_column} heatmap")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout()
        return f"plot_type=heatmap, x_column={x_column}, y_column={y_column}, rows={len(plot_rows)}"

    if config.plot_type == "pairplot":
        columns = tuple(getattr(config, "columns", ()) or ())
        missing = [item for item in columns if item not in df.columns]
        if missing:
            raise ValueError(f"pairplot 컬럼을 찾을 수 없습니다: {missing}")
        plot_df = df[list(columns)].dropna()
        if plot_df.empty:
            raise ValueError("pairplot에 사용할 값이 없습니다.")
        sampled, total_rows = _sample_for_plot(plot_df, MAX_PAIRPLOT_ROWS)
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(max(7, 2.6 * n_cols), max(7, 2.4 * n_cols)))
        if n_cols == 1:
            axes = [[axes]]
        for row_idx, row_col in enumerate(columns):
            for col_idx, col_col in enumerate(columns):
                axis = axes[row_idx][col_idx]
                if row_idx == col_idx:
                    axis.hist(sampled[row_col], bins=20)
                else:
                    axis.scatter(sampled[col_col], sampled[row_col], s=8, alpha=0.55)
                if row_idx == n_cols - 1:
                    axis.set_xlabel(col_col)
                else:
                    axis.set_xticklabels([])
                if col_idx == 0:
                    axis.set_ylabel(row_col)
                else:
                    axis.set_yticklabels([])
        fig.suptitle("Pairplot", y=1.02)
        plt.tight_layout()
        return f"plot_type=pairplot, columns={list(columns)}, rows={total_rows}, plotted_rows={len(sampled)}"

    series = df[column].dropna()
    if series.empty:
        raise ValueError(f"`{column}` 컬럼에 시각화 가능한 값이 없습니다.")

    plt.figure(figsize=(10, 5))
    if config.plot_type == "histogram":
        series.plot(kind="hist", bins=min(50, max(10, int(len(series) ** 0.5))))
        plt.ylabel("Frequency")
    elif config.plot_type == "boxplot":
        series.plot(kind="box")
        plt.ylabel(column)
    elif config.plot_type == "violin":
        plt.violinplot(series)
        plt.ylabel(column)
        plt.xticks([1], [column])
    elif config.plot_type == "bar":
        stat_column = _stat_column(df)
        if stat_column:
            counts = (
                df[[column, stat_column]]
                .dropna()
                .assign(**{column: lambda frame: frame[column].astype(str)})
                .groupby(column, sort=False)[stat_column]
                .sum()
                .sort_values(ascending=False)
            )
        else:
            counts = series.astype(str).value_counts()
        _top_n_series(counts, getattr(config, "top_n", None)).head(30).plot(kind="bar")
        plt.ylabel("Count")
    else:
        raise ValueError(f"지원하지 않는 plot_type입니다: {config.plot_type}")
    plt.title(f"{column} distribution")
    plt.xlabel(column)
    plt.tight_layout()
    return (
        f"plot_type={config.plot_type}, column={column}, "
        f"rows={len(df)}, non_null={len(series)}"
    )


def collect_matplotlib_figure_payloads(*, close: bool = True) -> List[Dict[str, Any]]:
    """Convert currently open matplotlib figures into chat-log image payloads."""

    import matplotlib.pyplot as plt

    payloads: List[Dict[str, Any]] = []
    for fig_id in plt.get_fignums():
        fig = plt.figure(fig_id)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        suptitle = getattr(fig, "_suptitle", None)
        if suptitle is not None:
            title_text = suptitle.get_text()
        elif fig.axes:
            title_text = fig.axes[0].get_title()
        else:
            title_text = "Matplotlib Figure"
        payloads.append(
            {
                "kind": "matplotlib",
                "title": title_text,
                "image": base64.b64encode(buffer.read()).decode("utf-8"),
            }
        )
    if close:
        plt.close("all")
    return payloads


__all__ = ["collect_matplotlib_figure_payloads", "plot_controlled_visualization"]
