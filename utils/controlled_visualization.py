from __future__ import annotations

import pandas as pd


def plot_controlled_visualization(df: pd.DataFrame, config) -> str:
    import matplotlib.pyplot as plt

    column = config.column
    if config.plot_type == "grouped_bar":
        group_column = getattr(config, "group_column", "") or ""
        if not group_column or group_column not in df.columns:
            raise ValueError(f"`{group_column}` 그룹 컬럼을 찾을 수 없습니다.")
        if column not in df.columns:
            raise ValueError(f"`{column}` 컬럼을 찾을 수 없습니다.")

        if "stat_count" in df.columns and pd.api.types.is_numeric_dtype(df["stat_count"]):
            plot_rows = (
                df[[group_column, column, "stat_count"]]
                .dropna()
                .assign(
                    **{
                        group_column: lambda frame: frame[group_column].astype(str),
                        column: lambda frame: frame[column].astype(str),
                    }
                )
                .groupby([group_column, column], sort=False)["stat_count"]
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

        if getattr(config, "group_mode", "") == "separate" and len(pivot.columns) > 1:
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
    elif config.plot_type == "bar":
        if "stat_count" in df.columns and pd.api.types.is_numeric_dtype(df["stat_count"]):
            counts = (
                df[[column, "stat_count"]]
                .dropna()
                .assign(**{column: lambda frame: frame[column].astype(str)})
                .groupby(column, sort=False)["stat_count"]
                .sum()
                .sort_values(ascending=False)
                .head(30)
            )
            counts.plot(kind="bar")
        else:
            series.astype(str).value_counts().head(30).plot(kind="bar")
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


__all__ = ["plot_controlled_visualization"]
