from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any, List, Optional, Sequence, Tuple

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.tools import BaseTool, tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool

from utils.session import (
    DEFAULT_DATA_DIR,
    DFB_DEFAULT_NAME,
    parse_float,
    parse_int,
    resolve_time_column,
    read_table,
)

# Optional deps
try:
    from sklearn.ensemble import IsolationForest  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    IsolationForest = None

try:
    from statsmodels.tsa.seasonal import STL  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    STL = None


pytool: Optional[PythonAstREPLTool] = None


def _init_pytool(df_a: pd.DataFrame, df_b: Optional[pd.DataFrame]) -> PythonAstREPLTool:
    global pytool
    pytool = PythonAstREPLTool(
        globals={
            "pd": pd,
            "np": np,
            "plt": plt,
            "df": df_a,
            "df_A": df_a,
            "df_B": df_b,
            "df_join": None,
            "duckdb": duckdb,
        },
        name="python_repl_ast",
        description="Execute Python on df_A/df_B/df_join with pandas/matplotlib.",
    )
    return pytool


def _ensure_pytool() -> PythonAstREPLTool:
    if pytool is None:
        raise RuntimeError("Toolset not initialised. Call build_tools first.")
    return pytool


def build_tools(
    df_a: pd.DataFrame,
    df_b: Optional[pd.DataFrame],
) -> Tuple[PythonAstREPLTool, Sequence[BaseTool]]:
    """Initialise the Python REPL tool and return the full tool list."""
    pt = _init_pytool(df_a, df_b)
    tools: List[BaseTool] = [
        pt,
        load_loading_csv,
        describe_columns,
        save_plots_zip,
        load_df_b,
        describe_columns_on,
        sql_on_dfs,
        propose_join_keys,
        align_time_buckets,
        compare_on_keys,
        mismatch_report,
        make_timesafe,
        create_features,
        rolling_stats,
        stl_decompose,
        anomaly_iqr,
        anomaly_isoforest,
        cohort_compare,
        topn_machines,
        select_numeric_candidates,
        rank_outlier_columns,
        plot_outliers,
        plot_outlier_overview,
        plot_outliers_multi,
        plot_compare_timeseries,
        stl_plot,
        corr_heatmap,
        auto_outlier_eda,
    ]
    return pt, tools


@tool
def load_loading_csv(filename: str) -> str:
    """Load a CSV or Parquet from DATA_DIR into 'loading_df'. Pass only file name."""
    pt = _ensure_pytool()
    current_data_dir = st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)
    path = os.path.join(current_data_dir, filename)
    try:
        new_df = read_table(path)
    except Exception as exc:  # pragma: no cover - surface message to agent
        return f"Failed to load {path}: {exc}"
    pt.globals["loading_df"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    shape = f"{new_df.shape[0]} rows x {new_df.shape[1]} cols"
    return (
        f"Loaded {filename} from {current_data_dir} into loading_df\n"
        f"Shape: {shape}\n\nPreview (head):\n{preview}"
    )


@tool
def describe_columns(cols: str = "") -> str:
    """
    Describe selected columns (comma-separated).
    Uses 'loading_df' if available; otherwise uses 'df_A'.
    """
    pt = _ensure_pytool()
    if "loading_df" in pt.globals and pt.globals["loading_df"] is not None:
        current_df = pt.globals["loading_df"]
        source = "loading_df"
    else:
        current_df = pt.globals.get("df_A")
        source = "df_A"
    if current_df is None:
        return "df_A not loaded."
    use_cols = [c.strip() for c in (cols or "").split(",") if c.strip()] or list(
        current_df.columns
    )
    missing = [c for c in use_cols if c not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"
    desc = current_df[use_cols].describe(include="all").transpose()
    shape = f"{current_df.shape[0]} rows x {current_df.shape[1]} cols"
    return f"[source={source} | shape={shape}]\n\n" + desc.to_markdown()


@tool
def save_plots_zip() -> str:
    """Zip all current matplotlib figures. Use after plotting with python_repl_ast."""
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if not figs:
        return "No figures to save."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, fig in enumerate(figs, 1):
            img = io.BytesIO()
            fig.savefig(img, format="png", dpi=110, bbox_inches="tight")
            img.seek(0)
            zf.writestr(f"plot_{i}.png", img.read())
    return f"Zipped {len(figs)} plots. Bytes={len(buf.getvalue())}"


@tool
def load_df_b(filename: str = "") -> str:
    """
    Load a CSV or Parquet into 'df_B'. If empty, defaults to telemetry_raw.csv under DATA_DIR.
    """
    pt = _ensure_pytool()
    current_data_dir = st.session_state.get("DATA_DIR", DEFAULT_DATA_DIR)
    if not filename:
        path = os.path.join(current_data_dir, DFB_DEFAULT_NAME)
        show_name = os.path.basename(path)
    else:
        path = (
            os.path.join(current_data_dir, filename)
            if not os.path.isabs(filename)
            else filename
        )
        show_name = os.path.basename(path)
    try:
        new_df = read_table(path)
    except Exception as exc:  # pragma: no cover
        return f"Failed to load df_B from {path}: {exc}"
    pt.globals["df_B"] = new_df
    preview = new_df.head(10).to_markdown(index=False)
    return (
        f"Loaded df_B from '{show_name}' (full: {path}) with shape {new_df.shape}\n\n"
        f"Preview:\n{preview}"
    )


@tool
def describe_columns_on(target: str = "A", cols: str = "") -> str:
    """
    Describe columns from df_A or df_B. target: 'A' | 'B'; cols: comma-separated; empty -> all
    """
    pt = _ensure_pytool()
    selector = (target or "A").strip().upper()
    if selector == "B":
        current_df = pt.globals.get("df_B")
        if current_df is None:
            return "df_B is not loaded. Use load_df_b() or sidebar."
        source = "df_B"
    else:
        current_df = pt.globals.get("df_A")
        source = "df_A"
    if current_df is None:
        return "df_A not loaded."
    use_cols = [c.strip() for c in (cols or "").split(",") if c.strip()] or list(
        current_df.columns
    )
    missing = [c for c in use_cols if c not in current_df.columns]
    if missing:
        return f"Missing columns: {missing}\nAvailable: {list(current_df.columns)}"
    desc = current_df[use_cols].describe(include="all").transpose()
    return f"[source={source} | shape={current_df.shape}]\n\n" + desc.to_markdown()


@tool
def sql_on_dfs(query: str) -> str:
    """
    Run DuckDB SQL over df_A, df_B (if loaded), df_join (if created).
    Tables: df_A, df_B, df_join
    """
    pt = _ensure_pytool()
    try:
        if pt.globals.get("df_A") is not None:
            duckdb.register("df_A", pt.globals["df_A"])
        if pt.globals.get("df_B") is not None:
            duckdb.register("df_B", pt.globals["df_B"])
        if pt.globals.get("df_join") is not None:
            duckdb.register("df_join", pt.globals["df_join"])
        out = duckdb.sql(query).df()
        return out.head(200).to_markdown(index=False)
    except Exception as exc:  # pragma: no cover
        return f"SQL error: {exc}"


@tool
def propose_join_keys() -> str:
    """Suggest same-name & compatible-dtype join key candidates between df_A and df_B."""
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    df_b = pt.globals.get("df_B")
    if df_a is None:
        return "df_A not loaded."
    if df_b is None:
        return "df_B is not loaded."

    def dtype_sig(series):
        t = str(series.dtype)
        if "int" in t:
            return "int"
        if "float" in t:
            return "float"
        if "datetime" in t or "date" in t or "time" in t:
            return "datetime"
        return "str"

    pairs = []
    for col in df_a.columns:
        if col in df_b.columns and dtype_sig(df_a[col]) == dtype_sig(df_b[col]):
            pairs.append((col, dtype_sig(df_a[col])))
    if not pairs:
        return "No obvious same-name & same-type keys. Consider casting or mapping."
    md = "| key | dtype |\n|---|---|\n" + "\n".join(
        [f"| {k} | {t} |" for k, t in pairs]
    )
    return f"Candidate join keys:\n{md}"


@tool
def align_time_buckets(
    target: str = "A", column: str = "datetime", freq: str = "H"
) -> str:
    """
    Resample time-like column to buckets and store as df_A_bucketed or df_B_bucketed.
    freq like 'H','D','15min'
    """
    pt = _ensure_pytool()
    selector = (target or "A").strip().upper()
    current_df = pt.globals.get("df_B") if selector == "B" else pt.globals.get("df_A")
    if current_df is None:
        return f"df_{selector} not loaded."
    time_col = resolve_time_column(current_df, column)
    if time_col is None:
        return f"Column '{column}' not in df_{selector}."
    tmp = current_df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    if column not in tmp.columns:
        tmp[column] = tmp[time_col]
    bucket_col = f"{column}_bucket"
    tmp[bucket_col] = tmp[column].dt.to_period(freq).dt.to_timestamp()
    pt.globals[f"df_{selector}_bucketed"] = tmp
    preview = tmp[[bucket_col]].head().to_markdown(index=False)
    return (
        f"Created df_{selector}_bucketed with '{bucket_col}' at freq={freq} using '{time_col}'.\n"
        f"Preview:\n{preview}"
    )


@tool
def compare_on_keys(
    keys: str, how: str = "inner", atol: Any = 0.0, rtol: Any = 0.0
) -> str:
    """
    Join df_A & df_B on comma-separated `keys`, then compare shared columns.
    Creates global 'df_join'.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    df_b = pt.globals.get("df_B")
    if df_a is None:
        return "df_A not loaded."
    if df_b is None:
        return "df_B is not loaded."

    atol_val = parse_float(atol, 0.0)
    rtol_val = parse_float(rtol, 0.0)

    ks = (keys or "").strip()
    if ks.lower().startswith("keys="):
        ks = ks.split("=", 1)[1].strip()
    if (ks.startswith("'") and ks.endswith("'")) or (
        ks.startswith('"') and ks.endswith('"')
    ):
        ks = ks[1:-1]
    key_cols = [k.strip() for k in ks.split(",") if k.strip()]
    if not key_cols:
        return "Please provide one or more keys (comma-separated)."
    for key in key_cols:
        if key not in df_a.columns or key not in df_b.columns:
            return f"Key '{key}' not found in both dataframes."

    shared_cols = [c for c in df_a.columns if c in df_b.columns and c not in key_cols]
    a_subset = df_a[key_cols + shared_cols].copy()
    b_subset = df_b[key_cols + shared_cols].copy()
    a_subset.columns = [*key_cols] + [f"{c}__A" for c in shared_cols]
    b_subset.columns = [*key_cols] + [f"{c}__B" for c in shared_cols]

    df_join = pd.merge(a_subset, b_subset, on=key_cols, how=how)
    pt.globals["df_join"] = df_join
    try:
        duckdb.register("df_join", df_join)
    except Exception:
        pass

    numeric, categorical = [], []
    for col in shared_cols:
        col_a = df_join.get(f"{col}__A")
        col_b = df_join.get(f"{col}__B")
        if col_a is None or col_b is None:
            continue
        if pd.api.types.is_numeric_dtype(col_a) and pd.api.types.is_numeric_dtype(
            col_b
        ):
            diff = (col_a - col_b).astype("float64")
            eq = (
                diff.abs()
                <= (atol_val + rtol_val * col_b.abs().fillna(0))
            ).fillna(False)
            numeric.append(
                {
                    "column": col,
                    "count": int(diff.notna().sum()),
                    "mean_A": float(col_a.mean(skipna=True)),
                    "mean_B": float(col_b.mean(skipna=True)),
                    "mean_diff": float(diff.mean(skipna=True)),
                    "abs_mean_diff": float(diff.abs().mean(skipna=True)),
                    "pct_equal_with_tol": float(eq.mean() * 100.0),
                }
            )
        else:
            eq = col_a.astype("string") == col_b.astype("string")
            categorical.append(
                {
                    "column": col,
                    "count": int(eq.notna().sum()),
                    "match_rate_%": float(eq.mean(skipna=True) * 100.0),
                    "n_unique_A": int(col_a.nunique(dropna=True)),
                    "n_unique_B": int(col_b.nunique(dropna=True)),
                }
            )

    out = [
        f"[compare_on_keys] how={how}, rows={len(df_join)}, keys={key_cols}, "
        f"atol={atol_val}, rtol={rtol_val}"
    ]
    if numeric:
        df_num = pd.DataFrame(numeric).sort_values(
            "abs_mean_diff", ascending=False
        )
        out.append(
            "**Numeric comparison (top 20 by abs_mean_diff):**\n"
            + df_num.head(20).to_markdown(index=False)
        )
    else:
        out.append("**Numeric comparison:** None")
    if categorical:
        df_cat = pd.DataFrame(categorical).sort_values("match_rate_%")
        out.append(
            "**Categorical comparison (lowest 20 match first):**\n"
            + df_cat.head(20).to_markdown(index=False)
        )
    else:
        out.append("**Categorical comparison:** None")
    out.append("\nTip: 시각화가 필요하면 python_repl_ast에서 df_join을 사용하세요.")
    return "\n\n".join(out)


@tool
def mismatch_report(column: str, top_k: Any = 20) -> str:
    """
    Compute mismatch stats for a specific column after compare_on_keys.
    """
    pt = _ensure_pytool()
    dj = pt.globals.get("df_join")
    if dj is None:
        return "df_join not found. Run compare_on_keys() first."

    topk_val = parse_int(top_k, 20)

    col_a = f"{column}__A"
    col_b = f"{column}__B"
    if col_a not in dj.columns or col_b not in dj.columns:
        return f"Column '{column}' not found in df_join."
    series_a, series_b = dj[col_a], dj[col_b]
    if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(
        series_b
    ):
        diff = (series_a - series_b).abs()
        res = (
            dj.assign(abs_diff=diff)
            .sort_values("abs_diff", ascending=False)
            .head(topk_val)
        )
        return res.to_markdown(index=False)
    neq = dj[series_a.astype("string") != series_b.astype("string")]
    if len(neq) == 0:
        return "No mismatches."
    counts = (
        neq[[col_a, col_b]]
        .astype("string")
        .value_counts()
        .reset_index(name="count")
        .head(topk_val)
    )
    return counts.to_markdown(index=False)


@tool
def make_timesafe(column: str = "datetime", tz: str = "UTC") -> str:
    """
    Convert df_A time column to timezone-aware pandas datetime.
    """
    pt = _ensure_pytool()
    changed = []
    preferred = column or "datetime"
    for name in ["df_A", "df_B"]:
        current_df = pt.globals.get(name)
        if current_df is None:
            continue
        time_col = resolve_time_column(current_df, preferred)
        if time_col is None:
            continue
        tmp = current_df.copy()
        tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
        target_col = preferred
        if target_col not in tmp.columns or target_col != time_col:
            tmp[target_col] = tmp[time_col]
        if tz:
            try:
                if tmp[target_col].dt.tz is None:
                    tmp[target_col] = tmp[target_col].dt.tz_localize(tz)
                else:
                    tmp[target_col] = tmp[target_col].dt.tz_convert(tz)
            except Exception:
                pass
        pt.globals[name] = tmp
        changed.append(
            f"{name}({len(tmp)} rows, time_col='{time_col}')"
        )
    if not changed:
        return f"No target column '{column}' found in df_A/df_B."
    return (
        f"[make_timesafe] column='{column}', tz='{tz}' → updated: {', '.join(changed)}"
    )


@tool
def create_features(kind: str = "wear,temp,error,perf") -> str:
    """
    Create domain features on df_A if sources exist:
    - wear: rolling std etc.
    - temp: delta & rolling p95
    - error: per TB / per hour
    - perf: WAF, tail-latency ratio
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."
    out_cols: List[str] = []
    dfw = df_a.copy()
    kinds = {k.strip() for k in (kind or "").split(",") if k.strip()}

    if "wear" in kinds and "wear_leveling_count" in dfw.columns:
        dfw["wear_leveling_std_5"] = dfw["wear_leveling_count"].rolling(
            5, min_periods=1
        ).std()
        out_cols += ["wear_leveling_std_5"]

    if "temp" in kinds and "temperature" in dfw.columns:
        dfw["temp_delta"] = dfw["temperature"].diff()
        out_cols += ["temp_delta"]
        if "datetime" in dfw.columns:
            try:
                ts = pd.to_datetime(dfw["datetime"], errors="coerce")
                dfw = dfw.assign(__ts=ts).sort_values("__ts")
                dfw["temp_p95_12"] = dfw["temperature"].rolling(
                    12, min_periods=3
                ).apply(lambda x: np.nanpercentile(x, 95), raw=True)
                dfw = dfw.drop(columns=["__ts"])
                out_cols += ["temp_p95_12"]
            except Exception:
                pass

    if "error" in kinds:
        if "uncorrectable_error_count" in dfw.columns and "tbw" in dfw.columns:
            safe_tbw = dfw["tbw"].replace(0, np.nan)
            dfw["uncorrectable_per_tb"] = (
                dfw["uncorrectable_error_count"] / safe_tbw
            )
            out_cols += ["uncorrectable_per_tb"]
        if (
            "datetime" in dfw.columns
            and "uncorrectable_error_count" in dfw.columns
        ):
            ts = pd.to_datetime(dfw["datetime"], errors="coerce")
            dfw = dfw.assign(__ts=ts).sort_values("__ts")
            delta = dfw["uncorrectable_error_count"].diff()
            dt_hours = dfw["__ts"].diff().dt.total_seconds() / 3600.0
            dfw["uncorr_per_hour"] = delta / dt_hours.replace(0, np.nan)
            dfw = dfw.drop(columns=["__ts"])
            out_cols += ["uncorr_per_hour"]

    if "perf" in kinds:
        if "nand_writes" in dfw.columns and "host_writes" in dfw.columns:
            denom = dfw["host_writes"].replace(0, np.nan)
            dfw["waf"] = dfw["nand_writes"] / denom
            out_cols += ["waf"]
        if all(c in dfw.columns for c in ["latency_p50", "latency_p99"]):
            denom = dfw["latency_p50"].replace(0, np.nan)
            dfw["latency_tail_ratio"] = dfw["latency_p99"] / denom
            out_cols += ["latency_tail_ratio"]

    if not out_cols:
        return "No feature created (missing source columns)."
    pt.globals["df_A"] = dfw
    return f"[create_features] created: {out_cols}"


@tool
def rolling_stats(cols: str, window: str = "24H", on: str = "datetime") -> str:
    """
    Rolling mean/std for numeric cols in df_A using time window like '24H','7D'.
    Saves to pytool.globals['df_A_rolling'].
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."
    targets = [c.strip() for c in (cols or "").split(",") if c.strip()]
    if not targets:
        return "Please provide cols (comma-separated)."
    for col in targets:
        if col not in df_a.columns:
            return f"Column '{col}' not found in df_A."
    time_col = resolve_time_column(df_a, on)
    if time_col is None:
        return f"Time column '{on}' not found."
    tmp = df_a.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    if on not in tmp.columns:
        tmp[on] = tmp[time_col]
    tmp = tmp.sort_values(on).set_index(on)
    out = pd.DataFrame(index=tmp.index)
    for col in targets:
        if pd.api.types.is_numeric_dtype(tmp[col]):
            out[f"{col}_roll_mean"] = tmp[col].rolling(
                window, min_periods=3
            ).mean()
            out[f"{col}_roll_std"] = tmp[col].rolling(
                window, min_periods=3
            ).std()
    out = out.reset_index().rename(columns={on: on})
    pt.globals["df_A_rolling"] = out
    return f"[rolling_stats] window={window}, cols={targets}"


@tool
def stl_decompose(col: str, period: Any = 24, on: str = "datetime") -> str:
    """
    STL decomposition on df_A[col] with seasonal period (e.g., 24 for hourly daily).
    Saves components to pytool.globals['df_A_stl'].
    """
    if STL is None:
        return "statsmodels가 설치되어 있지 않습니다. (pip install statsmodels)"
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    period_val = parse_int(period, 24)

    s = str(col).strip()
    try:
        if s.startswith("{") and s.endswith("}"):
            obj = json.loads(s)
            s = obj.get("col") or obj.get("column") or obj.get("name") or ""
    except Exception:
        pass
    for prefix in ("col=", "column=", "name="):
        if s.lower().startswith(prefix):
            s = s.split("=", 1)[1].strip()
            break
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]

    if s not in df_a.columns:
        return f"Column '{s}' not found."
    time_col = resolve_time_column(df_a, on)
    if time_col is None:
        return f"Time column '{on}' not found."
    ts = pd.to_datetime(df_a[time_col], errors="coerce")
    values = pd.to_numeric(df_a[s], errors="coerce")
    ok = ts.notna() & values.notna()
    ts, values = ts[ok], values[ok]
    series = pd.Series(values.values, index=ts).sort_index()
    need = max(2 * period_val, 30)
    if len(series) < need:
        return f"Not enough points for STL (need ~{need})."
    res = STL(series, period=period_val, robust=True).fit()
    out = pd.DataFrame(
        {
            on: series.index,
            f"{s}_trend": res.trend.values,
            f"{s}_seasonal": res.seasonal.values,
            f"{s}_resid": res.resid.values,
        }
    )
    pt.globals["df_A_stl"] = out
    return (
        f"[stl_decompose] column='{s}', period={period_val}, rows={len(out)}. "
        "Call stl_plot() to visualise."
    )


@tool
def anomaly_iqr(col: str) -> str:
    """
    Mark IQR-based outliers on df_A[col]. Adds '{col}_is_outlier_iqr' boolean.
    Accepts flexible column specifications.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    s = str(col).strip()
    try:
        if s.startswith("{") and s.endswith("}"):
            obj = json.loads(s)
            s = obj.get("col") or obj.get("column") or obj.get("name") or ""
    except Exception:
        pass
    for prefix in ("col=", "column=", "name="):
        if s.lower().startswith(prefix):
            s = s.split("=", 1)[1].strip()
            break
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]

    if s == "":
        return "Please provide a column name."
    if s not in df_a.columns:
        return f"Column '{s}' not found."

    x = pd.to_numeric(df_a[s], errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (x < lo) | (x > hi)
    df_a[f"{s}_is_outlier_iqr"] = flags
    pt.globals["df_A"] = df_a
    return (
        f"[anomaly_iqr] col='{s}', bounds=({lo:.3f},{hi:.3f}), "
        f"outliers={int(flags.sum())}"
    )


@tool
def anomaly_isoforest(
    cols: str, contamination: Any = 0.01, random_state: Any = 42
) -> str:
    """
    IsolationForest anomalies on df_A[cols]. Adds 'isoforest_outlier' boolean.
    Accepts:
      - \"temperature,waf\"
      - \"cols=temperature,waf\"
      - {\"cols\": \"temperature,waf\"}
    """
    if IsolationForest is None:
        return "scikit-learn이 설치되어 있지 않습니다. (pip install scikit-learn)"
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    contamination_val = parse_float(contamination, 0.01)
    rs_val = parse_int(random_state, 42)

    raw_cols = cols
    targets: List[str] = []

    def _normalize_iterable(values) -> List[str]:
        return [str(c).strip() for c in values if str(c).strip()]

    if isinstance(raw_cols, (list, tuple, set)):
        targets = _normalize_iterable(raw_cols)
        s = ""
    else:
        s = str(raw_cols).strip()
        try:
            if s.startswith("{") and s.endswith("}"):
                obj = json.loads(s)
                extracted = (
                    obj.get("cols")
                    or obj.get("columns")
                    or obj.get("features")
                    or ""
                )
                if isinstance(extracted, (list, tuple, set)):
                    targets = _normalize_iterable(extracted)
                    s = ""
                else:
                    s = str(extracted or "").strip()
        except Exception:
            pass
        if not targets:
            for prefix in ("cols=", "columns=", "features="):
                if s.lower().startswith(prefix):
                    s = s.split("=", 1)[1].strip()
                    break
        if not targets:
            if (s.startswith("'") and s.endswith("'")) or (
                s.startswith('"') and s.endswith('"')
            ):
                s = s[1:-1]
            targets = [c.strip() for c in s.split(",") if c.strip()]

    if not targets:
        return "Please provide cols."
    for col in targets:
        if col not in df_a.columns:
            return f"Column '{col}' not found."

    X = df_a[targets].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 20:
        return "Not enough rows for IsolationForest (>=20 recommended)."
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination_val,
        random_state=rs_val,
    )
    pred = clf.fit_predict(X.values)  # -1 = outlier
    flags = pd.Series(pred == -1, index=X.index)
    df_a["isoforest_outlier"] = False
    df_a.loc[flags.index, "isoforest_outlier"] = flags
    pt.globals["df_A"] = df_a
    return (
        f"[anomaly_isoforest] cols={targets}, contamination={contamination_val}, "
        f"outliers={int(flags.sum())}"
    )


@tool
def cohort_compare(by: str = "model,fw", agg: str = "mean", cols: str = "") -> str:
    """
    Group df_A by categorical columns and aggregate numeric metrics.
    Saves to pytool.globals['df_cohort'].
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."
    keys = [k.strip() for k in (by or "").split(",") if k.strip()]
    for key in keys:
        if key not in df_a.columns:
            return f"Group key '{key}' not found."
    if cols:
        metrics = [c.strip() for c in cols.split(",") if c.strip()]
    else:
        metrics = [
            c for c in df_a.columns if pd.api.types.is_numeric_dtype(df_a[c])
        ]
    if not metrics:
        return "No numeric metrics to aggregate."
    aggfn = {
        "mean": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "count": "count",
    }.get(agg.lower())
    if aggfn is None:
        return "Unsupported agg. Use mean|median|max|min|count."
    grouped = (
        df_a.groupby(keys, dropna=False)[metrics].agg(aggfn).reset_index()
    )
    pt.globals["df_cohort"] = grouped
    return (
        f"[cohort_compare] by={keys}, agg={agg}, metrics={metrics}\nPreview:\n"
        f"{grouped.head().to_markdown(index=False)}"
    )


@tool
def topn_machines(
    metric: str = "uncorrectable_per_tb",
    n: Any = 20,
    machine_col: str = "machineID",
) -> str:
    """
    List top-N machines in df_A by a metric (descending).
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."
    n_val = parse_int(n, 20)
    if machine_col not in df_a.columns:
        return f"machine_col '{machine_col}' not found."
    if metric not in df_a.columns:
        return f"metric '{metric}' not found."
    sub = df_a[[machine_col, metric]].copy()
    sub = sub.sort_values(metric, ascending=False).head(n_val)
    pt.globals["df_topN"] = sub
    return (
        f"[topn_machines] metric='{metric}', n={n_val}\n"
        f"{sub.to_markdown(index=False)}"
    )


@tool
def select_numeric_candidates(
    min_unique: Any = 10, min_std: Any = 1e-9
) -> str:
    """
    Return numeric candidate columns in df_A with >= min_unique and std > min_std.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    min_unique_val = parse_int(min_unique, 10)
    min_std_val = parse_float(min_std, 1e-9)

    candidates = []
    for col in df_a.columns:
        if pd.api.types.is_numeric_dtype(df_a[col]):
            unique_count = df_a[col].nunique(dropna=True)
            std_val = pd.to_numeric(df_a[col], errors="coerce").std(skipna=True)
            if unique_count >= min_unique_val and (
                std_val is not None and std_val > min_std_val and np.isfinite(std_val)
            ):
                candidates.append(col)
    if not candidates:
        return "No numeric candidates."
    return (
        "Numeric candidates:\n"
        + pd.DataFrame({"column": candidates}).to_markdown(index=False)
    )


@tool
def rank_outlier_columns(method: str = "iqr_ratio", top_n: Any = 20) -> str:
    """
    Rank numeric columns by outlier ratio (IQR). Returns a table (head top_n).
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    top_n_val = parse_int(top_n, 20)

    num_cols = [
        c
        for c in df_a.columns
        if pd.api.types.is_numeric_dtype(df_a[c])
        and df_a[c].nunique(dropna=True) >= 10
    ]
    rows = []
    for col in num_cols:
        x = pd.to_numeric(df_a[col], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rate = float(((x < lo) | (x > hi)).mean() * 100.0)
        rows.append(
            {"column": col, "outlier_rate_%": rate, "lo": float(lo), "hi": float(hi)}
        )
    if not rows:
        return "No IQR-detectable outliers."
    rank_df = (
        pd.DataFrame(rows)
        .sort_values("outlier_rate_%", ascending=False)
        .head(top_n_val)
    )
    pt.globals["df_outlier_rank"] = rank_df
    return rank_df.to_markdown(index=False)


@tool
def plot_outliers(col: str, on: str = "datetime", sample: Any = 2000) -> str:
    """
    Plot boxplot and time series with IQR outlier highlighting for df_A[col].
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."
    sample_val = parse_int(sample, 2000)
    s = str(col).strip()
    if s not in df_a.columns:
        return f"Column '{s}' not found."
    x = pd.to_numeric(df_a[s], errors="coerce")
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return "IQR is zero or invalid."
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flags = (x < lo) | (x > hi)

    plt.figure()
    pd.DataFrame({s: x}).plot(kind="box")
    plt.title(f"Boxplot: {s} (IQR bounds: {lo:.2f}, {hi:.2f})")
    plt.xlabel("")
    plt.tight_layout()

    time_col = resolve_time_column(df_a, on)
    if time_col in df_a.columns:
        ts = pd.to_datetime(df_a[time_col], errors="coerce")
        dfv = pd.DataFrame({time_col: ts, s: x, "__out": flags}).dropna().sort_values(
            time_col
        )
        if on not in dfv.columns:
            dfv[on] = dfv[time_col]
        if sample_val and len(dfv) > sample_val:
            step = max(1, len(dfv) // sample_val)
            dfv = dfv.iloc[::step, :]
        plt.figure()
        plt.plot(dfv[on], dfv[s], linestyle="-", marker="", alpha=0.7)
        out = dfv[dfv["__out"]]
        if len(out) > 0:
            plt.scatter(out[on], out[s], marker="o", s=12)
        plt.title(f"Timeseries: {s} (outliers highlighted)")
        plt.tight_layout()

    return (
        f"[plot_outliers] col='{s}', outliers={int(flags.sum())}, "
        f"bounds=({lo:.3f},{hi:.3f})"
    )


@tool
def plot_outlier_overview(top_n: Any = 20) -> str:
    """
    Summary bar chart for top IQR outlier columns.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    top_n_val = parse_int(top_n, 20)

    rank_df = pt.globals.get("df_outlier_rank")
    if (
        rank_df is None
        or not isinstance(rank_df, pd.DataFrame)
        or "outlier_rate_%" not in rank_df.columns
    ):
        rows = []
        for col in df_a.columns:
            if pd.api.types.is_numeric_dtype(df_a[col]) and df_a[col].nunique(
                dropna=True
            ) >= 10:
                x = pd.to_numeric(df_a[col], errors="coerce")
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                iqr = q3 - q1
                if not np.isfinite(iqr) or iqr == 0:
                    continue
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                rate = float(((x < lo) | (x > hi)).mean() * 100.0)
                rows.append({"column": col, "outlier_rate_%": rate})
        if not rows:
            return "No IQR-detectable outliers."
        rank_df = pd.DataFrame(rows).sort_values(
            "outlier_rate_%", ascending=False
        )
        pt.globals["df_outlier_rank"] = rank_df

    top = rank_df.head(top_n_val)
    if top.empty:
        return "No IQR-detectable outliers."

    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top["column"][::-1], top["outlier_rate_%"][::-1])
    plt.xlabel("Outlier rate (%)")
    plt.title(f"Top {min(top_n_val, len(top))} outlier columns (IQR)")
    plt.tight_layout()
    return f"[plot_outlier_overview] top_n={top_n_val}"


@tool
def plot_outliers_multi(cols: str, on: str = "datetime", sample: Any = 1500) -> str:
    """
    Render small multiple line charts highlighting IQR outliers for columns.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    targets = [c.strip() for c in (cols or "").split(",") if c.strip()]
    if not targets:
        return "Please provide cols (comma-separated)."
    for col in targets:
        if col not in df_a.columns:
            return f"Column '{col}' not found in df_A."

    ts = None
    time_col = resolve_time_column(df_a, on)
    if time_col in df_a.columns:
        ts = pd.to_datetime(df_a[time_col], errors="coerce")

    n = len(targets)
    sample_val = parse_int(sample, 1500)
    fig, axes = plt.subplots(
        n, 1, figsize=(10, max(2.5, 1.5 * n)), sharex=ts is not None
    )
    if n == 1:
        axes = [axes]

    any_plotted = False
    for ax, col in zip(axes, targets):
        x = pd.to_numeric(df_a[col], errors="coerce")
        dfv = pd.DataFrame({col: x})
        if ts is not None:
            dfv[time_col] = ts
            dfv = dfv.dropna(subset=[time_col, col]).sort_values(time_col)
            if on not in dfv.columns:
                dfv[on] = dfv[time_col]
        else:
            dfv = dfv.dropna(subset=[col])

        if dfv.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_ylabel(col, rotation=0, labelpad=35, ha="right", va="center")
            continue

        if sample_val and len(dfv) > sample_val:
            step = max(1, len(dfv) // sample_val)
            dfv = dfv.iloc[::step, :]

        q1, q3 = dfv[col].quantile(0.25), dfv[col].quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            lo, hi = q1, q3
            flags = pd.Series(False, index=dfv.index)
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags = (dfv[col] < lo) | (dfv[col] > hi)

        if ts is not None:
            ax.plot(dfv[on], dfv[col], linewidth=1)
            out = dfv[flags]
            if len(out) > 0:
                ax.scatter(out[on], out[col], s=10)
        else:
            ax.plot(dfv.index, dfv[col], linewidth=1)
            out = dfv[flags]
            if len(out) > 0:
                ax.scatter(out.index, out[col], s=10)

        ax.set_ylabel(col, rotation=0, labelpad=35, ha="right", va="center")
        ax.grid(False)
        any_plotted = True

    if ts is not None:
        axes[-1].set_xlabel(on)
    if any_plotted:
        fig.suptitle("Outliers overview (IQR) — small multiples", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return f"[plot_outliers_multi] cols={targets}"


@tool
def plot_compare_timeseries(col: str, on: str = "datetime") -> str:
    """
    Plot df_join column comparison across time for df_A and df_B columns.
    """
    pt = _ensure_pytool()
    dj = pt.globals.get("df_join")
    if dj is None:
        return "df_join not found. Run compare_on_keys() first."

    col_a, col_b = f"{col}__A", f"{col}__B"
    if col_a not in dj.columns or col_b not in dj.columns:
        return f"Column '{col}' not found in df_join."
    time_col = resolve_time_column(dj, on)
    if time_col is None:
        return f"Time column '{on}' not found in df_join."

    dfv = dj[[time_col, col_a, col_b]].copy()
    dfv[time_col] = pd.to_datetime(dfv[time_col], errors="coerce")
    dfv[col_a] = pd.to_numeric(dfv[col_a], errors="coerce")
    dfv[col_b] = pd.to_numeric(dfv[col_b], errors="coerce")
    dfv = dfv.dropna().sort_values(time_col)
    if on not in dfv.columns:
        dfv[on] = dfv[time_col]
    if dfv.empty:
        return "No comparable rows with valid timestamps."

    dfv["abs_diff"] = (dfv[col_a] - dfv[col_b]).abs()

    plt.figure(figsize=(10, 4))
    plt.plot(dfv[on], dfv[col_a], label="A")
    plt.plot(dfv[on], dfv[col_b], label="B", alpha=0.8)
    plt.title(f"{col}: A vs B")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 2.8))
    plt.plot(dfv[on], dfv["abs_diff"])
    plt.title(f"{col}: |A - B|")
    plt.tight_layout()
    return f"[plot_compare_timeseries] col='{col}'"


@tool
def stl_plot(col: str, on: str = "datetime") -> str:
    """
    Visualise STL decomposition components captured by stl_decompose.
    """
    pt = _ensure_pytool()
    stl_df = pt.globals.get("df_A_stl")
    if stl_df is None or not isinstance(stl_df, pd.DataFrame) or stl_df.empty:
        return "Run stl_decompose first."
    time_col = resolve_time_column(stl_df, on)
    if time_col is None:
        return f"Time column '{on}' not found in df_A_stl."
    components = [f"{col}_trend", f"{col}_seasonal", f"{col}_resid"]
    missing = [c for c in components if c not in stl_df.columns]
    if missing:
        return f"Missing STL components: {missing}"
    chart = stl_df.copy()
    chart[time_col] = pd.to_datetime(chart[time_col], errors="coerce")
    if on not in chart.columns:
        chart[on] = chart[time_col]
    chart = chart.dropna(subset=[on]).sort_values(on)
    if chart.empty:
        return "No valid timestamps in df_A_stl."
    plt.figure(figsize=(10, 5))
    for comp, label in zip(components, ["trend", "seasonal", "resid"]):
        plt.plot(chart[on], chart[comp], label=label)
    plt.title(f"STL components: {col}")
    plt.legend()
    plt.tight_layout()
    return f"[stl_plot] col='{col}'"


@tool
def corr_heatmap(cols: str = "") -> str:
    """
    Render correlation heatmap for selected numeric columns.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    targets = [c.strip() for c in (cols or "").split(",") if c.strip()]
    if targets:
        for col in targets:
            if col not in df_a.columns:
                return f"Column '{col}' not found."
        dfv = df_a[targets].apply(pd.to_numeric, errors="coerce")
    else:
        dfv = df_a.select_dtypes(include=[np.number])

    if dfv.shape[1] < 2:
        return "Need at least 2 numeric columns."

    corr = dfv.corr(numeric_only=True)
    plt.figure(figsize=(max(5, 0.6 * corr.shape[1]), max(4, 0.6 * corr.shape[0])))
    plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.colorbar(label="corr")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return "[corr_heatmap] done"


@tool
def auto_outlier_eda(top_n: Any = 10, on: str = "datetime") -> str:
    """
    Run an outlier-first EDA pipeline on df_A.
    """
    pt = _ensure_pytool()
    df_a = pt.globals.get("df_A")
    if df_a is None:
        return "df_A not loaded."

    top_n_val = parse_int(top_n, 10)

    num_cols = [
        c for c in df_a.columns if pd.api.types.is_numeric_dtype(df_a[c])
    ]
    num_cols = [
        c for c in num_cols if df_a[c].nunique(dropna=True) >= 10
    ]
    if not num_cols:
        return "No numeric candidates."

    rows = []
    for col in num_cols:
        x = pd.to_numeric(df_a[col], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flags = (x < lo) | (x > hi)
        rate = float(flags.mean() * 100.0)
        rows.append(
            {"column": col, "outlier_rate_%": rate, "lo": float(lo), "hi": float(hi)}
        )
    if not rows:
        return "No IQR-detectable outliers."
    rank_df = pd.DataFrame(rows).sort_values(
        "outlier_rate_%", ascending=False
    )
    top_cols = rank_df.head(max(1, min(top_n_val, len(rank_df))))[
        "column"
    ].tolist()
    pt.globals["df_outlier_rank"] = rank_df

    summary = [
        "[auto_outlier_eda] IQR scan complete.",
        "**Top outlier columns:**\n"
        + rank_df.head(20).to_markdown(index=False),
    ]

    time_col = resolve_time_column(df_a, on)
    if time_col is not None and len(top_cols) > 0 and STL is not None:
        try:
            col = top_cols[0]
            ts = pd.to_datetime(df_a[time_col], errors="coerce")
            y = pd.to_numeric(df_a[col], errors="coerce")
            ok = ts.notna() & y.notna()
            ts, y = ts[ok], y[ok]
            series = pd.Series(y.values, index=ts).sort_index()
            if len(series) >= 48:
                res = STL(series, period=24, robust=True).fit()
                resid = pd.Series(res.resid, index=series.index)
                max_abs = float(resid.abs().max())
                when = str(resid.abs().idxmax())
                summary.append(
                    f"**STL residual spike** for '{col}': max |resid|={max_abs:.3f} at {when}"
                )
        except Exception:
            pass

    if IsolationForest is not None:
        try:
            k = min(4, len(top_cols))
            if k >= 2:
                X = df_a[top_cols[:k]].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna()
                if len(X) >= 50:
                    clf = IsolationForest(
                        n_estimators=200, contamination=0.02, random_state=42
                    ).fit(X.values)
                    out_rate = float((clf.predict(X.values) == -1).mean() * 100.0)
                    summary.append(
                        f"**IsolationForest** on {top_cols[:k]} → outlier_rate≈{out_rate:.1f}%"
                    )
        except Exception:
            pass

    return "\n\n".join(summary)


__all__ = ["build_tools", "pytool"]
