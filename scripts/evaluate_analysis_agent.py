"""Evaluate the production graph using original test_set questions and independent oracles.

No Databricks executor is connected or approved. Fixtures are preloaded in a fresh
temporary scope per question. This measures local analysis, not live DB access,
browser rendering, conversation continuity, or final-prose factual consistency.
Only --live-local-model makes model calls; --list/--all without it reports coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlparse
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

GRADING_PATH = ROOT / "tests/fixtures/analysis_agent_grading.json"


def load_specs():
    """Import definitions only; importing the old runner would execute fixture setup."""
    return [spec for level in (1, 2) for part in range(1, 5)
            for spec in getattr(import_module(f"test_set.level{level}.definitions_part{part}"),
                                f"get_level{level}_part{part}")()]


def load_grading():
    return json.loads(GRADING_PATH.read_text())["cases"]


def load_frames():
    return {name: pd.read_csv(ROOT / "test_set/data" / f"{name}.csv")
            for name in ("bank_loan", "titanic")}


def fixture_reference_context(table, frame, context_dir=None):
    """Describe actual fixture types and bounded category values, without answers.

    Optional external aliases are table-wide fixture context. Never inject a
    question's synonym_mapping: doing so would leak that question's reference.
    """
    context_dir = Path(context_dir) if context_dir else ROOT / "test_set/data_context"
    aliases = {}
    alias_source = None
    context_path = context_dir / (table + ".json")
    if context_path.exists():
        external = json.loads(context_path.read_text())
        if external.get("table") != table:
            raise ValueError("External fixture context belongs to another table")
        aliases = {column["name"]: column.get("aliases", []) for column in external.get("columns", [])}
        alias_source = external.get("source")
    columns = []
    for name, dtype in frame.dtypes.items():
        series = frame[name]
        distinct = int(series.nunique(dropna=True))
        top_values = []
        if distinct <= 10:
            for value, count in series.dropna().value_counts().head(10).items():
                value = value.item() if isinstance(value, np.generic) else value
                top_values.append({"value": value, "count": int(count)})
        columns.append({"name": name, "dtype": str(dtype), "distinct_count": distinct,
                        "null_count": int(series.isna().sum()), "top_values": top_values,
                        "aliases": [value for value in aliases.get(name, []) if isinstance(value, str)][:20]})
    return {"table": table, "training_status": "fixture_profile", "columns": columns,
            "alias_source": alias_source,
            "source": "local test_set fixture; not a live Databricks table"}


def _distribution(values, weights=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Only finite one-dimensional histogram inputs are graded")
    weights = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != values.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid histogram weights")
    grouped = pd.DataFrame({"value": values, "weight": weights}).groupby("value")["weight"].sum()
    return grouped[grouped != 0]


def _frame_digest(frame):
    encoded_columns = json.dumps(list(frame.columns), ensure_ascii=False).encode()
    hashed = pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False).values.tobytes()
    return sha256(encoded_columns + hashed).hexdigest()


class HistogramCapture:
    """Observe real Matplotlib input and bind it to the bytes actually persisted.

    This changes no plotting arguments. Keep evaluations sequential because the
    Matplotlib method instrumentation is process-global. Unsupported plots remain
    ungraded rather than passing by virtue of an image being present.
    """
    def __init__(self):
        self.histograms = []
        self.scatters = []
        self.boxplots = []
        self.saved = {}

    def __enter__(self):
        self.stack = ExitStack()
        original_hist, original_scatter = Axes.hist, Axes.scatter
        original_boxplot, original_save = Axes.boxplot, Figure.savefig

        def observed_hist(axis, x, *args, **kwargs):
            result = original_hist(axis, x, *args, **kwargs)
            try:
                self.histograms.append({"figure": id(axis.figure),
                    "distribution": _distribution(x, kwargs.get("weights")),
                    "bins": len(result[1]) - 1,
                    "rendered_total": float(np.asarray(result[0]).sum())})
            except (TypeError, ValueError):
                pass  # An unsupported plot is never accepted by the grader.
            return result

        def observed_scatter(axis, x, y, *args, **kwargs):
            result = original_scatter(axis, x, y, *args, **kwargs)
            try:
                left, right = np.asarray(x), np.asarray(y)
                if left.ndim == right.ndim == 1 and len(left) == len(right):
                    self.scatters.append({"figure": id(axis.figure), "x": left, "y": right})
            except (TypeError, ValueError):
                pass
            return result

        def observed_boxplot(axis, x, *args, **kwargs):
            result = original_boxplot(axis, x, *args, **kwargs)
            try:
                if isinstance(x, (list, tuple)) and x and all(
                        np.asarray(group).ndim == 1 for group in x):
                    groups = [np.asarray(group) for group in x]
                    labels = kwargs.get("tick_labels") or kwargs.get("labels")
                    self.boxplots.append({"figure": id(axis.figure), "groups": groups,
                                          "labels": list(labels) if labels is not None else None})
                else:
                    values = np.asarray(x)
                    if values.ndim == 1:
                        self.boxplots.append({"figure": id(axis.figure), "values": values})
            except (TypeError, ValueError):
                pass
            return result

        def observed_save(figure, destination, *args, **kwargs):
            result = original_save(figure, destination, *args, **kwargs)
            if hasattr(destination, "getvalue"):
                content = destination.getvalue()
                if isinstance(content, bytes) and content.startswith(b"\x89PNG\r\n\x1a\n"):
                    self.saved[sha256(content).hexdigest()] = id(figure)
            return result

        self.stack.enter_context(patch.object(Axes, "hist", observed_hist))
        self.stack.enter_context(patch.object(Axes, "scatter", observed_scatter))
        self.stack.enter_context(patch.object(Axes, "boxplot", observed_boxplot))
        self.stack.enter_context(patch.object(Figure, "savefig", observed_save))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def reference_oracle(spec, grading, frames):
    """Run trusted, checked-in reference code, never model-authored code.

    stdout is discarded and is never parsed for a guessed answer. Selectors name
    existing variables explicitly. Missing/changed contracts fail closed.
    """
    namespace = {"pd": pd, "np": np, "plt": plt, "matplotlib": matplotlib,
                 "df_bank": frames["bank_loan"].copy(),
                 "df_titanic": frames["titanic"].copy()}
    try:
        plt.close("all")
        with HistogramCapture() as capture, redirect_stdout(StringIO()):
            exec(compile(spec["python_code"], f"reference:{spec['id']}", "exec"), namespace)
        if grading["kind"] == "histogram":
            if len(capture.histograms) != 1:
                raise ValueError("Reference must produce exactly one supported histogram")
            return capture.histograms[0]
        if grading["kind"] == "chart_scatter":
            if len(capture.scatters) != 1:
                raise ValueError("Reference must produce exactly one scatter plot")
            observed = capture.scatters[0]
            plotted = pd.DataFrame({grading["x"]: observed["x"], grading["y"]: observed["y"]})
            return {"data_sha256": _frame_digest(plotted), "rows": len(plotted)}
        if grading["kind"] == "chart_boxplot":
            if len(capture.boxplots) != 1:
                raise ValueError("Reference must produce exactly one boxplot")
            plotted = pd.DataFrame({grading["column"]: capture.boxplots[0]["values"]})
            return {"data_sha256": _frame_digest(plotted), "rows": len(plotted)}
        if grading["kind"] == "chart_grouped_boxplot":
            grouped = [item for item in capture.boxplots if item.get("groups")]
            if len(grouped) != 1:
                raise ValueError("Reference must produce exactly one grouped boxplot")
            observed = grouped[0]
            source = frames[spec["target_table"]][[grading["category"], grading["column"]]].dropna()
            group_values = sorted(source[grading["category"]].unique(), key=lambda value: str(value))
            if observed.get("labels") and [str(value) for value in group_values] != [str(value) for value in observed["labels"]]:
                raise ValueError("Reference grouped boxplot labels differ from source groups")
            if len(observed["groups"]) != len(group_values):
                raise ValueError("Reference grouped boxplot labels do not match groups")
            plotted = pd.concat([
                pd.DataFrame({grading["category"]: [group] * len(values),
                              grading["column"]: pd.to_numeric(values)})
                for group, values in zip(group_values, observed["groups"])
            ], ignore_index=True)
            return {"data_sha256": _frame_digest(plotted), "rows": len(plotted),
                    "groups": [str(value) for value in group_values]}
        if grading["kind"] == "chart_bar_counts":
            value = namespace[grading["variable"]]
            if not isinstance(value, pd.Series) or value.empty:
                raise ValueError("Reference bar counts must be a non-empty Series")
            return {str(key): float(count) for key, count in value.items()}
        if grading["kind"] == "metadata_columns":
            columns = namespace[grading["variable"]]
            if not isinstance(columns, list) or not columns or not all(isinstance(value, str) for value in columns):
                raise ValueError("Reference metadata columns must be a non-empty string list")
            return columns
        if grading["kind"] == "metadata_dtypes":
            frame = namespace[grading["variable"]]
            if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 2 or frame.empty:
                raise ValueError("Reference dtype metadata must be a non-empty two-column frame")
            return [{"name": str(row.iloc[0]), "dtype": str(row.iloc[1])}
                    for _, row in frame.iterrows()]
        if grading["kind"] == "metadata_column_subset":
            columns = namespace[grading["variable"]]
            if not isinstance(columns, list) or not all(isinstance(value, str) for value in columns):
                raise ValueError("Reference metadata subset must be a string list")
            return columns
        if grading["kind"] in {"null_counts", "distinct_counts"}:
            value = namespace[grading["variable"]]
            if isinstance(value, pd.Series):
                result = {str(key): int(count) for key, count in value.items()}
            elif isinstance(value, pd.DataFrame) and value.shape[1] == 2:
                result = {str(row.iloc[0]): int(row.iloc[1]) for _, row in value.iterrows()}
            else:
                raise ValueError("Reference profile counts must be a Series or two-column frame")
            if grading.get("positive_only"):
                result = {key: count for key, count in result.items() if count > 0}
            return result
        if grading["kind"] == "scalar":
            value = namespace[grading["variable"]]
            reduction = grading.get("reduction")
            if reduction == "rows":
                value = len(value)
            elif reduction == "column_mean_percent":
                value = value[grading["column"]].mean() * 100
            elif reduction is not None:
                raise ValueError("Unsupported scalar reduction")
            if not np.isscalar(value) or not np.isfinite(float(value)):
                raise ValueError("Reference scalar is not finite")
            return float(value)
        if grading["kind"] == "scalar_set":
            values = {}
            for result_column, variable in grading["variables"].items():
                value = namespace[variable]
                if not np.isscalar(value) or not np.isfinite(float(value)):
                    raise ValueError("Reference scalar set contains a non-finite value")
                values[result_column] = float(value)
            if len(values) < 2:
                raise ValueError("Scalar set must verify at least two requested values")
            return values
        if grading["kind"] == "statistical":
            statistic = namespace[grading["statistic_variable"]]
            p_value = namespace[grading["p_value_variable"]]
            if not np.isscalar(statistic) or not np.isscalar(p_value):
                raise ValueError("Reference statistic and p-value must be scalar")
            result = {"statistic": float(statistic), "p_value": float(p_value)}
            if not all(np.isfinite(value) for value in result.values()):
                raise ValueError("Reference statistical result is not finite")
            if grading.get("df_variable"):
                df = namespace[grading["df_variable"]]
                if not np.isscalar(df) or not np.isfinite(float(df)):
                    raise ValueError("Reference degrees of freedom is not finite")
                result["degrees_of_freedom"] = float(df)
            return result
        if grading["kind"] == "statistical_mean_ci":
            estimate = namespace[grading["estimate_variable"]]
            interval = namespace[grading["ci_variable"]]
            if not np.isscalar(estimate) or len(interval) != 2:
                raise ValueError("Reference mean confidence interval has an invalid shape")
            result = {"estimate": float(estimate), "lower": float(interval[0]), "upper": float(interval[1])}
            if not all(np.isfinite(value) for value in result.values()):
                raise ValueError("Reference mean confidence interval is not finite")
            return result
        if grading["kind"] == "outlier":
            result = {}
            for path, definition in grading["fields"].items():
                value = namespace[definition["variable"]]
                reduction = definition.get("reduction")
                if reduction == "rows":
                    value = len(value)
                elif reduction == "row_percent":
                    value = len(value) / len(namespace[definition["denominator"]]) * 100
                elif reduction == "column_min":
                    value = value[definition["column"]].min()
                elif reduction == "column_max":
                    value = value[definition["column"]].max()
                elif reduction is not None:
                    raise ValueError("Unsupported outlier oracle reduction")
                if not np.isscalar(value) or not np.isfinite(float(value)):
                    raise ValueError("Reference outlier metric is not finite")
                result[path] = float(value)
            return result
        if grading["kind"] == "outlier_cohort_scalar_set":
            selected = namespace[grading["selection_variable"]]
            if not isinstance(selected, pd.DataFrame) or selected.empty:
                raise ValueError("Reference outlier cohort must be a non-empty DataFrame")
            values = {}
            for result_column, definition in grading["variables"].items():
                value = namespace[definition["variable"]]
                reduction = definition.get("reduction")
                if reduction == "rows":
                    value = len(value)
                elif reduction == "column_mean_percent":
                    value = value[definition["column"]].mean() * 100
                else:
                    raise ValueError("Unsupported outlier cohort reduction")
                if not np.isscalar(value) or not np.isfinite(float(value)):
                    raise ValueError("Reference outlier cohort metric is not finite")
                values[result_column] = float(value)
            threshold = namespace[grading["threshold_variable"]]
            if not np.isscalar(threshold) or not np.isfinite(float(threshold)):
                raise ValueError("Reference outlier threshold is not finite")
            return {"data_sha256": _frame_digest(selected), "rows": len(selected),
                    "threshold": float(threshold), "values": values}
        if grading["kind"] == "category_counts":
            return _category_counts(namespace[grading["variable"]])
        raise ValueError("Unsupported grading kind")
    finally:
        plt.close("all")


def _category_counts(frame):
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 2:
        raise ValueError("Expected a two-column category/count result")
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if len(numeric) != 1:
        raise ValueError("Expected one count and one category column")
    category = next(c for c in frame.columns if c != numeric[0])
    if frame[category].isna().any() or frame[category].duplicated().any():
        raise ValueError("Category keys must be unique and non-null")
    return frame.set_index(category)[numeric[0]].sort_index().astype(float)


def _same_series(left, right):
    return (left.index.equals(right.index) and
            bool(np.allclose(left.to_numpy(), right.to_numpy(), rtol=1e-8, atol=1e-8)))


def _fixture_dtype_family(value):
    """Independent pandas-based interpretation of fixture dtype strings."""
    try:
        dtype = pd.api.types.pandas_dtype(str(value))
    except (TypeError, ValueError):
        return "other"
    if pd.api.types.is_bool_dtype(dtype):
        return "categorical"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)):
        return "categorical"
    return "other"


def _fixture_descendant(runtime, dataset_id, fixture_id):
    visited, pending = set(), [dataset_id]
    while pending:
        current = pending.pop()
        if current == fixture_id:
            return True
        if not current or current in visited:
            continue
        visited.add(current)
        info = runtime.datasets.metadata.get(current)
        if info is None:
            continue
        pending.extend(getattr(info, "parent_ids", ()) or (() if not info.parent_id else (info.parent_id,)))
    return False


def _computed_projection(query):
    """A table mention in WHERE cannot turn a literal projection into evidence."""
    from sqlglot import exp
    from core.analysis_sql import validate_query
    try:
        tree = validate_query(query, dialect="duckdb")
    except (TypeError, ValueError):
        return False
    has_data = any(table.name == "data" for table in tree.find_all(exp.Table))
    return bool(tree.expressions) and all(
        expression.find(exp.Column) is not None or expression.find(exp.AggFunc) is not None
        or (has_data and isinstance(expression, exp.Star)) for expression in tree.expressions)


def _replay_calculation(runtime, dataset_id, fixture_id, fixture_frame, seen=None):
    """Reexecute recorded local lineage only, with no model or remote executor."""
    from core.analysis_sql import local_query, validate_query
    from utils.analysis_provenance import query_conditions
    from utils.analysis_datasets import filter_frame
    if dataset_id == fixture_id:
        return fixture_frame.copy()
    seen = set() if seen is None else seen
    if dataset_id in seen:
        raise ValueError("Cyclic calculation lineage")
    seen.add(dataset_id)
    info = runtime.datasets.metadata[dataset_id]
    if not info.parent_id:
        raise ValueError("Calculation has no fixture ancestor")
    parent = _replay_calculation(runtime, info.parent_id, fixture_id, fixture_frame, seen)
    parent_info = runtime.datasets.metadata[info.parent_id]
    if info.query:
        # Explicit requested_conditions are applied before SQL by the tool. The
        # saved metadata also includes SQL WHERE conditions; exclude those here
        # because the unchanged query applies them itself during replay.
        sql_conditions = query_conditions(validate_query(info.query, dialect="duckdb")) or ()
        residual = tuple(condition for condition in info.conditions
                         if condition not in parent_info.conditions and condition not in sql_conditions)
        if residual:
            parent = filter_frame(parent, residual)
        result, truncated, _ = local_query(parent, info.query)
        if truncated:
            raise ValueError("Counterfactual calculation was truncated")
        return result
    residual = tuple(condition for condition in info.conditions if condition not in parent_info.conditions)
    if not residual:
        raise ValueError("Derived calculation has no replayable transformation")
    return filter_frame(parent, residual)


def verify_counterfactuals(runtime, dataset_id, fixture_id, spec, grading, frames):
    """A guessed answer cannot pass just because the fixture happens to match.

    Replay the actual query/derivation chain against two deterministic alternative
    fixtures and independently rerun the unchanged reference code on each. These
    are grader probes, never extra agent executions or production data mutations.
    """
    target = spec["target_table"]
    shifted = frames[target].copy()
    for column in shifted.select_dtypes(include=["number"]).columns:
        shifted[column] = shifted[column] * 1.1 + 7
    variations = (shifted, frames[target].iloc[::2].copy())
    for index, variant in enumerate(variations, 1):
        oracle = reference_oracle(spec, grading, {**frames, target: variant})
        actual = _replay_calculation(runtime, dataset_id, fixture_id, variant)
        if grading["kind"] == "scalar":
            column = grading.get("result_column")
            ok = (len(actual) == 1 and
                  (column in actual.columns if column else actual.shape[1] == 1) and
                  np.isclose(float(actual.iloc[0][column] if column else actual.iloc[0, 0]),
                             oracle, rtol=1e-8, atol=1e-8))
        elif grading["kind"] == "scalar_set":
            ok = (len(actual) == 1 and set(actual.columns) == set(oracle)
                  and all(np.isclose(float(actual.iloc[0][column]), value,
                                    rtol=1e-8, atol=1e-8)
                          for column, value in oracle.items()))
        else:
            ok = _same_series(_category_counts(actual), oracle)
        if not ok:
            return False, index
    return True, len(variations)


def grade_evidence(runtime, outcome, spec, grading, oracle, fixture_id, capture):
    from langchain_core.messages import ToolMessage
    state = runtime.inspect()
    if outcome.get("status") not in {"answered", "complete"} or state["state"] != "idle":
        return "NOT_COMPLETE", "Agent stopped or awaits approval; no implicit approval", {}
    if state["recovery"].get("status") not in {None, "complete"}:
        return "NOT_COMPLETE", "Recovery has not established completion", {}
    if grading["kind"] in {"metadata_columns", "metadata_dtypes", "metadata_column_subset"}:
        matches = []
        for message in runtime.events():
            if not isinstance(message, ToolMessage) or message.name != "inspect_table_context":
                continue
            try:
                observation = json.loads(message.content)
                context = observation["table_context"]
                schema = [{"name": column["name"], "dtype": str(column.get("dtype") or "")}
                          for column in context["columns"]]
                columns = [column["name"] for column in schema]
                same_table = str(context["table"]).casefold().split(".")[-1] == spec["target_table"].casefold()
                if observation.get("status") != "ready" or not same_table:
                    continue
                if grading["kind"] == "metadata_columns":
                    actual, matches_oracle = columns, columns == oracle
                elif grading["kind"] == "metadata_dtypes":
                    actual, matches_oracle = schema, schema == oracle
                else:
                    family = grading["family"]
                    actual = [column["name"] for column in schema
                              if _fixture_dtype_family(column["dtype"]) == family]
                    matches_oracle = actual == oracle
                if matches_oracle:
                    matches.append({"table": context["table"], "columns": columns,
                                    "column_count": len(columns), "schema": schema,
                                    "selected_columns": actual,
                                    "authority": observation.get("authority")})
            except (KeyError, TypeError, ValueError):
                continue
        return ("PASS", "Inspected table context metadata matches the reference",
                {"metadata": matches[-1]}) if matches else (
                "FAIL", "Missing fresh table-context evidence or schema metadata differs from reference", {})
    if grading["kind"] in {"null_counts", "distinct_counts"}:
        matches = []
        field = "null_count" if grading["kind"] == "null_counts" else "distinct_count"
        for message in runtime.events():
            if not isinstance(message, ToolMessage) or message.name != "profile_dataset":
                continue
            try:
                observation = json.loads(message.content)
                dataset_id = observation["dataset_id"]
                info = runtime.datasets.metadata[dataset_id]
                profile_columns = observation["profile"]["columns"]
                actual = {str(column["name"]): int(column[field]) for column in profile_columns}
                if grading.get("positive_only"):
                    actual = {key: count for key, count in actual.items() if count > 0}
                same_source = info.source == spec["target_table"]
                complete = info.coverage == "complete" and info.predicate_known
                descendant = _fixture_descendant(runtime, dataset_id, fixture_id)
                expected_keys = set(oracle)
                selected = {key: actual[key] for key in expected_keys if key in actual}
                exact_columns = not grading.get("exact_columns") or set(actual) == expected_keys
                if (observation.get("status") == "ready" and same_source and complete and descendant
                        and selected == oracle and exact_columns):
                    matches.append({"dataset_id": dataset_id, "counts": selected,
                                    "scope": observation.get("scope")})
            except (KeyError, TypeError, ValueError):
                continue
        return ("PASS", "Structured dataset profile matches the independent reference",
                {"profile": matches[-1]}) if matches else (
                "FAIL", "Missing complete profile evidence or profile counts differ from reference", {})
    if grading["kind"] in {"chart_scatter", "chart_boxplot", "chart_grouped_boxplot", "chart_bar_counts"}:
        matches = []
        expected_kind = {"chart_scatter":"scatter", "chart_boxplot":"boxplot",
                         "chart_grouped_boxplot":"boxplot", "chart_bar_counts":"bar"}[grading["kind"]]
        for message in runtime.events():
            if not isinstance(message, ToolMessage) or message.name != "render_chart_spec":
                continue
            try:
                observation = json.loads(message.content)
                entry = observation["cards"][0]
                card = runtime.artifacts[entry["id"]]
                info = runtime.datasets.metadata[card.dataset_id]
                summary = observation["render_summary"]
                spec_result = observation["chart_spec"]
                digest = sha256(card.image).hexdigest()
                grounded = (observation.get("status") == "ready" and card.kind == expected_kind
                            and spec_result.get("kind") == expected_kind
                            and info.source == spec["target_table"] and info.coverage == "complete"
                            and _fixture_descendant(runtime, card.dataset_id, fixture_id)
                            and card.image.startswith(b"\x89PNG\r\n\x1a\n")
                            and digest in capture.saved)
                if grading["kind"] == "chart_bar_counts":
                    actual = {str(point["x"]): float(point["value"])
                              for point in summary.get("points", [])}
                    correct = (actual == oracle and spec_result.get("aggregation") == "count"
                               and spec_result.get("x") == grading["column"])
                elif grading["kind"] == "chart_grouped_boxplot":
                    expected_columns = [grading["column"], grading["category"]]
                    correct = (list(card.columns) == expected_columns
                               and spec_result.get("x") == grading["column"]
                               and spec_result.get("category") == grading["category"]
                               and summary.get("data_sha256") == oracle["data_sha256"]
                               and summary.get("rendered_rows") == oracle["rows"])
                else:
                    expected_columns = ([grading["column"]] if grading["kind"] == "chart_boxplot"
                                        else [grading["x"], grading["y"]])
                    correct = (list(card.columns) == expected_columns
                               and summary.get("data_sha256") == oracle["data_sha256"]
                               and summary.get("rendered_rows") == oracle["rows"])
                if grounded and correct:
                    matches.append({"chart_id":card.id,"png_sha256":digest,
                                    "kind":card.kind,"columns":list(card.columns),
                                    "data_sha256":summary.get("data_sha256")})
            except (KeyError, TypeError, ValueError):
                continue
        return ("PASS", "Real PNG and declarative chart data match the independent reference",
                {"charts": matches}) if matches else (
                "FAIL", "Missing grounded chart PNG or rendered data differs from reference", {})
    if grading["kind"] == "histogram":
        matches = []
        for chart_id in state["chart_ids"]:
            card = runtime.artifacts[chart_id]
            info = runtime.datasets.metadata[card.dataset_id]
            if (card.kind != "histogram" or tuple(card.columns) != (grading["column"],)
                    or info.source != spec["target_table"] or info.coverage != "complete"
                    or not _fixture_descendant(runtime, card.dataset_id, fixture_id)):
                continue
            digest = sha256(card.image).hexdigest()
            figure_id = capture.saved.get(digest)
            for observed in capture.histograms:
                if (observed["figure"] == figure_id and
                        _same_series(observed["distribution"], oracle["distribution"]) and
                        np.isclose(observed["rendered_total"], oracle["distribution"].sum())):
                    matches.append({"chart_id": chart_id, "png_sha256": digest,
                                    "observations": float(observed["distribution"].sum()),
                                    "actual_bins": observed["bins"], "reference_bins": oracle["bins"]})
        return ("PASS", "Real PNG and plotted weighted distribution match the reference", {"charts": matches}) if matches else (
            "FAIL", "Missing histogram PNG or plotted data differs from reference", {})
    if grading["kind"] in {"statistical", "statistical_mean_ci"}:
        observations = []
        for message in runtime.events():
            if not isinstance(message, ToolMessage) or message.name != "statistical_test":
                continue
            try:
                observation = json.loads(message.content)
                if observation.get("status") == "ready":
                    observations.append(observation)
            except (ValueError, TypeError, AttributeError):
                continue
        if not observations:
            return "FAIL", "No structured statistical-test result; assistant prose is not evidence", {}
        observation = observations[-1]
        try:
            dataset_id = observation["dataset_id"]
            result = observation["test_result"]
            info = runtime.datasets.metadata[dataset_id]
            sample = result["sample"]
            grounded = (
                info.source == spec["target_table"]
                and info.coverage == "complete"
                and info.predicate_known
                and info.grain == "raw"
                and _fixture_descendant(runtime, dataset_id, fixture_id)
                and result.get("kind") == grading["test"]
                and set(result.get("columns", [])) == set(grading["columns"])
                and sample.get("input_rows") == info.rows
                and sample.get("complete_rows", 0) >= 2
                and sample.get("complete_rows", 0) + sample.get("dropped_rows", -1) == info.rows
                and isinstance(result.get("assumptions"), dict)
                and bool(result.get("assumptions"))
                and isinstance(result.get("confidence_intervals"), list)
                and isinstance(result.get("warnings"), list)
            )
            if grading["kind"] == "statistical":
                statistic = float(result["statistic"])
                expected_statistic = float(oracle["statistic"])
                if grading["test"] == "independent_t":
                    statistic_matches = bool(np.isclose(
                        abs(statistic), abs(expected_statistic), rtol=1e-10, atol=1e-12))
                elif grading["test"] == "mann_whitney":
                    groups = result.get("groups", [])
                    total_pairs = int(groups[0]["n"]) * int(groups[1]["n"])
                    statistic_matches = bool(
                        np.isclose(statistic, expected_statistic, rtol=1e-10, atol=1e-12)
                        or np.isclose(total_pairs - statistic, expected_statistic, rtol=1e-10, atol=1e-12))
                else:
                    statistic_matches = bool(np.isclose(
                        statistic, expected_statistic, rtol=1e-10, atol=1e-12))
                p_value_matches = bool(np.isclose(
                    float(result["p_value"]), float(oracle["p_value"]), rtol=1e-10, atol=1e-12))
                df_matches = True
                if "degrees_of_freedom" in oracle:
                    df_matches = bool(np.isclose(
                        float(result["degrees_of_freedom"]), oracle["degrees_of_freedom"],
                        rtol=1e-10, atol=1e-12))
                correct = (statistic_matches and p_value_matches and df_matches
                           and isinstance(result.get("effect_size"), dict))
                details = {"expected": oracle,
                           "actual": {"statistic": statistic, "p_value": float(result["p_value"]),
                                      "degrees_of_freedom": result.get("degrees_of_freedom")},
                           "sample": sample, "effect_size": result.get("effect_size"),
                           "confidence_intervals": result.get("confidence_intervals"),
                           "dataset_id": dataset_id}
            else:
                interval = result["confidence_intervals"]
                correct = (
                    result.get("statistic") is None
                    and result.get("p_value") is None
                    and result.get("effect_size") is None
                    and len(interval) == 1
                    and np.isclose(float(result["estimate"]["value"]), oracle["estimate"],
                                   rtol=1e-10, atol=1e-12)
                    and np.isclose(float(interval[0]["lower"]), oracle["lower"],
                                   rtol=1e-10, atol=1e-12)
                    and np.isclose(float(interval[0]["upper"]), oracle["upper"],
                                   rtol=1e-10, atol=1e-12)
                )
                details = {"expected": oracle,
                           "actual": {"estimate": result["estimate"]["value"],
                                      "lower": interval[0]["lower"], "upper": interval[0]["upper"]},
                           "sample": sample, "confidence_intervals": interval,
                           "dataset_id": dataset_id}
        except (KeyError, TypeError, ValueError, IndexError):
            return "FAIL", "Statistical result shape or type does not match the grading contract", {}
        return (("PASS", "Structured statistical evidence matches the independent reference", details)
                if grounded and correct else
                ("FAIL", "Statistical evidence lacks complete provenance or differs from the reference", details))
    if grading["kind"] == "outlier_cohort_scalar_set":
        selections, calculations = [], []
        for message in runtime.events():
            if not isinstance(message, ToolMessage):
                continue
            try:
                observation = json.loads(message.content)
            except (ValueError, TypeError, AttributeError):
                continue
            if observation.get("status") != "ready":
                continue
            if message.name == "select_outlier_rows":
                selections.append(observation)
            elif message.name == "local_analysis_sql":
                calculations.append(observation)
        if not selections or not calculations:
            return "FAIL", "Outlier cohort selection and follow-up calculation evidence are both required", {}
        selection, calculation = selections[-1], calculations[-1]
        try:
            child_id = selection["dataset"]["id"]
            result_id = calculation["dataset"]["id"]
            child = runtime.datasets.metadata[child_id]
            result_info = runtime.datasets.metadata[result_id]
            result = selection["outlier_result"]
            summary = selection["selection_summary"]
            frame = runtime.datasets.frames[result_id]
            selected_frame = runtime.datasets.frames[child_id]
            actual = {column: float(frame.iloc[0][column]) for column in grading["variables"]}
            grounded = (
                child.source == spec["target_table"]
                and child.coverage == "complete" and not child.predicate_known
                and child.grain == "raw" and not child.aggregation
                and child.parent_id == fixture_id
                and _fixture_descendant(runtime, child_id, fixture_id)
                and result_info.parent_id == child_id
                and result_info.source == child.source
                and result_info.snapshot == child.snapshot
                and result_info.coverage == "complete"
                and _computed_projection(result_info.query)
                and result.get("kind") == "outlier_detection"
                and result.get("method") == grading["method"]
                and result.get("tail") == grading["tail"]
                and result.get("column") == grading["column"]
                and summary.get("selection") == grading["selection"]
                and summary.get("parent_dataset_id") == fixture_id
                and summary.get("selected_rows") == child.rows == oracle["rows"]
                and summary.get("data_sha256") == oracle["data_sha256"]
                and _frame_digest(selected_frame) == oracle["data_sha256"]
                and result.get("sample", {}).get("input_rows")
                    == runtime.datasets.metadata[fixture_id].rows
                and not selection.get("preview")
            )
            correct = (
                frame.shape[0] == 1
                and np.isclose(float(result["thresholds"]["upper"]), oracle["threshold"],
                               rtol=1e-10, atol=1e-12)
                and all(np.isclose(actual[column], expected, rtol=1e-10, atol=1e-12)
                        for column, expected in oracle["values"].items())
            )
            details = {"expected": oracle, "actual": actual,
                       "cohort_dataset_id": child_id, "result_dataset_id": result_id,
                       "scope": selection.get("scope")}
        except (KeyError, TypeError, ValueError, IndexError):
            return "FAIL", "Outlier cohort evidence shape or type does not match the grading contract", {}
        return (("PASS", "Lineage-safe outlier cohort and follow-up metrics match the independent reference", details)
                if grounded and correct else
                ("FAIL", "Outlier cohort lineage or follow-up metrics differ from the reference", details))
    if grading["kind"] == "outlier":
        observations = []
        for message in runtime.events():
            if not isinstance(message, ToolMessage) or message.name != "detect_outliers":
                continue
            try:
                observation = json.loads(message.content)
                if observation.get("status") == "ready":
                    observations.append(observation)
            except (ValueError, TypeError, AttributeError):
                continue
        if not observations:
            return "FAIL", "No structured outlier result; assistant prose is not evidence", {}
        observation = observations[-1]
        try:
            dataset_id = observation["dataset_id"]
            result = observation["outlier_result"]
            info = runtime.datasets.metadata[dataset_id]
            actual = {}
            for path in grading["fields"]:
                value = result
                for part in path.split("."):
                    value = value[part]
                actual[path] = float(value)
            grounded = (
                info.source == spec["target_table"]
                and info.coverage == "complete"
                and info.predicate_known
                and info.grain == "raw"
                and _fixture_descendant(runtime, dataset_id, fixture_id)
                and result.get("kind") == "outlier_detection"
                and result.get("method") == grading["method"]
                and result.get("tail") == grading["tail"]
                and result.get("column") == grading["column"]
                and result.get("sample", {}).get("input_rows") == info.rows
                and result.get("sample", {}).get("valid_rows", 0)
                    + result.get("sample", {}).get("missing_rows", -1) == info.rows
                and result.get("boundary_policy")
            )
            correct = all(np.isclose(actual[path], expected, rtol=1e-10, atol=1e-12)
                          for path, expected in oracle.items())
            details = {"expected": oracle, "actual": actual, "dataset_id": dataset_id,
                       "scope": observation.get("scope")}
        except (KeyError, TypeError, ValueError):
            return "FAIL", "Outlier result shape or type does not match the grading contract", {}
        return (("PASS", "Structured outlier evidence matches the independent reference", details)
                if grounded and correct else
                ("FAIL", "Outlier evidence lacks complete provenance or differs from the reference", details))
    candidates = []
    for message in runtime.events():
        if isinstance(message, ToolMessage) and message.name == "local_analysis_sql":
            try:
                observation = json.loads(message.content)
                if observation.get("status") == "ready":
                    candidates.append(observation["dataset"]["id"])
            except (ValueError, TypeError, KeyError):
                pass
    if not candidates:
        return "FAIL", "No structured calculation result; assistant prose is not evidence", {}
    # Grade the last calculation, not a correct intermediate followed by a wrong answer.
    dataset_id = candidates[-1]
    info = runtime.datasets.metadata[dataset_id]
    if (info.source != spec["target_table"] or info.coverage != "complete" or
            not _fixture_descendant(runtime, dataset_id, fixture_id)):
        return "FAIL", "Calculation lacks complete fixture provenance", {}
    if not _computed_projection(info.query):
        return "FAIL", "Literal projection is not evidence of data calculation", {}
    frame = runtime.datasets.frames[dataset_id]
    try:
        if grading["kind"] == "scalar":
            column = grading.get("result_column")
            if len(frame) != 1 or (column not in frame.columns if column else frame.shape[1] != 1):
                raise ValueError("Expected exactly one scalar cell")
            actual = float(frame.iloc[0][column] if column else frame.iloc[0, 0])
            ok = bool(np.isclose(actual, oracle, rtol=1e-8, atol=1e-8))
            details = {"expected": oracle, "actual": actual, "result_dataset_id": dataset_id}
        elif grading["kind"] == "scalar_set":
            if len(frame) != 1 or set(frame.columns) != set(oracle):
                raise ValueError("Expected one row with the declared scalar columns")
            actual = {column: float(frame.iloc[0][column]) for column in oracle}
            ok = all(np.isclose(actual[column], value, rtol=1e-8, atol=1e-8)
                     for column, value in oracle.items())
            details = {"expected": oracle, "actual": actual, "result_dataset_id": dataset_id}
        else:
            actual = _category_counts(frame)
            ok = _same_series(actual, oracle)
            details = {"expected": oracle.to_dict(), "actual": actual.to_dict(), "result_dataset_id": dataset_id}
    except (TypeError, ValueError):
        return "FAIL", "Calculation shape/type does not match reference contract", {}
    return ("PASS" if ok else "FAIL", "Compared final structured calculation with independent reference", details)


def preserve_runtime_metadata(runtime, spec, artifact_dir=None):
    """Retain diagnostics after temporary SQLite/Parquet storage is removed.

    Deliberately omit transcript/message content, model reasoning, preview rows,
    and failed tool bodies. Field allowlisting also protects future diagnostics
    additions from automatically becoming evaluation report content.
    """
    for handler in runtime.diagnostics.logger.handlers:
        handler.flush()
    fields = {"time", "event", "run_id", "error_id", "stage", "error_type", "http_status",
              "frames", "tool", "status", "elapsed_seconds", "span_id", "characters", "limit",
              "attempts", "operation", "missing_chart"}
    diagnostics = []
    for line in runtime.diagnostics.path.read_text().splitlines():
        try:
            event = json.loads(line)
            diagnostics.append({key: value for key, value in event.items() if key in fields})
        except (ValueError, TypeError, AttributeError):
            diagnostics.append({"event": "unreadable_diagnostic_record"})
    state = runtime.inspect()
    charts = []
    for chart_id in state["chart_ids"]:
        card = runtime.artifacts[chart_id]
        digest = sha256(card.image).hexdigest()
        chart = {"id": card.id, "dataset_id": card.dataset_id, "kind": card.kind,
                 "columns": list(card.columns), "png_sha256": digest,
                 "png_valid": card.image.startswith(b"\x89PNG\r\n\x1a\n")}
        if artifact_dir and chart["png_valid"]:
            destination = Path(artifact_dir) / f"{spec['id']}-{digest[:12]}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(card.image)
            chart["path"] = str(destination.resolve())
        charts.append(chart)
    metadata = {"case_id": spec["id"], "run_id": runtime.diagnostics.run_id,
                "state": state["state"], "message_count": state["message_count"],
                "pending_requests": len(state["requests"]),
                "recovery_status": state["recovery"].get("status"),
                "recovery_attempts": state["recovery"].get("attempts"),
                "recovery_model_calls": state["recovery"].get("model_calls"),
                "reference_context": runtime.context.reference_context,
                "datasets": [asdict(info) for info in runtime.datasets.metadata.values()],
                "charts": charts, "diagnostics": diagnostics}
    if artifact_dir:
        run_suffix = (runtime.diagnostics.run_id or "before-submit")[:12]
        destination = Path(artifact_dir) / f"{spec['id']}-{run_suffix}-metadata.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n")
        metadata["path"] = str(destination.resolve())
    return metadata


def evaluate_case(spec, grading, model, *, frames=None, artifact_dir=None):
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    from langchain_core.messages import AIMessage, ToolMessage
    base = {"id": spec["id"], "prompt": spec["prompt"], "target_table": spec["target_table"],
            "reference_sha256": sha256(spec["python_code"].encode()).hexdigest()}
    if grading is None:
        return {**base, "status": "UNGRADED", "reason": "No explicit machine-comparable oracle; no model call"}
    frames = frames if frames is not None else load_frames()
    try:
        oracle = reference_oracle(spec, grading, frames)
    except Exception as exc:
        return {**base, "status": "UNGRADED", "reason": "Reference oracle unavailable",
                "error_type": type(exc).__name__}
    remote_attempts = []
    def forbidden_factory(_datasets):
        def forbidden(envelope):
            remote_attempts.append(envelope["query"])
            raise AssertionError("Databricks execution is forbidden in fixture evaluation")
        return forbidden
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-agent-eval-") as temporary:
        runtime = GraphAnalysisRuntime(temporary, "evaluation", spec["id"], model,
            connection_identity="fixture-only:no-databricks", remote_factory=forbidden_factory)
        record = {**base, "status": "FAIL", "reason": "Evaluation interrupted before completion"}
        try:
            frame = frames[spec["target_table"]]
            payload = frame.to_csv(index=False).encode()
            info = runtime.datasets.register(frame.copy(), source=spec["target_table"],
                coverage="complete", predicate_known=True, snapshot="fixture:" + sha256(payload).hexdigest())
            runtime.context.reference_context[:] = [fixture_reference_context(spec["target_table"], frame)]
            # Tests can bind deterministic calls after the actual fixture ID exists.
            if hasattr(model, "evaluation_dataset_id"):
                model.evaluation_dataset_id = info.id
            with HistogramCapture() as capture:
                outcome = runtime.submit(spec["prompt"])
            status, reason, evidence = grade_evidence(runtime, outcome, spec, grading, oracle, info.id, capture)
            if status == "PASS" and grading["kind"] in {"scalar", "scalar_set", "category_counts"}:
                try:
                    verified, probes = verify_counterfactuals(runtime, evidence["result_dataset_id"],
                        info.id, spec, grading, frames)
                    evidence["counterfactual_probes"] = probes
                    if not verified:
                        status, reason = "FAIL", "Calculation fails independent counterfactual fixture checks"
                except (KeyError, TypeError, ValueError):
                    status, reason = "FAIL", "Calculation lineage cannot be independently replayed"
            events = runtime.events()
            calls = [call for message in events if isinstance(message, AIMessage) for call in message.tool_calls]
            observations = []
            for message in events:
                if isinstance(message, ToolMessage):
                    try:
                        value = json.loads(message.content)
                        observations.append({"tool": message.name, "status": value.get("status"),
                                             "error_code": value.get("error_code")})
                    except (ValueError, TypeError, AttributeError):
                        observations.append({"tool": message.name, "status": "unstructured"})
            if remote_attempts:
                status, reason = "FAIL", "Unexpected forbidden executor invocation"
            if artifact_dir and evidence.get("charts"):
                artifact_dir = Path(artifact_dir)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                for chart in evidence["charts"]:
                    destination = artifact_dir / f"{spec['id']}-{chart['png_sha256'][:12]}.png"
                    destination.write_bytes(runtime.artifacts[chart["chart_id"]].image)
                    chart["path"] = str(destination.resolve())
            record = {**base, "status": status, "reason": reason, "evidence": evidence,
                    "agent_status": outcome.get("status"), "runtime_state": runtime.inspect()["state"],
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "tools": dict(Counter(call["name"] for call in calls)),
                    "tool_calls": [{"id": call["id"], "tool": call["name"], "arguments": call["args"]}
                                   for call in calls],
                    "queries": [{"tool": call["name"], "query": call["args"].get("query")}
                                for call in calls if call["name"] in {"local_analysis_sql", "query_databricks"}],
                    "observations": observations, "remote_executions": 0,
                    "forbidden_executor_invocations": len(remote_attempts),
                    "error_type": outcome.get("error_type"), "error_id": outcome.get("error_id")}
        except Exception as exc:
            record = {**base, "status": "FAIL", "reason": "Evaluation exception",
                      "error_type": type(exc).__name__,
                      "elapsed_seconds": round(time.monotonic() - started, 3),
                      "remote_executions": 0, "forbidden_executor_invocations": len(remote_attempts)}
        finally:
            try:
                record["runtime_metadata"] = preserve_runtime_metadata(runtime, spec, artifact_dir)
            except Exception as exc:
                record.update(status="FAIL", reason="Runtime evidence could not be preserved",
                              metadata_error_type=type(exc).__name__)
            finally:
                runtime.close()
                for handler in list(runtime.diagnostics.logger.handlers):
                    runtime.diagnostics.logger.removeHandler(handler)
                    handler.close()
        return record


def build_report(results, specs, grading, mode, model_name=None):
    statuses = Counter(result["status"] for result in results)
    durations = [result["elapsed_seconds"] for result in results if "elapsed_seconds" in result]
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime", "model": model_name,
            "data_scope": "preloaded complete local test_set fixtures in temporary per-case scope",
            "coverage": {"total_reference_cases": len(specs), "supported_oracles": len(grading),
                         "selected": len(results), "statuses": dict(statuses),
                         "all_selected_passed": bool(results) and all(result["status"] == "PASS" for result in results),
                         "ungraded_is_never_pass": True},
            "latency_seconds": {"total": round(sum(durations), 3),
                                "max": max(durations, default=None)},
            "remote_executions": sum(result.get("remote_executions", 0) for result in results),
            "results": results,
            "limitations": ["Not a Databricks or browser end-to-end test",
                            "Single-turn questions; follow-up journeys require separate evaluation",
                            "Final prose is not graded; PASS establishes tool-result/artifact accuracy only",
                            "Unsupported reference cases stay UNGRADED and are excluded from success claims",
                            "Scripted-model tests validate the harness, not natural-language model quality"]}


def evaluation_exit_status(results):
    if any(result["status"] in {"FAIL", "NOT_COMPLETE"} for result in results):
        return 1
    if not results or any(result["status"] != "PASS" for result in results):
        return 2  # A partially graded suite is not a successful full-suite gate.
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--id", action="append", help="Reference ID; may be repeated")
    choice.add_argument("--all", action="store_true", help="Include all references; unsupported cases remain UNGRADED")
    parser.add_argument("--list", action="store_true", help="List support status without model calls")
    parser.add_argument("--live-local-model", action="store_true", help="Run actual configured localhost ChatOllama")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/actual_agent_evaluation.json")
    args = parser.parse_args(argv)
    specs, grading = load_specs(), load_grading()
    selected = specs if args.all or args.list else [spec for spec in specs if spec["id"] in (args.id or grading)]
    unknown = set(args.id or ()) - {spec["id"] for spec in specs}
    if unknown:
        parser.error("Unknown IDs: " + ", ".join(sorted(unknown)))
    if args.list:
        for spec in selected:
            print(f"{spec['id']}\t{grading.get(spec['id'], {}).get('kind', 'UNGRADED')}\t{spec['prompt']}")
        return 0
    if not args.live_local_model:
        results = [{"id": spec["id"], "status": "NOT_RUN" if spec["id"] in grading else "UNGRADED"}
                   for spec in selected]
        report = build_report(results, specs, grading, "coverage-only")
    else:
        from dotenv import load_dotenv
        from langchain_ollama import ChatOllama
        load_dotenv(ROOT / ".env")
        endpoint = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if urlparse(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
            parser.error("Live fixture evaluation permits only a localhost Ollama endpoint")
        # Fixture data and model reasoning must not be shipped to a tracing backend.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        model_name = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
        model = ChatOllama(model=model_name, base_url=endpoint, reasoning=True,
            temperature=0, num_ctx=16384, num_predict=4096, client_kwargs={"timeout": 60})
        results, frames = [], load_frames()
        for spec in selected:
            try:
                result = evaluate_case(spec, grading.get(spec["id"]), model, frames=frames,
                                       artifact_dir=args.output.parent / "actual_agent_eval_artifacts")
            except Exception as exc:
                result = {"id": spec["id"], "status": "FAIL", "reason": "Evaluation exception",
                          "error_type": type(exc).__name__}
            results.append(result)
            print(json.dumps({k: result[k] for k in ("id", "status", "elapsed_seconds") if k in result}), flush=True)
            # Keep partial failures visible if a later model request is interrupted.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(build_report(results, specs, grading, "live-local-model", model_name),
                                             ensure_ascii=False, indent=2) + "\n")
        report = build_report(results, specs, grading, "live-local-model", model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["coverage"], ensure_ascii=False))
    print(f"Report: {args.output}")
    if not args.live_local_model:
        return 0
    return evaluation_exit_status(results)


if __name__ == "__main__":
    raise SystemExit(main())
