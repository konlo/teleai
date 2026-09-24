"""Wide file-backed datasets feed local analysis without full-frame decoding."""
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_aggregate import aggregate_dataset
from utils.analysis_charts import render_chart_spec, render_count_rate_chart
from utils.analysis_compare import compare_group_aggregates
from utils.analysis_group_summary import summarize_groups
from utils.analysis_outliers import detect_outliers, winsorize_numeric_summary
from utils.analysis_pivot import pivot_dataset
from utils.analysis_statistics import statistical_test
from utils.analysis_timeseries import prepare_time_series


class ProjectionTests(unittest.TestCase):
    def test_agent_histogram_uses_file_backed_column_without_remote_or_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "projected-agent", ForbiddenModel())
            info = runtime.datasets.register_batches(
                [pd.DataFrame({"signal": [float(n) for n in range(300)],
                               "unused": ["wide"] * 300})],
                columns=["signal", "unused"], source="arbitrary.runtime_source",
                max_rows=500, coverage="complete", predicate_known=True)
            try:
                with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                    outcome = runtime.submit("signal 히스토그램을 보여줘")
                self.assertEqual(outcome["status"], "answered", outcome)
                card = runtime.artifacts[runtime.inspect()["chart_ids"][0]]
                self.assertEqual(runtime.datasets.metadata[card.dataset_id].parent_id, info.id)
                self.assertTrue(card.image.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(runtime.inspect()["recovery"]["model_calls"], 0)
            finally:
                runtime.close()

    def test_file_backed_charts_aggregate_and_statistics_read_only_needed_columns(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "projected-analysis")
            store = PersistentDatasets(db, budget=0)
            frame = pd.DataFrame({
                "measure": [1., 2., 3., 4., 5., 6.],
                "cohort": ["a", "a", "a", "b", "b", "b"],
                "outcome": ["yes", "no", "yes", "no", "yes", "yes"],
                "event_time": pd.date_range("2026-08-01", periods=6, freq="D"),
                **{f"unused_{n}": [f"payload-{n}"] * 6 for n in range(24)},
            })
            info = store.register_batches([frame.iloc[:3], frame.iloc[3:]],
                columns=list(frame.columns), source="arbitrary.runtime_source",
                max_rows=100, coverage="complete", predicate_known=True)
            db.select_dataset(info.id)
            cohort = store.register(frame.iloc[:3].copy(), source=info.source,
                coverage="complete", predicate_known=True, parent_id=info.id)
            original_file = db.dataset_file(info.id).read_bytes()
            with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                with patch.object(store.frames, "project", wraps=store.frames.project) as projected:
                    histogram, histogram_summary, _ = render_chart_spec(
                        store, info.id, kind="histogram", x="measure", bins=3)
                    self.assertTrue(histogram.image.startswith(b"\x89PNG\r\n\x1a\n"))
                    self.assertEqual(histogram_summary["source_rows"], 6)
                    grouped = aggregate_dataset(store, info.id, aggregation="mean",
                                                value_column="measure", group_column="cohort")
                    self.assertEqual(grouped["aggregation_result"]["complete_rows"], 6)
                    counted = aggregate_dataset(store, info.id, aggregation="count")
                    self.assertEqual(counted["aggregation_result"]["complete_rows"], 6)
                    self.assertEqual(store.frames.project(counted["dataset"]["id"], ["count"])
                                     ["count"].iloc[0], 6)
                    test = statistical_test(store, info.id, test="mean_ci",
                                            value_column="measure")
                    self.assertEqual(test["test_result"]["sample"]["complete_rows"], 6)
                    rate, summary, _ = render_count_rate_chart(
                        store, info.id, group_column="cohort", outcome_column="outcome",
                        success_value="yes")
                    self.assertTrue(rate.image.startswith(b"\x89PNG\r\n\x1a\n"))
                    self.assertEqual(summary["rendered_groups"], 2)
                    series = prepare_time_series(store, info.id, time_column="event_time",
                        frequency="day", aggregation="count")
                    self.assertEqual(series["time_series_result"]["complete_rows"], 6)
                    compared = compare_group_aggregates(store, info.id, cohort.id,
                        aggregation="count", group_column="cohort")
                    self.assertEqual(compared["comparison_result"]["baseline_complete_rows"], 6)
                    pivot = pivot_dataset(store, info.id, index_columns=["cohort"],
                        column_columns=["outcome"], aggregation="count")
                    self.assertEqual(pivot["pivot_result"]["complete_rows"], 6)
                    grouped = summarize_groups(store, info.id, group_columns=["cohort"],
                        metrics=[{"name": "observations", "aggregation": "count"},
                                 {"name": "average", "aggregation": "mean",
                                  "value_column": "measure"}])
                    self.assertEqual(grouped["group_summary_result"]["group_count"], 2)
                    detected = detect_outliers(store, info.id, column="measure", method="iqr")
                    self.assertEqual(detected["outlier_result"]["sample"]["input_rows"], 6)
                    winsorized = winsorize_numeric_summary(store, info.id, column="measure",
                        lower_quantile=0.1, upper_quantile=0.9)
                    self.assertEqual(winsorized["winsorization_result"]["sample"]["input_rows"], 6)
                    selected = [call.args[1] for call in projected.call_args_list]
            self.assertIn(["measure"], selected)
            self.assertIn(["measure", "cohort"], selected)
            self.assertIn(["cohort", "outcome"], selected)
            self.assertIn(["event_time"], selected)
            self.assertIn(["cohort"], selected)
            self.assertTrue(all(not any(column.startswith("unused_") for column in cols)
                                for cols in selected))
            self.assertEqual(db.selected_dataset_id(), info.id)
            self.assertEqual(db.dataset_file(info.id).read_bytes(), original_file)
            db.close()

    def test_invalid_column_does_not_decode_legacy_blob(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "legacy-projection")
            store = PersistentDatasets(db, budget=0)
            info = store.register(pd.DataFrame({"measure": [1., 2., 3.],
                                                "unused": ["x", "y", "z"]}),
                                  source="arbitrary.source", coverage="complete",
                                  predicate_known=True)
            with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                with self.assertRaises(ValueError):
                    render_chart_spec(store, info.id, kind="histogram", x="missing")
                chart, _, _ = render_chart_spec(store, info.id, kind="histogram", x="measure")
            self.assertTrue(chart.image.startswith(b"\x89PNG\r\n\x1a\n"))
            db.close()

    def test_local_sql_wildcard_keeps_all_source_columns(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "wildcard-projection")
            store = PersistentDatasets(db, budget=0)
            info = store.register_batches([pd.DataFrame({
                "measure": [1, 2, 3], "unused": ["a", "b", "c"]})],
                columns=["measure", "unused"], source="arbitrary.source",
                max_rows=10, coverage="complete", predicate_known=True)
            context = AnalysisToolContext(store, {}, [], lambda **_: None)
            sql = next(tool.run for tool in build_analysis_tools(context)
                       if tool.name == "local_analysis_sql")
            with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                counted = sql(info.id, "SELECT COUNT(*) AS total FROM data")
            self.assertEqual(counted["status"], "ready")
            self.assertEqual(store.frames.project(counted["dataset"]["id"], ["total"])
                             ["total"].iloc[0], 3)
            with patch.object(store.frames, "project", side_effect=AssertionError("unexpected projection")):
                result = sql(info.id, "SELECT * FROM data LIMIT 2")
                alias_collision = sql(info.id,
                    "SELECT measure AS unused FROM data WHERE unused = 'a'")
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["dataset"]["columns"], ("measure", "unused"))
            self.assertEqual(store.frames[result["dataset"]["id"]]["unused"].tolist(), ["a", "b"])
            self.assertEqual(alias_collision["status"], "ready")
            self.assertEqual(store.frames[alias_collision["dataset"]["id"]]["unused"].tolist(), [1])
            db.close()


if __name__ == "__main__":
    unittest.main()
