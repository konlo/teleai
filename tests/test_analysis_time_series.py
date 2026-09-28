"""Bounded datetime preparation, multi-series charting, and graph recovery."""
import tempfile
import unittest

import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import DatasetStore


SOURCE = "arbitrary.runtime_events"


def events():
    return pd.DataFrame({
        "occurred_at": [
            "2026-01-01 01:00", "2026-01-01 01:00", "2026-01-01 03:00",
            "2026-01-03 01:00", "2026-01-03 02:00",
        ],
        "cohort_key": ["alpha", "alpha", "beta", "alpha", "beta"],
        "metric_amount": [2, 4, 11, 7, 13],
    })


class TimeSeriesTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(self.context)}
        self.parent = self.store.register(
            events(), source=SOURCE, coverage="complete", predicate_known=True,
            snapshot="2026-01-03T00:00:00+00:00")

    def prepare(self):
        return self.tools["prepare_time_series"](
            self.parent.id, time_column="occurred_at", value_column="metric_amount",
            group_column="cohort_key", frequency="day", aggregation="sum",
            gap_policy="zero", timezone="Asia/Seoul", max_output_rows=5000)

    def test_grouped_daily_sum_gap_timezone_and_lineage_match_oracle(self):
        result = self.prepare()
        self.assertEqual(result["status"], "ready")
        summary = result["time_series_result"]
        child = self.store.metadata[result["dataset"]["id"]]
        actual = self.store.frames[child.id]
        expected = pd.DataFrame({
            "occurred_at": pd.to_datetime([
                "2026-01-01", "2026-01-01", "2026-01-02",
                "2026-01-02", "2026-01-03", "2026-01-03",
            ]).tz_localize("Asia/Seoul"),
            "cohort_key": ["alpha", "beta", "alpha", "beta", "alpha", "beta"],
            "metric_amount": [6.0, 11.0, 0.0, 0.0, 7.0, 13.0],
        })
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(child.parent_id, self.parent.id)
        self.assertEqual(child.snapshot, self.parent.snapshot)
        self.assertEqual(child.grain, "aggregate")
        self.assertEqual(summary["duplicate_time_rows"], 2)
        self.assertEqual(summary["gap_rows_added"], 2)
        self.assertEqual(summary["timezone"], "Asia/Seoul")
        self.assertEqual(len(summary["data_sha256"]), 64)

    def test_prepared_dataset_renders_two_real_series(self):
        prepared = self.prepare()
        child_id = prepared["dataset"]["id"]
        result = self.tools["render_chart_spec"](
            child_id, kind="line", x="occurred_at", y="metric_amount",
            category="cohort_key", aggregation="none", sort="ascending")
        self.assertEqual(result["status"], "ready")
        card = self.context.artifacts[result["cards"][0]["id"]]
        self.assertTrue(card.image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(card.columns, ("occurred_at", "metric_amount", "cohort_key"))
        points = result["render_summary"]["points"]
        self.assertEqual({point["category"] for point in points}, {"alpha", "beta"})
        self.assertEqual(len(points), 6)

    def test_nan_gap_policy_preserves_visible_line_breaks(self):
        prepared = self.tools["prepare_time_series"](
            self.parent.id, time_column="occurred_at", value_column="metric_amount",
            group_column="cohort_key", frequency="day", aggregation="sum",
            gap_policy="nan", timezone="Asia/Seoul", max_output_rows=5000)
        result = self.tools["render_chart_spec"](
            prepared["dataset"]["id"], kind="line", x="occurred_at",
            y="metric_amount", category="cohort_key", aggregation="none",
            sort="ascending")
        points = result["render_summary"]["points"]
        self.assertEqual(sum(point["value"] is None for point in points), 2)

    def test_unsafe_time_inputs_fail_closed_with_structured_errors(self):
        structured = {tool.name: tool for tool in local_tools(self.context)}
        epoch = self.store.register(
            pd.DataFrame({"tick": [1, 2, 3], "value": [1, 2, 3]}),
            source="arbitrary.numeric_time", coverage="complete", predicate_known=True)
        rejected = structured["prepare_time_series"].invoke({
            "dataset_id": epoch.id, "time_column": "tick", "value_column": "value",
            "frequency": "day", "aggregation": "sum"})
        self.assertEqual(rejected["status"], "error")
        self.assertEqual(rejected["error_code"], "invalid_tool_input")
        self.assertFalse(rejected["retryable"])

        too_many = self.store.register(
            pd.DataFrame({
                "when": pd.date_range("2026-01-01", periods=21, freq="h"),
                "group": [f"g{i}" for i in range(21)],
            }), source="arbitrary.groups", coverage="complete", predicate_known=True)
        rejected = structured["prepare_time_series"].invoke({
            "dataset_id": too_many.id, "time_column": "when", "group_column": "group",
            "frequency": "hour", "aggregation": "count"})
        self.assertEqual(rejected["status"], "error")
        self.assertEqual(rejected["error_code"], "invalid_tool_input")

    def test_production_graph_prepares_and_renders_without_model_or_remote(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "time-series", ForbiddenModel())
            runtime.datasets.register(
                events(), source=SOURCE, coverage="complete", predicate_known=True,
                snapshot="2026-01-03T00:00:00+00:00")
            result = runtime.submit(
                "보유 데이터에서 occurred_at 날짜를 기준으로 cohort_key별 일별 "
                "metric_amount 합계를 빈 날짜는 0으로 채워 선 그래프로 보여줘. "
                "시간대는 Asia/Seoul")
            self.assertEqual(result["status"], "answered", result)
            state = runtime.inspect()["recovery"]
            self.assertEqual(state["model_calls"], 0)
            self.assertEqual(len(state["sent_calls"]), 2)
            self.assertEqual(state["time_series_evidence"]["time_series_result"]["output_rows"], 6)
            chart_id = runtime.inspect()["chart_ids"][0]
            child_id = state["time_series_dataset"]
            self.assertEqual(runtime.artifacts[chart_id].dataset_id, child_id)
            runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "time-series", ForbiddenModel())
            self.assertIn(child_id, reopened.datasets.metadata)
            self.assertIn(chart_id, reopened.inspect()["chart_ids"])
            self.assertTrue(reopened.artifacts[chart_id].image.startswith(b"\x89PNG\r\n\x1a\n"))
            reopened.close()


if __name__ == "__main__":
    unittest.main()
