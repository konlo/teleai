"""Parent-versus-cohort grouped comparison contracts."""
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_agent import fixture_reference_context
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_compare import dataset_digest
from utils.analysis_datasets import DatasetStore


class AnalysisComparisonTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "anomaly_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 110, 120],
            "segment_code": ["north", "south", "east", "north", "south", "east",
                             "north", "south", "east", "north", "north", "south"],
            "metric_value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        })
        self.store = DatasetStore()
        self.parent = self.store.register(
            self.frame.copy(), source="arbitrary.runtime_measurements",
            coverage="complete", predicate_known=True, snapshot="fixture:v1")
        context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(context)}

    def test_comparison_preserves_two_parent_lineage_and_exact_values(self):
        selected = self.tools["select_outlier_rows"](
            self.parent.id, "anomaly_value", "iqr", selection="inliers", tail="upper")
        cohort_id = selected["dataset"]["id"]
        compared = self.tools["compare_group_aggregates"](
            self.parent.id, cohort_id, "mean", "segment_code",
            value_column="metric_value")
        result = self.store.frames[compared["dataset"]["id"]]
        baseline = (self.frame.groupby("segment_code")["metric_value"].mean()
                    .rename("baseline_mean_metric_value"))
        cohort = self.store.frames[cohort_id]
        cohort_mean = (cohort.groupby("segment_code")["metric_value"].mean()
                       .rename("cohort_mean_metric_value"))
        expected = pd.concat([baseline, cohort_mean], axis=1).reset_index()
        expected["difference"] = (expected["cohort_mean_metric_value"]
                                  - expected["baseline_mean_metric_value"])
        expected["percent_change"] = (
            expected["difference"] / expected["baseline_mean_metric_value"].abs() * 100)
        expected = expected.sort_values("segment_code").reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected)
        info = self.store.metadata[compared["dataset"]["id"]]
        self.assertEqual(info.parent_ids, (self.parent.id, cohort_id))
        self.assertEqual(info.snapshot, self.parent.snapshot)
        self.assertEqual(compared["comparison_result"]["data_sha256"], dataset_digest(result))
        with duckdb.connect() as connection:
            connection.register("baseline_data", self.frame)
            connection.register("cohort_data", cohort)
            replayed = connection.execute(info.query).fetchdf()
        pd.testing.assert_frame_equal(replayed, result)

    def test_unrelated_snapshot_and_non_numeric_measure_fail_closed(self):
        unrelated = self.store.register(
            self.frame.copy(), source=self.parent.source,
            coverage="complete", predicate_known=True, snapshot="fixture:v1")
        with self.assertRaises(ValueError):
            self.tools["compare_group_aggregates"](
                self.parent.id, unrelated.id, "mean", "segment_code",
                value_column="metric_value")
        child = self.store.register(
            self.frame.iloc[:5].copy(), source=self.parent.source,
            coverage="complete", predicate_known=False, snapshot="fixture:v2",
            parent_id=self.parent.id)
        with self.assertRaises(ValueError):
            self.tools["compare_group_aggregates"](
                self.parent.id, child.id, "mean", "segment_code",
                value_column="metric_value")
        valid_child = self.store.register(
            self.frame.iloc[:5].copy(), source=self.parent.source,
            coverage="complete", predicate_known=False, snapshot=self.parent.snapshot,
            parent_id=self.parent.id)
        with self.assertRaises(ValueError):
            self.tools["compare_group_aggregates"](
                self.parent.id, valid_child.id, "mean", "segment_code",
                value_column="segment_code")

    def test_zero_baseline_and_removed_group_have_explicit_nan_changes(self):
        frame = pd.DataFrame({"group": ["zero", "zero", "removed"], "value": [-1.0, 1.0, 5.0]})
        parent = self.store.register(
            frame, source="arbitrary.zero_case", coverage="complete",
            predicate_known=True, snapshot="fixture:zero")
        cohort = self.store.register(
            frame.iloc[[1]].copy(), source=parent.source, coverage="complete",
            predicate_known=False, snapshot=parent.snapshot, parent_id=parent.id)
        result = self.tools["compare_group_aggregates"](
            parent.id, cohort.id, "mean", "group", value_column="value")
        actual = self.store.frames[result["dataset"]["id"]].set_index("group")
        self.assertTrue(np.isnan(actual.loc["zero", "percent_change"]))
        self.assertTrue(np.isnan(actual.loc["removed", "cohort_mean_value"]))
        self.assertTrue(np.isnan(actual.loc["removed", "difference"]))

    def test_agent_compares_parent_and_inlier_cohort_without_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "cohort-compare", ForbiddenModel())
            parent = runtime.datasets.register(
                self.frame.copy(), source="arbitrary.runtime_measurements",
                coverage="complete", predicate_known=True, snapshot="fixture:v1")
            runtime.context.reference_context[:] = [fixture_reference_context(
                "arbitrary.runtime_measurements", self.frame)]
            try:
                outcome = runtime.submit(
                    "anomaly_value IQR 상한 이상치를 제외한 일반 행의 segment_code별 "
                    "metric_value 평균을 계산하고 이상치 포함 전후 평균을 비교해줘")
                self.assertEqual(outcome["status"], "answered", outcome)
                recovery = runtime.inspect()["recovery"]
                self.assertEqual(recovery["model_calls"], 0)
                self.assertEqual(recovery["outlier_aggregate_mode"], "grouped_comparison")
                cohort_id = recovery["outlier_dataset"]
                evidence = recovery["outlier_aggregate_evidence"]["comparison"]
                result_id = evidence["dataset"]["id"]
                self.assertEqual(runtime.datasets.metadata[cohort_id].parent_id, parent.id)
                self.assertEqual(
                    runtime.datasets.metadata[result_id].parent_ids, (parent.id, cohort_id))
                actual = runtime.datasets.frames[result_id]
                self.assertTrue(np.isfinite(actual["difference"]).all())
                self.assertIn("baseline_mean_metric_value", outcome["text"])
                self.assertIn("cohort_mean_metric_value", outcome["text"])
            finally:
                runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "cohort-compare", ForbiddenModel())
            try:
                evidence = reopened.inspect()["recovery"]["outlier_aggregate_evidence"]["comparison"]
                result_info = reopened.datasets.metadata[evidence["dataset"]["id"]]
                self.assertEqual(len(result_info.parent_ids), 2)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
