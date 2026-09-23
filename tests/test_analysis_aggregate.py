"""Bounded aggregate tool and outlier-cohort recovery contracts."""
import tempfile
import unittest

import duckdb
import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_agent import fixture_reference_context
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_aggregate import dataset_digest
from utils.analysis_datasets import DatasetStore


class AnalysisAggregateTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "anomaly_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 110, 120],
            "segment_code": ["north", "south", "east", "north", "south", "east",
                             "north", "south", "east", "north", "north", "south"],
            "score_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
        })
        self.store = DatasetStore()
        self.info = self.store.register(
            self.frame.copy(), source="arbitrary.runtime_observations",
            coverage="complete", predicate_known=True, snapshot="fixture:v1")
        context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(context)}

    def test_grouped_top_n_and_scalar_aggregate_preserve_lineage(self):
        grouped = self.tools["aggregate_dataset"](
            self.info.id, "count", group_column="segment_code",
            sort="descending", top_n=2)
        child = grouped["dataset"]
        result = self.store.frames[child["id"]]
        self.assertEqual(grouped["status"], "ready")
        self.assertEqual(result.to_dict(orient="records"), [
            {"segment_code": "north", "count": 5},
            {"segment_code": "south", "count": 4},
        ])
        self.assertEqual(child["parent_id"], self.info.id)
        self.assertEqual(child["source"], self.info.source)
        self.assertEqual(child["snapshot"], self.info.snapshot)
        self.assertEqual(grouped["aggregation_result"]["data_sha256"], dataset_digest(result))

        scalar = self.tools["aggregate_dataset"](
            self.info.id, "mean", value_column="score_value")
        scalar_frame = self.store.frames[scalar["dataset"]["id"]]
        self.assertAlmostEqual(
            float(scalar_frame.iloc[0]["mean_score_value"]), self.frame["score_value"].mean())

    def test_invalid_grain_dtype_and_group_cardinality_fail_closed(self):
        aggregate = self.store.register(
            pd.DataFrame({"count": [12]}), source=self.info.source,
            coverage="complete", predicate_known=True, grain="aggregate",
            aggregation="count(*)")
        structured = {tool.name: tool for tool in local_tools(
            AnalysisToolContext(self.store, {}, [], lambda **_: None))}
        rejected = structured["aggregate_dataset"].invoke({
            "dataset_id": aggregate.id, "aggregation": "count"})
        self.assertEqual(rejected["status"], "error")
        self.assertEqual(rejected["error_code"], "invalid_tool_input")
        with self.assertRaises(ValueError):
            self.tools["aggregate_dataset"](
                self.info.id, "mean", value_column="segment_code")
        with self.assertRaises(ValueError):
            self.tools["aggregate_dataset"](
                self.info.id, "count", group_column="segment_code", max_groups=2)

    def test_runtime_column_names_with_quotes_are_escaped_in_replay_query(self):
        frame = pd.DataFrame({
            'segment"code': ["a", "a", "b"],
            'score"value': [1.0, 3.0, 8.0],
        })
        info = self.store.register(
            frame, source="arbitrary.quoted_columns", coverage="complete",
            predicate_known=True, snapshot="fixture:quoted")
        result = self.tools["aggregate_dataset"](
            info.id, "mean", value_column='score"value', group_column='segment"code')
        derived = self.store.frames[result["dataset"]["id"]]
        query = self.store.metadata[result["dataset"]["id"]].query
        with duckdb.connect() as connection:
            connection.register("data", frame)
            replayed = connection.execute(query).fetchdf()
        pd.testing.assert_frame_equal(replayed, derived)

    def test_outlier_top_group_and_overall_mean_complete_without_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "cohort-top", ForbiddenModel())
            parent = runtime.datasets.register(
                self.frame.copy(), source="arbitrary.runtime_observations",
                coverage="complete", predicate_known=True, snapshot="fixture:v1")
            runtime.context.reference_context[:] = [fixture_reference_context(
                "arbitrary.runtime_observations", self.frame)]
            try:
                outcome = runtime.submit(
                    "arbitrary.runtime_observations의 anomaly_value IQR 상한 1.5*IQR 초과 "
                    "이상치들의 score_value 평균과 주요 segment_code TOP 2를 알려줘")
                self.assertEqual(outcome["status"], "answered", outcome)
                recovery = runtime.inspect()["recovery"]
                self.assertEqual(recovery["model_calls"], 0)
                self.assertEqual(recovery["outlier_aggregate_mode"], "top_frequency")
                cohort_id = recovery["outlier_dataset"]
                self.assertEqual(runtime.datasets.metadata[cohort_id].parent_id, parent.id)
                evidence = recovery["outlier_aggregate_evidence"]
                self.assertEqual(set(evidence), {"overall", "grouped"})
                overall = runtime.datasets.frames[evidence["overall"]["dataset"]["id"]]
                grouped = runtime.datasets.frames[evidence["grouped"]["dataset"]["id"]]
                self.assertEqual(float(overall.iloc[0]["mean_score_value"]), 20.0)
                self.assertEqual(grouped.to_dict(orient="records"), [
                    {"segment_code": "north", "count": 2},
                    {"segment_code": "south", "count": 1},
                ])
                self.assertIn("mean_score_value", outcome["text"])
                self.assertIn("segment_code,count", outcome["text"])
            finally:
                runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "cohort-top", ForbiddenModel())
            try:
                recovery = reopened.inspect()["recovery"]
                for evidence in recovery["outlier_aggregate_evidence"].values():
                    aggregate_id = evidence["dataset"]["id"]
                    self.assertIn(aggregate_id, reopened.datasets.metadata)
                    self.assertEqual(
                        reopened.datasets.metadata[aggregate_id].parent_id,
                        recovery["outlier_dataset"])
            finally:
                reopened.close()

    def test_inlier_grouped_mean_uses_detector_column_without_guessing(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "cohort-group", ForbiddenModel())
            runtime.datasets.register(
                self.frame.copy(), source="arbitrary.runtime_observations",
                coverage="complete", predicate_known=True, snapshot="fixture:v1")
            runtime.context.reference_context[:] = [fixture_reference_context(
                "arbitrary.runtime_observations", self.frame)]
            try:
                outcome = runtime.submit(
                    "anomaly_value IQR 상한 이상치를 제외한 일반 행의 "
                    "segment_code별 anomaly_value 평균을 보여줘")
                self.assertEqual(outcome["status"], "answered", outcome)
                recovery = runtime.inspect()["recovery"]
                self.assertEqual(recovery["model_calls"], 0)
                self.assertEqual(recovery["outlier_selection"], "inliers")
                self.assertEqual(recovery["outlier_aggregate_mode"], "grouped_metric")
                evidence = recovery["outlier_aggregate_evidence"]["grouped"]
                actual = runtime.datasets.frames[evidence["dataset"]["id"]]
                clean = self.frame[self.frame["anomaly_value"] <= 73.75]
                expected = (clean.groupby("segment_code", sort=False)["anomaly_value"].mean()
                            .rename("mean_anomaly_value").reset_index()
                            .sort_values("mean_anomaly_value", ascending=False, kind="mergesort")
                            .reset_index(drop=True))
                pd.testing.assert_frame_equal(actual, expected)
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
