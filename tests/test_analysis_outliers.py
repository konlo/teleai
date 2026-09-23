"""Bounded outlier tool and production recovery contracts."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_agent import fixture_reference_context
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import DatasetStore


class AnalysisOutlierTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.frame = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 100, np.nan]})
        self.info = self.store.register(
            self.frame, source="fixture.outliers", coverage="complete",
            predicate_known=True, snapshot="fixture:v1")
        context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(context)}

    def detect(self, **arguments):
        return self.tools["detect_outliers"](self.info.id, "value", **arguments)

    def test_iqr_returns_threshold_counts_scope_and_no_rows(self):
        result = self.detect(method="iqr", tail="upper", threshold=1.5)
        evidence = result["outlier_result"]
        clean = self.frame["value"].dropna()
        q1, q3 = clean.quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        self.assertEqual(result["status"], "ready")
        self.assertAlmostEqual(evidence["thresholds"]["upper"], upper)
        self.assertEqual(evidence["counts"]["selected"], int((clean > upper).sum()))
        self.assertEqual(evidence["sample"], {"input_rows": 8, "valid_rows": 7, "missing_rows": 1})
        self.assertNotIn("rows", evidence)
        self.assertIn("coverage=complete", result["scope"])

    def test_zscore_mad_and_quantile_are_bounded_and_reproducible(self):
        clean = self.frame["value"].dropna()
        zscore = self.detect(method="zscore", tail="both", threshold=2)["outlier_result"]
        self.assertAlmostEqual(zscore["parameters"]["sample_std"], clean.std(ddof=1))
        self.assertEqual(zscore["parameters"]["ddof"], 1)
        mad = self.detect(method="mad", tail="upper", threshold=3.5)["outlier_result"]
        self.assertEqual(mad["parameters"]["median"], clean.median())
        quantile = self.tools["detect_outliers"](
            self.info.id, "value", "quantile", "upper", 1.5, 0.01, 0.9)["outlier_result"]
        self.assertAlmostEqual(quantile["thresholds"]["upper"], clean.quantile(0.9))

    def test_outlier_selection_materializes_lineage_without_exposing_rows(self):
        result = self.tools["select_outlier_rows"](
            self.info.id, "value", "iqr", "outliers", "upper", 1.5)
        child = result["dataset"]
        summary = result["selection_summary"]
        self.assertEqual(result["status"], "ready")
        self.assertEqual(child["parent_id"], self.info.id)
        self.assertEqual(child["source"], self.info.source)
        self.assertEqual(child["snapshot"], self.info.snapshot)
        self.assertFalse(child["predicate_known"])
        self.assertIn("isfinite", child["query"])
        self.assertEqual(child["rows"], 1)
        self.assertEqual(summary["selected_rows"], 1)
        self.assertEqual(len(summary["data_sha256"]), 64)
        self.assertEqual(self.store.frames[child["id"]]["value"].tolist(), [100.0])
        self.assertNotIn("rows", result)
        self.assertNotIn("preview", result)

    def test_invalid_grain_dtype_constant_and_parameters_fail_closed(self):
        aggregated = self.store.register(
            pd.DataFrame({"value": [1, 2, 3, 4]}), source="fixture.outliers",
            coverage="complete", predicate_known=True, grain="aggregate",
            aggregation="SELECT AVG(value) FROM data")
        structured = {tool.name: tool for tool in local_tools(
            AnalysisToolContext(self.store, {}, [], lambda **_: None))}
        invalid = structured["detect_outliers"].invoke({
            "dataset_id": aggregated.id, "column": "value", "method": "iqr"})
        self.assertEqual(invalid["status"], "error")
        self.assertEqual(invalid["error_code"], "invalid_tool_input")
        with self.assertRaises(ValueError):
            self.detect(method="zscore", threshold=20)
        constant = self.store.register(
            pd.DataFrame({"value": [1, 1, 1, 1]}), source="constant",
            coverage="complete", predicate_known=True)
        with self.assertRaises(ValueError):
            self.tools["detect_outliers"](constant.id, "value", "iqr")

    def test_winsorization_compares_exact_means_without_mutating_source(self):
        before = self.frame.copy(deep=True)
        result = self.tools["winsorize_numeric"](
            self.info.id, "value", lower_quantile=0.1, upper_quantile=0.9)
        evidence = result["winsorization_result"]
        clean = self.frame["value"].dropna().astype(float)
        lower, upper = clean.quantile([0.1, 0.9])
        expected = clean.clip(lower=lower, upper=upper)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(evidence["kind"], "winsorization_comparison")
        self.assertAlmostEqual(evidence["thresholds"]["lower"], lower)
        self.assertAlmostEqual(evidence["thresholds"]["upper"], upper)
        self.assertAlmostEqual(evidence["original"]["mean"], clean.mean())
        self.assertAlmostEqual(evidence["winsorized"]["mean"], expected.mean())
        self.assertEqual(evidence["clipped_counts"]["lower"], int(clean.lt(lower).sum()))
        self.assertEqual(evidence["clipped_counts"]["upper"], int(clean.gt(upper).sum()))
        pd.testing.assert_frame_equal(self.frame, before)
        self.assertNotIn("rows", result)
        self.assertNotIn("preview", result)

    def test_winsorization_rejects_aggregate_constant_and_unsafe_bounds(self):
        with self.assertRaises(ValueError):
            self.tools["winsorize_numeric"](self.info.id, "value", 0, 0.99)
        with self.assertRaises(ValueError):
            self.tools["winsorize_numeric"](self.info.id, "value", 0.3, 0.7)
        aggregated = self.store.register(
            pd.DataFrame({"value": [1, 2, 3, 4]}), source="fixture.outliers",
            coverage="complete", predicate_known=True, grain="aggregate",
            aggregation="SELECT AVG(value) FROM data")
        with self.assertRaises(ValueError):
            self.tools["winsorize_numeric"](aggregated.id, "value", 0.01, 0.99)

    def test_unambiguous_iqr_and_sigma_prompts_route_without_model(self):
        prompts = {
            "iqr": "fixture.outliers의 value IQR과 상한선 1.5*IQR 초과 이상치 수를 알려줘",
            "zscore": "fixture.outliers의 value 최솟값, 최댓값 및 3-시그마 범위를 점검해줘",
            "quantile": "fixture.outliers의 value 상위 10% 이상치 수를 알려줘",
        }
        for method, prompt in prompts.items():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                runtime = GraphAnalysisRuntime(root, "owner", method, ForbiddenModel())
                runtime.datasets.register(
                    self.frame.copy(), source="fixture.outliers", coverage="complete",
                    predicate_known=True, snapshot="fixture:v1")
                try:
                    outcome = runtime.submit(prompt)
                    self.assertEqual(outcome["status"], "answered", outcome)
                    recovery = runtime.inspect()["recovery"]
                    self.assertEqual(recovery["model_calls"], 0)
                    self.assertEqual(
                        recovery["outlier_evidence"]["outlier_result"]["method"], method)
                    if method == "quantile":
                        self.assertEqual(
                            recovery["outlier_evidence"]["outlier_result"]["thresholds"]["upper"],
                            self.frame["value"].dropna().quantile(0.9))
                finally:
                    runtime.close()

    def test_structured_evidence_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "restart-outlier", ForbiddenModel())
            runtime.datasets.register(
                self.frame.copy(), source="fixture.outliers", coverage="complete",
                predicate_known=True, snapshot="fixture:v1")
            outcome = runtime.submit(
                "fixture.outliers의 value IQR 상한선 1.5*IQR 초과 이상치 수를 알려줘")
            self.assertEqual(outcome["status"], "answered", outcome)
            before = runtime.inspect()["recovery"]["outlier_evidence"]
            runtime.close()
            reopened = GraphAnalysisRuntime(root, "owner", "restart-outlier", ForbiddenModel())
            try:
                self.assertEqual(reopened.inspect()["recovery"]["outlier_evidence"], before)
            finally:
                reopened.close()

    def test_unambiguous_winsorization_routes_without_model_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "winsorization", ForbiddenModel())
            runtime.datasets.register(
                self.frame.copy(), source="fixture.outliers", coverage="complete",
                predicate_known=True, snapshot="fixture:v1")
            try:
                outcome = runtime.submit(
                    "fixture.outliers의 value 극단치 왜곡을 줄이도록 상하위 10% 윈저화(Clipping)를 적용하고 원본 평균과 보정 평균을 비교해줘")
                self.assertEqual(outcome["status"], "answered", outcome)
                recovery = runtime.inspect()["recovery"]
                self.assertEqual(recovery["model_calls"], 0)
                evidence = recovery["winsor_evidence"]["winsorization_result"]
                self.assertEqual(evidence["parameters"], {
                    "lower_quantile": 0.1, "upper_quantile": 0.9})
                self.assertIsNone(recovery["outlier_spec"])
                before = recovery["winsor_evidence"]
            finally:
                runtime.close()
            reopened = GraphAnalysisRuntime(root, "owner", "winsorization", ForbiddenModel())
            try:
                self.assertEqual(reopened.inspect()["recovery"]["winsor_evidence"], before)
            finally:
                reopened.close()

    def test_compound_outlier_cohort_metric_completes_without_model_or_remote(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "compound-outlier", ForbiddenModel())
            parent = runtime.datasets.register(
                pd.DataFrame({"Fare": [1, 2, 3, 100], "Survived": [0, 1, 0, 1]}),
                source="titanic", coverage="complete", predicate_known=True, snapshot="fixture:v1")
            runtime.context.reference_context[:] = [fixture_reference_context(
                "titanic", runtime.datasets.frames[parent.id])]
            try:
                outcome = runtime.submit(
                    "타이타닉 요금(Fare)에서 IQR 상한 이상치 승객 수와 이들의 생존율을 구해줘")
                self.assertEqual(outcome["status"], "answered", outcome)
                recovery = runtime.inspect()["recovery"]
                self.assertEqual(recovery["model_calls"], 0)
                child_id = recovery["outlier_dataset"]
                result_id = recovery["evidence_ids"][-1]
                self.assertEqual(runtime.datasets.metadata[child_id].parent_id, parent.id)
                self.assertEqual(runtime.datasets.metadata[result_id].parent_id, child_id)
                result = runtime.datasets.frames[result_id]
                self.assertEqual(int(result.iloc[0]["count"]), 1)
                self.assertAlmostEqual(float(result.iloc[0]["percent"]), 100.0)
            finally:
                runtime.close()
            reopened = GraphAnalysisRuntime(root, "owner", "compound-outlier", ForbiddenModel())
            try:
                recovery = reopened.inspect()["recovery"]
                self.assertIn(recovery["outlier_dataset"], reopened.datasets.metadata)
                self.assertIn(recovery["evidence_ids"][-1], reopened.datasets.metadata)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
