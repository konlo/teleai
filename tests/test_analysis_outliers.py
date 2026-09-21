"""Bounded outlier tool and production recovery contracts."""
import tempfile
import unittest

import numpy as np
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import DatasetStore


class OutlierOnlyModel(BaseChatModel):
    dataset_id: str
    calls: int = 0

    @property
    def _llm_type(self):
        return "compound-outlier-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.calls == 0:
            message = AIMessage(content="", tool_calls=[{
                "name": "detect_outliers",
                "args": {"dataset_id": self.dataset_id, "column": "Fare",
                         "method": "iqr", "tail": "upper", "threshold": 1.5},
                "id": "compound-outlier-call",
            }])
        else:
            message = AIMessage(content="이상치와 생존율 분석을 완료했습니다.")
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


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

    def test_compound_outlier_cohort_metric_is_not_falsely_completed(self):
        with tempfile.TemporaryDirectory() as root:
            model = OutlierOnlyModel(dataset_id="pending")
            runtime = GraphAnalysisRuntime(root, "owner", "compound-outlier", model)
            info = runtime.datasets.register(
                pd.DataFrame({"Fare": [1, 2, 3, 100], "Survived": [0, 1, 0, 1]}),
                source="titanic", coverage="complete", predicate_known=True, snapshot="fixture:v1")
            model.dataset_id = info.id
            try:
                outcome = runtime.submit(
                    "타이타닉 요금(Fare)에서 IQR 상한 이상치 승객 수와 이들의 생존율을 구해줘")
                self.assertNotEqual(outcome["status"], "answered", outcome)
                recovery = runtime.inspect()["recovery"]
                self.assertIsNone(recovery.get("outlier_spec"))
                self.assertNotIn("outlier_evidence", recovery)
            finally:
                runtime.close()


if __name__ == "__main__":
    unittest.main()
