"""Bounded count/rate composite chart contracts."""
import tempfile
import unittest

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore


SOURCE = "arbitrary.metrics.runtime_events"


def frame():
    return pd.DataFrame({
        "bucket_code": ["mar", "jan", "feb", "feb", "jan", "feb"],
        "outcome_flag": ["yes", "yes", "no", "yes", "no", None],
        "unrelated_value": [7, 2, 8, 3, 1, 9],
    })


class NoModelCall(BaseChatModel):
    @property
    def _llm_type(self):
        return "count-rate-recovery-must-not-run-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("deterministic count/rate recovery should not call the model")


class CountRateChartTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(self.context)}
        self.info = self.store.register(
            frame(), source=SOURCE, coverage="complete", predicate_known=True)

    def test_dual_axis_uses_explicit_numerator_denominator_and_calendar_order(self):
        first = self.tools["render_count_rate_chart"](
            self.info.id, "bucket_code", "outcome_flag", "yes",
            layout="dual_axis", sort="calendar_month")
        second = self.tools["render_count_rate_chart"](
            self.info.id, "bucket_code", "outcome_flag", "yes",
            layout="dual_axis", sort="calendar_month")
        self.assertEqual(first["status"], "ready")
        card = self.context.artifacts[first["cards"][0]["id"]]
        self.assertEqual(card.kind, "dual_axis")
        self.assertTrue(card.image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(first["render_summary"]["missing_outcome_rows"], 1)
        self.assertEqual(first["render_summary"]["points"], [
            {"group": "jan", "row_count": 2, "denominator_count": 2,
             "success_count": 1, "rate_percent": 50.0},
            {"group": "feb", "row_count": 3, "denominator_count": 2,
             "success_count": 1, "rate_percent": 50.0},
            {"group": "mar", "row_count": 1, "denominator_count": 1,
             "success_count": 1, "rate_percent": 100.0},
        ])
        self.assertEqual(first["render_summary"]["data_sha256"],
                         second["render_summary"]["data_sha256"])

    def test_split_panel_and_fail_closed_boundaries(self):
        result = self.tools["render_count_rate_chart"](
            self.info.id, "bucket_code", "outcome_flag", "yes",
            layout="split_panel", sort="count_descending", top_n=2)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["chart_spec"]["layout"], "split_panel")
        self.assertEqual(result["render_summary"]["rendered_groups"], 2)
        with self.assertRaises(ValueError):
            self.tools["render_count_rate_chart"](
                self.info.id, "bucket_code", "outcome_flag", "missing")
        aggregate = self.store.register(
            pd.DataFrame({"bucket_code": ["jan"], "outcome_flag": ["yes"]}),
            source=SOURCE, coverage="complete", predicate_known=True,
            grain="aggregate", aggregation="COUNT(*)")
        with self.assertRaises(ValueError):
            self.tools["render_count_rate_chart"](
                aggregate.id, "bucket_code", "outcome_flag", "yes")

    def test_production_graph_runs_without_model_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "count-rate", NoModelCall())
            runtime.datasets.register(
                frame(), source=SOURCE, coverage="complete", predicate_known=True)
            result = runtime.submit(
                "bucket_code별 전체 건수(막대)와 outcome_flag='yes' 성공률(선)을 "
                "이중 축(Dual Y-axis)으로 시각화해줘")
            self.assertEqual(result["status"], "answered", result)
            state = runtime.inspect()
            self.assertEqual(len(state["chart_ids"]), 1)
            card = runtime.artifacts[state["chart_ids"][0]]
            self.assertEqual(card.kind, "dual_axis")
            self.assertEqual(card.columns, ("bucket_code", "outcome_flag"))
            chart_id = card.id
            runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "count-rate", NoModelCall())
            self.assertIn(chart_id, reopened.inspect()["chart_ids"])
            self.assertTrue(reopened.artifacts[chart_id].image.startswith(b"\x89PNG"))
            reopened.close()

    def test_split_panel_request_runs_without_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "count-rate-panel", NoModelCall())
            runtime.datasets.register(
                frame(), source=SOURCE, coverage="complete", predicate_known=True)
            result = runtime.submit(
                "bucket_code별 전체 건수와 outcome_flag='yes' 성공률을 "
                "2열 서브플롯 패널로 시각화해줘")
            self.assertEqual(result["status"], "answered", result)
            card = runtime.artifacts[runtime.inspect()["chart_ids"][0]]
            self.assertEqual(card.kind, "split_panel")
            runtime.close()


if __name__ == "__main__":
    unittest.main()
