"""Bounded explicit chart rendering and production-agent completion contracts."""
import tempfile
import unittest
from uuid import uuid4

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore


SOURCE = "catalog.analytics.events"


def frame():
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=12, freq="D"),
        "segment": ["A", "B", "A", "C"] * 3,
        "value": [1, 4, 2, 8, 3, 7, 5, 9, 6, 10, 11, 12],
        "score": [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11],
    })


class NoModelCall(BaseChatModel):
    @property
    def _llm_type(self):
        return "chart-recovery-must-not-run"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("deterministic chart recovery should not call the model")


class ChartSpecModel(BaseChatModel):
    dataset_id: str = ""
    position: int = 0

    @property
    def _llm_type(self):
        return "explicit-chart-spec-contract"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.position == 0:
            message = AIMessage(content="", tool_calls=[{
                "name": "render_chart_spec",
                "args": {"dataset_id": self.dataset_id, "kind": "histogram", "x": "value",
                         "bins": 12, "title": "Value distribution", "x_label": "Value"},
                "id": str(uuid4()),
            }])
        else:
            message = AIMessage(content="차트를 생성했습니다.")
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class ChartSpecTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(self.context)}
        self.info = self.store.register(frame(), source=SOURCE, coverage="complete", predicate_known=True)

    def assert_png(self, result, kind):
        self.assertEqual(result["status"], "ready")
        card = self.context.artifacts[result["cards"][0]["id"]]
        self.assertEqual(card.kind, kind)
        self.assertTrue(card.image.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(len(result["render_summary"]["data_sha256"]), 64)
        return card

    def test_supported_chart_kinds_render_real_pngs(self):
        cases = [
            ("histogram", {"x":"value", "bins":7, "title":"Distribution"}),
            ("bar", {"x":"segment", "aggregation":"count", "sort":"descending",
                     "top_n":2, "orientation":"horizontal"}),
            ("line", {"x":"time", "y":"value", "sort":"ascending"}),
            ("scatter", {"x":"value", "y":"score", "category":"segment"}),
            ("boxplot", {"x":"value"}),
            ("boxplot", {"x":"value", "category":"segment"}),
        ]
        for kind, arguments in cases:
            with self.subTest(kind=kind):
                result = self.tools["render_chart_spec"](self.info.id, kind, **arguments)
                self.assert_png(result, kind)

    def test_bar_aggregation_top_n_labels_and_digest_are_deterministic(self):
        arguments = dict(kind="bar", x="segment", y="value", aggregation="mean",
                         sort="descending", top_n=2, title="Segment mean",
                         x_label="Segment", y_label="Mean value")
        first = self.tools["render_chart_spec"](self.info.id, **arguments)
        second = self.tools["render_chart_spec"](self.info.id, **arguments)
        card = self.assert_png(first, "bar")
        self.assertEqual(card.title, "Segment mean")
        self.assertEqual(first["chart_spec"]["aggregation"], "mean")
        self.assertEqual(first["render_summary"]["rendered_rows"], 2)
        self.assertEqual(first["render_summary"]["data_sha256"],
                         second["render_summary"]["data_sha256"])

    def test_invalid_spec_and_reaggregation_return_structured_errors(self):
        structured = {tool.name: tool for tool in local_tools(self.context)}
        bad_column = structured["render_chart_spec"].invoke({
            "dataset_id": self.info.id, "kind": "scatter", "x": "missing", "y": "score"})
        self.assertEqual(bad_column["status"], "error")
        self.assertEqual(bad_column["error_code"], "invalid_tool_input")
        self.assertFalse(bad_column["retryable"])

        aggregate = self.store.register(
            pd.DataFrame({"segment":["A", "B"], "value":[1.0, 2.0]}),
            source=SOURCE, coverage="complete", predicate_known=True,
            grain="aggregate", aggregation="AVG(value)")
        repeated = structured["render_chart_spec"].invoke({
            "dataset_id": aggregate.id, "kind": "bar", "x": "segment", "y": "value",
            "aggregation": "mean"})
        self.assertEqual(repeated["status"], "error")
        self.assertEqual(repeated["error_code"], "invalid_tool_input")

    def test_scatter_request_is_completed_deterministically_without_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "scatter", NoModelCall())
            runtime.datasets.register(frame(), source=SOURCE, coverage="complete", predicate_known=True)
            result = runtime.submit("value와 score 산점도를 보여줘")
            self.assertEqual(result["status"], "answered", result)
            chart_ids = runtime.inspect()["chart_ids"]
            self.assertEqual(len(chart_ids), 1)
            card = runtime.artifacts[chart_ids[0]]
            self.assertEqual(card.kind, "scatter")
            self.assertTrue(card.image.startswith(b"\x89PNG\r\n\x1a\n"))
            runtime.close()

    def test_grouped_boxplot_is_completed_deterministically_without_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "grouped-boxplot", NoModelCall())
            runtime.datasets.register(frame(), source=SOURCE, coverage="complete", predicate_known=True)
            result = runtime.submit("segment별 value 분포를 박스플롯으로 보여줘")
            self.assertEqual(result["status"], "answered", result)
            card = runtime.artifacts[runtime.inspect()["chart_ids"][0]]
            self.assertEqual(card.kind, "boxplot")
            self.assertEqual(card.columns, ("value", "segment"))
            runtime.close()

    def test_group_labels_must_cover_every_observed_level_to_avoid_filtering(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "group-label-scope", NoModelCall())
            info = runtime.datasets.register(
                frame(), source=SOURCE, coverage="complete", predicate_known=True)
            current = {"scope": {"conditions": [{
                "column": "segment", "op": "in", "value": ["A", "B", "C"]}],
                "any_conditions": [], "measure_conditions": [], "unresolved": [],
                "ratio": None}}
            self.assertTrue(runtime.recovery._all_group_levels_scope_valid(
                info, "segment", current))
            current["scope"]["conditions"][0]["value"] = ["A", "B"]
            self.assertFalse(runtime.recovery._all_group_levels_scope_valid(
                info, "segment", current))
            runtime.close()

    def test_custom_histogram_uses_explicit_spec_instead_of_cached_default(self):
        with tempfile.TemporaryDirectory() as root:
            model = ChartSpecModel()
            runtime = GraphAnalysisRuntime(root, "owner", "custom-histogram", model)
            info = runtime.datasets.register(frame(), source=SOURCE,
                                             coverage="complete", predicate_known=True)
            model.dataset_id = info.id
            result = runtime.submit("value 히스토그램을 12개 구간으로 그리고 제목도 바꿔줘")
            self.assertEqual(result["status"], "answered", result)
            self.assertEqual(model.position, 1)
            chart_ids = runtime.inspect()["chart_ids"]
            self.assertEqual(len(chart_ids), 1)
            card = runtime.artifacts[chart_ids[0]]
            self.assertEqual(card.title, "Value distribution")
            self.assertEqual(card.kind, "histogram")
            runtime.close()

    def test_explicit_chart_png_and_lineage_survive_runtime_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "restart-chart", NoModelCall())
            runtime.datasets.register(frame(), source=SOURCE, coverage="complete", predicate_known=True)
            result = runtime.submit("value와 score 산점도를 보여줘")
            self.assertEqual(result["status"], "answered", result)
            chart_id = runtime.inspect()["chart_ids"][0]
            dataset_id = runtime.artifacts[chart_id].dataset_id
            runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "restart-chart", NoModelCall())
            self.assertIn(chart_id, reopened.inspect()["chart_ids"])
            card = reopened.artifacts[chart_id]
            self.assertEqual(card.dataset_id, dataset_id)
            self.assertTrue(card.image.startswith(b"\x89PNG\r\n\x1a\n"))
            reopened.close()


if __name__ == "__main__":
    unittest.main()
