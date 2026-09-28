"""Structured statistical-test tool and production recovery contracts."""
import tempfile
import unittest

import pandas as pd
from scipy import stats

from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_statistics import ForbiddenModel, evaluate
from utils.analysis_datasets import DatasetStore


class AnalysisStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.info = self.store.register(
            pd.DataFrame({
                "group": ["A"] * 5 + ["B"] * 5,
                "value": [1, 2, 3, 4, 5, 4, 5, 6, 7, 8],
                "before": [10, 12, 9, 11, 8, 7, 9, 8, 10, 6],
                "after": [9, 10, 8, 9, 7, 7, 8, 7, 8, 5],
                "category_a": ["x", "x", "y", "y", "x", "x", "y", "y", "y", "x"],
                "category_b": ["yes", "no", "yes", "no", "yes", "yes", "yes", "no", "no", "no"],
                "three_groups": ["G1", "G1", "G1", "G2", "G2", "G2", "G3", "G3", "G3", "G3"],
            }),
            source="fixture.statistics",
            coverage="complete",
            predicate_known=True,
            snapshot="fixture:v1",
        )
        context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(context)}

    def run_test(self, **arguments):
        return self.tools["statistical_test"](self.info.id, **arguments)

    def test_independent_t_matches_scipy_and_returns_effect_ci_assumptions(self):
        result = self.run_test(test="independent_t", value_column="value", group_column="group")
        observed = result["test_result"]
        a = self.store.frames[self.info.id].query("group == 'A'")["value"]
        b = self.store.frames[self.info.id].query("group == 'B'")["value"]
        reference = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(observed["statistic"], reference.statistic)
        self.assertAlmostEqual(observed["p_value"], reference.pvalue)
        self.assertEqual(observed["effect_size"]["name"], "hedges_g")
        self.assertEqual(len(observed["confidence_intervals"]), 1)
        self.assertIn("normality", observed["assumptions"])

    def test_paired_t_and_mean_ci_return_grounded_intervals(self):
        paired = self.run_test(test="paired_t", value_column="before", paired_column="after")["test_result"]
        reference = stats.ttest_rel(
            self.store.frames[self.info.id]["before"], self.store.frames[self.info.id]["after"])
        self.assertAlmostEqual(paired["statistic"], reference.statistic)
        self.assertAlmostEqual(paired["p_value"], reference.pvalue)
        self.assertEqual(paired["effect_size"]["name"], "cohen_dz")
        interval = self.run_test(test="mean_ci", value_column="value")["test_result"]
        self.assertIsNone(interval["p_value"])
        self.assertEqual(interval["estimate"]["value"], 4.5)
        self.assertLess(interval["confidence_intervals"][0]["lower"], 4.5)
        self.assertGreater(interval["confidence_intervals"][0]["upper"], 4.5)

    def test_chi_square_matches_scipy_and_bounds_contingency(self):
        result = self.run_test(
            test="chi_square", value_column="category_a", group_column="category_b")["test_result"]
        table = pd.crosstab(
            self.store.frames[self.info.id]["category_a"],
            self.store.frames[self.info.id]["category_b"],
        )
        reference = stats.chi2_contingency(table)
        self.assertAlmostEqual(result["statistic"], reference.statistic)
        self.assertAlmostEqual(result["p_value"], reference.pvalue)
        self.assertEqual(result["contingency"]["observed"], table.to_numpy().tolist())
        self.assertEqual(result["effect_size"]["name"], "cramers_v")
        self.assertEqual(result["confidence_intervals"][0]["parameter"], "odds_ratio")

    def test_anova_and_mann_whitney_match_scipy(self):
        frame = self.store.frames[self.info.id]
        anova = self.run_test(
            test="one_way_anova", value_column="value", group_column="three_groups")["test_result"]
        arrays = [group["value"].to_numpy() for _, group in frame.groupby("three_groups", sort=False)]
        reference_anova = stats.f_oneway(*arrays)
        self.assertAlmostEqual(anova["statistic"], reference_anova.statistic)
        self.assertAlmostEqual(anova["p_value"], reference_anova.pvalue)
        self.assertEqual(anova["effect_size"]["name"], "eta_squared")
        mann = self.run_test(
            test="mann_whitney", value_column="value", group_column="group")["test_result"]
        a = frame.query("group == 'A'")["value"]
        b = frame.query("group == 'B'")["value"]
        reference_mann = stats.mannwhitneyu(a, b, alternative="two-sided")
        self.assertAlmostEqual(mann["statistic"], reference_mann.statistic)
        self.assertAlmostEqual(mann["p_value"], reference_mann.pvalue)
        self.assertEqual(mann["effect_size"]["name"], "rank_biserial_correlation")

    def test_invalid_grain_dtype_group_count_and_variation_fail_closed(self):
        aggregated = self.store.register(
            pd.DataFrame({"group":["A", "B"], "value":[1, 2]}),
            source="fixture.statistics", coverage="complete", predicate_known=True,
            grain="aggregate", aggregation="SELECT group, AVG(value) FROM data GROUP BY group")
        structured = {tool.name: tool for tool in local_tools(
            AnalysisToolContext(self.store, {}, [], lambda **_: None))}
        invalid = structured["statistical_test"].invoke({
            "dataset_id":aggregated.id, "test":"independent_t",
            "value_column":"value", "group_column":"group"})
        self.assertEqual(invalid["status"], "error")
        self.assertEqual(invalid["error_code"], "invalid_tool_input")
        with self.assertRaises(ValueError):
            self.run_test(test="independent_t", value_column="category_a", group_column="group")

        constant = self.store.register(
            pd.DataFrame({"group":["A", "A", "B", "B"], "value":[1, 1, 1, 1]}),
            source="constant", coverage="complete", predicate_known=True)
        with self.assertRaises(ValueError):
            self.tools["statistical_test"](
                constant.id, "independent_t", "value", "group")

    def test_production_graph_matches_independent_fixture_oracle_without_model(self):
        report = evaluate()
        self.assertEqual(report["status"], "PASS", report)
        self.assertTrue(report["evidence"]["statistic_matches"])
        self.assertTrue(report["evidence"]["p_value_matches"])
        self.assertEqual(report["evidence"]["model_calls"], 0)
        self.assertEqual(report["evidence"]["remote_executions"], 0)

    def test_structured_statistical_evidence_survives_runtime_restart(self):
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "restart-statistics", ForbiddenModel())
            runtime.datasets.register(
                self.store.frames[self.info.id].copy(), source="fixture.statistics",
                coverage="complete", predicate_known=True, snapshot="fixture:v1")
            outcome = runtime.submit(
                "fixture.statistics의 group 그룹별 value 차이에 대해 독립표본 t-검정을 수행해줘")
            self.assertEqual(outcome["status"], "answered", outcome)
            before = runtime.inspect()["recovery"]["statistical_evidence"]["test_result"]
            runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "restart-statistics", ForbiddenModel())
            state = reopened.inspect()
            after = state["recovery"]["statistical_evidence"]["test_result"]
            self.assertEqual(state["state"], "idle")
            self.assertEqual(state["recovery"]["status"], "complete")
            self.assertEqual(after, before)
            reopened.close()

    def test_every_supported_intent_routes_without_model_when_columns_are_unambiguous(self):
        prompts = {
            "paired_t": "fixture.statistics의 before와 after 차이에 대해 대응표본 t-검정을 수행해줘",
            "chi_square": "fixture.statistics의 category_a와 category_b 카이제곱 독립성 검정을 수행해줘",
            "one_way_anova": "fixture.statistics의 three_groups 그룹별 value 일원분산분석을 수행해줘",
            "mann_whitney": "fixture.statistics의 group 그룹별 value 맨-휘트니 U 검정을 수행해줘",
            "mean_ci": "fixture.statistics의 value 평균에 대한 95% 신뢰구간을 계산해줘",
        }
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        for kind, prompt in prompts.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                runtime = GraphAnalysisRuntime(root, "owner", kind, ForbiddenModel())
                runtime.datasets.register(
                    self.store.frames[self.info.id].copy(), source="fixture.statistics",
                    coverage="complete", predicate_known=True, snapshot="fixture:v1")
                try:
                    outcome = runtime.submit(prompt)
                    self.assertEqual(outcome["status"], "answered", outcome)
                    recovery = runtime.inspect()["recovery"]
                    self.assertEqual(recovery["model_calls"], 0)
                    self.assertEqual(
                        recovery["statistical_evidence"]["test_result"]["kind"], kind)
                finally:
                    runtime.close()


if __name__ == "__main__":
    unittest.main()
