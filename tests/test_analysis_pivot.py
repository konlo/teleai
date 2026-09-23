"""Bounded pivot tool and deterministic production recovery contracts."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_agent import fixture_reference_context, load_frames
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import DatasetStore


class AnalysisPivotTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.frame = pd.DataFrame({
            "row_axis": ["b", "a", "a", "b", "b", "a"],
            "column_axis": ["x", "x", "y", "y", "x", "y"],
            "second_axis": ["u", "u", "u", "v", "v", "v"],
            "measure": [1.0, 2.0, 4.0, 8.0, np.nan, 6.0],
            "outcome": ["yes", "no", "yes", "yes", "no", "yes"],
            "scope": ["keep", "keep", "keep", "keep", "drop", "drop"],
        })
        self.info = self.store.register(
            self.frame.copy(), source="fixture.runtime_matrix", coverage="complete",
            predicate_known=True, snapshot="fixture:v1")
        context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(context)}

    def test_count_mean_margin_condition_and_multi_column_axes_are_exact(self):
        count = self.tools["pivot_dataset"](
            self.info.id, ["row_axis"], ["column_axis", "second_axis"], "count",
            conditions=[{"column": "scope", "op": "eq", "value": "keep"}],
            margins=True, margins_name="total")
        result = self.store.frames[count["dataset"]["id"]]
        subset = self.frame.loc[self.frame.scope.eq("keep")]
        expected = pd.crosstab(
            [subset["row_axis"]], [subset["column_axis"], subset["second_axis"]],
            margins=True, margins_name="total").reset_index()
        expected.columns = [" | ".join(str(part) for part in item if str(part))
                            for item in expected.columns.to_flat_index()]
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(count["pivot_result"]["filtered_rows"], 4)
        self.assertEqual(self.store.metadata[count["dataset"]["id"]].parent_id, self.info.id)

        mean = self.tools["pivot_dataset"](
            self.info.id, ["row_axis"], ["column_axis"], "mean",
            value_column="measure", margins=True, margins_name="overall")
        actual_mean = self.store.frames[mean["dataset"]["id"]].set_index("row_axis")
        expected_mean = self.frame.pivot_table(
            index="row_axis", columns="column_axis", values="measure",
            aggfunc="mean", margins=True, margins_name="overall", observed=True)
        pd.testing.assert_frame_equal(actual_mean, expected_mean, check_names=False)

    def test_success_rate_and_overall_percent_match_independent_oracles(self):
        rate = self.tools["pivot_dataset"](
            self.info.id, ["row_axis"], ["column_axis"], "success_rate",
            value_column="outcome", success_value="yes")
        actual_rate = self.store.frames[rate["dataset"]["id"]].set_index("row_axis")
        expected_rate = (self.frame.assign(flag=self.frame.outcome.eq("yes"))
                         .pivot_table(index="row_axis", columns="column_axis",
                                      values="flag", aggfunc="mean", observed=True) * 100)
        pd.testing.assert_frame_equal(actual_rate, expected_rate, check_names=False)

        percent = self.tools["pivot_dataset"](
            self.info.id, ["row_axis"], ["column_axis"], "overall_percent")
        actual_percent = self.store.frames[percent["dataset"]["id"]].set_index("row_axis")
        expected_percent = pd.crosstab(
            self.frame.row_axis, self.frame.column_axis, normalize="all") * 100
        pd.testing.assert_frame_equal(actual_percent, expected_percent, check_names=False)

    def test_calendar_month_sort_keeps_margin_last(self):
        month_frame = pd.DataFrame({
            "period": ["mar", "jan", "feb", "jan"],
            "channel": ["x", "x", "y", "y"],
        })
        month_info = self.store.register(
            month_frame, source="fixture.month_matrix", coverage="complete",
            predicate_known=True, snapshot="fixture:v1")
        result = self.tools["pivot_dataset"](
            month_info.id, ["period"], ["channel"], "count",
            margins=True, margins_name="total", sort="calendar_month")
        actual = self.store.frames[result["dataset"]["id"]]
        self.assertEqual(actual["period"].tolist(), ["jan", "feb", "mar", "total"])

    def test_invalid_schema_grain_and_parameters_fail_closed(self):
        structured = {tool.name: tool for tool in local_tools(
            AnalysisToolContext(self.store, {}, [], lambda **_: None))}
        invalid = structured["pivot_dataset"].invoke({
            "dataset_id": self.info.id, "index_columns": ["missing"],
            "column_columns": ["column_axis"], "aggregation": "count"})
        self.assertEqual(invalid["status"], "error")
        with self.assertRaises(ValueError):
            self.tools["pivot_dataset"](
                self.info.id, ["row_axis"], ["row_axis"], "count")
        with self.assertRaises(ValueError):
            self.tools["pivot_dataset"](
                self.info.id, ["row_axis"], ["column_axis"], "mean",
                value_column="outcome")
        with self.assertRaises(ValueError):
            self.tools["pivot_dataset"](
                self.info.id, ["row_axis"], ["column_axis"], "success_rate",
                value_column="outcome")
        aggregate = self.store.register(
            pd.DataFrame({"row_axis": ["a"], "column_axis": ["x"]}),
            source="fixture.runtime_matrix", coverage="complete", predicate_known=True,
            grain="aggregate", aggregation="count")
        with self.assertRaises(ValueError):
            self.tools["pivot_dataset"](
                aggregate.id, ["row_axis"], ["column_axis"], "count")

    def test_explicit_pivot_prompts_run_without_model_and_survive_restart(self):
        frames = load_frames()
        prompts = [
            ("bank_loan", "직업(job)과 혼인상태(marital)별 고객 수 피벗 테이블을 생성하고, 행과 열의 전체 총합계(margins=True)를 표기해줘."),
            ("bank_loan", "교육수준(education)과 주택대출(housing)별 평균 잔액(balance) 피벗 테이블을 만들고, 전체 평균 총계(margins=True)를 포함해줘."),
            ("titanic", "타이타닉 승객 등급(Pclass)과 성별(Sex)에 따른 생존율 피벗 테이블을 백분율로 출력해줘."),
            ("bank_loan", "월(month)과 통신 수단(contact)에 따른 마케팅 예금 가입 성공 건수(y='yes') 피벗 테이블을 출력해줘."),
            ("bank_loan", "이전 마케팅 결과(poutcome)와 이번 가입 결과(y) 간의 교차 빈도표를 작성하고 전체 대비 백분율(normalize=True)을 구해줘."),
        ]
        for index, (source, prompt) in enumerate(prompts):
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as root:
                runtime = GraphAnalysisRuntime(root, "owner", f"pivot-{index}", ForbiddenModel())
                runtime.datasets.register(
                    frames[source].copy(), source=source, coverage="complete",
                    predicate_known=True, snapshot="fixture:v1")
                runtime.context.reference_context[:] = [fixture_reference_context(source, frames[source])]
                try:
                    outcome = runtime.submit(prompt)
                    self.assertEqual(outcome["status"], "answered", outcome)
                    recovery = runtime.inspect()["recovery"]
                    self.assertEqual(recovery["model_calls"], 0)
                    self.assertIsNotNone(recovery["pivot_evidence"])
                    before = recovery["pivot_evidence"]
                finally:
                    runtime.close()
                reopened = GraphAnalysisRuntime(root, "owner", f"pivot-{index}", ForbiddenModel())
                try:
                    self.assertEqual(reopened.inspect()["recovery"]["pivot_evidence"], before)
                finally:
                    reopened.close()


if __name__ == "__main__":
    unittest.main()
