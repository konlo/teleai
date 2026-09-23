"""Harness contracts using the production graph, real DuckDB and real PNGs.

Scripted models below are not natural-language model acceptance evidence.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import pandas as pd

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from scripts.evaluate_analysis_agent import (evaluate_case, evaluation_exit_status, fixture_reference_context,
                                            load_frames, load_grading, load_specs, reference_oracle)


class EvaluationModel(BaseChatModel):
    evaluation_dataset_id: str = ""
    calls: list = []
    position: int = 0
    answer: str = "요청한 분석 결과입니다."

    @property
    def _llm_type(self):
        return "scripted-evaluation-harness-contract"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.position < len(self.calls):
            call = self.calls[self.position]
            arguments = {k: self.evaluation_dataset_id if v == "$fixture" else v
                         for k, v in call["args"].items()}
            message = AIMessage(content="", tool_calls=[{"name": call["name"],
                                "args": arguments, "id": str(uuid4())}])
        else:
            message = AIMessage(content=self.answer)
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class ActualAgentEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {spec["id"]: spec for spec in load_specs()}
        cls.grading = load_grading()
        cls.frames = load_frames()

    def evaluate(self, case, model):
        return evaluate_case(self.specs[case], self.grading.get(case), model, frames=self.frames)

    def test_correct_real_sql_result_passes(self):
        result = self.evaluate("L1_016", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT AVG(balance) AS average FROM data"}}]))
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["tools"]["local_analysis_sql"], 1)
        self.assertEqual(result["remote_executions"], 0)

    def test_profile_cases_use_structured_local_evidence(self):
        for case in ("L1_006", "L1_008", "L1_009"):
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"]["profile_dataset"], 1)
                self.assertEqual(result["remote_executions"], 0)

    def test_explicit_chart_cases_use_real_png_and_declarative_data(self):
        expected = {"L1_077":"boxplot", "L1_078":"bar", "L1_086":"scatter"}
        for case, kind in expected.items():
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"]["render_chart_spec"], 1)
                self.assertEqual(result["evidence"]["charts"][0]["kind"], kind)
                self.assertEqual(result["remote_executions"], 0)

    def test_wrong_scalar_tool_result_fails_even_with_correct_sounding_prose(self):
        result = self.evaluate("L1_016", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT AVG(balance) + 999 AS average FROM data"}}],
            answer="정확한 평균 잔액을 계산했습니다."))
        self.assertEqual(result["status"], "FAIL", result)

    def test_literal_equal_to_reference_is_not_computation_evidence(self):
        expected = float(self.frames["bank_loan"]["balance"].mean())
        result = self.evaluate("L1_016", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": f"SELECT {expected!r} AS average FROM data LIMIT 1"}}]))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_constant_hidden_behind_aggregate_fails_counterfactual_replay(self):
        expected = float(self.frames["bank_loan"]["balance"].mean())
        result = self.evaluate("L1_016", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": f"SELECT AVG(balance) * 0 + {expected!r} AS average FROM data"}}]))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)
        if result["agent_status"] in {"answered", "complete"}:
            self.assertIn("counterfactual", result["reason"])

    def test_unexecuted_answer_is_never_pass(self):
        result = self.evaluate("L1_016", EvaluationModel(answer="평균은 999입니다."))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_reference_category_counts_match_alias_and_order_independently(self):
        result = self.evaluate("L1_029", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT COUNT(*) AS n, contact AS category FROM data GROUP BY contact ORDER BY contact DESC"}}]))
        self.assertEqual(result["status"], "PASS", result)

    def test_explicit_conditions_are_preserved_and_replayed(self):
        conditions = [{"column": "loan", "op": "eq", "value": "yes"}]
        result = self.evaluate("L1_019", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT COUNT(*) AS n FROM data",
            "requested_conditions": conditions}}]))
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["tool_calls"][0]["arguments"]["requested_conditions"], conditions)
        self.assertEqual(result["evidence"]["counterfactual_probes"], 2)

    def test_every_declared_reference_oracle_is_executable_without_a_model(self):
        self.assertEqual(len(self.grading), 72)
        self.assertTrue(set(self.grading).issubset(self.specs))
        for case, grading in self.grading.items():
            with self.subTest(case=case):
                reference_oracle(self.specs[case], grading, self.frames)

    def test_statistical_cases_use_structured_evidence_and_match_reference(self):
        for case in (f"L2_{number:03d}" for number in range(51, 61)):
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"], {"statistical_test": 1})
                self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
                self.assertEqual(result["remote_executions"], 0)

    def test_statistical_prose_without_tool_evidence_never_passes(self):
        with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
            result = self.evaluate(
                "L2_051", EvaluationModel(answer="t-test와 p-value 계산을 완료했습니다."))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_outlier_cases_use_structured_evidence_and_match_reference(self):
        for case in ("L2_036", "L2_039", "L2_048"):
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"], {"detect_outliers": 1})
                self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
                self.assertEqual(result["remote_executions"], 0)

    def test_outlier_prose_without_tool_evidence_never_passes(self):
        with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
            result = self.evaluate(
                "L2_036", EvaluationModel(answer="IQR과 이상치 수를 계산했습니다."))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_outlier_cohort_followup_uses_lineage_and_matches_reference(self):
        result = self.evaluate("L2_040", EvaluationModel())
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["tools"], {
            "select_outlier_rows": 1, "local_analysis_sql": 1})
        self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
        self.assertEqual(result["remote_executions"], 0)

    def test_outlier_cohort_aggregates_use_lineage_and_match_reference(self):
        result = self.evaluate("L2_037", EvaluationModel())
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["tools"], {
            "select_outlier_rows": 1, "aggregate_dataset": 2})
        self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
        self.assertEqual(result["remote_executions"], 0)
        self.assertEqual(len(result["evidence"]["result_dataset_ids"]), 2)

    def test_parent_versus_outlier_cohort_comparison_matches_reference(self):
        result = self.evaluate("L2_038", EvaluationModel())
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["tools"], {
            "select_outlier_rows": 1, "compare_group_aggregates": 1})
        self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
        self.assertEqual(result["remote_executions"], 0)
        self.assertEqual(len(result["evidence"]["actual"]), 12)

    def test_grouped_boxplot_cases_use_real_png_and_exact_grouped_data(self):
        for case in ("L2_042", "L2_045", "L2_050"):
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"], {"render_chart_spec": 1})
                self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
                self.assertEqual(result["remote_executions"], 0)

    def test_ordered_frequency_line_uses_real_png_and_exact_counts(self):
        for case in ("L1_098", "L2_068"):
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"], {"render_chart_spec": 1})
                self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
                self.assertEqual(result["remote_executions"], 0)

    def test_count_rate_dual_axis_uses_real_png_and_exact_denominators(self):
        for case in ("L2_061", "L2_064", "L2_065", "L2_066"):
            with self.subTest(case=case):
                result = self.evaluate(case, EvaluationModel())
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"], {"render_count_rate_chart": 1})
                self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
                self.assertEqual(result["remote_executions"], 0)

    def test_scalar_reductions_are_computed_from_reference_objects(self):
        self.assertEqual(reference_oracle(self.specs["L1_005"], self.grading["L1_005"], self.frames),
                         float(len(self.frames["bank_loan"])))
        high_fare=self.frames["titanic"][self.frames["titanic"]["Fare"]>=100]
        self.assertAlmostEqual(reference_oracle(self.specs["L1_045"], self.grading["L1_045"], self.frames),
                               float(high_fare["Survived"].mean()*100))

    def test_metadata_columns_use_structured_inspection_without_model_or_remote(self):
        model = EvaluationModel()
        result = self.evaluate("L1_001", model)
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["tools"], {"inspect_table_context": 1})
        self.assertEqual(result["evidence"]["metadata"]["columns"], list(self.frames["bank_loan"].columns))
        self.assertEqual(result["evidence"]["metadata"]["column_count"], len(self.frames["bank_loan"].columns))
        self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
        self.assertEqual(result["remote_executions"], 0)

    def test_metadata_dtypes_and_type_subsets_use_structured_inspection(self):
        expected = {
            "L1_002": list(self.frames["bank_loan"].columns),
            "L1_003": self.frames["bank_loan"].select_dtypes(include=["number"]).columns.tolist(),
            "L1_004": self.frames["bank_loan"].select_dtypes(include=["object"]).columns.tolist(),
            "L1_007": list(self.frames["titanic"].columns),
        }
        for case, columns in expected.items():
            with self.subTest(case=case):
                model = EvaluationModel()
                result = self.evaluate(case, model)
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["tools"], {"inspect_table_context": 1})
                evidence = result["evidence"]["metadata"]
                if case in {"L1_003", "L1_004"}:
                    self.assertEqual(evidence["selected_columns"], columns)
                else:
                    self.assertEqual([item["name"] for item in evidence["schema"]], columns)
                self.assertEqual(result["runtime_metadata"]["recovery_model_calls"], 0)
                self.assertEqual(result["remote_executions"], 0)

    def test_multi_scalar_contract_checks_every_requested_statistic(self):
        expected = reference_oracle(self.specs["L1_017"], self.grading["L1_017"], self.frames)
        self.assertEqual(set(expected), {"average", "maximum"})
        correct = self.evaluate("L1_017", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture",
            "query": "SELECT AVG(age) AS average, MAX(age) AS maximum FROM data"}}]))
        self.assertEqual(correct["status"], "PASS", correct)
        with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
            wrong = self.evaluate("L1_017", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
                "dataset_id": "$fixture",
                "query": "SELECT AVG(age) AS average, MIN(age) AS maximum FROM data"}}]))
        self.assertIn(wrong["status"], {"FAIL", "NOT_COMPLETE"}, wrong)

    def test_reduction_oracle_accepts_recovery_only_with_real_calculation_lineage(self):
        result=self.evaluate("L1_032",EvaluationModel(calls=[{"name":"local_analysis_sql","args":{
            "dataset_id":"$fixture","query":"SELECT COUNT(*) FROM data WHERE balance < 0"}}]))
        self.assertEqual(result["status"],"PASS",result)
        recovered=self.evaluate("L1_032",EvaluationModel(answer="마이너스 잔액 고객 수를 계산했습니다."))
        self.assertEqual(recovered["status"],"PASS",recovered)
        self.assertEqual(recovered["tools"]["local_analysis_sql"],1)

    def test_new_titanic_scalar_passes_with_real_fixture_calculation(self):
        result = self.evaluate("L1_025", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT AVG(Fare) AS mean_fare FROM data"}}]))
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["evidence"]["counterfactual_probes"], 2)

    def test_new_percentage_requires_correct_subset_and_percent_scale(self):
        for case, condition in [("L1_038", "female"), ("L1_039", "male")]:
            with self.subTest(case=case):
                # Exercise the scripted model query itself; the production
                # deterministic ratio fallback is covered by integration and
                # live-model tests and would otherwise mask this harness test.
                with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
                    result = self.evaluate(case, EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
                        "dataset_id": "$fixture", "query": f"SELECT 100.0 * AVG(Survived) AS percent FROM data WHERE Sex = '{condition}'"}}]))
                self.assertEqual(result["status"], "PASS", result)
                self.assertEqual(result["evidence"]["counterfactual_probes"], 2)

    def test_percentage_prose_cannot_hide_wrong_denominator_or_scale(self):
        queries = ["SELECT AVG(Survived) FROM data WHERE Sex = 'female'",
                   "SELECT 100.0 * AVG(Survived) FROM data"]
        for query in queries:
            with self.subTest(query=query):
                with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
                    result = self.evaluate("L1_038", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
                        "dataset_id": "$fixture", "query": query}}], answer="여성 승객의 생존율을 정확히 계산했습니다."))
                self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_new_group_distribution_checks_subset_and_each_category(self):
        result = self.evaluate("L1_047", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT job AS category, COUNT(*) AS n FROM data WHERE age >= 60 GROUP BY job"}}]))
        self.assertEqual(result["status"], "PASS", result)
        wrong = self.evaluate("L1_047", EvaluationModel(calls=[{"name": "local_analysis_sql", "args": {
            "dataset_id": "$fixture", "query": "SELECT job AS category, COUNT(*) AS n FROM data GROUP BY job"}}],
            answer="60세 이상 시니어 고객의 직업 분포입니다."))
        self.assertIn(wrong["status"], {"FAIL", "NOT_COMPLETE"}, wrong)

    def test_new_supported_question_still_rejects_prose_only_answer(self):
        result = self.evaluate("L1_024", EvaluationModel(answer="실제 생존자 수를 계산했습니다."))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_compound_result_can_grade_a_named_scalar_without_ignoring_other_output(self):
        result = self.evaluate("L2_046", EvaluationModel())
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["evidence"]["counterfactual_probes"], 2)

    def test_fixture_context_bounds_values_and_loads_only_external_aliases(self):
        frame = pd.DataFrame({"flag": ["yes", "no"] * 10, "unique_value": list(range(20))})
        with tempfile.TemporaryDirectory() as context_dir:
            path = Path(context_dir) / "synthetic.json"
            path.write_text(json.dumps({"table": "synthetic", "columns": [
                {"name": "flag", "aliases": ["external description"]}]}))
            context = fixture_reference_context("synthetic", frame, context_dir)
        columns = {column["name"]: column for column in context["columns"]}
        self.assertEqual(columns["flag"]["dtype"], "object")
        self.assertEqual({item["value"] for item in columns["flag"]["top_values"]}, {"yes", "no"})
        self.assertEqual(columns["flag"]["aliases"], ["external description"])
        self.assertEqual(columns["unique_value"]["top_values"], [])

    def test_external_table_context_is_complete_and_has_a_documented_source(self):
        for table, frame in self.frames.items():
            with self.subTest(table=table):
                context = fixture_reference_context(table, frame)
                self.assertIn("test_set/README.md", context["alias_source"])
                self.assertEqual({column["name"] for column in context["columns"]}, set(frame.columns))
                self.assertTrue(all(column["aliases"] for column in context["columns"]))
        with tempfile.TemporaryDirectory() as context_dir:
            (Path(context_dir) / "synthetic.json").write_text(json.dumps({"table": "another_table", "columns": []}))
            with self.assertRaises(ValueError):
                fixture_reference_context("synthetic", pd.DataFrame({"x": [1]}), context_dir)

    def test_histogram_requires_real_plotted_values_and_png(self):
        result = self.evaluate("L1_076", EvaluationModel(calls=[{"name": "recommend_chart_images", "args": {
            "dataset_id": "$fixture", "columns": ["age"]}}]))
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["evidence"]["charts"][0]["observations"], len(self.frames["bank_loan"]))
        self.assertEqual(len(result["evidence"]["charts"][0]["png_sha256"]), 64)

    def test_titanic_histogram_grades_non_null_age_distribution(self):
        result = self.evaluate("L1_093", EvaluationModel(calls=[{"name": "recommend_chart_images", "args": {
            "dataset_id": "$fixture", "columns": ["Age"]}}]))
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["evidence"]["charts"][0]["observations"],
                         int(self.frames["titanic"]["Age"].notna().sum()))

    def test_no_chart_cannot_pass(self):
        with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
            result = self.evaluate("L1_076", EvaluationModel(answer="히스토그램을 완성했습니다."))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_wrong_column_chart_cannot_pass(self):
        with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
            result = self.evaluate("L1_076", EvaluationModel(calls=[{"name": "recommend_chart_images", "args": {
                "dataset_id": "$fixture", "columns": ["duration"]}}]))
        self.assertIn(result["status"], {"FAIL", "NOT_COMPLETE"}, result)

    def test_failed_chart_png_and_metadata_survive_temporary_runtime_cleanup(self):
        with tempfile.TemporaryDirectory() as artifacts:
            with patch("core.analysis_agent.recovery.RecoveryMiddleware._next_local", return_value=None):
                result = evaluate_case(self.specs["L1_076"], self.grading["L1_076"],
                    EvaluationModel(calls=[{"name": "recommend_chart_images", "args": {
                        "dataset_id": "$fixture", "columns": ["duration"]}}]),
                    frames=self.frames, artifact_dir=artifacts)
            self.assertNotEqual(result["status"], "PASS", result)
            metadata = result["runtime_metadata"]
            self.assertTrue(Path(metadata["path"]).exists())
            self.assertTrue(metadata["charts"])
            for chart in metadata["charts"]:
                self.assertTrue(Path(chart["path"]).read_bytes().startswith(b"\x89PNG"))
            self.assertTrue(any(event["event"] == "run_started" for event in metadata["diagnostics"]))

    def test_grading_exception_preserves_metadata_without_model_transcript(self):
        marker = "MODEL_PRIVATE_TEXT_MUST_NOT_BE_PERSISTED"
        with tempfile.TemporaryDirectory() as artifacts, patch(
                "scripts.evaluate_analysis_agent.grade_evidence", side_effect=RuntimeError(marker)):
            result = evaluate_case(self.specs["L1_016"], self.grading["L1_016"],
                EvaluationModel(answer=marker), frames=self.frames, artifact_dir=artifacts)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["error_type"], "RuntimeError")
            metadata_text = Path(result["runtime_metadata"]["path"]).read_text()
            self.assertNotIn(marker, metadata_text)
            self.assertNotIn(marker, json.dumps(result))
            self.assertTrue(json.loads(metadata_text)["diagnostics"])

    def test_databricks_proposal_stays_unapproved_and_not_complete(self):
        result = self.evaluate("L1_016", EvaluationModel(calls=[{"name": "query_databricks", "args": {
            "source": "bank_loan", "query": "SELECT AVG(balance) FROM bank_loan", "reason": "평균 계산"}}]))
        self.assertEqual(result["status"], "NOT_COMPLETE", result)
        self.assertEqual(result["agent_status"], "awaiting_approval")
        self.assertEqual(result["forbidden_executor_invocations"], 0)

    def test_ungraded_reference_does_not_call_model_or_pass(self):
        model = EvaluationModel()
        result = self.evaluate("L2_100", model)
        self.assertEqual(result["status"], "UNGRADED")
        self.assertEqual(model.position, 0)

    def test_partial_coverage_never_exits_as_fully_passing_suite(self):
        self.assertEqual(evaluation_exit_status([{"status": "PASS"}, {"status": "UNGRADED"}]), 2)
        self.assertEqual(evaluation_exit_status([{"status": "PASS"}, {"status": "NOT_COMPLETE"}]), 1)
        self.assertEqual(evaluation_exit_status([]), 2)
        self.assertEqual(evaluation_exit_status([{"status": "PASS"}]), 0)

    def test_noop_reference_does_not_silently_become_pass(self):
        spec = {**self.specs["L1_076"], "python_code": "pass"}
        model = EvaluationModel()
        result = evaluate_case(spec, self.grading["L1_076"], model, frames=self.frames)
        self.assertEqual(result["status"], "UNGRADED")
        self.assertEqual(model.position, 0)


if __name__ == "__main__":
    unittest.main()
