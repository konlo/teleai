import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_statistics import ForbiddenModel


FIXTURE = json.loads(Path("tests/fixtures/analysis_acceptance.json").read_text())


class ScalarRecoveryTests(unittest.TestCase):
    def test_model_node_checkpoint_does_not_answer_full_population_from_sample(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "model-node-full-population", ForbiddenModel())
            source = runtime.datasets.register(
                pd.DataFrame({"measure_847": [2, 4, 6]}),
                source="arbitrary.runtime_table", coverage="unknown", predicate_known=True)
            runtime.select_dataset(source.id)
            human = HumanMessage(
                content="전체 원본 테이블의 measure_847 중앙값을 알려줘.",
                id="saved-model-node-full-population")
            recovery, _ = runtime.recovery._state({"messages": [human]})
            recovery.update(status="working", model_calls=1)
            runtime.agent.update_state(runtime.config, {
                "messages": [human], "recovery": recovery},
                as_node="ObservedSummarizationMiddleware.before_model")
            self.assertEqual(runtime.agent.get_state(runtime.config).next, ("model",))
            try:
                outcome = runtime.resume()
                self.assertEqual(outcome["status"], "incomplete", outcome)
                self.assertEqual(runtime.context.selected_dataset_id, source.id)
                self.assertFalse(runtime.inspect()["requests"])
            finally:
                runtime.close()

    def test_exhausted_checkpoint_cannot_promote_sample_to_full_population(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "resume-full-population", ForbiddenModel())
            source = runtime.datasets.register(
                pd.DataFrame({"measure_847": [2, 4, 6]}),
                source="arbitrary.runtime_table", coverage="unknown", predicate_known=True)
            runtime.select_dataset(source.id)
            human = HumanMessage(
                content="전체 원본 테이블의 measure_847 중앙값을 알려줘.",
                id="saved-full-population-request")
            recovery, _ = runtime.recovery._state({"messages": [human]})
            recovery.update(status="working", model_calls=10, model_seconds=181.0)
            runtime.agent.update_state(runtime.config, {
                "messages": [human, AIMessage(content="Unverified prior answer")],
                "recovery": recovery}, as_node="model")

            outcome = runtime.resume()
            self.assertEqual(outcome["status"], "exhausted", outcome)
            self.assertEqual(runtime.context.selected_dataset_id, source.id)
            self.assertFalse(runtime.inspect()["requests"])
            self.assertEqual(len(runtime.datasets.metadata), 1)
            runtime.close()

    def test_exhausted_checkpoint_resumes_exact_loaded_sample_locally(self):
        original = pd.DataFrame({"measure_847": [2, 4, 6, 8, 10, 12]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "resume-sample", ForbiddenModel())
            source = runtime.datasets.register(
                original, source="arbitrary.runtime_table",
                coverage="unknown", predicate_known=True)
            runtime.datasets.register(
                original.iloc[:3].copy(), source="arbitrary.runtime_table",
                coverage="unknown", predicate_known=True, parent_id=source.id)
            runtime.select_dataset(source.id)
            human = HumanMessage(
                content="현재 로딩된 6행 표본의 measure_847 중앙값을 알려줘.",
                id="saved-request")
            recovery, _ = runtime.recovery._state({"messages": [human]})
            recovery.update(status="working", model_calls=10, model_seconds=181.0)
            runtime.agent.update_state(runtime.config, {
                "messages": [human, AIMessage(content="Unverified prior answer")],
                "recovery": recovery}, as_node="model")

            outcome = runtime.resume()
            self.assertEqual(outcome["status"], "answered", outcome)
            result = runtime.datasets.frames[runtime.inspect()["recovery"]["evidence_ids"][-1]]
            self.assertEqual(float(result.iloc[0, 0]), 7.0)
            self.assertEqual(runtime.context.selected_dataset_id, source.id)
            pd.testing.assert_frame_equal(runtime.datasets.frames[source.id], original)
            self.assertFalse(runtime.inspect()["requests"])
            runtime.close()

    def test_explicit_loaded_sample_size_selects_original_after_derived_result(self):
        original = pd.DataFrame({"measure_847": [2, 4, 6, 8, 10, 12]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "sample-median", ForbiddenModel())
            source = runtime.datasets.register(
                original, source="arbitrary.runtime_table",
                coverage="unknown", predicate_known=True)
            runtime.datasets.register(
                original.iloc[:3].copy(), source="arbitrary.runtime_table",
                coverage="unknown", predicate_known=True, parent_id=source.id)
            runtime.select_dataset(source.id)

            outcome = runtime.submit(
                "현재 로딩된 6행 표본의 measure_847 중앙값을 알려줘.")
            self.assertEqual(outcome["status"], "answered", outcome)
            state = runtime.inspect()["recovery"]
            result = runtime.datasets.frames[state["evidence_ids"][-1]]
            self.assertEqual(float(result.iloc[0, 0]), 7.0)
            self.assertEqual(state["model_calls"], 0)
            self.assertEqual(runtime.context.selected_dataset_id, source.id)
            pd.testing.assert_frame_equal(runtime.datasets.frames[source.id], original)
            runtime.close()

    def test_unqualified_single_scalar_reuses_unique_complete_raw_dataset(self):
        frame = pd.DataFrame({"segment_code": ["X", "Y", "X"],
                              "measure_847": [4, 10, 16]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "single-scalar", ForbiddenModel())
            raw = runtime.datasets.register(
                frame, source="arbitrary.new_schema", coverage="complete",
                predicate_known=True)
            other = runtime.datasets.register(
                pd.DataFrame({"measure_847": [100, 200]}),
                source="arbitrary.other_schema", coverage="complete",
                predicate_known=True)
            runtime.select_dataset(raw.id)

            outcome = runtime.submit("measure_847 평균을 계산해줘.")
            self.assertEqual(outcome["status"], "answered", outcome)
            state = runtime.inspect()["recovery"]
            result = runtime.datasets.frames[state["evidence_ids"][-1]]
            self.assertEqual(float(result.iloc[0, 0]), 10.0)
            self.assertEqual(state["model_calls"], 0)
            self.assertEqual(runtime.context.selected_dataset_id, raw.id)
            pd.testing.assert_frame_equal(runtime.datasets.frames[raw.id], frame)
            self.assertEqual(runtime.datasets.frames[other.id]["measure_847"].tolist(), [100, 200])
            runtime.close()

    def test_loaded_dataframe_scalar_followups_never_need_the_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "scalar-followups", ForbiddenModel())
            runtime.datasets.register(
                pd.DataFrame(FIXTURE["rows"]), source=FIXTURE["source"],
                coverage="complete", predicate_known=True)

            for turn in FIXTURE["turns"]:
                with self.subTest(prompt=turn["prompt"]):
                    outcome = runtime.submit(turn["prompt"])
                    self.assertEqual(outcome["status"], "answered", outcome)
                    state = runtime.inspect()["recovery"]
                    result = runtime.datasets.frames[state["evidence_ids"][-1]]
                    self.assertEqual(result.shape, (1, 1))
                    self.assertEqual(float(result.iloc[0, 0]), float(turn["expected"]))
                    self.assertEqual(state["model_calls"], 0)

            runtime.close()

    def test_scalar_recovery_uses_runtime_schema_instead_of_known_table_columns(self):
        frame = pd.DataFrame({
            "batch_code": ["W1", "W1", "W2"],
            "cohort_key": ["alpha", "beta", "alpha"],
            "metric_amount": [3, 9, 30],
        })
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "schema-neutral", ForbiddenModel())
            runtime.datasets.register(
                frame, source="arbitrary.runtime_table",
                coverage="complete", predicate_known=True)

            first = runtime.submit(
                "보유 데이터에서 batch_code W1의 metric_amount 평균을 계산해줘.")
            self.assertEqual(first["status"], "answered", first)
            state = runtime.inspect()["recovery"]
            self.assertEqual(float(runtime.datasets.frames[state["evidence_ids"][-1]].iloc[0, 0]), 6.0)
            self.assertEqual(state["model_calls"], 0)

            followup = runtime.submit("그중 cohort_key alpha만 계산해줘.")
            self.assertEqual(followup["status"], "answered", followup)
            state = runtime.inspect()["recovery"]
            self.assertEqual(float(runtime.datasets.frames[state["evidence_ids"][-1]].iloc[0, 0]), 3.0)
            self.assertEqual(state["model_calls"], 0)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
