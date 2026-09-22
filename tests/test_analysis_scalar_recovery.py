import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_statistics import ForbiddenModel


FIXTURE = json.loads(Path("tests/fixtures/analysis_acceptance.json").read_text())


class ScalarRecoveryTests(unittest.TestCase):
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
