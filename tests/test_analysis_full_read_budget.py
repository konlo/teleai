"""Whole-frame operations reject oversized file-backed inputs before decoding."""
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from core.analysis_agent.recovery import RecoveryMiddleware
from core.analysis_tool_contract import AnalysisToolContext
from core.analysis_runtime_tools import build_analysis_tools


class FullReadBudgetTests(unittest.TestCase):
    def test_recovery_reports_budget_failure_and_safe_next_step(self):
        class Diagnostics:
            def emit(self, *_args, **_kwargs):
                pass

        recovery = RecoveryMiddleware({}, Diagnostics())
        current = {
            "failed": {"local_analysis_sql": {
                "status": "rejected", "error_code": "full_frame_budget",
                "scope": "원본 전체 복원 전 중단했습니다. 기존 데이터는 보존됩니다.",
                "user_action": "필요한 컬럼과 조건을 명시해주세요."}},
            "remote_rejected": False, "scope_error": None, "request_id": "budget-test",
            "attempts": 1, "model_calls": 0, "sent_calls": [], "model_seconds": 0,
            "evidence_ids": [], "artifact_ids": [],
        }
        result = recovery._finish(current, reason="repeated_failed_tool")
        self.assertEqual(result["recovery"]["status"], "blocked")
        self.assertEqual(result["recovery"]["stop_reason"], "full_frame_budget")
        self.assertIn("필요한 컬럼과 조건", result["messages"][0].content)
        self.assertIn("기존 데이터는 보존", result["messages"][0].content)

    def test_complex_sql_join_and_row_cohort_reject_before_full_decode(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "full-read-budget")
            store = PersistentDatasets(db, budget=0, max_full_read_bytes=5_000)
            columns = ["key", "measure", *[f"unused_{n}" for n in range(30)]]
            frame = pd.DataFrame({"key": range(300), "measure": range(300),
                                  **{f"unused_{n}": [n] * 300 for n in range(30)}})
            first = store.register_batches([frame], columns=columns, source="fixture.left",
                max_rows=500, coverage="complete", predicate_known=True)
            second = store.register_batches([frame], columns=columns, source="fixture.right",
                max_rows=500, coverage="complete", predicate_known=True)
            db.select_dataset(first.id)
            originals = {info.id: db.dataset_file(info.id).read_bytes()
                         for info in (first, second)}
            context = AnalysisToolContext(store, {}, [], lambda **_: None)
            tools = {tool.name: tool.run for tool in build_analysis_tools(context)}
            with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                wildcard = tools["local_analysis_sql"](
                    first.id, "SELECT * FROM data LIMIT 2")
                narrow = tools["local_analysis_sql"](
                    first.id, "SELECT measure FROM data LIMIT 2")
                joined = tools["join_datasets"](
                    first.id, second.id, ["key"], ["key"], "inner")
                cohort = tools["select_outlier_rows"](
                    first.id, "measure", "iqr", selection="inliers")
            for result in (wildcard, joined, cohort):
                self.assertEqual(result["status"], "rejected", result)
                self.assertEqual(result["error_code"], "full_frame_budget")
                self.assertFalse(result["retryable"])
                self.assertGreater(result["estimated_bytes"], result["budget_bytes"])
            self.assertEqual(narrow["status"], "ready", narrow)
            self.assertEqual(narrow["dataset"]["rows"], 2)
            self.assertEqual(db.selected_dataset_id(), first.id)
            self.assertEqual({info.id: db.dataset_file(info.id).read_bytes()
                              for info in (first, second)}, originals)
            self.assertEqual(set(store.metadata) - {first.id, second.id},
                             {narrow["dataset"]["id"]})
            db.close()


if __name__ == "__main__":
    unittest.main()
