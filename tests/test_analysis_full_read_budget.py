"""Whole-frame operations reject oversized file-backed inputs before decoding."""
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from uuid import uuid4

from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from core.analysis_agent.recovery import RecoveryMiddleware
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_catalog import resolve_table_context
from core.analysis_tool_contract import AnalysisToolContext
from core.analysis_runtime_tools import build_analysis_tools
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_pivot import dataset_digest


class AdaptiveBudgetModel(BaseChatModel):
    """Scripted recovery contract; not evidence of real-model reasoning quality."""
    dataset_id: str = ""
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted-full-read-budget-recovery"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        observations = []
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    observations.append(json.loads(message.content))
                except ValueError:
                    pass
        self.calls += 1
        if not observations:
            query = "SELECT * FROM data LIMIT 2"
        elif any(item.get("error_code") == "full_frame_budget" for item in observations):
            if any(item.get("status") == "ready" and item.get("dataset", {}).get("rows") == 2
                   for item in observations):
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content="보유 데이터의 event_key 앞 두 값을 확인했습니다."))])
            query = "SELECT event_key FROM data LIMIT 2"
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="결과가 없습니다."))])
        message = AIMessage(content="", tool_calls=[{
            "name": "local_analysis_sql", "args": {"dataset_id": self.dataset_id, "query": query},
            "id": str(uuid4()),
        }])
        return ChatResult(generations=[ChatGeneration(message=message)])


class FullReadBudgetTests(unittest.TestCase):
    def test_agent_can_recover_with_narrow_local_query_without_remote_reload(self):
        with tempfile.TemporaryDirectory() as root:
            model = AdaptiveBudgetModel()
            remote_calls = []
            def remote_factory(_datasets):
                def execute(envelope):
                    remote_calls.append(envelope)
                    raise AssertionError("unexpected remote query")
                return execute
            runtime = GraphAnalysisRuntime(root, "owner", "budget-recovery", model,
                policy=RuntimePolicy(max_full_read_bytes=5_000),
                connection_identity="synthetic-connector", remote_factory=remote_factory)
            frame = pd.DataFrame({"event_key": range(300),
                                  **{f"field_{n}": [n] * 300 for n in range(31)}})
            info = runtime.datasets.register_batches([frame], columns=list(frame.columns),
                source="fixture.events", max_rows=500, coverage="complete",
                predicate_known=True)
            runtime.db.select_dataset(info.id)
            runtime.context.selected_dataset_id = info.id
            model.dataset_id = info.id
            original = runtime.db.dataset_file(info.id).read_bytes()
            result = runtime.submit(
                "현재 보유 데이터에서 event_key의 앞 두 값을 보여줘. 추가 원격 조회 없이 진행해줘.")
            observations = [json.loads(message.content)
                for message in runtime.agent.get_state(runtime.config).values["messages"]
                if isinstance(message, ToolMessage)]
            self.assertEqual(result["status"], "answered", result)
            self.assertEqual(model.calls, 3)
            self.assertTrue(any(item.get("error_code") == "full_frame_budget"
                                for item in observations))
            self.assertTrue(any(item.get("status") == "ready" and
                                item.get("preview") == [{"event_key": 0}, {"event_key": 1}]
                                for item in observations))
            self.assertEqual(runtime.db.selected_dataset_id(), info.id)
            self.assertEqual(runtime.db.dataset_file(info.id).read_bytes(), original)
            self.assertEqual(remote_calls, [])
            self.assertEqual(runtime.inspect()["requests"], [])
            self.assertFalse(any(message.name == "query_databricks"
                                 for message in runtime.agent.get_state(runtime.config).values["messages"]
                                 if isinstance(message, ToolMessage)))
            runtime.close()

    def test_streaming_evidence_digest_matches_full_frame_for_file_and_legacy(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "streaming-digest")
            store = PersistentDatasets(db, budget=0)
            frame = pd.DataFrame({"measure": range(2_500),
                                  "segment": ["east", "west", None, "north", "south"] * 500,
                                  "ratio": [0.5, None, 2.5, 3.0, 4.5] * 500})
            file_info = store.register_batches([frame.iloc[:1_250], frame.iloc[1_250:]],
                columns=list(frame.columns), source="fixture.events", max_rows=3_000,
                coverage="complete", predicate_known=True)
            legacy_info = store.register(frame, source="fixture.events",
                coverage="complete", predicate_known=True)
            expected = dataset_digest(frame)
            with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                self.assertEqual(stored_dataset_digest(store, file_info.id), expected)
                self.assertEqual(stored_dataset_digest(store, legacy_info.id), expected)
            db.close()

    def test_schema_and_answer_preview_do_not_decode_whole_file(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "bounded-metadata")
            store = PersistentDatasets(db, budget=0, max_full_read_bytes=5_000)
            frame = pd.DataFrame({"event_key": range(300),
                                  **{f"field_{n}": [n] * 300 for n in range(31)}})
            info = store.register_batches([frame], columns=list(frame.columns),
                source="fixture.events", query="SELECT * FROM fixture.events",
                max_rows=500, coverage="complete", predicate_known=True)
            context = AnalysisToolContext(store, {}, [], lambda **_: None)

            class Diagnostics:
                def emit(self, *_args, **_kwargs):
                    pass

            recovery = RecoveryMiddleware({}, Diagnostics(), context=context)
            with patch.object(FrameCache, "__getitem__", side_effect=AssertionError("full decode")):
                inspected = store.inspect(info.id)
                table = resolve_table_context([], store, "fixture.events")
                answer = recovery._answer({"calculation": True, "evidence_ids": [info.id]})
            self.assertEqual(len(inspected["dtypes"]), 32)
            self.assertEqual(table["status"], "ready")
            self.assertEqual(len(table["table_context"]["columns"]), 32)
            self.assertIn("총 300행 중 앞 15행", answer)
            self.assertIn("0,0", answer)
            db.close()

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
