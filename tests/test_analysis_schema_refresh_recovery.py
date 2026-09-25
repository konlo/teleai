"""A stale schema question must produce a bounded approval, not an LLM loop."""
from dataclasses import asdict
from datetime import datetime, timezone
import tempfile
import unittest
from uuid import uuid4

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.approvals import QueryNotSubmitted
from core.analysis_catalog import compact_catalog, resolve_table_context


TARGET = "catalog.schema.events"
OTHER = "catalog.schema.other"


class InspectOnce(BaseChatModel):
    calls: int = 0
    target: str = TARGET

    @property
    def _llm_type(self):
        return "schema-inspection-fixture"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("stale schema recovery called the model again")
        message = AIMessage(content="", tool_calls=[{
            "name": "inspect_table_context",
            "args": {"table": self.target}, "id": str(uuid4()),
        }])
        return ChatResult(generations=[ChatGeneration(message=message)])


class NoModelCall(InspectOnce):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("fresh schema should not call the model")


class CatalogThenForbidden(InspectOnce):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("the follow-up should resolve from the prior table list")
        message = AIMessage(content=(
            "1. **events**: 은행 대출 관련 데이터입니다.\n"
            "2. **other**: 항공 운항 관련 데이터입니다."))
        return ChatResult(generations=[ChatGeneration(message=message)])


class SchemaRefreshRecoveryTests(unittest.TestCase):
    def test_forbidden_schema_probe_stops_without_model_retry(self):
        stale = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "old_field", "dtype": "int64"}]}

        def forbidden(_envelope):
            raise QueryNotSubmitted(403)

        with tempfile.TemporaryDirectory() as root:
            model = InspectOnce()
            runtime = GraphAnalysisRuntime(root, "owner", "schema-forbidden", model,
                connection_identity="test-connection",
                remote_factory=lambda _datasets: forbidden,
                reference_context_loader=lambda: [stale])
            proposal = runtime.submit("events의 어떤 항목들을 볼 수 있지?")
            self.assertEqual(proposal["status"], "awaiting_approval", proposal)
            result = runtime.respond(proposal["requests"][0]["id"], approved=True)
            self.assertEqual(result["status"], "blocked", result)
            self.assertIn("HTTP 403", result["text"])
            self.assertIn("SQL은 제출되지", result["text"])
            self.assertNotIn("히스토그램", result["text"])
            self.assertEqual(model.calls, 0)
            self.assertEqual(runtime.inspect()["dataset_ids"], [])
            runtime.close()

    def test_elliptical_followup_uses_prior_table_label_without_another_model_call(self):
        stale = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "old_field", "dtype": "int64"}]}
        other = {"table": OTHER, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "other_field", "dtype": "string"}]}
        remote_calls = []
        with tempfile.TemporaryDirectory() as root:
            model = CatalogThenForbidden()
            runtime = GraphAnalysisRuntime(root, "owner", "elliptical-schema", model,
                connection_identity="test-connection",
                remote_factory=lambda _datasets: lambda envelope: remote_calls.append(envelope),
                reference_context_loader=lambda: [stale, other])
            listed = runtime.submit("어떤 데이타를 볼 수 있지 ?")
            self.assertEqual(listed["status"], "answered", listed)
            proposal = runtime.submit(
                "은행 대출 관련 데이타를 보고 싶은데 어떤 항목들을 볼 수 있지 ?")
            self.assertEqual(proposal["status"], "awaiting_approval", proposal)
            self.assertEqual(proposal["requests"][0]["source"], TARGET)
            self.assertEqual(model.calls, 1)
            self.assertEqual(remote_calls, [])
            runtime.close()

    def test_stale_followup_proposes_exact_zero_row_query_and_preserves_root(self):
        stale = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "old_field", "dtype": "int64"}]}
        fresh_other = {"table": OTHER, "observed_at": datetime.now(timezone.utc).isoformat(),
                       "columns": [{"name": "other_field", "dtype": "string"}]}
        remote_calls = []

        def factory(datasets):
            def execute(envelope):
                remote_calls.append(envelope)
                info = datasets.register_batches([], columns=["fresh_field", "amount"],
                    source=TARGET, max_rows=100, query=envelope["query"],
                    snapshot=datetime.now(timezone.utc).isoformat(), coverage="sampled",
                    predicate_known=True)
                return {"status": "ready", "dataset": asdict(info), "preview": []}
            return execute

        with tempfile.TemporaryDirectory() as root:
            model = InspectOnce()
            runtime = GraphAnalysisRuntime(root, "owner", "stale-schema", model,
                connection_identity="test-connection", remote_factory=factory,
                reference_context_loader=lambda: [stale, fresh_other])
            original = runtime.datasets.register(pd.DataFrame({"preserve": [1, 2]}),
                source="catalog.schema.loaded", coverage="complete", predicate_known=True)
            runtime.select_dataset(original.id)

            proposal = runtime.submit("은행 대출 관련 데이터의 어떤 항목들을 볼 수 있지?")
            self.assertEqual(proposal["status"], "awaiting_approval", proposal)
            self.assertEqual(len(proposal["requests"]), 1)
            request = proposal["requests"][0]
            self.assertEqual(request["source"], TARGET)
            self.assertEqual(request["query"],
                "SELECT * FROM `catalog`.`schema`.`events` LIMIT 0")
            self.assertEqual(remote_calls, [])
            self.assertEqual(runtime.context.selected_dataset_id, original.id)

            result = runtime.respond(request["id"], approved=True)
            self.assertEqual(result["status"], "answered", result)
            self.assertIn("fresh_field", result["text"])
            self.assertIn("amount", result["text"])
            self.assertNotIn("old_field", result["text"])
            self.assertEqual(len(remote_calls), 1)
            self.assertEqual(model.calls, 1)
            self.assertEqual(runtime.context.selected_dataset_id, original.id)
            pd.testing.assert_frame_equal(runtime.datasets.frames[original.id],
                pd.DataFrame({"preserve": [1, 2]}))
            inspected = resolve_table_context([stale, fresh_other], runtime.datasets, TARGET)
            self.assertTrue(inspected["schema_changed"])
            self.assertEqual([column["dtype"] for column in inspected["table_context"]["columns"]],
                             ["", ""])
            self.assertIn("데이터 타입은 확인되지 않았", inspected["scope"])
            runtime.close()

    def test_zero_row_probe_does_not_report_dtype_only_change_as_schema_drift(self):
        saved = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "amount", "dtype": "decimal(20,2)"}]}
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "schema-dtype-unknown", NoModelCall(),
                reference_context_loader=lambda: [saved])
            runtime.datasets.register_batches([], columns=["amount"], source=TARGET,
                max_rows=100, query="SELECT * FROM `catalog`.`schema`.`events` LIMIT 0",
                snapshot=datetime.now(timezone.utc).isoformat(), coverage="sampled",
                predicate_known=True)
            inspected = resolve_table_context([saved], runtime.datasets, TARGET)
            self.assertEqual(inspected["status"], "ready")
            self.assertFalse(inspected["schema_changed"])
            self.assertEqual(inspected["table_context"]["columns"][0]["dtype"], "")
            runtime.close()

    def test_zero_row_probe_does_not_invent_column_types(self):
        stale = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "amount", "dtype": "decimal(20,2)"}]}
        remote_calls = []

        def factory(datasets):
            def execute(envelope):
                remote_calls.append(envelope)
                info = datasets.register_batches([], columns=["amount"], source=TARGET,
                    max_rows=100, query=envelope["query"],
                    snapshot=datetime.now(timezone.utc).isoformat(), coverage="sampled",
                    predicate_known=True)
                return {"status": "ready", "dataset": asdict(info), "preview": []}
            return execute

        with tempfile.TemporaryDirectory() as root:
            model = NoModelCall()
            runtime = GraphAnalysisRuntime(root, "owner", "unknown-dtype", model,
                connection_identity="test-connection", remote_factory=factory,
                reference_context_loader=lambda: [stale])
            proposal = runtime.submit("events의 컬럼 데이터 타입은 어떤 것들이 있지?")
            self.assertEqual(proposal["status"], "awaiting_approval", proposal)
            result = runtime.respond(proposal["requests"][0]["id"], approved=True)
            self.assertEqual(result["status"], "blocked", result)
            self.assertIn("데이터 타입은 확인되지 않았", result["text"])
            self.assertNotIn("decimal(20,2)", result["text"])
            self.assertEqual(len(remote_calls), 1)
            runtime.close()

    def test_fresh_schema_answers_without_approval_or_model(self):
        fresh = {"table": TARGET, "observed_at": datetime.now(timezone.utc).isoformat(),
                 "columns": [{"name": "current_field", "dtype": "int64"}]}
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "fresh-schema", NoModelCall(),
                reference_context_loader=lambda: [fresh])
            result = runtime.submit("events의 어떤 항목들을 볼 수 있지?")
            self.assertEqual(result["status"], "answered", result)
            self.assertIn("current_field", result["text"])
            self.assertEqual(runtime.inspect()["recovery"]["model_calls"], 0)
            runtime.close()

    def test_rejected_schema_refresh_does_not_query_or_retry_model(self):
        stale = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "old_field", "dtype": "int64"}]}
        remote_calls = []
        with tempfile.TemporaryDirectory() as root:
            model = InspectOnce()
            runtime = GraphAnalysisRuntime(root, "owner", "schema-rejected", model,
                connection_identity="test-connection",
                remote_factory=lambda _datasets: lambda envelope: remote_calls.append(envelope),
                reference_context_loader=lambda: [stale])
            proposal = runtime.submit("events의 어떤 항목들을 볼 수 있지?")
            self.assertEqual(proposal["status"], "awaiting_approval", proposal)
            result = runtime.respond(proposal["requests"][0]["id"], approved=False)
            self.assertEqual(result["status"], "blocked", result)
            self.assertIn("조회가 취소", result["text"])
            self.assertEqual(remote_calls, [])
            self.assertEqual(model.calls, 0)
            runtime.close()

    def test_stale_schema_without_remote_connection_stops_clearly(self):
        stale = {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
                 "columns": [{"name": "old_field", "dtype": "int64"}]}
        with tempfile.TemporaryDirectory() as root:
            model = InspectOnce()
            runtime = GraphAnalysisRuntime(root, "owner", "schema-offline", model,
                reference_context_loader=lambda: [stale])
            result = runtime.submit("events의 어떤 항목들을 볼 수 있지?")
            self.assertEqual(result["status"], "blocked", result)
            self.assertIn("스키마 조회 연결", result["text"])
            self.assertNotIn("old_field", result["text"])
            self.assertEqual(model.calls, 0)
            runtime.close()

    def test_stale_catalog_does_not_advertise_old_column_count(self):
        result = compact_catalog({"datasets": [], "skills": [], "available_tables": [
            {"table": TARGET, "observed_at": "2025-01-01T00:00:00Z",
             "columns": [{"name": "old_field"}]},
            {"table": OTHER, "observed_at": datetime.now(timezone.utc).isoformat(),
             "columns": [{"name": "current_field"}]},
        ]})
        self.assertIsNone(result["available_tables"][0]["column_count"])
        self.assertEqual(result["available_tables"][1]["column_count"], 1)


if __name__ == "__main__":
    unittest.main()
