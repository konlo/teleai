"""Common tool envelopes, profiling and approval-safe source discovery."""
import json
import tempfile
import unittest
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext, COMMON_TOOL_RESULT_SCHEMA
from core.analysis_sql import validate_query
from utils.analysis_datasets import DatasetStore


SOURCE = "catalog.analytics.bank_loan"


class DiscoveryModel(BaseChatModel):
    position: int = 0

    @property
    def _llm_type(self):
        return "discovery-approval-contract"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.position == 0:
            call = {"name": "plan_source_discovery", "args": {"pattern": "loan"}}
            message = AIMessage(content="", tool_calls=[{**call, "id": str(uuid4())}])
        elif self.position == 1:
            query = (
                "SELECT table_catalog, table_schema, table_name, table_type "
                "FROM `catalog`.information_schema.tables "
                "WHERE instr(lower(table_name), lower('loan')) > 0 "
                "ORDER BY table_catalog, table_schema, table_name LIMIT 100"
            )
            message = AIMessage(content="", tool_calls=[{
                "name": "query_databricks",
                "args": {
                    "source": "catalog.information_schema.tables",
                    "query": query,
                    "reason": "사용 가능한 테이블 목록을 information_schema에서 조회합니다.",
                },
                "id": str(uuid4()),
            }])
        else:
            message = AIMessage(content="조회 결과를 확인했습니다.")
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class NoModelCall(BaseChatModel):
    @property
    def _llm_type(self):
        return "must-not-run"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("deterministic profile recovery should not call the model")


class AnalysisToolContractTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.remote_proposal = Mock()
        self.context = AnalysisToolContext(
            self.store,
            {},
            [{"table": SOURCE, "columns": [{"name": "age", "dtype": "int64"}]}],
            self.remote_proposal,
        )
        self.definitions = build_analysis_tools(self.context)
        self.tools = {definition.name: definition.run for definition in self.definitions}

    def test_every_tool_declares_common_output_contract(self):
        required = set(COMMON_TOOL_RESULT_SCHEMA["required"])
        for definition in self.definitions:
            with self.subTest(tool=definition.name):
                schema = definition.schema()
                self.assertTrue(required.issubset(schema["output_schema"]["required"]))
                self.assertIn("ready", schema["statuses"])
                self.assertFalse(schema["parameters"]["additionalProperties"])

    def test_dataset_profile_is_bounded_structured_and_scope_aware(self):
        frame = pd.DataFrame({
            "age": [20, 20, 40, None],
            "segment": ["A", "A", "B", None],
            "customer_id": [f"private-{index}" for index in range(4)],
        })
        info = self.store.register(
            frame,
            source=SOURCE,
            coverage="complete",
            predicate_known=True,
            snapshot="2026-09-18T00:00:00Z",
        )
        result = self.tools["profile_dataset"](info.id, limit=3)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["evidence_ids"], [info.id])
        self.assertIn("coverage=complete", result["scope"])
        columns = {column["name"]: column for column in result["profile"]["columns"]}
        self.assertEqual(columns["age"]["null_count"], 1)
        self.assertEqual(columns["age"]["numeric_summary"]["median"], 20.0)
        self.assertEqual(columns["segment"]["distinct_count"], 2)
        self.assertEqual(columns["segment"]["top_values"][0], {"value": "A", "count": 2})

        high_cardinality = pd.DataFrame({"customer_id": [f"private-{index}" for index in range(60)]})
        high = self.store.register(high_cardinality, source=SOURCE, coverage="sampled")
        private_profile = self.tools["profile_dataset"](high.id)
        column = private_profile["profile"]["columns"][0]
        self.assertEqual(column["top_values_omitted"], "high_cardinality")
        self.assertNotIn("private-0", json.dumps(private_profile, ensure_ascii=False))

    def test_profile_validates_columns_and_paginates(self):
        info = self.store.register(pd.DataFrame({f"c{i}": [i] for i in range(70)}), source=SOURCE)
        page = self.tools["profile_dataset"](info.id, offset=60, limit=10)
        self.assertEqual(page["profile"]["column_page"]["returned"], 10)
        self.assertFalse(page["profile"]["column_page"]["has_more"])
        with self.assertRaises(ValueError):
            self.tools["profile_dataset"](info.id, columns=["missing"])

    def test_source_discovery_only_builds_a_read_only_approval_plan(self):
        result = self.tools["plan_source_discovery"](pattern="loan", limit=25)
        self.assertEqual(result["status"], "planned")
        plan = result["discovery_plan"]
        validate_query(plan["query"])
        self.assertIn("`catalog`.information_schema.tables", plan["query"])
        self.assertIn("LIMIT 25", plan["query"])
        self.remote_proposal.assert_not_called()

    def test_unknown_or_ambiguous_catalog_is_not_guessed(self):
        unknown = self.tools["plan_source_discovery"](catalog="another")
        self.assertEqual(unknown["status"], "needs_context")
        self.context.reference_context.append({"table": "second.schema.events", "columns": []})
        ambiguous = self.tools["plan_source_discovery"]()
        self.assertEqual(ambiguous["status"], "needs_context")

    def test_show_chart_reuses_verified_png_and_missing_id_has_common_error(self):
        info = self.store.register(
            pd.DataFrame({"age": [20, 30, 40, 50]}),
            source=SOURCE,
            coverage="complete",
            predicate_known=True,
        )
        recommendation = self.tools["recommend_chart_images"](info.id, ["age"])
        chart_id = recommendation["cards"][0]["id"]
        shown = self.tools["show_chart"](chart_id)
        self.assertEqual(shown["status"], "ready")
        self.assertEqual(shown["cards"][0]["id"], chart_id)
        self.assertTrue(self.context.artifacts[chart_id].image.startswith(b"\x89PNG\r\n\x1a\n"))

        structured = {tool.name: tool for tool in local_tools(self.context)}
        missing = structured["show_chart"].invoke({"chart_id": "missing"})
        self.assertEqual(missing["status"], "error")
        self.assertEqual(missing["error_code"], "invalid_tool_input")
        self.assertFalse(missing["retryable"])

    def test_production_graph_pauses_before_discovery_query_execution(self):
        remote_calls = []

        def factory(_datasets):
            def execute(envelope):
                remote_calls.append(envelope)
                return {"status": "ready", "dataset": {"id": "unexpected"}}
            return execute

        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(
                root,
                "owner",
                "source-discovery",
                DiscoveryModel(),
                connection_identity="test-connection",
                remote_factory=factory,
            )
            runtime.context.reference_context[:] = [{"table": SOURCE, "columns": []}]
            result = runtime.submit("loan 이름이 포함된 테이블을 찾아줘")
            self.assertEqual(result["status"], "awaiting_approval", result)
            self.assertEqual(remote_calls, [])
            self.assertEqual(len(result["requests"]), 1)
            self.assertIn("information_schema.tables", result["requests"][0]["query"])
            runtime.close()

    def test_production_graph_profiles_one_loaded_dataset_without_model_or_remote(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "profile", NoModelCall())
            runtime.datasets.register(
                pd.DataFrame({"age": [20, None, 40], "segment": ["A", "B", "A"]}),
                source=SOURCE,
                coverage="complete",
                predicate_known=True,
            )
            result = runtime.submit("보유 데이터의 결측치를 보여줘")
            self.assertEqual(result["status"], "answered", result)
            self.assertIn("age: 결측 1건", result["text"])
            self.assertEqual(runtime.inspect()["recovery"]["profile_kind"], "missing")
            runtime.close()


if __name__ == "__main__":
    unittest.main()
