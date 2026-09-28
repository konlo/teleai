"""Bounded multi-dataset join, lineage and recovery contracts."""
import tempfile
import unittest
import json
from uuid import uuid4

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore
from utils.analysis_join import join_datasets
from scripts.evaluate_analysis_join import evaluate


LEFT_SOURCE = "catalog.analytics.customers"
RIGHT_SOURCE = "catalog.analytics.transactions"


class NoModelCall(BaseChatModel):
    @property
    def _llm_type(self):
        return "join-recovery-must-not-run"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("unambiguous local join should not call the model")


class JoinAggregateModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "join-then-aggregate-contract"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        joined_id = None
        for message in messages:
            if isinstance(message, ToolMessage) and message.name == "join_datasets":
                observation = json.loads(message.content)
                if observation.get("status") == "ready":
                    joined_id = observation["dataset"]["id"]
        if joined_id:
            call = {"name":"local_analysis_sql", "args":{
                "dataset_id":joined_id,
                "query":"SELECT job, SUM(amount) AS total_amount FROM data GROUP BY job ORDER BY job",
                "current_result_only":True,
            }}
        else:
            raise AssertionError("deterministic recovery should create the unambiguous join first")
        message = AIMessage(content="", tool_calls=[{**call, "id":str(uuid4())}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class AnalysisJoinTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(self.context)}

    def register(self, left, right):
        left_info = self.store.register(left, source=LEFT_SOURCE, coverage="complete",
                                        predicate_known=True, snapshot="left-v1")
        right_info = self.store.register(right, source=RIGHT_SOURCE, coverage="complete",
                                         predicate_known=True, snapshot="right-v1")
        return left_info, right_info

    def test_one_to_one_inner_join_preserves_both_parents_and_suffixes(self):
        left, right = self.register(
            pd.DataFrame({"customer_id":[1, 2], "value":["A", "B"]}),
            pd.DataFrame({"customer_id":[1, 2], "value":[10, 20], "amount":[5, 7]}),
        )
        result = self.tools["join_datasets"](
            left.id, right.id, ["customer_id"], ["customer_id"], "inner")
        self.assertEqual(result["status"], "ready", result)
        info = self.store.metadata[result["dataset"]["id"]]
        self.assertEqual(info.parent_ids, (left.id, right.id))
        self.assertEqual(info.parent_id, left.id)
        self.assertEqual(info.grain, "joined")
        self.assertEqual(info.columns, ("customer_id", "value_left", "value_right", "amount"))
        self.assertEqual(result["join_summary"]["relationship"], "one_to_one")
        self.assertEqual(result["join_summary"]["expected_rows"], 2)
        self.assertEqual(result["join_summary"]["actual_rows"], 2)

    def test_one_to_many_left_join_uses_sql_null_semantics(self):
        left, right = self.register(
            pd.DataFrame({"customer_id":[1, 2, None], "segment":["A", "B", "C"]}),
            pd.DataFrame({"customer_id":[1, 1, 3, None], "amount":[5, 7, 9, 99]}),
        )
        result = self.tools["join_datasets"](
            left.id, right.id, ["customer_id"], ["customer_id"], "left")
        summary = result["join_summary"]
        self.assertEqual(summary["relationship"], "one_to_many")
        self.assertEqual(summary["left_null_key_rows"], 1)
        self.assertEqual(summary["right_null_key_rows"], 1)
        self.assertEqual(summary["expected_rows"], 4)
        joined = self.store.frames[result["dataset"]["id"]]
        self.assertEqual(len(joined), 4)
        self.assertEqual(joined["amount"].notna().sum(), 2)

    def test_outer_join_preserves_same_named_key_for_right_only_rows(self):
        left, right = self.register(
            pd.DataFrame({"customer_id":[1], "segment":["A"]}),
            pd.DataFrame({"customer_id":[2], "amount":[9]}),
        )
        result = self.tools["join_datasets"](
            left.id, right.id, ["customer_id"], ["customer_id"], "outer")
        self.assertEqual(result["status"], "ready", result)
        joined = self.store.frames[result["dataset"]["id"]]
        self.assertEqual(set(joined["customer_id"].dropna().astype(int)), {1, 2})
        self.assertEqual(result["join_summary"]["unmatched_left_rows"], 1)
        self.assertEqual(result["join_summary"]["unmatched_right_rows"], 1)

    def test_many_to_many_is_rejected_before_creating_a_dataset(self):
        left, right = self.register(
            pd.DataFrame({"customer_id":[1, 1], "left_value":[1, 2]}),
            pd.DataFrame({"customer_id":[1, 1], "right_value":[3, 4]}),
        )
        before = set(self.store.metadata)
        result = self.tools["join_datasets"](
            left.id, right.id, ["customer_id"], ["customer_id"], "inner")
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["error_code"], "many_to_many_join")
        self.assertEqual(set(self.store.metadata), before)
        self.assertEqual(result["join_summary"]["expected_rows"], 4)

    def test_explicit_aggregate_parent_can_resolve_many_to_many(self):
        left, right = self.register(
            pd.DataFrame({"customer_id":[1, 1], "left_value":[1, 2]}),
            pd.DataFrame({"customer_id":[1, 1], "amount":[3, 4]}),
        )
        aggregated = self.store.register(
            pd.DataFrame({"customer_id":[1], "amount":[7]}),
            source=RIGHT_SOURCE,
            coverage="complete",
            predicate_known=True,
            grain="aggregate",
            aggregation="SELECT customer_id, SUM(amount) AS amount FROM data GROUP BY customer_id",
            snapshot="right-v1",
            parent_id=right.id,
        )
        result = self.tools["join_datasets"](
            left.id, aggregated.id, ["customer_id"], ["customer_id"], "inner")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["join_summary"]["relationship"], "many_to_one")
        self.assertEqual(result["join_summary"]["actual_rows"], 2)
        joined = self.store.frames[result["dataset"]["id"]]
        self.assertEqual(joined["amount"].tolist(), [7, 7])

    def test_output_limit_and_incompatible_keys_fail_closed(self):
        left, right = self.register(
            pd.DataFrame({"customer_id":range(4)}),
            pd.DataFrame({"customer_id":range(4, 8)}),
        )
        limited = join_datasets(self.store, left.id, right.id,
            left_on=["customer_id"], right_on=["customer_id"], how="outer", max_rows=5)
        self.assertEqual(limited["status"], "rejected")
        self.assertEqual(limited["error_code"], "join_output_limit")

        text = self.store.register(pd.DataFrame({"customer_id":["1"]}), source="text.keys",
                                   coverage="complete", predicate_known=True)
        structured = {tool.name: tool for tool in local_tools(self.context)}
        invalid = structured["join_datasets"].invoke({
            "left_dataset_id": left.id, "right_dataset_id": text.id,
            "left_on":["customer_id"], "right_on":["customer_id"], "how":"inner"})
        self.assertEqual(invalid["status"], "error")
        self.assertEqual(invalid["error_code"], "invalid_tool_input")
        self.assertFalse(invalid["retryable"])

    def test_unknown_parent_id_is_reported_by_the_common_adapter(self):
        structured = {tool.name: tool for tool in local_tools(self.context)}
        result = structured["join_datasets"].invoke({
            "left_dataset_id":"missing-left", "right_dataset_id":"missing-right",
            "left_on":["id"], "right_on":["id"], "how":"inner"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "dataset_not_loaded")
        self.assertEqual(result["missing_dataset_arguments"],
                         ["left_dataset_id", "right_dataset_id"])

    def test_unambiguous_join_completes_without_model_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "join", NoModelCall())
            left = runtime.datasets.register(
                pd.DataFrame({"customer_id":[1, 2], "job":["A", "B"]}),
                source=LEFT_SOURCE, coverage="complete", predicate_known=True, snapshot="left-v1")
            right = runtime.datasets.register(
                pd.DataFrame({"customer_id":[1, 1, 2], "amount":[5, 7, 9]}),
                source=RIGHT_SOURCE, coverage="complete", predicate_known=True, snapshot="right-v1")
            outcome = runtime.submit(
                "customers와 transactions를 customer_id로 내부 조인해서 조인 결과 행 개수와 컬럼 수를 보여줘")
            self.assertEqual(outcome["status"], "answered", outcome)
            joined = [info for info in runtime.datasets.metadata.values() if info.grain == "joined"]
            self.assertEqual(len(joined), 1)
            self.assertEqual(joined[0].parent_ids, (left.id, right.id))
            joined_id = joined[0].id
            runtime.close()

            reopened = GraphAnalysisRuntime(root, "owner", "join", NoModelCall())
            restored = reopened.datasets.metadata[joined_id]
            self.assertEqual(restored.parent_ids, (left.id, right.id))
            self.assertEqual(len(reopened.datasets.frames[joined_id]), 3)
            reopened.close()

    def test_real_fixture_join_matches_independent_oracle(self):
        report = evaluate()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["evidence"]["model_calls"], 0)
        self.assertEqual(report["evidence"]["remote_executions"], 0)
        self.assertTrue(report["evidence"]["same_values"])

    def test_joined_dataset_can_feed_grounded_followup_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "join-aggregate", JoinAggregateModel())
            runtime.datasets.register(
                pd.DataFrame({"customer_id":[1, 2], "job":["A", "B"]}),
                source=LEFT_SOURCE, coverage="complete", predicate_known=True, snapshot="left-v1")
            runtime.datasets.register(
                pd.DataFrame({"customer_id":[1, 1, 2], "amount":[5, 7, 9]}),
                source=RIGHT_SOURCE, coverage="complete", predicate_known=True, snapshot="right-v1")
            outcome = runtime.submit(
                "customers와 transactions를 customer_id로 내부 조인해서 job별 amount 합계를 보여줘")
            self.assertEqual(outcome["status"], "answered", outcome)
            aggregates = [info for info in runtime.datasets.metadata.values()
                          if info.grain == "aggregate" and info.parent_id]
            self.assertEqual(len(aggregates), 1)
            result = runtime.datasets.frames[aggregates[0].id].sort_values("job").reset_index(drop=True)
            self.assertEqual(result.to_dict(orient="records"), [
                {"job":"A", "total_amount":12.0}, {"job":"B", "total_amount":9.0}])
            self.assertEqual(runtime.inspect()["recovery"]["model_calls"], 1)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
