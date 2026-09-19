import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event, Thread

import pandas as pd

from core.analysis_approval import ApprovalQueue
from core.analysis_loop import AnalysisSession
from core.analysis_runtime_tools import build_analysis_tools, build_runtime_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import AnalysisNeed, Condition, DatasetStore, assess_reuse
from utils.analysis_skill_registry import AnalysisSkillRegistry
from utils.analysis_charts import recommend_charts


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.queue = ApprovalQueue("session-one")
        self.request = self.queue.propose(source="table", query="SELECT x FROM table",
                                         reason="비교 기간 필요", goal="이전 기간 비교")

    def test_no_execution_before_approval_and_no_reuse_afterwards(self):
        calls = []
        run = lambda request: calls.append(request.query)
        with self.assertRaises(PermissionError):
            self.queue.execute(self.request.id, run)
        self.assertEqual(calls, [])
        self.queue.approve(self.request.id)
        self.queue.execute(self.request.id, run)
        with self.assertRaises(PermissionError):
            self.queue.execute(self.request.id, run)
        self.assertEqual(calls, [self.request.query])

    def test_revision_change_invalidates_previous_approval(self):
        self.queue.approve(self.request.id)
        self.queue.advance()
        with self.assertRaises(PermissionError):
            self.queue.execute(self.request.id, lambda _: self.fail("executed stale SQL"))

    def test_decline_and_failure_cannot_execute_again(self):
        self.queue.decline(self.request.id)
        with self.assertRaises(PermissionError):
            self.queue.execute(self.request.id, lambda _: self.fail("executed declined SQL"))
        second = self.queue.propose(source="t", query="SELECT y FROM t", reason="retry", goal="goal")
        self.queue.approve(second.id)
        def fail(_):
            raise RuntimeError("backend failed")
        with self.assertRaises(RuntimeError):
            self.queue.execute(second.id, fail)
        with self.assertRaises(PermissionError):
            self.queue.execute(second.id, fail)

    def test_double_click_while_running_does_not_resubmit(self):
        entered, release = Event(), Event()
        def run(_):
            entered.set()
            release.wait(2)
        self.queue.approve(self.request.id)
        worker = Thread(target=lambda: self.queue.execute(self.request.id, run))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaises(PermissionError):
                self.queue.execute(self.request.id, run)
        finally:
            release.set()
            worker.join()

    def test_query_is_immutable_and_duplicate_proposal_is_one_card(self):
        with self.assertRaises(FrozenInstanceError):
            self.request.query = "SELECT changed"
        duplicate = self.queue.propose(source=self.request.source, query=self.request.query,
                                      reason="again", goal="again")
        self.assertEqual(self.request.id, duplicate.id)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.a = Condition("model", "eq", "A")
        self.info = self.store.register(pd.DataFrame({"model": ["A", "A"], "value": [1, 9]}),
            source="events", coverage="complete", conditions=(self.a,), predicate_known=True)

    def need(self, conditions=(), **kwargs):
        return AnalysisNeed("events", ("value",), conditions=conditions, **kwargs)

    def test_other_model_and_wider_population_require_source(self):
        for conditions in [(), (Condition("model", "eq", "B"),)]:
            self.assertEqual(assess_reuse(self.info, self.need(conditions)).action, "query_source")

    def test_narrower_scope_is_filtered_locally_without_mutating_parent(self):
        child = self.store.derive(self.info.id, self.need((self.a, Condition("value", "gt", 5))))
        self.assertEqual(self.store.frames[child.id]["value"].tolist(), [9])
        self.assertEqual(len(self.store.frames[self.info.id]), 2)
        self.assertEqual(child.parent_id, self.info.id)

    def test_complete_empty_result_is_reusable(self):
        empty = self.store.register(pd.DataFrame({"value": pd.Series(dtype=float)}),
                                   source="events", coverage="complete", predicate_known=True)
        self.assertEqual(assess_reuse(empty, self.need()).action, "reuse")

    def test_truncated_only_usable_when_explicitly_analyzing_current_result(self):
        info = self.store.register(pd.DataFrame({"value": [1]}), source="events", coverage="truncated")
        self.assertEqual(assess_reuse(info, self.need()).action, "query_source")
        self.assertEqual(assess_reuse(info, self.need(current_result_only=True)).action, "reuse")

    def test_aggregate_cannot_supply_raw_distribution(self):
        info = self.store.register(pd.DataFrame({"value": [5]}), source="events",
                                  coverage="complete", predicate_known=True, grain="daily", aggregation="avg")
        self.assertEqual(assess_reuse(info, self.need()).action, "query_source")

    def test_narrow_interval_and_exclusive_boundary(self):
        info = self.store.register(pd.DataFrame({"value": [11, 20, 30]}), source="events",
            coverage="complete", predicate_known=True, conditions=(Condition("value", "gt", 10),))
        child = self.store.derive(info.id, self.need((Condition("value", "ge", 20),)))
        self.assertEqual(self.store.frames[child.id]["value"].tolist(), [20, 30])
        self.assertEqual(assess_reuse(info, self.need((Condition("value", "ge", 10),))).action, "query_source")


class SkillTests(unittest.TestCase):
    def test_discovery_and_on_demand_content(self):
        registry = AnalysisSkillRegistry()
        self.assertEqual(len(registry.list()), 4)
        self.assertTrue(all("body" not in entry for entry in registry.list()))
        self.assertIn("Databricks", registry.read("dataframe-reuse")["body"])

    def test_paths_and_symlinks_cannot_escape_registry(self):
        with self.assertRaises(ValueError):
            AnalysisSkillRegistry().read("../../secrets")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "skills"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                AnalysisSkillRegistry(root).read("escape")

    def test_chart_skill_matches_current_renderer_and_does_not_claim_edit_controls(self):
        body = AnalysisSkillRegistry().read("chart-recommendations")["body"]
        for supported in ("히스토그램", "막대", "선 그래프", "산점도", "박스플롯"):
            self.assertIn(supported, body)
        for unsupported in ("그룹별 박스플롯", "상관 히트맵", "bin 개수", "색상", "축", "제목"):
            self.assertIn(unsupported, body)
        self.assertIn("아직 지원하지 않는다", body)
        self.assertIn("Databricks 재조회를 요청하지 않는다", body)
        context = AnalysisToolContext(DatasetStore(), {}, [], lambda **_: None)
        recommend = next(tool for tool in build_analysis_tools(context)
                         if tool.name == "recommend_chart_images")
        self.assertEqual(set(recommend.parameters["properties"]), {"dataset_id", "columns"})


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.session = AnalysisSession("session", "Analyze the user's data.", [])
        self.session.tools = build_runtime_tools(self.session, self.store)

    @staticmethod
    def call(name, arguments, call_id="call-1"):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "name": name, "arguments": arguments}]}

    def test_skill_observation_and_prior_user_request_reach_next_turn(self):
        seen = []
        def first(messages, schemas):
            seen.append(messages)
            if messages[-1]["role"] == "user":
                return self.call("read_analysis_skill", {"name": "period-comparison"})
            self.assertIn("기간 비교", messages[-1]["content"])
            return {"role": "assistant", "content": "월별로 비교했습니다."}
        self.session.submit("월별로 비교해줘", first)
        def followup(messages, schemas):
            self.assertTrue(any(m.get("content") == "월별로 비교해줘" for m in messages))
            self.assertTrue(any(m.get("name") == "read_analysis_skill" for m in messages))
            self.assertEqual(messages[-1]["content"], "중앙값으로 바꿔줘")
            return {"role": "assistant", "content": "같은 기간의 중앙값으로 비교합니다."}
        self.assertEqual(self.session.submit("중앙값으로 바꿔줘", followup)["status"], "answered")

    def test_approval_suspends_then_resumes_with_actual_observation(self):
        calls = []
        def propose(messages, schemas):
            return self.call("propose_databricks_query", {
                "source": "events", "query": "SELECT count(*) FROM events", "reason": "전체 건수 필요"})
        result = self.session.submit("전체 건수를 비교해줘", propose)
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(calls, [])
        def execute(request):
            calls.append(request.query)
            return {"rows": [{"count": 42}]}
        def resumed(messages, schemas):
            self.assertIn('42', messages[-1]["content"])
            self.assertTrue(any(m.get("content") == "전체 건수를 비교해줘" for m in messages))
            return {"role": "assistant", "content": "총 42건입니다."}
        answer = self.session.resume_after_approval(result["requests"][0], execute=execute,
            model=resumed, approved=True)
        self.assertEqual(answer["text"], "총 42건입니다.")
        self.assertEqual(len(calls), 1)

    def test_decline_never_invokes_remote_executor(self):
        result = self.session.submit("재조회", lambda *_: self.call("propose_databricks_query", {
            "source": "events", "query": "SELECT x FROM events", "reason": "필요"}))
        self.session.resume_after_approval(result["requests"][0],
            execute=lambda _: self.fail("declined query executed"), approved=False,
            model=lambda *_: {"role": "assistant", "content": "현재 데이터로 가능한 분석을 안내합니다."})

    def test_endless_tool_calls_stop_at_limit(self):
        self.session.max_steps = 2
        result = self.session.submit("목록", lambda *_: self.call("list_analysis_context", {}))
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(sum(m["role"] == "tool" for m in self.session.history), 2)


class ChartTests(unittest.TestCase):
    def test_actual_images_and_dataset_identity(self):
        store = DatasetStore()
        info = store.register(pd.DataFrame({"value": [1, 2, 3, 4, 5, 6],
                                           "group": ["a", "a", "b", "b", "a", "b"]}),
                              source="fixture", coverage="complete", predicate_known=True)
        cards = recommend_charts(store, info.id)
        self.assertEqual(len(cards), 3)
        self.assertTrue(all(card.image.startswith(b"\x89PNG\r\n\x1a\n") for card in cards))
        self.assertTrue(all(card.dataset_id == info.id for card in cards))

    def test_aggregate_values_are_not_histogrammed(self):
        store = DatasetStore()
        info = store.register(pd.DataFrame({"group": ["a", "b"], "mean": [3, 9]}),
                              source="fixture", grain="group", aggregation="avg")
        cards = recommend_charts(store, info.id)
        self.assertEqual([card.kind for card in cards], ["bar"])

    def test_empty_result_and_missing_columns(self):
        store = DatasetStore()
        info = store.register(pd.DataFrame({"value": []}), source="fixture")
        self.assertEqual(recommend_charts(store, info.id), [])
        with self.assertRaises(ValueError):
            recommend_charts(store, info.id, ["missing"])


if __name__ == "__main__":
    unittest.main()
