import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import AnalysisNeed, Condition, DatasetStore, assess_reuse
from utils.analysis_skill_registry import AnalysisSkillRegistry
from utils.analysis_charts import recommend_charts


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
        self.assertEqual(len(registry.list()), 5)
        self.assertIn('search_analysis_tools', registry.read('autonomous-recovery')['body'])
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
