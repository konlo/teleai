"""Bounded grouped multi-metric summary contracts."""
import unittest

import numpy as np
import pandas as pd
import tempfile

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore
from utils.analysis_datasets import stored_dataset_digest
from scripts.evaluate_analysis_agent import fixture_reference_context
from scripts.evaluate_analysis_statistics import ForbiddenModel


class AnalysisGroupSummaryTests(unittest.TestCase):
    def test_explicit_group_summary_completes_without_model_or_remote_reload(self):
        frame = pd.DataFrame({
            'segment': ['A', 'A', 'B', 'B'],
            'value': [2.0, 4.0, 10.0, 20.0],
        })
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'group-summary', ForbiddenModel())
            parent = runtime.datasets.register(frame.copy(), source='synthetic.events',
                coverage='complete', predicate_known=True, snapshot='fixture:v1')
            runtime.context.reference_context[:] = [fixture_reference_context(parent.source, frame)]
            digest = stored_dataset_digest(runtime.datasets, parent.id)
            try:
                result = runtime.submit('각 segment별 value 평균과 건수를 계산해서 비교해줘')
                self.assertEqual(result['status'], 'answered', result)
                recovery = runtime.inspect()['recovery']
                self.assertEqual(recovery['model_calls'], 0)
                evidence = recovery['group_summary_evidence']
                self.assertEqual(evidence['group_summary_result']['parent_dataset_id'], parent.id)
                self.assertIn('mean_value', result['text'])
                self.assertIn('count', result['text'])
                child = runtime.datasets.frames[evidence['dataset']['id']]
                self.assertEqual(child['mean_value'].tolist(), [3.0, 15.0])
                self.assertEqual(child['count'].tolist(), [2, 2])
                self.assertEqual(stored_dataset_digest(runtime.datasets, parent.id), digest)
                self.assertIsNone(runtime.remote_execute)
                wrong = {
                    'group_columns': ['segment'],
                    'metrics': [{'name': 'mean_value', 'aggregation': 'mean',
                                 'value_column': 'value'},
                                {'name': 'count', 'aggregation': 'count',
                                 'value_column': 'value'}],
                    'conditions': [],
                }
                self.assertFalse(runtime.recovery._group_scope_valid(
                    parent, wrong, recovery))
            finally:
                runtime.close()

    def test_named_group_comparison_filters_only_requested_observed_values(self):
        frame = pd.DataFrame({
            'cohort': ['A', 'A', 'B', 'B', 'C'],
            'measure': [2.0, 4.0, 10.0, 20.0, 1000.0],
        })
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'named-groups', ForbiddenModel())
            parent = runtime.datasets.register(frame.copy(), source='fixture.dynamic_table',
                coverage='complete', predicate_known=True, snapshot='fixture:v1')
            runtime.context.reference_context[:] = [fixture_reference_context(parent.source, frame)]
            runtime.select_dataset(parent.id)
            try:
                result = runtime.submit('cohort A와 B의 measure 평균을 비교하고 결과를 표로 보여줘')
                self.assertEqual(result['status'], 'answered', result)
                recovery = runtime.inspect()['recovery']
                self.assertEqual(recovery['model_calls'], 0)
                evidence = recovery['group_summary_evidence']
                self.assertEqual(evidence['group_summary_result']['conditions'], [
                    {'column': 'cohort', 'op': 'in', 'value': ['A', 'B']}])
                child = runtime.datasets.frames[evidence['dataset']['id']]
                self.assertEqual(child['cohort'].tolist(), ['A', 'B'])
                self.assertEqual(child['mean_measure'].tolist(), [3.0, 15.0])
                self.assertEqual(runtime.datasets.frames[parent.id]['measure'].tolist(),
                                 frame['measure'].tolist())
            finally:
                runtime.close()

    def setUp(self):
        self.store = DatasetStore()
        self.frame = pd.DataFrame({
            "segment_code": ["b", "a", "a", "b", "c", "c"],
            "metric_value": [-10.0, -2.0, 4.0, 8.0, np.nan, 6.0],
            "event_count": [1, 2, 3, 4, 5, 6],
        })
        self.info = self.store.register(
            self.frame.copy(), source="arbitrary.runtime_events", coverage="complete",
            predicate_known=True, snapshot="fixture:v1")
        context = AnalysisToolContext(self.store, {}, [], lambda **_: None)
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(context)}
        self.structured = {tool.name: tool for tool in local_tools(context)}

    def test_multiple_metrics_and_global_filter_match_independent_pandas(self):
        result = self.tools["summarize_groups"](
            self.info.id,
            ["segment_code"],
            [
                {"name": "rows", "aggregation": "count"},
                {"name": "average", "aggregation": "mean", "value_column": "metric_value"},
                {"name": "median_events", "aggregation": "median", "value_column": "event_count"},
                {"name": "max_events", "aggregation": "max", "value_column": "event_count"},
            ],
            conditions=[{"column": "metric_value", "op": "ge", "value": 0}],
        )
        actual = self.store.frames[result["dataset"]["id"]]
        source = self.frame[self.frame.metric_value.ge(0)]
        expected = source.groupby("segment_code").agg(
            rows=("segment_code", "size"), average=("metric_value", "mean"),
            median_events=("event_count", "median"), max_events=("event_count", "max"),
        ).reset_index()
        pd.testing.assert_frame_equal(actual, expected)
        child = self.store.metadata[result["dataset"]["id"]]
        self.assertEqual(child.parent_id, self.info.id)
        self.assertEqual(result["group_summary_result"]["filtered_rows"], len(source))

    def test_conditional_percentage_and_mean_keep_zero_groups(self):
        condition = {"column": "metric_value", "op": "lt", "value": 0}
        result = self.tools["summarize_groups"](
            self.info.id,
            ["segment_code"],
            [
                {"name": "negative_percent", "aggregation": "conditional_percent",
                 "condition": condition},
                {"name": "negative_mean", "aggregation": "conditional_mean",
                 "value_column": "metric_value", "condition": condition, "empty_value": 0.0},
            ],
        )
        actual = self.store.frames[result["dataset"]["id"]].set_index("segment_code")
        self.assertEqual(actual.loc["a", "negative_percent"], 50.0)
        self.assertEqual(actual.loc["a", "negative_mean"], -2.0)
        self.assertEqual(actual.loc["c", "negative_percent"], 0.0)
        self.assertEqual(actual.loc["c", "negative_mean"], 0.0)

    def test_invalid_columns_metric_shapes_and_aggregate_grain_fail_closed(self):
        rejected = self.structured["summarize_groups"].invoke({
            "dataset_id": self.info.id,
            "group_columns": ["missing"],
            "metrics": [{"name": "rows", "aggregation": "count"}],
        })
        self.assertEqual(rejected["status"], "error")
        with self.assertRaises(ValueError):
            self.tools["summarize_groups"](
                self.info.id, ["segment_code"],
                [{"name": "bad", "aggregation": "mean", "value_column": "segment_code"}])
        with self.assertRaises(ValueError):
            self.tools["summarize_groups"](
                self.info.id, ["segment_code"],
                [{"name": "bad", "aggregation": "conditional_percent"}])
        aggregate = self.store.register(
            pd.DataFrame({"segment_code": ["a"], "rows": [1]}),
            source="arbitrary.runtime_events", coverage="complete", predicate_known=True,
            grain="aggregate", aggregation="count")
        with self.assertRaises(ValueError):
            self.tools["summarize_groups"](
                aggregate.id, ["segment_code"],
                [{"name": "rows", "aggregation": "count"}])


if __name__ == "__main__":
    unittest.main()
