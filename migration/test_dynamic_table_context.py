"""Table-neutral schema drift contracts for the production analysis runtime."""
import tempfile
import unittest

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from uuid import uuid4

from core.analysis_agent.intent_scope import resolve_request_scope
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_catalog import resolve_table_context
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from migration.test_persistent_runtime import QuietModel
from utils.analysis_datasets import DatasetStore
from utils.analysis_charts import histogram_from_counts


SOURCE = 'catalog_dynamic.schema_dynamic.rotating_table'


def stale_context(columns):
    return {'table':SOURCE, 'training_status':'trained',
            'trained_at':'2020-01-01T00:00:00Z',
            'columns':[{'name':name, 'dtype':'string',
                        'aliases':['업무 '+name],
                        'top_values':[{'value':'known'}]} for name in columns]}


class DynamicTableContextTests(unittest.TestCase):
    def test_stale_schema_requires_approval_gated_zero_row_refresh(self):
        result = resolve_table_context([stale_context(['removed_column'])], DatasetStore(), SOURCE)

        self.assertEqual(result['status'], 'needs_refresh')
        self.assertEqual(result['refresh_query'],
                         'SELECT * FROM `catalog_dynamic`.`schema_dynamic`.`rotating_table` LIMIT 0')
        self.assertIn('현재 컬럼이라고 보장할 수 없습니다', result['message'])

    def test_unique_short_table_name_resolves_but_ambiguous_name_does_not(self):
        unique = resolve_table_context([stale_context(['old'])], DatasetStore(), 'rotating_table')
        duplicate = {**stale_context(['other']), 'table':'another_catalog.other_schema.rotating_table'}
        ambiguous = resolve_table_context([stale_context(['old']), duplicate], DatasetStore(), 'rotating_table')

        self.assertEqual(unique['status'], 'needs_refresh')
        self.assertEqual(ambiguous['status'], 'needs_context')
        self.assertIn('모호', ambiguous['message'])

    def test_approved_select_star_schema_replaces_stale_column_set(self):
        datasets = DatasetStore()
        info = datasets.register(pd.DataFrame(columns=['new_metric', 'new_group']),
            source=SOURCE, query=f'SELECT * FROM {SOURCE} LIMIT 0',
            coverage='truncated', predicate_known=True)

        result = resolve_table_context([stale_context(['removed_column', 'new_group'])], datasets, SOURCE)

        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['authority'], 'approved_select_star_result')
        self.assertTrue(result['schema_changed'])
        self.assertEqual([column['name'] for column in result['table_context']['columns']],
                         ['new_metric', 'new_group'])
        self.assertEqual(result['table_context']['dataset_id'], info.id)

    def test_approved_schema_detects_dtype_change_with_same_column_name(self):
        datasets = DatasetStore()
        datasets.register(pd.DataFrame({'stable_name':pd.Series(dtype='int64')}),
            source=SOURCE, query=f'SELECT * FROM {SOURCE} LIMIT 0',
            coverage='truncated', predicate_known=True)
        saved = stale_context(['stable_name'])
        saved['columns'][0]['dtype'] = 'string'

        result = resolve_table_context([saved], datasets, SOURCE)

        self.assertTrue(result['schema_changed'])
        self.assertEqual(result['table_context']['columns'][0]['dtype'], 'int64')

    def test_stale_alias_cannot_ground_a_removed_column(self):
        datasets = DatasetStore()
        datasets.register(pd.DataFrame({'new_metric':[1, 2]}), source=SOURCE,
                          coverage='complete', predicate_known=True)
        context = AnalysisToolContext(datasets, {}, [stale_context(['removed_column'])], lambda **_: None)

        scope = resolve_request_scope("업무 removed_column = 'known'인 행", context)

        self.assertEqual(scope['conditions'], [])
        self.assertNotIn('removed_column', scope['columns'])

    def test_inspection_tool_uses_arbitrary_runtime_schema_without_table_rules(self):
        datasets = DatasetStore()
        datasets.register(pd.DataFrame(columns=['feature_2027', 'segment_v2']),
            source=SOURCE, query=f'SELECT * FROM {SOURCE} LIMIT 0',
            coverage='truncated', predicate_known=True)
        context = AnalysisToolContext(datasets, {}, [stale_context(['legacy_feature'])], lambda **_: None)
        inspect = next(tool.run for tool in build_analysis_tools(context)
                       if tool.name == 'inspect_table_context')

        result = inspect(SOURCE)

        self.assertEqual(result['status'], 'ready')
        self.assertEqual([column['name'] for column in result['table_context']['columns']],
                         ['feature_2027', 'segment_v2'])

    def test_running_runtime_reloads_context_provider_without_restart(self):
        supplied = [[{'table':'catalog.one.initial', 'columns':[]}]]
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'dynamic-context', QuietModel(),
                reference_context_loader=lambda:supplied[0])
            self.assertEqual(runtime.context.reference_context[0]['table'], 'catalog.one.initial')
            supplied[0] = [{'table':'catalog.two.replaced', 'columns':[]}]

            runtime.inspect()

            self.assertEqual(runtime.context.reference_context[0]['table'], 'catalog.two.replaced')
            runtime.close()

    def test_agent_turns_stale_schema_into_approval_card_without_execution(self):
        class RefreshModel(QuietModel):
            position: int = 0

            def _generate(self, messages, **kwargs):
                calls = [
                    {'name':'inspect_table_context', 'args':{'table':SOURCE}},
                    {'name':'query_databricks', 'args':{
                        'source':SOURCE,
                        'query':'SELECT * FROM `catalog_dynamic`.`schema_dynamic`.`rotating_table` LIMIT 0',
                        'reason':'현재 스키마 컬럼을 확인합니다.'}},
                ]
                call = calls[min(self.position, len(calls) - 1)]
                self.position += 1
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content='', tool_calls=[{**call, 'id':str(uuid4())}]))])

        remote_calls = []
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'stale-approval', RefreshModel(),
                connection_identity='connection',
                remote_factory=lambda _:lambda envelope:remote_calls.append(envelope))
            runtime.context.reference_context = [stale_context(['removed_column'])]

            result = runtime.submit(f'{SOURCE}의 컬럼 목록을 알려줘')

            self.assertEqual(result['status'], 'awaiting_approval', result)
            self.assertEqual(remote_calls, [])
            self.assertEqual(len(runtime.inspect()['requests']), 1)
            self.assertEqual(runtime.inspect()['requests'][0]['query'],
                             'SELECT * FROM `catalog_dynamic`.`schema_dynamic`.`rotating_table` LIMIT 0')
            runtime.close()

    def test_latest_source_request_bypasses_cached_chart_and_waits_for_approval(self):
        class FreshHistogramModel(QuietModel):
            position: int = 0

            def _generate(self, messages, **kwargs):
                if self.position == 0:
                    message = AIMessage(content='', tool_calls=[{
                        'name':'prepare_histogram',
                        'args':{'source':SOURCE, 'column':'metric_dynamic',
                                'where_sql':'', 'fresh_source_required':True},
                        'id':str(uuid4())}])
                else:
                    message = AIMessage(content='준비했습니다.')
                self.position += 1
                return ChatResult(generations=[ChatGeneration(message=message)])

        remote_calls = []
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'fresh-histogram', FreshHistogramModel(),
                connection_identity='connection',
                remote_factory=lambda _:lambda envelope:remote_calls.append(envelope))
            runtime.context.reference_context = [{
                'table':SOURCE, 'columns':[{'name':'metric_dynamic', 'dtype':'int64'}]}]
            query = (f'SELECT metric_dynamic, COUNT(*) AS __frequency FROM {SOURCE} '
                     'WHERE metric_dynamic IS NOT NULL GROUP BY metric_dynamic')
            info = runtime.datasets.register(
                pd.DataFrame({'metric_dynamic':[1, 2], '__frequency':[10, 20]}),
                source=SOURCE, query=query, coverage='complete', predicate_known=True,
                grain='aggregate', aggregation=query, snapshot='2020-01-01T00:00:00+00:00')
            card = histogram_from_counts(runtime.datasets, info.id, 'metric_dynamic', '__frequency')
            runtime.artifacts[card.id] = card

            result = runtime.submit(
                f'현재 원본 테이블 {SOURCE}의 metric_dynamic 히스토그램을 보여줘')

            self.assertEqual(result['status'], 'awaiting_approval', result)
            self.assertEqual(remote_calls, [])
            self.assertEqual(len(runtime.inspect()['requests']), 1)
            self.assertIn('COUNT(*)', runtime.inspect()['requests'][0]['query'])
            self.assertNotEqual(runtime.inspect()['recovery'].get('artifact_ids'), [card.id])
            runtime.close()


if __name__ == '__main__':
    unittest.main()
