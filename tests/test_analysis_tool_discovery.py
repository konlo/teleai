"""Search permissions and recoverable local outages are distinct from remote failures."""
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from langchain_core.messages import ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.failure_messages import remote_blocked
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext, normalize_tool_result
from utils.analysis_datasets import DatasetStore, stored_dataset_digest
from tests.test_actual_agent_evaluation import EvaluationModel


class DiscoveryTests(unittest.TestCase):
    def test_predicate_measure_followup_completes_without_model_and_recovers_old_timeout(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as root:
                r=GraphAnalysisRuntime(root,'owner','local-followup',EvaluationModel())
                try:
                    raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),
                        source='custom.readings',coverage='unknown',predicate_known=True)
                    r.select_dataset(raw.id)
                    first=r.submit('현재 보유한 4행 표본에서 reading >= 10인 행의 건수를 알려줘')
                    self.assertEqual(first['status'],'answered',first)
                    if interrupted:
                        with patch.object(r.recovery,'_next_local',return_value=None), \
                                patch.object(EvaluationModel,'_generate',side_effect=TimeoutError('injected')):
                            result=r.submit('그중 reading 평균을 알려줘')
                        self.assertEqual(result['status'],'incomplete',result)
                        result=r.resume()
                    else:
                        result=r.submit('그중 reading 평균을 알려줘')
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery']
                    self.assertEqual(state['model_calls'],0)
                    self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],15.)
                    self.assertFalse(r.inspect()['requests'])
                finally:r.close()

    def test_summarization_counts_toward_budget_and_reserves_last_analysis_call(self):
        from core.analysis_agent.memory import memory_middleware
        from langchain.agents.middleware import SummarizationMiddleware
        middleware = memory_middleware(EvaluationModel())
        state = {'messages': [], 'recovery': {'model_calls': 8, 'model_seconds': 0}}
        with patch.object(SummarizationMiddleware, 'before_model', return_value={'messages': []}) as summarize:
            result = middleware.before_model(state, None)
            self.assertEqual(result['recovery']['model_calls'], 9)
            self.assertEqual(result['recovery']['summary_model_calls'], 1)
            self.assertIsNone(middleware.before_model({**state, 'recovery': result['recovery']}, None))
            self.assertEqual(summarize.call_count, 1)
            self.assertEqual(state['recovery']['model_calls'], 8)

    def test_search_returns_actual_bounded_schema_without_execution(self):
        store = DatasetStore()
        context = AnalysisToolContext(store, {}, [], lambda **_: self.fail('No remote call'))
        definitions = build_analysis_tools(context)
        tool = next(t for t in definitions if t.name == 'search_analysis_tools')
        result = tool.run(query='pivot_dataset', limit=1)
        self.assertEqual(result['matches'][0]['name'], 'pivot_dataset')
        actual = next(t for t in definitions if t.name == 'pivot_dataset')
        self.assertEqual(result['matches'][0]['parameters'], actual.parameters)
        self.assertFalse(store.metadata)
        self.assertLess(len(json.dumps(tool.run(query='dataset'))), 22000)

    def test_allowlist_applies_to_search_and_remote_permission_is_explicit(self):
        context = AnalysisToolContext(DatasetStore(), {}, [], lambda **_: None,
            allowed_tool_names=frozenset({'inspect_dataset','search_analysis_tools'}))
        tool = next(t for t in build_analysis_tools(context) if t.name == 'search_analysis_tools')
        self.assertFalse(tool.run(query='query_databricks')['matches'])
        self.assertFalse(tool.run(query='pivot_dataset')['matches'])
        context.allowed_tool_names = frozenset({'query_databricks'})
        found = tool.run(query='query_databricks')['matches'][0]
        self.assertEqual(found['permission'], 'exact_sql_approval_required')
        self.assertEqual(found['name'], 'query_databricks')

    def test_local_unavailable_does_not_block_alternative_or_report_remote_failure(self):
        model = EvaluationModel(calls=[
            {'name':'aggregate_dataset','args':{'dataset_id':'$fixture','aggregation':'mean','value_column':'reading'}},
            {'name':'search_analysis_tools','args':{'query':'local_analysis_sql','limit':1}},
            {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT AVG(reading) AS mean FROM data'}}])
        with tempfile.TemporaryDirectory() as root:
            with patch('core.analysis_runtime_tools.build_aggregate_dataset',return_value=normalize_tool_result(
                    {'status':'unavailable','error_code':'local_worker_unavailable','retryable':False})):
                r = GraphAnalysisRuntime(root,'owner','outage',model)
                try:
                    raw=r.datasets.register(pd.DataFrame({'reading':[1.,2.,9.]}), source='custom.measurements',coverage='complete',predicate_known=True)
                    model.evaluation_dataset_id=raw.id
                    digest=stored_dataset_digest(r.datasets,raw.id)
                    with patch.object(r.recovery,'_next_local',return_value=None), patch.object(r.recovery,'_budget_local_rescue',return_value=None):
                        result=r.submit('reading 평균을 알려줘')
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery']
                    self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],4.)
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                    self.assertFalse(r.inspect()['requests'])
                    self.assertIn('search_analysis_tools',[m.name for m in r.events() if isinstance(m,ToolMessage)])
                finally:r.close()

    def test_only_remote_unavailability_is_an_automatic_remote_stop(self):
        self.assertFalse(remote_blocked({'failed':{'render_chart_spec':{'status':'unavailable'}}}))
        self.assertTrue(remote_blocked({'failed':{'query_databricks':{'status':'unavailable'}}}))
        self.assertTrue(remote_blocked({'remote_rejected':True}))

    def test_followup_measure_may_also_be_the_inherited_predicate_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','followup',EvaluationModel())
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),
                source='custom.readings',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id)
            first=r.submit('reading >= 10인 행의 건수를 알려줘')
            self.assertEqual(first['status'],'answered',first)
            r.close()
            model=EvaluationModel(calls=[{'name':'local_analysis_sql','args':{
                'dataset_id':raw.id,'query':'SELECT AVG(reading) AS mean FROM data WHERE reading >= 10'}}])
            r=GraphAnalysisRuntime(root,'owner','followup',model)
            try:
                with patch.object(r.recovery,'_next_local',return_value=None),patch.object(r.recovery,'_budget_local_rescue',return_value=None):
                    result=r.submit('그중 reading 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],15.)
                self.assertEqual(state['scope']['conditions'],[{'column':'reading','op':'ge','value':10}])
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()
