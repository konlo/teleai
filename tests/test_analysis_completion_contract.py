"""Completion cannot silently skip a requested output or publish model claims."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from core.analysis_agent.completion import (CONTRACTS, CompletionContract,
    completion_ready, recovery_instruction)
from core.analysis_agent.recovery import RecoveryMiddleware
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_runtime_tools import build_analysis_tools
from scripts.evaluate_analysis_agent import fixture_reference_context
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import stored_dataset_digest


class Diagnostics:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class CompletionContractTests(unittest.TestCase):
    def test_repair_contracts_reference_real_registered_tools(self):
        from core.analysis_tool_contract import AnalysisToolContext
        from utils.analysis_datasets import DatasetStore
        context = AnalysisToolContext(DatasetStore(), {}, [], lambda **_: None)
        names = {tool.name for tool in build_analysis_tools(context)}
        names.add('query_databricks')  # Runtime's approval-gated replacement.
        for contract in CONTRACTS:
            with self.subTest(capability=contract.name):
                self.assertTrue(set(contract.tools).issubset(names))

    def test_invalid_enum_returns_repair_constraint_before_any_mutation(self):
        from core.analysis_tool_contract import AnalysisToolContext
        from utils.analysis_datasets import DatasetStore
        store = DatasetStore()
        raw = store.register(pd.DataFrame({'category':['a','b','a']}), source='fixture.events',
                             coverage='complete', predicate_known=True)
        tool = next(t for t in local_tools(AnalysisToolContext(store, {}, [], lambda **_: None))
                    if t.name == 'aggregate_dataset')
        result = tool.invoke({'dataset_id':raw.id, 'aggregation':'count',
                              'group_column':'category', 'sort':'none'})
        self.assertEqual(result['error_code'], 'invalid_tool_arguments')
        self.assertEqual(result['validation_issues'], [{'path':['sort'], 'rule':'enum',
                                                        'expected':['ascending','descending']}])
        self.assertEqual(len(store.metadata), 1)
        corrected = tool.invoke({'dataset_id':raw.id, 'aggregation':'count',
                                 'group_column':'category', 'sort':'ascending'})
        self.assertEqual(corrected['status'], 'ready')
        self.assertEqual(len(store.metadata), 2)

    def test_unbound_business_condition_cannot_publish_model_chosen_population(self):
        from tests.test_actual_agent_evaluation import EvaluationModel
        frame = pd.DataFrame({'prior_signal':['ok','bad','bad'],
                              'current_signal':['ok','ok','bad']})
        model = EvaluationModel(calls=[{'name':'local_analysis_sql', 'args':{
            'dataset_id':'$fixture', 'query':"SELECT COUNT(*) AS count FROM data WHERE current_signal = 'ok'"}}],
            answer='이전 통과 인원은 2명입니다.')
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'unbound-condition', model)
            try:
                raw = runtime.datasets.register(frame, source='fixture.decisions',
                    coverage='complete', predicate_known=True, snapshot='fixture:v1')
                model.evaluation_dataset_id = raw.id
                runtime.context.reference_context[:] = [fixture_reference_context(raw.source, frame)]
                result = runtime.submit("이전 평가에서 '통과' 판정을 받은 사람이 몇 명인지 알려줘")
                self.assertEqual(result['status'], 'blocked', result)
                self.assertEqual(runtime.inspect()['recovery']['stop_reason'], 'analysis_target_unresolved')
                self.assertNotIn('2명', result['text'])
                self.assertIn('어떤 컬럼과 조건값', result['text'])
                pd.testing.assert_frame_equal(runtime.datasets.frames[raw.id], frame)
            finally:
                runtime.close()

    def test_load_or_preview_does_not_short_circuit_other_outputs(self):
        for early in ({'data_load': True, 'load_evidence_id': 'raw'},
                      {'preview_limit': 3, 'preview_evidence': {'values': [1, 2, 3]}}):
            with self.subTest(early=early):
                current = {**early, 'chart': True, 'artifact_ids': [], 'failed': {}}
                self.assertFalse(completion_ready(current))
                current['artifact_ids'] = ['verified-card']
                self.assertTrue(completion_ready(current))
                current.update(calculation=True, evidence_ids=[])
                self.assertFalse(completion_ready(current))

    def test_unregistered_renderer_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CompletionContract('new', 'new_requested', 'new_evidence', ('tool',), None)

    def test_recovery_targets_missing_output_and_preserves_completed_work(self):
        current = {'group_summary_requested': True, 'group_summary_evidence': None,
                   'chart': True, 'artifact_ids': ['verified']}
        instruction = recovery_instruction(current)
        self.assertIn('summarize_groups', instruction)
        self.assertNotIn('local_analysis_sql', instruction)
        self.assertNotIn('render_chart_spec', instruction)

    def test_final_exit_rechecks_evidence_even_when_called_as_success(self):
        diagnostics = Diagnostics()
        middleware = RecoveryMiddleware({}, diagnostics)
        current = dict(chart=True, artifact_ids=[], evidence_ids=[], failed={},
                       attempts=0, model_calls=1, model_seconds=0, sent_calls=[])
        result = middleware._finish(current, AIMessage(content='차트를 생성했습니다.'))
        self.assertNotEqual(result['recovery']['status'], 'complete')
        self.assertNotIn('차트를 생성했습니다', result['messages'][0].content)
        self.assertEqual(diagnostics.events[-1][1]['missing_capabilities'], ['chart'])

    def test_graph_rejects_missing_renderer_output_without_losing_raw(self):
        frame = pd.DataFrame({'bucket': ['x', 'x', 'y'], 'reading': [1., 3., 8.]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'render-failure', ForbiddenModel())
            try:
                parent = runtime.datasets.register(frame, source='fixture.observations',
                    coverage='complete', predicate_known=True, snapshot='fixture:v1')
                runtime.context.reference_context[:] = [fixture_reference_context(parent.source, frame)]
                digest = stored_dataset_digest(runtime.datasets, parent.id)
                broken = tuple(replace(c, render=lambda *_: '') if c.name == 'group_summary'
                               else c for c in CONTRACTS)
                with patch('core.analysis_agent.completion.CONTRACTS', broken):
                    result = runtime.submit('각 bucket별 reading 평균과 건수를 표로 보여줘')
                self.assertNotEqual(result['status'], 'answered', result)
                recovery = runtime.inspect()['recovery']
                self.assertTrue(recovery['group_summary_evidence'])
                self.assertEqual(recovery['stop_reason'], 'completion_output_missing')
                self.assertEqual(stored_dataset_digest(runtime.datasets, parent.id), digest)
                # Next request must still work with the saved raw data and a real table.
                result = runtime.submit('각 bucket별 reading 평균과 건수를 계산해줘')
                self.assertEqual(result['status'], 'answered', result)
                self.assertIn('mean_reading', result['text'])
                self.assertEqual(stored_dataset_digest(runtime.datasets, parent.id), digest)
            finally:
                runtime.close()

    def test_compound_statistics_and_chart_select_raw_among_prior_subsets(self):
        frame = pd.DataFrame({'reading': [2., 4., 10., 20., 30., 40.]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'compound', ForbiddenModel())
            try:
                raw = runtime.datasets.register(frame, source='fixture.signals',
                    coverage='unknown', predicate_known=True, snapshot='fixture:v1')
                subset = runtime.datasets.register(frame.iloc[:3].copy(), source=raw.source,
                    coverage='unknown', predicate_known=True, snapshot='fixture:v1',
                    parent_id=raw.id, conditions=[{'column':'reading','op':'le','value':10}])
                runtime.context.reference_context[:] = [fixture_reference_context(raw.source, frame)]
                runtime.select_dataset(raw.id)
                render = next(t.run for t in build_analysis_tools(runtime.context) if t.name == 'render_chart_spec')
                # Both an original and a filtered chart exist from previous EDA.
                render(dataset_id=raw.id, kind='histogram', x='reading')
                render(dataset_id=subset.id, kind='histogram', x='reading')
                digest = stored_dataset_digest(runtime.datasets, raw.id)
                result = runtime.submit('현재 로딩된 6행 표본에서 reading 평균을 계산하고 reading 히스토그램도 보여줘')
                self.assertEqual(result['status'], 'answered', result)
                recovery = runtime.inspect()['recovery']
                self.assertEqual(recovery['model_calls'], 0)
                self.assertIn(str(frame.reading.mean()), result['text'])
                self.assertIn('이미지를 생성했습니다', result['text'])
                self.assertTrue(recovery['evidence_ids'])
                self.assertTrue(recovery['artifact_ids'])
                for card_id in recovery['artifact_ids']:
                    self.assertEqual(runtime.artifacts[card_id].dataset_id, raw.id)
                self.assertEqual(stored_dataset_digest(runtime.datasets, raw.id), digest)
            finally:
                runtime.close()

    def test_timed_out_model_checkpoint_resumes_compound_outputs_without_model(self):
        frame = pd.DataFrame({'reading': [2., 4., 10., 20., 30., 40.]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'compound-resume', ForbiddenModel())
            raw = runtime.datasets.register(frame, source='fixture.signals',
                coverage='unknown', predicate_known=True, snapshot='fixture:v1')
            runtime.datasets.register(frame.iloc[:3].copy(), source=raw.source,
                coverage='unknown', predicate_known=True, snapshot='fixture:v1', parent_id=raw.id)
            runtime.select_dataset(raw.id)
            human = HumanMessage(content='현재 로딩된 6행 표본에서 reading 평균과 reading 히스토그램을 보여줘',
                                 id='saved-compound-request')
            recovery, _ = runtime.recovery._state({'messages': [human]})
            recovery.update(status='working', model_calls=10, model_seconds=181.)
            runtime.agent.update_state(runtime.config, {'messages': [human], 'recovery': recovery},
                                       as_node='ObservedSummarizationMiddleware.before_model')
            runtime.close()
            reopened = GraphAnalysisRuntime(root, 'owner', 'compound-resume', ForbiddenModel())
            try:
                self.assertEqual(reopened.agent.get_state(reopened.config).next, ('model',))
                result = reopened.resume()
                self.assertEqual(result['status'], 'answered', result)
                rec = reopened.inspect()['recovery']
                self.assertEqual(rec['model_calls'], 10)
                self.assertTrue(rec['artifact_ids'])
                self.assertTrue(rec['evidence_ids'])
                self.assertIn(str(frame.reading.mean()), result['text'])
                self.assertFalse(reopened.inspect()['requests'])
                pd.testing.assert_frame_equal(reopened.datasets.frames[raw.id], frame)
            finally:
                reopened.close()


if __name__ == '__main__':
    unittest.main()
