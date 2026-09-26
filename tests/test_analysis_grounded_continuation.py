"""Schema-neutral local planning and persistent clarification journeys."""
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_agent.clarification import continue_analysis
from core.analysis_agent.intent_scope import _mentioned
from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_agent import fixture_reference_context
from scripts.evaluate_analysis_statistics import ForbiddenModel
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_datasets import stored_dataset_digest


class GroundedContinuationTests(unittest.TestCase):
    def test_alias_boundaries_keep_particles_but_not_currency_words(self):
        self.assertFalse(_mentioned('10,000달러를 넘는 값', '달'))
        for text, name in [('이전 달 중앙값', '달'), ('5월의 평균', '월'), ('나이가 40대', '나이'),
                           ('집 대출이나 신용 대출', '집 대출')]:
            self.assertTrue(_mentioned(text, name), (text, name))

    def test_compound_label_uses_one_numeric_measure_and_preserves_raw(self):
        frame = pd.DataFrame({'tag': ['a', 'b', 'c'], 'cost': [2., 4., 9.]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'measure', ForbiddenModel())
            try:
                raw = runtime.datasets.register(frame, source='custom.observations',
                    coverage='complete', predicate_known=True)
                runtime.context.reference_context[:] = [fixture_reference_context(raw.source, frame)]
                digest = stored_dataset_digest(runtime.datasets, raw.id)
                result = runtime.submit('tag cost 평균을 알려줘')
                self.assertEqual(result['status'], 'answered', result)
                state = runtime.inspect()['recovery']
                self.assertEqual(state['model_calls'], 0)
                self.assertEqual(runtime.datasets.frames[state['evidence_ids'][-1]].iloc[0, 0], 5.)
                self.assertEqual(stored_dataset_digest(runtime.datasets, raw.id), digest)
                self.assertEqual(runtime.inspect()['requests'], [])
            finally:
                runtime.close()

    def test_two_measures_or_grouped_request_cannot_become_one_overall_average(self):
        for prompt in ('left_value right_value 평균을 알려줘', 'tag별 left_value 평균을 알려줘'):
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as root:
                runtime = GraphAnalysisRuntime(root, 'owner', 'ambiguous', EvaluationModel())
                try:
                    frame = pd.DataFrame({'tag':['a','b'], 'left_value':[2.,8.], 'right_value':[4.,10.]})
                    raw = runtime.datasets.register(frame, source='fixture.measurements',
                        coverage='complete', predicate_known=True)
                    runtime.context.reference_context[:] = [fixture_reference_context(raw.source, frame)]
                    result = runtime.submit(prompt)
                    state = runtime.inspect()['recovery']
                    if result['status'] == 'answered':
                        # A real grouped plan is valid; collapsing it to one
                        # scalar is not. Two numeric targets remain ambiguous.
                        self.assertIn('tag별', prompt)
                        self.assertTrue(state.get('group_summary_evidence'))
                    self.assertFalse(state.get('evidence_ids'))
                finally:
                    runtime.close()

    def test_filtered_frequency_is_table_neutral_includes_null_and_keeps_raw(self):
        frame = pd.DataFrame({'reading':[1,5,6,8,9], 'class_key':['a','a','a','b',None]})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'frequency', ForbiddenModel())
            try:
                raw = runtime.datasets.register(frame, source='custom.events',
                    coverage='complete', predicate_known=True)
                digest = stored_dataset_digest(runtime.datasets, raw.id)
                result = runtime.submit('reading >= 5인 class_key 분포를 표로 보여줘')
                self.assertEqual(result['status'], 'answered', result)
                state = runtime.inspect()['recovery']
                actual = runtime.datasets.frames[state['evidence_ids'][-1]]
                self.assertEqual(actual['count'].tolist(), [2,1,1])
                self.assertTrue(pd.isna(actual['class_key'].iloc[-1]))
                self.assertEqual(stored_dataset_digest(runtime.datasets, raw.id), digest)
                self.assertEqual(state['model_calls'], 0)
            finally:
                runtime.close()

    def test_qualified_count_clarification_survives_restart_and_reuses_original(self):
        frame = pd.DataFrame({'prior_signal':['ok','bad','bad'], 'current_signal':['ok','ok','bad']})
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'clarify', EvaluationModel())
            raw = runtime.datasets.register(frame, source='custom.events',
                coverage='complete', predicate_known=True)
            runtime.select_dataset(raw.id)
            digest = stored_dataset_digest(runtime.datasets, raw.id)
            try:
                result = runtime.submit("이전 평가에서 '통과'한 사람이 몇 명인지 알려줘")
                self.assertEqual(result['status'], 'blocked', result)
                self.assertTrue(runtime.inspect()['recovery']['pending_clarification'])
            finally:
                runtime.close()
            runtime = GraphAnalysisRuntime(root, 'owner', 'clarify', ForbiddenModel())
            try:
                result = runtime.submit("prior_signal = 'ok'")
                self.assertEqual(result['status'], 'answered', result)
                state = runtime.inspect()['recovery']
                self.assertEqual(runtime.datasets.frames[state['evidence_ids'][-1]].iloc[0,0], 1)
                self.assertEqual(state['model_calls'], 0)
                self.assertEqual(stored_dataset_digest(runtime.datasets, raw.id), digest)
                self.assertEqual(runtime.inspect()['requests'], [])
                self.assertFalse(state.get('pending_clarification'))
            finally:
                runtime.close()

    def test_new_request_approval_word_unknown_column_and_selection_change_do_not_bind(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'boundary', EvaluationModel())
            try:
                raw = runtime.datasets.register(pd.DataFrame({'signal':['ok']}),
                    source='custom.events', coverage='complete', predicate_known=True)
                runtime.select_dataset(raw.id)
                runtime.submit('이전 평가를 통과한 사람이 몇 명이야?')
                state = runtime.inspect()['recovery']
                for reply in ('승인', 'signal', "unknown = 'ok'", 'signal의 평균을 알려줘'):
                    self.assertIsNone(continue_analysis(reply, state, runtime.context), reply)
                runtime.context.selected_dataset_id = None
                self.assertIsNone(continue_analysis("signal = 'ok'", state, runtime.context))
            finally:
                runtime.close()
