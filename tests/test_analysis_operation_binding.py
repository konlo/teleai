"""Production graph contracts. Scripted interpretations are not live accuracy scores."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.operation_binding import metadata_for, validate
from core.analysis_agent.intent_scope import resolve_request_scope
from tests.test_analysis_semantic_binding import SemanticModel
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_datasets import stored_dataset_digest

FIXTURE=json.loads((Path(__file__).parent/'fixtures/natural_language_journeys.json').read_text())


class OperationBindingTests(unittest.TestCase):
    def runtime(self, root, replies=None, null=False):
        column=FIXTURE['columns'][0]
        plans=replies or [self.plan()]*2
        runtime=GraphAnalysisRuntime(root,'owner','operations',SemanticModel(replies=plans),intent_mode='contract_fixture')
        frame=pd.DataFrame(FIXTURE['rows'],columns=FIXTURE['columns'])
        if null:frame.loc[0,column]=None
        raw=runtime.datasets.register(frame,source=FIXTURE['source'],coverage='complete',predicate_known=True)
        runtime.select_dataset(raw.id)
        return runtime,raw,frame

    def plan(self, operation='MEDIAN'):
        return {'uncertain':False,'operation':operation,'column':FIXTURE['columns'][0], 'request_span':'중위수'}

    def test_interpretation_executes_calculation_preserves_original_and_restarts(self):
        prompt=FIXTURE['cases'][1]['turns'][0]['prompt']
        with tempfile.TemporaryDirectory() as root:
            r,raw,frame=self.runtime(root)
            digest=stored_dataset_digest(r.datasets,raw.id)
            try:
                result=r.submit(prompt);state=r.inspect()['recovery']
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(state['operations'],['MEDIAN'])
                self.assertEqual(state['model_calls'],2)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),10)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()
            r=GraphAnalysisRuntime(root,'owner','operations',EvaluationModel(),intent_mode='contract_fixture')
            try:
                self.assertEqual(r.inspect()['recovery']['operation_binding']['dataset_id'],raw.id)
                pd.testing.assert_frame_equal(r.datasets.frames[raw.id],frame)
            finally:r.close()

    def test_conflicting_or_fabricated_interpretations_cannot_become_an_answer(self):
        for mutation in ('disagree','column','scope','uncertain','invented_span'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as root:
                wrong=self.plan()
                if mutation=='disagree':wrong['operation']='MAX'
                if mutation=='column':wrong['column']='invented'
                if mutation=='scope':wrong['conditions']=[]
                if mutation=='uncertain':wrong['uncertain']=True
                if mutation=='invented_span':wrong['request_span']='nonexistent wording'
                r,raw,_=self.runtime(root,[self.plan(),wrong])
                try:
                    result=r.submit(FIXTURE['cases'][1]['turns'][0]['prompt'])
                    self.assertEqual(result['status'],'blocked',result)
                    self.assertEqual(len(r.datasets.metadata),1)
                    self.assertFalse(r.inspect()['recovery'].get('operation_binding'))
                finally:r.close()

    def test_bound_operation_keeps_fixed_filter_and_snapshot(self):
        column=FIXTURE['columns'][0]
        with tempfile.TemporaryDirectory() as root:
            r,raw,_=self.runtime(root)
            try:
                result=r.submit(f'{column}가 10 미만인 행의 {column} 중위수를 알려줘')
                state=r.inspect()['recovery']
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),3)
                self.assertEqual(state['scope']['conditions'],[{'column':column,'op':'lt','value':10}])
            finally:r.close()

    def test_exclusion_inverts_single_predicate_without_model_or_raw_decode(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw,frame=self.runtime(root)
            try:
                result=r.submit(FIXTURE['cases'][5]['turns'][0]['prompt']);state=r.inspect()['recovery']
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),3)
                self.assertEqual(state['model_calls'],0)
                pd.testing.assert_frame_equal(r.datasets.frames[raw.id],frame)
                with patch('pandas.read_parquet',side_effect=AssertionError('row decode')):
                    self.assertEqual(r.datasets.inspect(raw.id)['null_counts'][FIXTURE['columns'][0]],0)
            finally:r.close()

    def test_compound_exclusion_and_null_complement_never_use_positive_filter(self):
        col,group=FIXTURE['columns'][:2]
        prompts=[f'{col}가 10 이상 30 미만인 행을 제외하고 평균',
                 f'{group} = \'blue\' 그리고 {col} >= 10 인 것은 제외하고 평균']
        with tempfile.TemporaryDirectory() as root:
            r,raw,_=self.runtime(root)
            try:
                for text in prompts:
                    scope=resolve_request_scope(text,r.context)
                    self.assertIn('compound_exclusion_unresolved',scope['unresolved'],scope)
            finally:r.close()
        with tempfile.TemporaryDirectory() as root:
            r,raw,_=self.runtime(root,null=True)
            try:
                scope=resolve_request_scope(FIXTURE['cases'][5]['turns'][0]['prompt'],r.context)
                self.assertIn('exclusion_null_policy_unresolved',scope['unresolved'])
            finally:r.close()

    def test_unfamiliar_followup_changes_operator_instead_of_inheriting_max(self):
        col=FIXTURE['columns'][0]
        with tempfile.TemporaryDirectory() as root:
            r,raw,_=self.runtime(root)
            try:
                first=r.submit(f'{col}가 10 이상인 행의 {col} 최댓값을 알려줘')
                self.assertEqual(first['status'],'answered',first)
                result=r.submit(f'그중 {col} 중위수도 알려줘')
                state=r.inspect()['recovery']
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(state['operations'],['MEDIAN'])
                self.assertEqual(state['scope']['conditions'],[{'column':col,'op':'ge','value':10}])
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),20.)
            finally:r.close()
