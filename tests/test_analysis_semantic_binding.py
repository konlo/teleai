"""External-definition binding contracts; scripted models are not live accuracy evidence."""
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.semantic import semantic_metadata, validate_interpretation
from core.analysis_catalog import load_saved_reference_context
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_datasets import stored_dataset_digest
from utils.table_context import ColumnContext, TableContext, save_table_context


class SemanticModel(EvaluationModel):
    replies: list = []

    def _generate(self, messages, **kwargs):
        reply = self.replies[min(self.position, len(self.replies)-1)]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(reply)))])


class SemanticBindingTests(unittest.TestCase):
    def setup_runtime(self, root, replies, *, coverage='complete'):
        model = SemanticModel(replies=replies)
        runtime = GraphAnalysisRuntime(root, 'owner', 'semantic', model)
        frame = pd.DataFrame({'historical_flag':[0,1,1,1], 'recent_flag':[0,0,0,1], 'metric_z':[2.,4.,8.,10.]})
        raw = runtime.datasets.register(frame, source='private.unfamiliar', coverage=coverage,
            predicate_known=True, snapshot='fixture:v1')
        runtime.context.reference_context[:] = [{'table':raw.source,
            'observed_at':datetime.now(timezone.utc).isoformat(), 'columns':[
            {'name':'historical_flag','dtype':'int64','description':'이전 평가 승인 상태: 0=통과, 1=거절'},
            {'name':'recent_flag','dtype':'int64','description':'현재 평가 승인 상태: 0=통과, 1=거절'},
            {'name':'metric_z','dtype':'float64','description':'이번 서비스 이용 시간(분)'}]}]
        return runtime, raw, frame

    @staticmethod
    def plan(column='historical_flag', value=0):
        return {'uncertain':False,'operation':'COUNT','column':column,
                'conditions':[{'column':column,'op':'eq','value':value}],
                'definition': ('이전' if column == 'historical_flag' else '현재')+' 평가 승인 상태: 0=통과, 1=거절'}

    def test_reversed_encoding_prior_current_and_raw_preservation(self):
        with tempfile.TemporaryDirectory() as root:
            plan = self.plan()
            runtime, raw, frame = self.setup_runtime(root, [plan,plan])
            digest = stored_dataset_digest(runtime.datasets, raw.id)
            try:
                result = runtime.submit('이전 평가에서 통과한 사람이 몇 명이야?')
                self.assertEqual(result['status'], 'answered', result)
                state = runtime.inspect()['recovery']
                self.assertEqual(state['model_calls'], 2)
                self.assertEqual(state['semantic_binding']['conditions'][0]['value'], 0)
                self.assertEqual(runtime.datasets.frames[state['evidence_ids'][-1]].iloc[0,0], 1)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id), digest)
                self.assertFalse(runtime.inspect()['requests'])
            finally: runtime.close()
            runtime = GraphAnalysisRuntime(root, 'owner', 'semantic', EvaluationModel())
            try:
                self.assertEqual(runtime.inspect()['recovery']['semantic_binding']['source'],raw.source)
                pd.testing.assert_frame_equal(runtime.datasets.frames[raw.id],frame)
            finally: runtime.close()

    def test_independent_disagreement_never_executes_calculation(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, raw, _ = self.setup_runtime(root, [self.plan(), self.plan('recent_flag')])
            try:
                result = runtime.submit('이전 평가에서 통과한 사람이 몇 명이야?')
                self.assertEqual(result['status'],'blocked',result)
                self.assertFalse(runtime.inspect()['recovery'].get('semantic_binding'))
                self.assertEqual(len(runtime.datasets.metadata),1)
            finally: runtime.close()

    def test_two_agreeing_models_cannot_substitute_duration_for_event_count(self):
        wrong={'uncertain':False,'operation':'AVG','column':'metric_z','conditions':[],
               'definition':'이번 서비스 이용 시간(분)'}
        with tempfile.TemporaryDirectory() as root:
            runtime,raw,_=self.setup_runtime(root,[wrong,wrong])
            try:
                runtime.context.reference_context[0]['columns'][1]['description']='이번 서비스 이용 횟수'
                result=runtime.submit('이번 서비스 이용 횟수의 평균을 알려줘')
                self.assertEqual(result['status'],'blocked',result)
                self.assertFalse(runtime.inspect()['recovery'].get('semantic_binding'))
                self.assertEqual(len(runtime.datasets.metadata),1)
                self.assertEqual(runtime.inspect()['recovery']['model_calls'],2)
            finally:runtime.close()

    def test_numeric_mean_requires_two_grounded_interpretations_then_real_calculation(self):
        plan={'uncertain':False,'operation':'AVG','column':'metric_z','conditions':[],
              'definition':'이번 서비스 이용 시간(분)'}
        with tempfile.TemporaryDirectory() as root:
            runtime,raw,_=self.setup_runtime(root,[plan,plan])
            try:
                result=runtime.submit('이번 서비스에 소비한 분량의 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual(runtime.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],6.)
                self.assertEqual(state['model_calls'],2)
                self.assertEqual(state['semantic_binding']['dataset_id'],raw.id)
            finally:runtime.close()

    def test_missing_stale_mismatched_schema_or_source_description_is_not_grounding(self):
        for mutation in ('missing','stale','source','dtype','duplicate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                runtime, raw, _ = self.setup_runtime(root,[self.plan()])
                try:
                    item=runtime.context.reference_context[0]
                    if mutation=='missing':
                        for c in item['columns']:c.pop('description')
                    elif mutation=='stale':item['observed_at']='2000-01-01T00:00:00+00:00'
                    elif mutation=='source':item['table']='different.table'
                    elif mutation=='dtype':
                        for c in item['columns']:c['dtype']='obsolete'
                    else:runtime.context.reference_context.append(deepcopy(item))
                    self.assertIsNone(semantic_metadata(runtime.context,raw.id))
                finally:runtime.close()

    def test_wrong_operation_value_definition_or_column_cannot_bind(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, raw, _ = self.setup_runtime(root,[self.plan()])
            try:
                metadata=semantic_metadata(runtime.context,raw.id)
                plans=[]
                for key,value in [('column','missing'),('operation','AVG'),('definition','invented'),('uncertain',True)]:
                    p=self.plan();p[key]=value;plans.append(p)
                p=self.plan(value=999);plans.append(p)
                for plan in plans:self.assertIsNone(validate_interpretation(plan,metadata,'COUNT'))
            finally:runtime.close()

    def test_bound_scope_cannot_override_explicit_predicate_or_sample_population(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, raw, _ = self.setup_runtime(root,[self.plan(),self.plan()],coverage='unknown')
            try:
                state,_=runtime.recovery._state({'messages':[HumanMessage(id='request',content='이전 평가에서 통과한 사람이 몇 명이야?')]})
                self.assertIsNone(runtime.recovery._next_local(state,{}))
                state['scope']['conditions']=[{'column':'recent_flag','op':'eq','value':1}]
                runtime.context.semantic_resolver.request=state
                result=runtime.context.semantic_resolver.resolve(raw.id)
                self.assertNotEqual(result['status'],'ready')
            finally:runtime.close()

    def test_table_context_persistence_preserves_description(self):
        with tempfile.TemporaryDirectory() as root:
            context=TableContext(table_fqn='catalog.schema.events', columns=[ColumnContext(
                name='phase',dtype='int64',description='評価結果: 7=accepted, 2=rejected')])
            save_table_context(context,storage_dir=root)
            loaded=load_saved_reference_context(root)
            self.assertEqual(loaded[0]['columns'][0]['description'],context.columns[0].description)
