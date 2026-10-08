"""Distinct value journeys complete from real tool evidence without inference."""
from dataclasses import asdict
from datetime import datetime,timezone
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
import pandas as pd
from sqlglot import exp
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.value_list import prepare,requested,retained
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_sql import validate_query
from utils.analysis_datasets import stored_dataset_digest
from tests.test_mysql_metadata_contract import NoInference

TABLE='catalog.lab.observations'
REFERENCE=[{'table':TABLE,'observed_at':datetime.now(timezone.utc).isoformat(),
    'columns':[{'name':'label','dtype':'string'},{'name':'reading','dtype':'double'}]}]


class ValueListTests(unittest.TestCase):
    def test_saved_categorical_count_timeout_resumes_without_model_or_sql(self):
        from httpx import ReadTimeout
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatResult,ChatGeneration
        class TimeoutModel(NoInference):
            calls:int=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[{
                        'name':'query_databricks','id':str(uuid4()),'args':{'source':TABLE,
                        'query':'SELECT label, COUNT(*) AS __frequency FROM '+TABLE+' GROUP BY label',
                        'reason':'범주별 실제 빈도 확인'}}]))])
                raise ReadTimeout('fixture timeout')
        with tempfile.TemporaryDirectory() as root:
            queries=[];model=TimeoutModel()
            reference=[{**REFERENCE[0],'columns':[{'name':'label','dtype':'opaque'}]}]
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    info=store.register(pd.DataFrame({'label':['A','B'],'__frequency':[200,100]}),
                        source=TABLE,query=envelope['query'],coverage='complete',grain='aggregate',
                        aggregation=envelope['query'],predicate_known=False,
                        snapshot=datetime.now(timezone.utc).isoformat())
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            runtime=GraphAnalysisRuntime(root,'owner','count-resume',model,connection_identity='fixture',
                remote_factory=factory,reference_context_loader=lambda:reference,intent_mode='contract_fixture')
            try:
                failed=runtime.submit(TABLE+' label histogram을 그려줘')
                self.assertEqual(failed['status'],'incomplete',failed)
                self.assertEqual(failed['error_type'],'ReadTimeout')
                before=model.calls
                reference[0]['columns'][0]['dtype']='longtext'
                runtime.context.reference_context=reference
                result=runtime.resume()
                self.assertEqual(result['status'],'answered',result)
                card=runtime.artifacts[runtime.inspect()['recovery']['artifact_ids'][0]]
                self.assertEqual(card.kind,'bar')
                self.assertEqual(card.render_spec['counts'],[200,100])
                self.assertEqual(model.calls,before)
                self.assertEqual(len(queries),1)
            finally:runtime.close()

    def test_chart_followup_uses_recent_values_not_older_numeric_chart_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root)
            original=runtime.datasets.register(pd.DataFrame({'label':['A','A','B',None]*100,
                'reading':[1.,2.,3.,4.]*100}),source=TABLE,coverage='complete',predicate_known=True)
            runtime.select_dataset(original.id)
            digest=stored_dataset_digest(runtime.datasets,original.id)
            try:
                self.assertEqual(runtime.submit('reading histogram을 그려줘')['status'],'answered')
                self.assertEqual(runtime.submit('label은 어떤 값들로 되어 있지 ?')['status'],'answered')
            finally:runtime.close()
            runtime,queries=self.runtime(root)
            try:
                result=runtime.submit('histogram을 그려줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual(state['required_columns'],['label'])
                self.assertEqual(state['model_calls'],0)
                card=runtime.artifacts[state['artifact_ids'][0]]
                self.assertEqual(card.kind,'bar')
                self.assertEqual(card.columns,('label',))
                self.assertEqual(dict(zip(card.render_spec['labels'],card.render_spec['counts'])),{'A':200,'B':100})
                self.assertTrue(card.image.startswith(b'\x89PNG'))
                self.assertEqual(runtime.context.selected_dataset_id,original.id)
                self.assertEqual(stored_dataset_digest(runtime.datasets,original.id),digest)
                self.assertEqual(queries,[])
                self.assertFalse(runtime.recovery._proposed_scope_valid(
                    {'name':'prepare_histogram','args':{'source':TABLE,'column':'reading'}},state))
                self.assertFalse(runtime.recovery._proposed_scope_valid(
                    {'name':'show_chart','args':{'chart_id':next(c.id for c in runtime.artifacts.values() if c.columns==('reading',))}},state))
                self.assertEqual(runtime.submit('reading histogram을 그려줘')['status'],'answered')
                self.assertEqual(runtime.artifacts[runtime.inspect()['recovery']['artifact_ids'][0]].columns,('reading',))
            finally:runtime.close()

    def test_remote_followup_counts_population_not_distinct_rows(self):
        with tempfile.TemporaryDirectory() as root:
            data=pd.DataFrame({'label':['A','A','B',None]*100,'reading':[1.,2.,3.,4.]*100})
            queries=[]
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    tree=validate_query(envelope['query'])
                    if tree.args.get('distinct'):frame=data[['label']].drop_duplicates()
                    elif tree.args.get('group'):
                        column=tree.args['group'].expressions[0].name
                        frame=data.groupby(column).size().rename('__frequency').reset_index()
                    else:frame=data.head(0)
                    info=store.register(frame,source=TABLE,query=envelope['query'],coverage='complete',
                        grain='aggregate' if tree.args.get('group') else 'raw',
                        aggregation=envelope['query'] if tree.args.get('group') else '',
                        predicate_known=False,snapshot=datetime.now(timezone.utc).isoformat())
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            runtime=GraphAnalysisRuntime(root,'owner','remote-focus',NoInference(),remote_factory=factory,
                connection_identity='fixture',reference_context_loader=lambda:[{**REFERENCE[0],
                    'columns':[{'name':'label','dtype':'longtext'},{'name':'reading','dtype':'double'}]}],intent_mode='contract_fixture')
            try:
                self.assertEqual(runtime.propose_query(TABLE,'SELECT * FROM '+TABLE+' LIMIT 0','schema only')['status'],'answered')
                self.assertEqual(runtime.submit('reading histogram을 그려줘')['status'],'answered')
                self.assertEqual(runtime.submit('label은 어떤 값들로 되어 있지 ?')['status'],'answered')
                result=runtime.submit('histogram을 그려줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery'];card=runtime.artifacts[state['artifact_ids'][0]]
                self.assertEqual(card.columns,('label',))
                self.assertEqual(dict(zip(card.render_spec['labels'],card.render_spec['counts'])),{'A':200,'B':100})
                self.assertEqual(len(queries),4)
                self.assertIn('COUNT(*)',queries[-1])
                self.assertEqual(state['model_calls'],0)
                result=runtime.submit('histogram을 다시 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),4)
            finally:runtime.close()

    def test_followup_preserves_value_list_filter_and_does_not_bind_unknown_named_axis(self):
        from langchain_core.messages import HumanMessage
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root)
            try:
                original=runtime.datasets.register(pd.DataFrame({'label':['A','A','B',None]*100,
                    'reading':[1.,2.,3.,4.]*100}),source=TABLE,coverage='complete',predicate_known=True)
                runtime.select_dataset(original.id)
                self.assertEqual(runtime.submit('reading > 2 조건에서 label은 어떤 값들로 되어 있지 ?')['status'],'answered')
                saved=runtime.agent.get_state(runtime.config).values
                trial,_=runtime.recovery._state({**saved,'messages':[*saved['messages'],
                    HumanMessage(content='unknown_axis histogram을 그려줘',id=str(uuid4()))]})
                self.assertNotIn('target_inherited_from',trial)
                self.assertIsNone(runtime.recovery._cached_chart_call(trial))
                result=runtime.submit('histogram을 그려줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery'];card=runtime.artifacts[state['artifact_ids'][0]]
                self.assertEqual(dict(zip(card.render_spec['labels'],card.render_spec['counts'])),{'B':100})
                self.assertEqual(state['scope']['conditions'],[{'column':'reading','op':'gt','value':2}])
                self.assertEqual(queries,[])
            finally:runtime.close()

    def test_saved_timeout_resumes_from_distinct_receipt_without_model_or_db(self):
        from httpx import ReadTimeout
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatResult,ChatGeneration
        class TimeoutModel(NoInference):
            calls:int=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls==1:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[{
                        'name':'query_databricks','id':str(uuid4()),'args':{'source':TABLE,
                        'query':'SELECT DISTINCT label FROM '+TABLE,'reason':'실제 값 목록 확인'}}]))])
                raise ReadTimeout('fixture timeout')
        with tempfile.TemporaryDirectory() as root:
            model=TimeoutModel();runtime,queries=self.runtime(root,model=model)
            try:
                with patch('core.analysis_agent.value_list.requested',return_value=False):
                    failed=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(failed['status'],'incomplete',failed)
                self.assertEqual(failed['error_type'],'ReadTimeout')
                self.assertEqual(model.calls,4)  # one success, then failure + two bounded retries
                before=runtime.inspect()['recovery']
                runtime.model_attempts.sync(before)  # include the three recorded failed attempts
                result=runtime.resume()
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('고유값 3개를 모두',result['text'])
                self.assertEqual(model.calls,4)
                self.assertEqual(len(queries),1)
                after=runtime.inspect()['recovery']
                self.assertEqual(after['model_calls'],before['model_calls'])
                self.assertEqual(after['model_seconds'],before['model_seconds'])
            finally:runtime.close()

    def test_completed_filtered_receipt_is_not_reused_for_a_different_population(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root)
            try:
                result=runtime.propose_query(TABLE,'SELECT DISTINCT label FROM '+TABLE+' WHERE reading > 4',
                    '한 조건의 목록을 확인합니다.')
                self.assertEqual(result['status'],'answered',result)
                self.assertIsNone(retained(runtime.context,TABLE,'label',[]))
                self.assertIsNotNone(retained(runtime.context,TABLE,'label',[{'column':'reading','op':'gt','value':4}]))
                result=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),2)
            finally:runtime.close()

    def runtime(self,root,*,model=None,manual=False,values=None,coverage=None):
        executions=[]
        def factory(store):
            def execute(envelope):
                executions.append(envelope['query'])
                tree=validate_query(envelope['query'])
                self.assertTrue(tree.args.get('distinct'))
                frame=pd.DataFrame({'label':values if values is not None else ['A','B',None]})
                info=store.register(frame,source=TABLE,query=envelope['query'],
                    coverage=coverage or ('unknown' if tree.args.get('limit') else 'complete'),predicate_known=False,
                    snapshot=datetime.now(timezone.utc).isoformat())
                return {'status':'ready','dataset':asdict(info)}
            return execute
        runtime=GraphAnalysisRuntime(root,'owner','values',model or NoInference(),
            connection_identity='fixture',remote_factory=factory,
            reference_context_loader=lambda:REFERENCE,policy=RuntimePolicy(require_remote_approval=manual),intent_mode='contract_fixture')
        return runtime,executions

    def test_completely_fetched_limit_is_not_a_complete_population_list(self):
        for coverage in ('complete','unknown'):
            with self.subTest(coverage=coverage),tempfile.TemporaryDirectory() as root:
                runtime,queries=self.runtime(root,coverage=coverage)
                try:
                    result=runtime.propose_query(TABLE,'SELECT DISTINCT label FROM '+TABLE+' LIMIT 3',
                        '제한된 목록의 실제 조회 결과를 확인합니다.')
                    self.assertEqual(result['status'],'answered',result)
                    result=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                    self.assertEqual(result['status'],'answered',result)
                    self.assertIn('전체 목록이 아닐 수',result['text'])
                    self.assertNotIn('모두 표시',result['text'])
                    self.assertEqual(len(queries),1)
                finally:runtime.close()

    def test_values_are_not_counts_charts_or_scalar_operations(self):
        for text in ['label은 어떤 값들로 되어 있지 ?', 'label 고유값을 보여줘',
                     'show distinct values of label','what values does label have?']:
            self.assertTrue(requested(text),text)
        for text in ['label 고유값 개수를 알려줘','count distinct values of label',
                     'reading 최대값 보여줘','label 빈도를 보여줘']:
            self.assertFalse(requested(text),text)

    def test_remote_values_repeat_restart_filters_and_preserve_raw(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root)
            original=runtime.datasets.register(pd.DataFrame({'label':['keep'],'reading':[3.]}),
                source=TABLE,coverage='unknown',predicate_known=True)
            runtime.select_dataset(original.id)
            digest=stored_dataset_digest(runtime.datasets,original.id)
            try:
                result=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('고유값 3개를 모두',result['text'])
                self.assertIn('NULL',result['text'])
                self.assertEqual(len(queries),1)
                self.assertIn('LIMIT 101',queries[0])
                self.assertEqual(runtime.context.selected_dataset_id,original.id)
                self.assertEqual(stored_dataset_digest(runtime.datasets,original.id),digest)
                repeat=runtime.submit('label 고유값을 다시 보여줘')
                self.assertEqual(repeat['status'],'answered',repeat)
                self.assertEqual(len(queries),1)
            finally:runtime.close()
            reopened,other_queries=self.runtime(root)
            try:
                result=reopened.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(other_queries,[])
                result=reopened.submit('reading > 2 조건에서 label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(other_queries),1)
                self.assertIn('WHERE',other_queries[0])
                self.assertEqual(reopened.context.selected_dataset_id,original.id)
            finally:reopened.close()

    def test_local_complete_original_uses_projection_batches_without_sql(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root)
            try:
                original=runtime.datasets.register(pd.DataFrame({'label':['A','A','B',None]*1000,
                    'reading':[1.,2.,3.,4.]*1000}),source=TABLE,coverage='complete',predicate_known=True)
                runtime.select_dataset(original.id)
                result=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('고유값 3개를 모두',result['text'])
                self.assertEqual(queries,[])
                result=runtime.submit('label 고유값 개수를 알려줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertFalse(runtime.inspect()['recovery'].get('value_list_requested'))
            finally:runtime.close()

    def test_approval_and_truncated_values_are_not_full_population(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root,manual=True,values=[str(i) for i in range(101)])
            try:
                result=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'awaiting_approval',result)
                self.assertEqual(queries,[])
                result=runtime.respond(result['requests'][0]['id'],approved=True)
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('전체 목록이 아닐 수',result['text'])
                self.assertEqual(len(runtime.inspect()['recovery']['value_list_evidence']['values']),100)
            finally:runtime.close()

    def test_unreceipted_wrong_column_or_scope_cannot_complete(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,queries=self.runtime(root)
            try:
                for query in ['SELECT DISTINCT label FROM '+TABLE,
                    "SELECT DISTINCT 'forged' AS label FROM "+TABLE,
                    'SELECT DISTINCT reading FROM '+TABLE,
                    'SELECT DISTINCT label FROM '+TABLE+' WHERE reading > 4']:
                    runtime.datasets.register(pd.DataFrame({'label':['forged']}),source=TABLE,
                        query=query,coverage='complete',predicate_known=False)
                self.assertIsNone(retained(runtime.context,TABLE,'label',[]))
                result=runtime.submit('label은 어떤 값들로 되어 있지 ?')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),1)
                self.assertNotIn('forged',result['text'])
            finally:runtime.close()
