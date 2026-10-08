"""Meaning questions and target corrections preserve the live schema subject."""
from datetime import datetime,timezone
import tempfile
import unittest
import pandas as pd
from copy import deepcopy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.recovery import _column_listing_requested
from core.analysis_agent.schema_questions import corrective_goal,type_question
from langchain_core.messages import HumanMessage
from tests.test_actual_agent_evaluation import EvaluationModel


class SchemaQuestionContextTests(unittest.TestCase):
    def test_schema_role_is_not_column_list_or_row_disjunction(self):
        self.assertFalse(_column_listing_requested('여기에 time or date에 관련된 항목이 있을까 ? column들을 확인해줘'))
        self.assertFalse(_column_listing_requested('여기 column 중 시간 관련 column을 확인해줘'))
        self.assertTrue(_column_listing_requested('이 테이블의 column들 다시 보여줘'))
        msgs=[HumanMessage(id='prior',content='여기에 날짜 관련 컬럼이 있을까?'),
              HumanMessage(id='current',content='other 이 테이블에서 찾아줘야지')]
        goal=corrective_goal(msgs[-1].content,msgs,'current')
        self.assertIn('날짜 관련',goal)
        self.assertIn('other',goal)

    def test_metadata_subject_then_semantic_question_and_target_correction(self):
        with tempfile.TemporaryDirectory() as root:
            data=pd.DataFrame({'recorded_on':pd.to_datetime(['2025-01-01','2025-01-02']),'reading':[1,2]})
            sources=['lab.old','lab.sensor','lab.other']
            schema=[{'table':s,'observed_at':datetime.now(timezone.utc).isoformat(),
                'columns':[{'name':c,'dtype':str(data[c].dtype)} for c in data.columns]} for s in sources]
            model=EvaluationModel(calls=[{'name':'inspect_table_context','args':{'table':'lab.sensor'}}],
                answer='recorded_on은 실제 스키마에서 datetime 타입입니다.')
            r=GraphAnalysisRuntime(root,'owner','meaning',model,sql_dialect='mysql',reference_context_loader=lambda:schema,intent_mode='contract_fixture')
            try:
                old=r.datasets.register(data,source='lab.old',coverage='complete',predicate_known=True)
                r.select_dataset(old.id)
                self.assertEqual(r.submit('sensor 컬럼 목록 보여줘')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['required_sources'],['lab.sensor'])
                result=r.submit('여기에 time or date에 관련된 항목이 있을까 ? column들을 확인해줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertIsNone(state['metadata_kind'])
                self.assertEqual(state['required_sources'],['lab.sensor'])
                self.assertNotIn('unsupported_disjunction',state['scope']['unresolved'])
                model.position=0
                model.calls=[{'name':'inspect_table_context','args':{'table':'lab.other'}}]
                result=r.submit('other 이 테이블에서 찾아줘야지')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertIn('time or date',state['request_text'])
                self.assertEqual(state['required_sources'],['lab.other'])
                self.assertEqual(state['scope']['sources'],['lab.other'])
                self.assertEqual(r.context.selected_dataset_id,old.id)
                for prompt in ['table schema에 data type이 있지 않아 ?',
                               '테이블 스키마의 자료형을 알려줘','이 테이블 컬럼별 data type 보여줘']:
                    result=r.submit(prompt)
                    self.assertEqual(result['status'],'answered',result)
                    self.assertIn('lab.other',result['text'])
                    self.assertNotIn('lab.old',result['text'])
                    state=r.inspect()['recovery']
                    self.assertEqual(state['metadata_kind'],'dtypes')
                    self.assertEqual(state['required_sources'],['lab.other'])
                    self.assertEqual(state['model_calls'],0)
                    self.assertEqual(r.context.selected_dataset_id,old.id)
                r.close()
                r=GraphAnalysisRuntime(root,'owner','meaning',model,sql_dialect='mysql',reference_context_loader=lambda:schema,intent_mode='contract_fixture')
                self.assertIn('lab.other',r.submit('table schema에 data type이 있지 않아 ?')['text'])
                # An explicit UI selection supersedes the conversation subject.
                other=r.datasets.register(data,source='lab.sensor',coverage='complete',predicate_known=True)
                r.select_dataset(other.id)
                self.assertIn('lab.sensor',r.submit('테이블 컬럼 타입 목록 보여줘')['text'])
            finally:r.close()

    def test_legacy_semantic_observation_binds_types_without_trusting_prose(self):
        from langchain_core.messages import AIMessage,ToolMessage
        import json
        from core.analysis_agent.schema_questions import prior_subject
        with tempfile.TemporaryDirectory() as root:
            contexts=[{'table':s,'observed_at':datetime.now(timezone.utc).isoformat(),
                'columns':[{'name':'reading','dtype':'double'}]} for s in ['lab.old','lab.sensor']]
            r=GraphAnalysisRuntime(root,'owner','legacy-types',EvaluationModel(),sql_dialect='mysql',
                reference_context_loader=lambda:contexts,intent_mode='contract_fixture')
            try:
                agg=r.datasets.register(pd.DataFrame({'reading':[1],'__frequency':[3]}),
                    source='lab.old',grain='aggregate',coverage='complete',predicate_known=True)
                r.select_dataset(agg.id)
                messages=[HumanMessage(id='semantic',content='sensor 이 테이블에서 시간 관련 column 찾아줘'),
                    AIMessage(content='',tool_calls=[{'name':'inspect_table_context','args':{'table':'lab.sensor'},'id':'proof'}]),
                    ToolMessage(name='inspect_table_context',tool_call_id='proof',content=json.dumps({
                        'status':'ready','authority':'saved_snapshot','table_context':contexts[1]})),
                    AIMessage(content='lab.old가 현재 테이블입니다.'),
                    HumanMessage(id='types',content='table schema에 data type이 있지 않아 ?')]
                current,_=r.recovery._state({'messages':messages,'recovery':{
                    'request_id':'semantic','status':'complete','required_sources':[],
                    'confirmed_analysis':{'status':'complete','required_sources':['lab.old']}}})
                self.assertEqual(current['metadata_kind'],'dtypes')
                self.assertEqual(current['required_sources'],['lab.sensor'])
                self.assertFalse(r.recovery._proposed_scope_valid({'name':'inspect_dataset','args':{'dataset_id':agg.id}},current))
                self.assertFalse(r.recovery._proposed_scope_valid({'name':'inspect_table_context','args':{'table':'lab.old'}},current))
                self.assertIsNone(prior_subject({},[messages[3]],r.context))
                self.assertTrue(type_question(messages[-1].content))
                self.assertFalse(type_question('data type별 평균을 계산해줘'))
                # A later completed analysis has its own source and supersedes
                # the inherited schema-only topic without changing UI selection.
                later=deepcopy(current)
                later.update(request_id='analysis',status='complete',metadata_kind=None,
                    calculation=True,required_sources=['lab.old'],
                    selection_at_confirmation=agg.id)
                new_messages=messages[:-1]+[HumanMessage(id='after-analysis',content='table schema에 data type이 있지 않아 ?')]
                after,_=r.recovery._state({'messages':new_messages,'recovery':later})
                self.assertEqual(after['required_sources'],['lab.old'])
            finally:r.close()

    def test_database_types_are_distinct_from_loaded_frame_storage(self):
        from core.analysis_catalog import resolve_table_context
        from datetime import timedelta
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','db-types',EvaluationModel(),sql_dialect='mysql',intent_mode='contract_fixture')
            try:
                raw=r.datasets.register(pd.DataFrame({'captured_on':['2025-01-01'],'device_key':['1']}),
                    source='lab.events',query='SELECT * FROM lab.events LIMIT 1',coverage='unknown',
                    snapshot=datetime.now(timezone.utc).isoformat())
                later=(datetime.now(timezone.utc)+timedelta(seconds=2)).isoformat()
                context={'table':'lab.events','training_status':'observed_schema','observed_at':later,
                    'columns':[{'name':'captured_on','dtype':'date'},{'name':'device_key','dtype':'varchar'}]}
                r.context.reference_context=[context]
                proof=resolve_table_context([context],r.datasets,'lab.events')
                self.assertEqual(proof['table_context']['columns'][0]['dtype'],'object')
                self.assertEqual(proof['table_context']['columns'][0]['database_dtype'],'date')
                self.assertFalse(proof['schema_changed'])
                result=r.submit('lab.events table schema의 data type 알려줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('captured_on: date',result['text'])
                self.assertIn('현재 DB 메타데이터',result['text'])
                for bad in [dict(context,training_status='trained'),
                            dict(context,observed_at='2020-01-01T00:00:00+00:00'),
                            dict(context,columns=[{'name':'different','dtype':'date'}])]:
                    checked=resolve_table_context([bad],r.datasets,'lab.events')
                    self.assertFalse(checked.get('database_type_authority'))
            finally:r.close()
