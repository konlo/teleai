"""Row display must produce an actual table, without replacing analysis data."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import os,tempfile,unittest
from unittest.mock import patch
import pandas as pd
from langchain_core.messages import AIMessage,HumanMessage
from streamlit.testing.v1 import AppTest
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.row_preview import requested,frame
from tests.test_mysql_metadata_contract import NoInference
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_datasets import stored_dataset_digest

DATA=pd.DataFrame({'reading':range(20),'cohort':['A','B']*10,'event_time':range(20,40)})

def schema(source):
    return [{'table':source,'observed_at':datetime.now(timezone.utc).isoformat(),
             'columns':[{'name':c,'dtype':str(DATA[c].dtype)} for c in DATA.columns]}]

class EmptyModel(EvaluationModel):
    def _generate(self,*args,**kwargs):
        from langchain_core.outputs import ChatGeneration,ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=''))])

class RowPreviewTests(unittest.TestCase):
    def test_count_word_orders_and_inventory_are_distinct(self):
        from core.analysis_agent.remote_completion import table_list_requested
        for prompt in ['observations table row 10개 보여줘','observations table row10개 보여줘',
                       'observations 테이블 행 10개 보여줘','observations records 10 show',
                       'show 10 rows from observations','observations row를 10개만 보여줘',
                       'observations table에서 레코드 10건 보여줘','observations 데이타 10row만 보여줘',
                       'observations 데이터 10행 보여줘']:
            with self.subTest(prompt=prompt):
                self.assertEqual(requested(prompt),{'limit':10,'question':''})
                self.assertFalse(table_list_requested(prompt))
        for prompt in ['tables 목록 보여줘','테이블 10개 보여줘','어떤 table이 있어?']:
            self.assertIsNone(requested(prompt))
            self.assertTrue(table_list_requested(prompt))
        for prompt in ['observations table row -10개 보여줘','observations table row 0개 보여줘',
                       'observations table row 201개 보여줘']:
            self.assertTrue(requested(prompt)['question'])
            self.assertFalse(table_list_requested(prompt))
        self.assertIsNone(requested('row_id 10개 보여줘'))
        self.assertIsNone(requested('records_archive 10개 보여줘'))
        self.assertFalse(table_list_requested('observations table row 수 보여줘'))

    def test_local_bounded_prefix_preserves_data_selection_and_restarts(self):
        for dialect in ['mysql','databricks']:
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as root:
                source='lab.observations' if dialect=='mysql' else 'catalog.lab.observations'
                r=GraphAnalysisRuntime(root,'owner','local',NoInference(),sql_dialect=dialect,
                    reference_context_loader=lambda:schema(source),summary_trigger_tokens=1,intent_mode='contract_fixture')
                try:
                    raw=r.datasets.register(DATA,source=source,coverage='complete',predicate_known=True)
                    agg=r.datasets.register(pd.DataFrame({'reading':[1],'__frequency':[20]}),source=source,
                        grain='aggregate',coverage='complete',query=f'SELECT reading,COUNT(*) AS __frequency FROM {source} GROUP BY reading')
                    r.select_dataset(agg.id)
                    digest=stored_dataset_digest(r.datasets,raw.id)
                    for prompt in ['데이타 10개 row만 추출해서 table로 보여줘','데이터 10행을 표로 보여줘','show 10 rows as a table',
                                   'observations table row 10개 보여줘','observations table rows 10 보여줘']:
                        result=r.submit(prompt)
                        self.assertEqual(result['status'],'answered',result)
                        proof=r.events()[-1].additional_kwargs['analysis_table_preview']
                        pd.testing.assert_frame_equal(frame(r.datasets,proof),DATA.head(10))
                        self.assertEqual(r.inspect()['recovery']['model_calls'],0)
                        self.assertEqual(r.context.selected_dataset_id,agg.id)
                        self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                        self.assertEqual(len(r.datasets.metadata),2)
                    r.close()
                    r=GraphAnalysisRuntime(root,'owner','local',NoInference(),sql_dialect=dialect,
                        reference_context_loader=lambda:schema(source),summary_trigger_tokens=1,intent_mode='contract_fixture')
                    self.assertEqual(r.submit('데이타 10개 row만 추출해서 table로 보여줘')['status'],'answered')
                finally:r.close()

    def test_aggregate_is_not_raw_preview_remote_query_is_bounded_and_reused(self):
        for dialect in ['mysql','databricks']:
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as root:
                source='lab.observations' if dialect=='mysql' else 'catalog.lab.observations'
                calls=[]
                def factory(store):
                    def execute(envelope):
                        calls.append(envelope['query'])
                        expected='SELECT * FROM '+'.'.join('`'+p+'`' for p in source.split('.'))+' LIMIT 10'
                        self.assertEqual(envelope['query'],expected)
                        info=store.register(DATA.head(10),source=source,query=envelope['query'],
                            coverage='sampled',predicate_known=False)
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                r=GraphAnalysisRuntime(root,'owner','remote',NoInference(),sql_dialect=dialect,
                    reference_context_loader=lambda:schema(source),connection_identity='fixture',remote_factory=factory,intent_mode='contract_fixture')
                try:
                    agg=r.datasets.register(pd.DataFrame({'reading':[1],'__frequency':[20]}),source=source,
                        grain='aggregate',coverage='complete',query=f'SELECT reading,COUNT(*) AS __frequency FROM {source} GROUP BY reading')
                    r.select_dataset(agg.id)
                    result=r.submit('데이타 10개 row만 추출해서 table로 보여줘')
                    self.assertEqual(result['status'],'answered',result)
                    proof=r.events()[-1].additional_kwargs['analysis_table_preview']
                    self.assertEqual(proof['columns'],list(DATA.columns))
                    self.assertEqual(proof['rows'],10)
                    self.assertNotEqual(proof['dataset_id'],agg.id)
                    self.assertEqual(len(calls),1)
                    self.assertEqual(r.context.selected_dataset_id,agg.id)
                    self.assertEqual(r.submit('데이터 10행을 표로 보여줘')['status'],'answered')
                    self.assertEqual(len(calls),1)
                finally:r.close()

    def test_current_result_only_never_loads_raw_and_empty_reply_cannot_complete(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','current',NoInference(),reference_context_loader=lambda:schema('lab.observations'),intent_mode='contract_fixture')
            try:
                info=r.datasets.register(DATA.head(4),source='lab.observations',grain='aggregate')
                r.select_dataset(info.id)
                result=r.submit('현재 결과의 10개 row를 table로 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.events()[-1].additional_kwargs['analysis_table_preview']['rows'],4)
                self.assertEqual(len(r.datasets.metadata),1)
            finally:r.close()
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','empty',EmptyModel(),intent_mode='contract_fixture')
            try:
                result=r.submit('안녕하세요')
                self.assertNotEqual(result['status'],'answered')
                self.assertTrue(result['text'].strip())
            finally:r.close()

    def test_invalid_limits_and_ambiguous_table_do_not_query(self):
        for limit in [-10,0,201]:
            self.assertTrue(requested(f'{limit}개 row를 표로 보여줘')['question'])
        self.assertIsNone(requested('histogram을 10개 그려줘'))
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','ambiguous',NoInference(),reference_context_loader=lambda:
                schema('lab.observations')+schema('lab.another'),intent_mode='contract_fixture')
            try:
                result=r.submit('데이터 10개 row를 표로 보여줘')
                self.assertIn('어느 테이블',result['text'])
                self.assertFalse(r.inspect()['recovery'].get('table_preview_evidence'))
                self.assertEqual(len(r.datasets.metadata),0)
            finally:r.close()

    def test_missing_source_connection_is_reported_without_a_model_loop(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','disconnected',NoInference(),
                reference_context_loader=lambda:schema('lab.observations'),intent_mode='contract_fixture')
            try:
                result=r.submit('데이터 10개 row를 표로 보여줘')
                self.assertEqual(result['status'],'blocked',result)
                self.assertIn('데이터베이스 조회 연결도 없습니다',result['text'])
                self.assertEqual(r.inspect()['recovery']['stop_reason'],'row_preview_unavailable')
                self.assertEqual(r.inspect()['recovery']['model_calls'],0)
                self.assertEqual(len(r.datasets.metadata),0)
            finally:r.close()

    def test_file_backed_preview_does_not_materialize_whole_frame(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','bounded',NoInference(),
                reference_context_loader=lambda:schema('lab.observations'),
                policy=RuntimePolicy(max_full_read_bytes=1024),intent_mode='contract_fixture')
            try:
                original=pd.concat([DATA]*1000,ignore_index=True)
                info=r.datasets.register(original,source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(info.id)
                digest=stored_dataset_digest(r.datasets,info.id)
                with patch.object(type(r.datasets.frames),'__getitem__',side_effect=AssertionError('No full decode')):
                    result=r.submit('데이터 10개 row를 표로 보여줘')
                    self.assertEqual(result['status'],'answered',result)
                    proof=r.events()[-1].additional_kwargs['analysis_table_preview']
                    pd.testing.assert_frame_equal(frame(r.datasets,proof),original.head(10))
                self.assertEqual(stored_dataset_digest(r.datasets,info.id),digest)
                altered={**proof,'source':'different.table'}
                with self.assertRaises(ValueError):frame(r.datasets,altered)
            finally:r.close()

    def test_manual_approval_rejection_and_failed_model_checkpoint_preserve_original(self):
        with tempfile.TemporaryDirectory() as root:
            called=[]
            r=GraphAnalysisRuntime(root,'owner','reject',NoInference(),sql_dialect='mysql',
                reference_context_loader=lambda:schema('lab.observations'),connection_identity='fixture',
                remote_factory=lambda store:lambda envelope:called.append(envelope),
                policy=RuntimePolicy(require_remote_approval=True),intent_mode='contract_fixture')
            try:
                agg=r.datasets.register(DATA.head(1),source='lab.observations',grain='aggregate')
                r.select_dataset(agg.id)
                result=r.submit('데이터 10개 row를 table로 보여줘')
                self.assertEqual(result['status'],'awaiting_approval',result)
                result=r.respond(result['requests'][0]['id'],approved=False)
                self.assertEqual(result['status'],'blocked',result)
                self.assertEqual(called,[])
                self.assertEqual(r.context.selected_dataset_id,agg.id)
            finally:r.close()
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','paused',NoInference(),reference_context_loader=lambda:schema('lab.observations'),intent_mode='contract_fixture')
            try:
                raw=r.datasets.register(DATA,source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                human=HumanMessage(id='old-preview',content='데이타 10개 row만 추출해서 table로 보여줘')
                r.agent.update_state(r.config,{'messages':[human],'recovery':{'request_id':human.id,
                    'status':'working','model_calls':3,'model_seconds':200.,'required_columns':[]}},
                    as_node='ObservedSummarizationMiddleware.before_model')
                r.close()
                r=GraphAnalysisRuntime(root,'owner','paused',NoInference(),reference_context_loader=lambda:schema('lab.observations'),intent_mode='contract_fixture')
                self.assertEqual(r.resume()['status'],'answered')
                self.assertEqual(r.events()[-1].additional_kwargs['analysis_table_preview']['rows'],10)
                self.assertEqual(r.inspect()['recovery']['model_calls'],3)
            finally:r.close()

    def test_streamlit_displays_ten_rows_all_columns_as_native_table(self):
        from tests.test_llm_goal import GoalModel, goal
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root,'TELLY_DATA_BACKEND':'databricks'}),patch(
            'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=GoalModel(goals=[goal('row_preview',{'limit':10},sources=['catalog.lab.observations'])])):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=30).run()
            r=app.session_state['v1_runtime']
            try:
                r.context.reference_context[:]=schema('catalog.lab.observations')
                r.reference_context_loader=lambda:schema('catalog.lab.observations')
                info=r.datasets.register(DATA,source='catalog.lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(info.id)
                app.run()
                app.chat_input[0].set_value('데이타 10개 row만 추출해서 table로 보여줘').run()
                self.assertFalse(app.exception)
                displayed=[d.value for d in app.main.get('dataframe')]
                self.assertEqual(len(displayed),1)
                pd.testing.assert_frame_equal(displayed[0],DATA.head(10))
            finally:r.close()
