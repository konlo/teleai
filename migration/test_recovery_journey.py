import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from migration.test_persistent_runtime import QuietModel
from core.analysis_agent.runtime import GraphAnalysisRuntime

FIXTURE=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
SOURCE=FIXTURE['source']
COLUMN=next(k for k,v in FIXTURE['rows'][0].items() if isinstance(v,(int,float)))
QUERY=f'SELECT {COLUMN}, COUNT(*) AS frequency FROM {SOURCE} GROUP BY {COLUMN}'

class JourneyModel(QuietModel):
    def _generate(self,messages,**kwargs):
        observations=[]
        for m in messages:
            if isinstance(m,ToolMessage):
                try: observations.append((m.name,json.loads(m.content)))
                except ValueError: pass
        loaded=next((o['dataset']['id'] for name,o in reversed(observations) if name=='query_databricks' and o.get('status')=='ready'),None)
        recovered=any(m.type=='system' and '완료 증거가 없어' in str(m.content) for m in messages)
        rendered=any(name=='render_histogram' and o.get('cards') for name,o in observations)
        if rendered: msg=AIMessage(content='히스토그램을 생성했습니다.')
        elif loaded: msg=AIMessage(content='',tool_calls=[{'name':'render_histogram','args':{'dataset_id':loaded,'value_column':COLUMN,'weight_column':'frequency'},'id':'render'}])
        elif recovered: msg=AIMessage(content='',tool_calls=[{'name':'query_databricks','args':{'source':SOURCE,'query':QUERY,'reason':'히스토그램에 필요한 값별 빈도 조회'},'id':'query'}])
        else: msg=AIMessage(content='이미 히스토그램을 확인했습니다.')  # Deliberate false completion.
        return ChatResult(generations=[ChatGeneration(message=msg)])

class RecoveryJourneyTests(unittest.TestCase):
    def make(self,root,calls,model=None):
        def factory(datasets):
            def execute(envelope):
                calls.append(envelope['query'])
                frame=pd.DataFrame(FIXTURE['rows']).groupby(COLUMN).size().reset_index(name='frequency')
                info=datasets.register(frame,source=SOURCE,query=QUERY,coverage='complete',grain='aggregate',aggregation=QUERY)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        return GraphAnalysisRuntime(root,'owner','journey',model or JourneyModel(),connection_identity='test',remote_factory=factory)

    def test_false_completion_replans_approval_reopen_then_png(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]; r=self.make(root,calls)
            result=r.submit(f'{COLUMN} histogram을 보여줘')
            self.assertEqual(result['status'],'awaiting_approval');self.assertEqual(calls,[])
            self.assertEqual(r.agent.get_state(r.config).values['recovery']['attempts'],1)
            request=result['requests'][0]['id'];r.close()
            r=self.make(root,calls)
            result=r.respond(request,approved=True)
            self.assertEqual(result['status'],'answered',str(result)+r.diagnostics.path.read_text());self.assertEqual(len(calls),1)
            goal=r.agent.get_state(r.config).values['recovery']
            self.assertEqual(goal['status'],'complete');self.assertTrue(goal['artifact_ids'])
            self.assertEqual(r.artifacts[goal['artifact_ids'][0]].kind,'histogram')
            self.assertTrue(r.artifacts[goal['artifact_ids'][0]].image.startswith(b'\x89PNG'))
            self.assertEqual(r.inspect()['state'],'idle')
            self.assertFalse(any(m.content=='이미 히스토그램을 확인했습니다.' for m in r.events()))
            r.close()

    def test_refusal_does_not_execute(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls)
            request=r.submit('histogram')['requests'][0]['id']
            result=r.respond(request,approved=False)
            self.assertEqual(calls,[])
            self.assertEqual(result['status'],'blocked')
            self.assertEqual(r.inspect()['requests'],[])
            r.close()

    def test_no_tool_model_exhausts_bounded_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls,QuietModel())
            result=r.submit('histogram')
            self.assertEqual(result['status'],'exhausted');self.assertEqual(calls,[])
            self.assertEqual(r.agent.get_state(r.config).values['recovery']['attempts'],2)
            self.assertEqual(r.inspect()['state'],'idle');r.close()

class EvidenceTests(unittest.TestCase):
    def test_summary_human_does_not_replace_request_or_chart_obligation(self):
        from langchain_core.messages import HumanMessage
        from utils.analysis_charts import histogram_from_counts
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','compacted-chart',QuietModel())
            frame=pd.DataFrame(FIXTURE['rows']).groupby(COLUMN).size().reset_index(name='frequency')
            info=r.datasets.register(frame,source=SOURCE,coverage='complete',grain='aggregate')
            card=histogram_from_counts(r.datasets,info.id,COLUMN,'frequency')
            r.artifacts[card.id]=card
            r.transcript.record([HumanMessage(content=f'{COLUMN} histogram',id='request')])
            middleware=__import__('core.analysis_agent.recovery',fromlist=['RecoveryMiddleware']).RecoveryMiddleware(r.artifacts,r.diagnostics,transcript=r.transcript)
            observation={'status':'ready','cards':[{'id':card.id,'dataset_id':card.dataset_id,
                'title':card.title,'reason':card.reason,'kind':card.kind,
                'columns':card.columns,'scope':card.scope}]}
            messages=[HumanMessage(content='histogram other_column',id='summary',additional_kwargs={'lc_source':'summarization'}),
                AIMessage(content='',tool_calls=[{'name':'render_histogram','args':{'dataset_id':info.id,'value_column':COLUMN,'weight_column':'frequency'},'id':'compacted-call'}]),
                ToolMessage(content=json.dumps(observation),name='render_histogram',tool_call_id='compacted-call'),
                AIMessage(content='히스토그램을 생성했습니다.',id='answer')]
            result=middleware.after_model({'messages':messages},None)
            self.assertEqual(result['recovery']['status'],'complete')
            self.assertEqual(result['recovery']['artifact_ids'],[card.id])
            self.assertEqual(result['recovery']['request_id'],'request')
            r.close()

    def test_repeated_summaries_cannot_reset_recovery_budget(self):
        from langchain_core.messages import HumanMessage
        from core.analysis_agent.recovery import RecoveryMiddleware
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','summary-budget',QuietModel())
            r.transcript.record([HumanMessage(content='histogram',id='request')])
            middleware=RecoveryMiddleware(r.artifacts,r.diagnostics,transcript=r.transcript)
            recovery={'request_id':'request','attempts':2,'chart':True,'kind':'histogram','columns':[],'failed':{}}
            messages=[HumanMessage(content='histogram',id='new-summary',additional_kwargs={'lc_source':'summarization'}),AIMessage(content='done')]
            result=middleware.after_model({'messages':messages,'recovery':recovery},None)
            self.assertEqual(result['recovery']['status'],'exhausted')
            self.assertEqual(result['recovery']['attempts'],2)
            r.close()

    def test_metadata_after_chart_failure_is_not_completion(self):
        from langchain_core.messages import HumanMessage, ToolMessage
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','evidence',QuietModel())
            middleware=__import__('core.analysis_agent.recovery',fromlist=['RecoveryMiddleware']).RecoveryMiddleware(r.artifacts,r.diagnostics)
            messages=[HumanMessage(content='histogram',id='request'),
                ToolMessage(content='{"status":"no_valid_chart","cards":[]}',name='recommend_chart_images',tool_call_id='chart'),
                ToolMessage(content='{"status":"ready"}',name='inspect_table_context',tool_call_id='schema'),
                AIMessage(content='완료',id='answer')]
            result=middleware.after_model({'messages':messages},None)
            self.assertEqual(result['jump_to'],'model')
            self.assertEqual(result['recovery']['attempts'],1)
            r.close()

    def test_truncated_and_invalid_frequencies_never_render(self):
        from utils.analysis_charts import histogram_from_counts
        from utils.analysis_datasets import DatasetStore
        store=DatasetStore()
        frame=pd.DataFrame(FIXTURE['rows']).groupby(COLUMN).size().reset_index(name='frequency')
        info=store.register(frame,source=SOURCE,coverage='truncated',grain='aggregate')
        with self.assertRaises(ValueError):histogram_from_counts(store,info.id,COLUMN,'frequency')
        frame['frequency']=-1
        info=store.register(frame,source=SOURCE,coverage='complete',grain='aggregate')
        with self.assertRaises(ValueError):histogram_from_counts(store,info.id,COLUMN,'frequency')

class PlanOnlyModel(QuietModel):
    def _generate(self,messages,**kwargs):
        prepared=any(isinstance(m,ToolMessage) and m.name=='prepare_histogram' for m in messages)
        if prepared:message=AIMessage(content='이제 완료했습니다.')
        else:message=AIMessage(content='',tool_calls=[{'name':'prepare_histogram','args':{'source':SOURCE,'column':COLUMN,'where_sql':''},'id':'plan'}])
        return ChatResult(generations=[ChatGeneration(message=message)])

class PlannedJourneyTests(unittest.TestCase):
    def test_controller_connects_plan_approval_and_render_despite_model_stopping(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            def factory(datasets):
                def execute(envelope):
                    calls.append(envelope)
                    frame=pd.DataFrame(FIXTURE['rows']).groupby(COLUMN).size().reset_index(name='__frequency')
                    info=datasets.register(frame,source=SOURCE,coverage='complete',grain='aggregate',query=envelope['query'])
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            r=GraphAnalysisRuntime(root,'owner','plan',PlanOnlyModel(),connection_identity='test',remote_factory=factory)
            r.context.reference_context.append({'table':SOURCE,'columns':[{'name':COLUMN,'dtype':'int64'}]})
            result=r.submit(f'{COLUMN} histogram')
            self.assertEqual(result['status'],'awaiting_approval');self.assertEqual(calls,[])
            query=result['requests'][0]['query'];self.assertIn('COUNT(*)',query);self.assertNotIn('LIMIT',query)
            request_id=result['requests'][0]['id']
            r.close()
            r=GraphAnalysisRuntime(root,'owner','plan',PlanOnlyModel(),connection_identity='test',remote_factory=factory)
            result=r.respond(request_id,approved=True)
            self.assertEqual(result['status'],'answered',str(result)+r.diagnostics.path.read_text());self.assertEqual(len(calls),1)
            goal=r.agent.get_state(r.config).values['recovery']
            self.assertTrue(goal['artifact_ids']);self.assertEqual(goal['status'],'complete')
            r.close()
