"""Partial row display -> explicit axes -> image -> restart, without inference."""
from dataclasses import asdict
import json,tempfile,unittest,os
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from pathlib import Path
import pandas as pd
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from core.analysis_agent.completion import active_contracts
from tests.test_mysql_metadata_contract import NoInference
from tests.test_row_preview import schema
from utils.analysis_datasets import stored_dataset_digest

CASE=json.loads((Path(__file__).parent/'fixtures/chart_display_journeys.json').read_text())
DATA=pd.DataFrame({CASE['columns'][0]:range(10),CASE['columns'][1]:['A','B']*5,CASE['columns'][2]:range(10,20)})

class ChartDisplayJourneyTests(unittest.TestCase):
    def make(self,root,dialect='mysql',calls=None):
        source=CASE['source'] if dialect=='mysql' else 'catalog.'+CASE['source']
        def factory(store):
            def execute(env):
                calls.append(env['query'])
                info=store.register(DATA,source=source,query=env['query'],coverage='sampled',predicate_known=False)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        context=lambda:[{**schema(source)[0],'columns':[{'name':c,'dtype':str(DATA[c].dtype)} for c in DATA]}]
        return GraphAnalysisRuntime(root,'owner','journey',NoInference(),sql_dialect=dialect,
            reference_context_loader=context,summary_trigger_tokens=1,connection_identity='fixture',
            remote_factory=factory if calls is not None else None,intent_mode='contract_fixture')

    def seed(self,r):
        source=CASE['source'] if r.sql_dialect=='mysql' else 'catalog.'+CASE['source']
        agg=r.datasets.register(pd.DataFrame({CASE['columns'][0]:[1],'__frequency':[10]}),source=source,
            grain='aggregate',coverage='complete',query=f'SELECT {CASE["columns"][0]}, COUNT(*) AS __frequency FROM {source} GROUP BY {CASE["columns"][0]}')
        r.select_dataset(agg.id)
        return agg

    def test_partial_preview_scatter_reverse_repeat_and_restart(self):
        for dialect in ['mysql','databricks']:
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as root:
                calls=[];r=self.make(root,dialect,calls)
                try:
                    agg=self.seed(r)
                    self.assertEqual(r.submit(CASE['preview_prompt'])['status'],'answered')
                    proof=r.events()[-1].additional_kwargs['analysis_table_preview']
                    digest=stored_dataset_digest(r.datasets,proof['dataset_id'])
                    for prompt,axes in [(CASE['scatter_prompt'],CASE['columns'][::2]),(CASE['reverse_prompt'],CASE['columns'][::2][::-1]),(CASE['relationship_prompt'],CASE['columns'][::2])]:
                        result=r.submit(prompt)
                        self.assertEqual(result['status'],'answered',result)
                        cur=r.inspect()['recovery'];card=r.artifacts[cur['artifact_ids'][0]]
                        self.assertEqual([card.render_spec['x'],card.render_spec['y']],axes)
                        self.assertEqual(card.dataset_id,proof['dataset_id'])
                        self.assertEqual(cur['model_calls'],0)
                        self.assertEqual(r.context.selected_dataset_id,agg.id)
                    r.close();r=self.make(root,dialect,calls)
                    self.assertEqual(r.submit(CASE['scatter_prompt'])['status'],'answered')
                    self.assertEqual(len(calls),1)
                    self.assertEqual(stored_dataset_digest(r.datasets,proof['dataset_id']),digest)
                finally:r.close()

    def test_saved_image_is_finished_on_resume_even_when_model_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls=calls)
            try:
                self.seed(r);r.submit(CASE['preview_prompt'])
                confirmed=r.inspect()['recovery']['confirmed_analysis']
                info=confirmed['table_preview_evidence']['dataset_id']
                human=HumanMessage(id='interrupted-scatter',content=CASE['scatter_prompt'])
                cur,_=r.recovery._state({'messages':r.agent.get_state(r.config).values['messages']+[human],
                    'recovery':r.inspect()['recovery']})
                args={'dataset_id':info,'kind':'scatter','x':CASE['columns'][0],'y':CASE['columns'][2]}
                result=next(t for t in local_tools(r.context) if t.name=='render_chart_spec').invoke(args)
                if isinstance(result,str):result=json.loads(result)
                ai=AIMessage(content='',tool_calls=[{'id':'ready-chart','name':'render_chart_spec','args':args}])
                tool=ToolMessage(content=json.dumps(result),tool_call_id='ready-chart',name='render_chart_spec')
                cur.update(model_calls=4,model_seconds=200.,processed=['ready-chart'])
                r.agent.update_state(r.config,{'messages':[human,ai,tool],'recovery':cur},as_node='ObservedSummarizationMiddleware.before_model')
                r.close();r=self.make(root,calls=calls)
                before=len(r.artifacts)
                outcome=r.resume()
                self.assertEqual(outcome['status'],'answered',outcome)
                self.assertEqual(len(r.artifacts),before)
                self.assertEqual(len(calls),1)
            finally:r.close()

    def test_display_binding_rejects_widening_stale_source_and_changed_selection(self):
        from core.analysis_agent.chart_binding import bind,axes_match,raw_scatter_eligible
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls=calls)
            try:
                self.seed(r);r.submit(CASE['preview_prompt'])
                human=HumanMessage(id='followup',content=CASE['scatter_prompt'])
                cur,_=r.recovery._state({'messages':r.events()+[human],'recovery':r.inspect()['recovery']})
                self.assertTrue(cur.get('display_dataset_id'))
                proof=cur['chart_display_evidence'];info=r.datasets.metadata[proof['dataset_id']]
                self.assertFalse(axes_match(cur,{'x':CASE['columns'][2],'y':CASE['columns'][0]}))
                self.assertFalse(raw_scatter_eligible(info,{'current_result_only':False}))
                for changed in [
                    {'text':CASE['scatter_prompt']+' 전체 데이터'},
                    {'fresh_source_required':True},
                    {'required_sources':['lab.another']},
                    {'scope':{'conditions':[{'column':CASE['columns'][0],'op':'ge','value':4}]}},
                    {'confirmed_analysis':{**cur['confirmed_analysis'],'table_preview_evidence':{**proof,'snapshot':'stale'}}},
                    {'confirmed_analysis':{**cur['confirmed_analysis'],'selection_at_confirmation':'different'}},
                ]:
                    with self.subTest(changed=list(changed)):
                        candidate={**cur,'text':CASE['scatter_prompt'],**changed}
                        binding=bind(r.context,candidate)
                        self.assertFalse(binding.get('display_dataset_id'))
                        self.assertFalse(binding.get('current_result_only'))
                self.assertFalse(r.recovery._chart_scope_valid(info,
                    {'kind':'scatter','x':CASE['columns'][2],'y':CASE['columns'][0]},cur))
                self.assertEqual(len(calls),1)
            finally:r.close()

    def test_web_table_followup_renders_png_and_completes(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root,'TELLY_DATA_BACKEND':'databricks'}),patch(
                'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=NoInference()):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=30).run()
            r=app.session_state['v1_runtime']
            r.recovery.intent_mode = 'contract_fixture'  # UI execution/rendering fixture, not an intent score.
            try:
                source='catalog.'+CASE['source']
                context=[{**schema(source)[0],'columns':[{'name':c,'dtype':str(DATA[c].dtype)} for c in DATA]}]
                r.context.reference_context[:]=context;r.reference_context_loader=lambda:context
                raw=r.datasets.register(DATA,source=source,coverage='unknown',predicate_known=False,
                    query=f'SELECT * FROM {source} LIMIT 10')
                agg=r.datasets.register(DATA.head(1),source=source,grain='aggregate')
                r.select_dataset(agg.id);app.run()
                for prompt in [CASE['preview_prompt'],CASE['scatter_prompt']]:
                    app.chat_input[0].set_value(prompt).run()
                    self.assertFalse(app.exception)
                self.assertEqual(r.inspect()['recovery']['status'],'complete')
                self.assertEqual(r.context.selected_dataset_id,agg.id)
                self.assertEqual(len(app.get('image')),1)
                self.assertTrue(app.get('image')[0].proto.imgs[0].url)
                self.assertFalse(app.error)
                self.assertEqual(r.inspect()['recovery']['model_calls'],0)
            finally:r.close()
