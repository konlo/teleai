"""Run new obligations through the actual graph; scripted model is not an LLM score."""
import tempfile
import unittest
import pandas as pd
from tests.test_llm_goal import GoalModel,goal
from tests.test_analysis_extensions import rolling_contract
from core.analysis_agent.runtime import GraphAnalysisRuntime
from utils.analysis_datasets import stored_dataset_digest

class ExtensionAgentJourneyTests(unittest.TestCase):
    def test_small_eda_cap_and_exact_received_argument_support_graph_repair(self):
        import json
        for repeated in (False, True):
            with self.subTest(repeated=repeated),tempfile.TemporaryDirectory() as root:
                plan=goal('advanced_eda',{'kind':'correlation_heatmap'},columns=['reading','second'])
                args={'dataset_id':'$fixture','columns':['reading','second'],'kind':'correlation_heatmap','max_points':'two'}
                wrong={'name':'render_advanced_eda','args':args}
                fixed={'name':'render_advanced_eda','args':{**args,'max_points':2}}
                model=GoalModel(goals=[plan],calls=[wrong,wrong if repeated else fixed])
                runtime=GraphAnalysisRuntime(root,'extensions','small-eda',model,sql_dialect='mysql')
                raw=runtime.datasets.register(pd.DataFrame({'reading':[1.,2.,3.,4.],'second':[2.,4.,6.,8.]}),
                    source='lab.observations',coverage='complete',predicate_known=True)
                runtime.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
                before=stored_dataset_digest(runtime.datasets,raw.id)
                try:
                    result=runtime.submit('Draw the correlation heatmap of these four retained rows without reloading.')
                    messages=runtime.agent.get_state(runtime.config).values['messages']
                    feedback=[m for m in messages if m.additional_kwargs.get('lc_source')=='proposal_preflight']
                    detail=json.loads(feedback[-1].content.split(': ',1)[1])
                    issue=next(i for i in detail['validation_issues'] if i['path']==['max_points'])
                    self.assertEqual(issue['received'],'two');self.assertEqual(issue['expected'],'integer')
                    self.assertNotIn('local_analysis_sql',detail['message'])
                    if repeated:
                        self.assertEqual(result['status'],'blocked',result)
                        self.assertIn('도구 입력',str(result))
                        self.assertNotIn('생성된 SQL',str(result))
                        self.assertEqual(len(runtime.artifacts),0)
                    else:
                        self.assertEqual(result['status'],'answered',result)
                        proof=runtime.inspect()['recovery']['advanced_eda_evidence']['statistics']
                        self.assertEqual(proof['total_rows'],4)
                        self.assertEqual(proof['correlation']['reading']['second'],1.)
                        self.assertEqual(proof['pairwise_valid_counts']['reading']['second'],4)
                    self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),before)
                finally:runtime.close()

    def test_python_contract_error_is_repaired_without_reloading(self):
        with tempfile.TemporaryDirectory() as root:
            plan=goal('custom_analysis',{'description':'Rolling mean','result_contract':rolling_contract('reading')},columns=['reading'])
            model=GoalModel(goals=[plan],calls=[
                {'name':'execute_analysis_python','args':{'dataset_id':'$fixture','columns':['reading'],'code':'import pandas as pd\nresult = df'}},
                {'name':'execute_analysis_python','args':{'dataset_id':'$fixture','columns':['reading'],
                    'code':"result = pd.DataFrame({'rolling_mean': df['reading'].rolling(2).mean()})"}}])
            runtime=GraphAnalysisRuntime(root,'extensions','repair',model,sql_dialect='mysql',
                reference_context_loader=lambda:[{'table':'lab.observations','columns':[{'name':'reading','dtype':'double'}]}])
            raw=runtime.datasets.register(pd.DataFrame({'reading':[1.,2.,3.,4.]}),source='lab.observations',coverage='complete',predicate_known=True)
            runtime.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            digest=stored_dataset_digest(runtime.datasets,raw.id)
            try:
                result=runtime.submit('Calculate a two-row rolling mean using custom Python.')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),digest)
                self.assertEqual(len(runtime.datasets.metadata),2)
                self.assertTrue(any(getattr(m,'name',None)=='execute_analysis_python' and
                    'python_contract_violation' in str(m.content) for m in runtime.agent.get_state(runtime.config).values['messages']))
            finally:runtime.close()
    def test_projected_row_request_executes_only_requested_columns(self):
        from core.analysis_agent.row_preview import frame
        from dataclasses import asdict
        for dialect in ('mysql','databricks'):
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as root:
                source='lab.observations' if dialect=='mysql' else 'catalog.lab.observations'
                plan=goal('row_preview',{'limit':10},sources=[source],columns=['reading'])
                executed=[]
                def factory(store):
                    def execute(envelope):
                        executed.append(envelope['query'])
                        info=store.register(pd.DataFrame({'reading':range(10)}),source=source,
                            query=envelope['query'],coverage='sampled',predicate_known=True)
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                runtime=GraphAnalysisRuntime(root,'extensions','projection',GoalModel(goals=[plan]),sql_dialect=dialect,
                    connection_identity='fixture',remote_factory=factory,
                    reference_context_loader=lambda:[{'table':source,'columns':[{'name':'reading','dtype':'double'},
                        {'name':'extra','dtype':'string'}]}])
                try:
                    result=runtime.submit('Show only reading, ten rows.')
                    self.assertEqual(result['status'],'answered',result)
                    self.assertEqual(len(executed),1)
                    self.assertIn('SELECT `reading` FROM',executed[0])
                    self.assertEqual(list(frame(runtime.datasets,runtime.inspect()['recovery']['table_preview_evidence']).columns),['reading'])
                finally:runtime.close()
    def test_custom_python_completed_then_restart_export_restores_original(self):
        with tempfile.TemporaryDirectory() as root:
            plan=goal('custom_analysis',{'description':'Calculate rolling mean','result_contract':rolling_contract('reading')},columns=['reading'])
            model=GoalModel(goals=[plan],calls=[{'name':'execute_analysis_python','args':{
                'dataset_id':'$fixture','columns':['reading'],
                'code':"result = pd.DataFrame({'rolling_mean': df['reading'].rolling(2).mean()})"}}])
            schema=lambda:[{'table':'lab.observations','columns':[{'name':'reading','dtype':'double'}]}]
            runtime=GraphAnalysisRuntime(root,'extensions','session',model,sql_dialect='mysql',reference_context_loader=schema)
            raw=runtime.datasets.register(pd.DataFrame({'reading':[1.,2.,3.,4.]}),source='lab.observations',coverage='complete',predicate_known=True)
            runtime.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            digest=stored_dataset_digest(runtime.datasets,raw.id)
            try:
                result=runtime.submit('Compute a two-row rolling mean with custom Python on reading.')
                self.assertEqual(result['status'],'answered',result)
                proof=runtime.inspect()['recovery']['custom_analysis_evidence']
                self.assertEqual(runtime.datasets.frames[proof['output_dataset_id']]['rolling_mean'].dropna().tolist(),[1.5,2.5,3.5])
                self.assertEqual(runtime.plans.inspect(runtime.inspect()['recovery']['request_id'])['plan']['tasks'][0]['status'],'verified')
            finally:runtime.close()
            export_goal=goal('export',{'format':'csv'},columns=['reading'])
            model=GoalModel(goals=[export_goal],calls=[{'name':'export_analysis_result','args':{'dataset_id':raw.id,'format':'csv'}}])
            restored=GraphAnalysisRuntime(root,'extensions','session',model,sql_dialect='mysql',reference_context_loader=schema)
            try:
                result=restored.submit('Export the saved original data as CSV.')
                self.assertEqual(result['status'],'answered',result)
                self.assertTrue(restored.inspect()['recovery']['export_evidence'])
                self.assertEqual(stored_dataset_digest(restored.datasets,raw.id),digest)
                self.assertEqual(restored.context.selected_dataset_id,raw.id)
            finally:restored.close()
    def test_advanced_eda_graph_attaches_actual_verified_image(self):
        with tempfile.TemporaryDirectory() as root:
            plan=goal('advanced_eda',{'kind':'correlation_heatmap'},columns=['reading','second'])
            model=GoalModel(goals=[plan],calls=[{'name':'render_advanced_eda','args':{
                'dataset_id':'$fixture','columns':['reading','second'],'kind':'correlation_heatmap'}}])
            runtime=GraphAnalysisRuntime(root,'extensions','chart',model,sql_dialect='mysql',
                reference_context_loader=lambda:[{'table':'lab.observations','columns':[
                    {'name':'reading','dtype':'double'},{'name':'second','dtype':'double'}]}])
            raw=runtime.datasets.register(pd.DataFrame({'reading':[1.,2.,3.,4.],'second':[2.,4.,6.,8.]}),
                source='lab.observations',coverage='complete',predicate_known=True)
            runtime.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            try:
                result=runtime.submit('Draw a correlation heatmap for reading and second.')
                self.assertEqual(result['status'],'answered',result)
                recovery=runtime.inspect()['recovery']
                self.assertEqual(len(recovery['artifact_ids']),1)
                self.assertEqual(recovery['advanced_eda_evidence']['statistics']['correlation']['reading']['second'],1.)
                self.assertEqual(runtime.events()[-1].additional_kwargs['analysis_artifact_ids'],recovery['artifact_ids'])
            finally:runtime.close()

if __name__=='__main__':unittest.main()
