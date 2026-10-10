"""Independent scope, numerical, isolation and artifact checks for new tools."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from io import BytesIO
import json
import zipfile
import subprocess
import unittest
import pandas as pd
from core.analysis_agent.assets import AssetDB,PersistentDatasets,PersistentCharts
from core.analysis_tool_contract import AnalysisToolContext
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_agent.execution_plan import ExecutionPlans
from core.analysis_agent.python_contract import validate_code
from core.analysis_agent.query_control import QueryControl
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_image_validation import validate_chart_image

def rolling_contract(column='measurement',output='rolling_mean',columns=None,sort=False):
    steps=([{'op':'sort','columns':[column],'ascending':True}] if sort else [])
    steps.append({'op':'rolling','column':column,'output_column':output,'window':2,'min_periods':2,'aggregation':'mean'})
    return {'steps':steps,'output_columns':columns or [output]}

class AnalysisExtensionsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.db=AssetDB(self.tmp.name,'owner','conversation')
        self.datasets=PersistentDatasets(self.db,max_full_read_bytes=128*1024*1024)
        self.raw=self.datasets.register(pd.DataFrame({'measurement':[1.,2.,3.,4.],'comparison':[2.,4.,6.,8.]}),
            source='arbitrary.samples',coverage='complete',predicate_known=True,snapshot='frozen')
        self.state={'request_id':'request1','required_sources':['arbitrary.samples'],
                    'required_columns':['measurement','comparison'],'scope':{'conditions':[]}}
        self.state['custom_analysis_spec']={'description':'Rolling mean','result_contract':rolling_contract()}
        self.context=AnalysisToolContext(self.datasets,PersistentCharts(self.db),[],lambda **_:None,
            selected_dataset_id=self.raw.id,runtime_services={'asset_db':self.db,'current':lambda:self.state,
                'policy':lambda:{'max_full_read_bytes':128*1024*1024}})
        self.tools={t.name:t.run for t in build_analysis_tools(self.context)}
        self.digest=stored_dataset_digest(self.datasets,self.raw.id)
    def tearDown(self):
        self.assertEqual(self.digest,stored_dataset_digest(self.datasets,self.raw.id))
        self.db.close();self.tmp.cleanup()
    def test_python_actual_worker_result_and_input_preservation(self):
        result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),
            code="result = pd.DataFrame({'rolling_mean': df['measurement'].rolling(2).mean()})")
        self.assertEqual(result['status'],'ready',result)
        output=self.datasets.frames[result['dataset']['id']]
        self.assertEqual(output['rolling_mean'].dropna().tolist(),[1.5,2.5,3.5])
        self.assertIn(self.raw.id,self.datasets.metadata[result['dataset']['id']].parent_ids or
                      (self.datasets.metadata[result['dataset']['id']].parent_id,))
        self.assertEqual(result['python_receipt']['input_digest'],self.digest)
    def test_worker_rejects_import_io_private_mutation_and_unbounded_control(self):
        for code in ("import os\nresult=df", "result=pd.read_csv('secret')", "result=df.__class__",
                     "pd['x']=1\nresult=df", "while True: pass\nresult=df", "result=eval('1')",
                     "result=df.agg('to_csv', '/tmp/forbidden.csv')", "result=df.agg({'measurement':'to_pickle'})"):
            with self.subTest(code=code),self.assertRaises(ValueError):validate_code(code)
        result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code='result = 1')
        self.assertEqual(result['error_code'],'python_execution_failed')
        with patch('core.analysis_agent.python_analysis.subprocess.run',side_effect=subprocess.TimeoutExpired('worker',12)):
            result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code='result = df.copy()')
        self.assertEqual(result['error_code'],'python_worker_timeout')
    def test_standard_generated_pandas_script_updates_only_private_worker_copy(self):
        self.state['custom_analysis_spec']['result_contract']=rolling_contract(output='new_mean',columns=['measurement','comparison','new_mean'],sort=True)
        result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),
            code="working = df.sort_values('measurement').copy()\nworking['new_mean'] = working['measurement'].rolling(2).mean()\nprint(working)")
        self.assertEqual(result['status'],'ready',result)
        output=self.datasets.frames[result['dataset']['id']]
        self.assertEqual(output['new_mean'].dropna().tolist(),[1.5,2.5,3.5])
        self.assertTrue(result['python_receipt']['terminal_table_output_adapted'])
        self.assertTrue(result['python_receipt']['working_copy_only'])
        self.assertNotIn('new_mean',self.datasets.metadata[self.raw.id].columns)
        with self.assertRaises(ValueError):validate_code("print(df, file='elsewhere')")
    def test_real_eda_png_correlations_bins_and_source_switch(self):
        for kind in ('correlation_heatmap','scatter_matrix','distribution_panels'):
            result=self.tools['render_advanced_eda'](dataset_id=self.raw.id,columns=list(self.raw.columns),kind=kind)
            self.assertEqual(result['status'],'ready')
            validate_chart_image(self.context.artifacts[result['cards'][0]['id']].image)
            stats=result['advanced_eda_receipt']['statistics']
            if kind=='correlation_heatmap':
                self.assertEqual(stats['correlation']['measurement']['comparison'],1.)
                self.assertEqual(stats['pairwise_valid_counts']['measurement']['comparison'],4)
            if kind=='distribution_panels':self.assertEqual(sum(stats['histograms']['measurement']['counts']),4)
        self.state['required_sources']=['other.samples']
        with self.assertRaises(ValueError):self.tools['render_advanced_eda'](dataset_id=self.raw.id,columns=list(self.raw.columns),kind='scatter_matrix')
    def test_filtered_population_applies_without_source_reload(self):
        self.state['scope']={'conditions':[{'column':'measurement','op':'ge','value':3}]}
        self.state['custom_analysis_spec']['result_contract']={'steps':[{'op':'aggregate','column':'measurement','output_column':'average','aggregation':'mean'}],'output_columns':['average']}
        result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),
            code="result = pd.DataFrame({'average': [df['measurement'].mean()]})")
        self.assertEqual(self.datasets.frames[result['dataset']['id']].iloc[0,0],3.5)
    def test_export_zip_manifest_and_restart(self):
        for format in ('csv','parquet'):
            result=self.tools['export_analysis_result'](dataset_id=self.raw.id,format=format)
            metadata,payload=self.db.get(result['export']['id'],'export')
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                manifest=json.loads(archive.read('manifest.json'))
                self.assertEqual(manifest['dataset']['source'],'arbitrary.samples')
                self.assertEqual(manifest['dataset']['snapshot'],'frozen')
                if format=='csv':self.assertEqual(len(pd.read_csv(BytesIO(archive.read('result.csv')))),4)
                else:self.assertEqual(len(pd.read_parquet(BytesIO(archive.read('result.parquet')))),4)
            restored=AssetDB(self.tmp.name,'owner','conversation')
            try:self.assertEqual(restored.get(metadata['id'],'export')[1],payload)
            finally:restored.close()
    def test_cost_plan_is_nonexecuting_unknown_remote_cost_and_contract_lookup(self):
        result=self.tools['plan_analysis_execution'](source='arbitrary.samples',columns=list(self.raw.columns),operation='aggregate')
        self.assertEqual(result['execution_plan']['route'],'reuse_local')
        self.assertIsNone(result['execution_plan']['remote_cost_estimate'])
        self.context.registered_contracts={'example':{'name':'example','parameters':{'required':['query']}}}
        self.assertEqual(self.tools['get_analysis_tool_contract'](name='example')['contract']['parameters']['required'],['query'])
        self.assertEqual(self.tools['get_analysis_tool_contract'](name='invented')['status'],'needs_context')
    def test_plan_dependency_graph_durability_and_cycle_rejection(self):
        plans=ExecutionPlans(self.db)
        state={**self.state,'goal':{'tasks':[{'capability':'custom_analysis','options':{}},{'capability':'export','options':{}}]},
            'export_spec':{'format':'csv'},'custom_analysis_spec':self.state['custom_analysis_spec']}
        plan=plans.sync(state);a,b=[t['id'] for t in plan['tasks']]
        plans.dependencies('request1',[{'task_id':b,'depends_on':[a]}])
        self.assertEqual(plans.sync(state)['tasks'][1]['depends_on'],[a])
        self.assertEqual(plans.admission(state,'export_analysis_result')['error_code'],'task_dependency_pending')
        with self.assertRaises(ValueError):plans.dependencies('request1',[{'task_id':a,'depends_on':[b]}])
        from core.analysis_agent.python_result_contract import digest
        state['custom_analysis_evidence']={'output_dataset_id':self.raw.id,'semantic_verified':True,'result_contract_hash':digest(rolling_contract())}
        plans.sync(state)
        self.assertIsNone(plans.admission(state,'export_analysis_result'))
    def test_owned_query_control_no_duplicate_or_foreign_cancellation(self):
        cancelled=[];control=QueryControl();control.bind('own');control.start(lambda:cancelled.append(True))
        self.assertEqual(control.cancel('other')['error_code'],'query_not_active')
        self.assertEqual(control.cancel('own')['execution_state'],'cancel_requested')
        self.assertEqual(cancelled,[True]);control.finish()
        self.assertEqual(control.cancel('own')['error_code'],'query_not_active')

if __name__=='__main__':unittest.main()
