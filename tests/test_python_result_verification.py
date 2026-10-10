"""Wrong-but-executable Python must never become a completed analysis."""
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from tests import test_analysis_extensions as fixtures
rolling_contract=fixtures.rolling_contract
from tests.test_llm_goal import GoalModel,goal
from core.analysis_agent.python_result_verification import verify,expected_frame
from core.analysis_agent.python_result_contract import verified,digest
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.progress_guard import instruction
from utils.analysis_datasets import stored_dataset_digest

class NumericOracleTests(unittest.TestCase):
    def test_numeric_operations_with_missing_values_and_sample_std(self):
        frame=pd.DataFrame({'x':[2.,np.nan,6.,10.]})
        answers={'mean':[2.,2.,6.,8.],'sum':[2.,2.,6.,16.],
                 'count':[1.,1.,1.,2.],'min':[2.,2.,6.,6.],
                 'max':[2.,2.,6.,10.],'median':[2.,2.,6.,8.],
                 'std':[np.nan,np.nan,np.nan,2.8284271247461903]}
        for kind,values in answers.items():
            contract=rolling_contract('x','out');contract['steps'][0].update(aggregation=kind,min_periods=1)
            with self.subTest(kind=kind):
                result=pd.DataFrame({'out':values})
                self.assertTrue(verify(frame,result,contract)['semantic_verified'])
                pd.testing.assert_frame_equal(expected_frame(frame,contract),pd.DataFrame({'out':frame.x.rolling(2,min_periods=1).agg(kind)}),check_exact=False)
        contract={'steps':[{'op':'diff','column':'x','output_column':'change','periods':2}], 'output_columns':['change']}
        self.assertTrue(verify(frame,pd.DataFrame({'change':[np.nan,np.nan,4.,np.nan]}),contract)['semantic_verified'])
    def test_sort_ties_and_nulls_and_wrong_order(self):
        frame=pd.DataFrame({'x':[4.,1.,1.,np.nan],'label':['last','first','second','null']})
        contract={'steps':[{'op':'sort','columns':['x'],'ascending':True}], 'output_columns':['label']}
        self.assertTrue(verify(frame,pd.DataFrame({'label':['first','second','last','null']}),contract)['semantic_verified'])
        self.assertFalse(verify(frame,pd.DataFrame({'label':['second','first','last','null']}),contract)['semantic_verified'])
    def test_scalar_aggregations_empty_count_sum_and_std(self):
        for values,kind,answer in (([2.,np.nan,6.],'mean',4.),([2.,6.],'std',2.8284271247461903),
                                   ([np.nan],'count',0.),([np.nan],'sum',0.),([np.nan],'mean',np.nan)):
            with self.subTest(values=values,kind=kind):
                contract={'steps':[{'op':'aggregate','column':'x','output_column':'out','aggregation':kind}],'output_columns':['out']}
                self.assertTrue(verify(pd.DataFrame({'x':values}),pd.DataFrame({'out':[answer]}),contract)['semantic_verified'])
    def test_partial_windows_cannot_be_invented_by_goal_interpretation(self):
        from core.analysis_agent.python_result_contract import validate_intent
        contract=rolling_contract('x');contract['steps'][0]['min_periods']=1
        with self.assertRaises(ValueError):validate_intent(contract,'Calculate a two-row rolling mean')
        contract['steps'][0]['partial_window_quote']='include partial windows'
        with self.assertRaises(ValueError):validate_intent(contract,'Calculate a two-row rolling mean')
        validate_intent(contract,'Calculate a two-row rolling mean; include partial windows')
    def test_legacy_or_changed_contract_receipt_cannot_complete(self):
        contract=rolling_contract('x')
        current={'custom_analysis_spec':{'result_contract':contract},'custom_analysis_evidence':{'output_dataset_id':'old'}}
        self.assertFalse(verified(current))
        current['custom_analysis_evidence'].update(semantic_verified=True,result_contract_hash=digest(contract))
        self.assertTrue(verified(current))
        contract['steps'][0]['window']=3
        self.assertFalse(verified(current))
    def test_sentence_punctuation_is_not_a_qualified_table(self):
        from core.analysis_agent.source_references import qualified_mentions
        self.assertEqual(qualified_mentions('Use retained data. Do not reload.'),[])
        self.assertEqual(qualified_mentions('`catalog` . `schema` . `table` rows')[0]['source'],'catalog.schema.table')
        self.assertEqual(qualified_mentions('unknown.schema.table rows')[0]['source'],'unknown.schema.table')
    def test_manual_and_automatic_recovery_follow_actual_policy(self):
        automatic=instruction({},[],False);manual=instruction({},[],True)
        self.assertIn('do not ask for approval or wait',automatic)
        self.assertNotIn('requires the configured exact-query approval',automatic)
        self.assertIn('requires the configured exact-query approval',manual)
        self.assertNotIn('do not ask for approval',manual)

class PythonPublicationTests(unittest.TestCase):
    setUp=fixtures.AnalysisExtensionsTests.setUp
    tearDown=fixtures.AnalysisExtensionsTests.tearDown
    def test_executable_wrong_code_wrong_window_and_wrong_values_not_published(self):
        for code in ('result=df.copy()',
                     "result=pd.DataFrame({'rolling_mean':df['measurement'].rolling(3).mean()})",
                     "result=pd.DataFrame({'rolling_mean':df['measurement'].rolling(2).sum()})"):
            with self.subTest(code=code):
                before=len(self.datasets.metadata)
                result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code=code)
                self.assertEqual(result['error_code'],'semantic_mismatch',result)
                self.assertTrue(result['retryable']);self.assertEqual(len(self.datasets.metadata),before)
    def test_reference_columns_are_projected_but_wrong_math_still_fails(self):
        for good in (True,False):
            with self.subTest(good=good):
                operation='mean' if good else 'sum'
                code="result=df.copy()\nresult['rolling_mean']=df['measurement'].rolling(2)."+operation+'()'
                count=len(self.datasets.metadata)
                result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code=code)
                if good:
                    self.assertEqual(result['status'],'ready',result)
                    self.assertEqual(result['dataset']['columns'],['rolling_mean'])
                    self.assertEqual(result['python_receipt']['omitted_input_reference_columns'],list(self.raw.columns))
                else:
                    self.assertEqual(result['error_code'],'semantic_mismatch')
                    self.assertEqual(len(self.datasets.metadata),count)
        from core.analysis_agent.tool_repair import diagnosis
        self.assertIn('execute_analysis_python',diagnosis({'tool':'execute_analysis_python','error_code':'semantic_mismatch'},['execute_analysis_python'])['candidate_tools'])
    def test_unexpected_derived_columns_are_not_silently_discarded(self):
        code="result=pd.DataFrame({'rolling_mean':df['measurement'].rolling(2).mean(), 'unrequested_total':df['measurement'].sum()})"
        result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code=code)
        self.assertEqual(result['error_code'],'semantic_mismatch');self.assertEqual(len(self.datasets.metadata),1)
    def test_resource_budget_prevents_worker_and_no_publication(self):
        self.state['custom_analysis_spec']['result_contract']=rolling_contract()
        self.state['custom_analysis_spec']['result_contract']['steps'][0].update(window=10000,aggregation='median')
        raw=self.datasets.register(pd.DataFrame({'measurement':np.arange(201,dtype=float),'comparison':np.arange(201,dtype=float)}),
            source=self.raw.source,coverage='complete',predicate_known=True)
        with patch('core.analysis_agent.python_analysis.subprocess.run') as run:
            result=self.tools['execute_analysis_python'](dataset_id=raw.id,columns=list(raw.columns),code='result=df.copy()')
        self.assertEqual(result['error_code'],'semantic_verification_budget');run.assert_not_called()
        self.assertEqual(len(self.datasets.metadata),2)
    def test_historical_derived_column_is_not_an_input_until_referenced(self):
        from core.analysis_agent.custom_input_context import names
        old=self.datasets.register(pd.DataFrame({'old_computed':[1.]}),source=self.raw.source,parent_id=self.raw.id)
        data={'verified_previous':{'export_evidence':{'dataset_id':self.raw.id}}}
        self.assertNotIn('old_computed',names(self.context,data,[self.raw.source]))
        data['verified_previous']={'custom_analysis_evidence':{'output_dataset_id':old.id}}
        self.assertIn('old_computed',names(self.context,data,[self.raw.source]))
        data['literal_table_subjects']=[{'name':self.raw.source}]
        self.assertNotIn('old_computed',names(self.context,data,[self.raw.source]))
    def test_missing_contract_does_not_execute_or_publish(self):
        self.state.pop('custom_analysis_spec')
        with patch('core.analysis_agent.python_analysis.subprocess.run') as run:
            result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code='result=df.copy()')
        self.assertEqual(result['error_code'],'result_contract_missing');run.assert_not_called()
        self.assertEqual(len(self.datasets.metadata),1)

class SemanticRecoveryJourneyTests(unittest.TestCase):
    def test_wrong_result_replans_or_stops_without_publishing_false_success(self):
        for repeated in (False,True):
            with self.subTest(repeated=repeated),tempfile.TemporaryDirectory() as root:
                contract=rolling_contract('reading')
                plan=goal('custom_analysis',{'description':'Two-row moving mean','result_contract':contract},columns=['reading'])
                wrong={'name':'execute_analysis_python','args':{'dataset_id':'$fixture','columns':['reading'],
                    'code':"result=pd.DataFrame({'rolling_mean':df['reading'].rolling(3).mean()})"}}
                right={'name':'execute_analysis_python','args':{'dataset_id':'$fixture','columns':['reading'],
                    'code':"result=pd.DataFrame({'rolling_mean':df['reading'].rolling(2).mean()})"}}
                model=GoalModel(goals=[plan],calls=[wrong,wrong if repeated else right])
                runtime=GraphAnalysisRuntime(root,'semantic','repeat' if repeated else 'repair',model,sql_dialect='mysql',
                    reference_context_loader=lambda:[{'table':'lab.observations','columns':[{'name':'reading','dtype':'double'}]}])
                raw=runtime.datasets.register(pd.DataFrame({'reading':[1.,2.,3.,4.]}),source='lab.observations',coverage='complete',predicate_known=True)
                runtime.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
                before=stored_dataset_digest(runtime.datasets,raw.id)
                try:
                    result=runtime.submit('Compute a two-row moving mean of reading on the retained data. Do not reload.')
                    recovery=runtime.inspect()['recovery']
                    self.assertIn('custom_analysis_spec',recovery,{'result':result,'error':recovery.get('goal_contract_error')})
                    self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),before)
                    self.assertEqual(recovery['custom_analysis_spec']['result_contract'],contract)
                    messages=runtime.agent.get_state(runtime.config).values['messages']
                    self.assertTrue(any(getattr(m,'name',None)=='execute_analysis_python' and 'semantic_mismatch' in str(m.content) for m in messages))
                    if repeated:
                        self.assertNotEqual(result['status'],'answered',result)
                        self.assertEqual(len(runtime.datasets.metadata),1)
                        self.assertFalse(verified(recovery))
                    else:
                        self.assertEqual(result['status'],'answered',result)
                        proof=recovery['custom_analysis_evidence'];self.assertTrue(proof['semantic_verified'])
                        self.assertEqual(runtime.datasets.frames[proof['output_dataset_id']].rolling_mean.dropna().tolist(),[1.5,2.5,3.5])
                        self.assertEqual(len(runtime.datasets.metadata),2)
                        task=runtime.plans.inspect(recovery['request_id'])['plan']['tasks'][0]
                        self.assertEqual(task['postconditions'],contract)
                        self.assertEqual(task['verification_receipt']['result_contract_hash'],digest(contract))
                        self.assertEqual(task['status'],'verified')
                finally:runtime.close()

if __name__=='__main__':unittest.main()
