"""Failure admission and independent model grammar checks, without external models."""
import json,time
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
from langchain_core.messages import ToolMessage
from tests import test_analysis_extensions as fixtures
from core.analysis_agent.extension_evidence import collect
from core.analysis_agent.query_control import QueryControl,QueryResultDiscarded
from core.analysis_agent.goal_interpreter import GoalInterpreter

class ExtensionFailureTests(unittest.TestCase):
    setUp=fixtures.AnalysisExtensionsTests.setUp
    tearDown=fixtures.AnalysisExtensionsTests.tearDown
    def test_eda_receipt_cannot_complete_an_unrelated_call(self):
        result=self.tools['render_advanced_eda'](dataset_id=self.raw.id,columns=list(self.raw.columns),kind='correlation_heatmap')
        state={**self.state,'advanced_eda_spec':{'kind':'correlation_heatmap'}}
        calls={'call':{'name':'render_advanced_eda','args':{'dataset_id':self.raw.id,'columns':list(self.raw.columns),'kind':'correlation_heatmap'}}}
        result['advanced_eda_receipt']['dataset_id']='invented'
        collect(self.context,state,[ToolMessage(content=json.dumps(result),tool_call_id='call')],calls)
        self.assertNotIn('advanced_eda_evidence',state)
    def test_export_filters_requested_population_and_discloses_scope(self):
        self.state['scope']={'conditions':[{'column':'measurement','op':'ge','value':3}]}
        result=self.tools['export_analysis_result'](dataset_id=self.raw.id,format='parquet')
        self.assertEqual(result['scope']['rows'],2)
        self.assertEqual(result['scope']['conditions'][0]['value'],3)
    def test_export_respects_explicit_columns_without_dropping_original_columns(self):
        from io import BytesIO
        import zipfile,pandas as pd
        self.state['required_columns']=['comparison']
        result=self.tools['export_analysis_result'](dataset_id=self.raw.id,format='csv')
        _,payload=self.db.get(result['export']['id'],'export')
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            self.assertEqual(pd.read_csv(BytesIO(archive.read('result.csv'))).columns.tolist(),['comparison'])
            self.assertEqual(json.loads(archive.read('manifest.json'))['exported_columns'],['comparison'])
        self.assertEqual(list(self.datasets.metadata[self.raw.id].columns),['measurement','comparison'])
        with self.assertRaises(ValueError):self.tools['export_analysis_result'](dataset_id=self.raw.id,format='csv',columns=['measurement'])
    def test_png_export_keeps_verified_frequency_grain_and_rejects_new_population(self):
        frame=self.datasets.frames[self.raw.id].groupby('measurement').size().reset_index(name='frequency')
        counts=self.datasets.register(frame,source=self.raw.source,parent_id=self.raw.id,snapshot=self.raw.snapshot,
            coverage='complete',predicate_known=True,grain='aggregate',aggregation='COUNT',
            query='SELECT measurement, COUNT(*) AS frequency FROM data GROUP BY measurement')
        rendered=self.tools['render_histogram'](dataset_id=counts.id,value_column='measurement',weight_column='frequency')
        result=self.tools['export_analysis_result'](chart_id=rendered['cards'][0]['id'],format='png')
        self.assertEqual(result['status'],'ready')
        self.assertEqual(result['scope']['aggregation'],counts.aggregation)
        self.state['scope']={'conditions':[{'column':'measurement','op':'ge','value':3}]}
        with self.assertRaises(ValueError):self.tools['export_analysis_result'](chart_id=rendered['cards'][0]['id'],format='png')
    def test_complex_scope_is_not_declared_local_reusable(self):
        self.state['scope']={'any_conditions':[{'column':'measurement','op':'ge','value':3}]}
        plan=self.tools['plan_analysis_execution'](source=self.raw.source,columns=list(self.raw.columns),operation='aggregate')
        self.assertEqual(plan['execution_plan']['route'],'database_aggregate')
        self.assertEqual(plan['execution_plan']['local_candidates'],[])
    def test_cancelled_result_never_publishes_even_if_driver_returns_success(self):
        control=QueryControl();called=[];control.bind('own',time.time()+2);control.start(lambda:called.append(1))
        try:
            control.cancel('own');control.cancel('own')
            self.assertEqual(called,[1])
            with self.assertRaises(QueryResultDiscarded):control.validate_result()
        finally:control.finish()
    def test_advanced_columns_are_required_by_model_grammar(self):
        schema={'properties':{'columns':{'type':'array'}}}
        GoalInterpreter.bind_required_columns(schema,{'capabilities':['advanced_eda'],'chart_kind':''})
        self.assertEqual(schema['properties']['columns']['minItems'],2)
        self.assertEqual(schema['properties']['columns']['maxItems'],6)
    def test_invalid_python_syntax_is_repairable_without_worker_or_dataset_publication(self):
        count=len(self.datasets.metadata)
        result=self.tools['execute_analysis_python'](dataset_id=self.raw.id,columns=list(self.raw.columns),code='result = df[')
        self.assertEqual(result['error_code'],'python_contract_violation')
        self.assertEqual(result['error_type'],'SyntaxError')
        self.assertTrue(result['retryable'])
        self.assertEqual(len(self.datasets.metadata),count)
    def test_export_receipt_supplies_followup_population_without_selected_dataset(self):
        from core.analysis_agent.population_basis import read
        request='Analyze only those four exported rows'
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps({
            'basis':'displayed_result','quote':'those four exported rows',
            'changes_filters':False,'filter_change_quote':''}))
        interpreter=SimpleNamespace(context=self.context,selection_model=model,
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)),
            model_recovery=None,diagnostics=Mock())
        result=read(interpreter,{'request_id':'followup','request_text':request},
            {'verified_previous':{'export_evidence':{'dataset_id':self.raw.id}}})
        self.assertTrue(result['current_result_only'])
        payload=json.loads(model.invoke.call_args.args[0][1].content)
        self.assertEqual(payload['available_displayed_result']['export_evidence']['rows'],4)
        self.assertEqual(payload['available_displayed_result']['export_evidence']['source'],self.raw.source)
        model.reset_mock()
        result=read(interpreter,{'request_id':'foreign','request_text':'Show other.samples'},
            {'verified_previous':{'export_evidence':{'dataset_id':self.raw.id}},
             'literal_table_subjects':[{'name':'other.samples'}]})
        self.assertFalse(result['current_result_only'])
        model.invoke.assert_not_called()
    def test_population_repair_explains_the_actual_filter_quote_violation(self):
        from core.analysis_agent.population_basis import read
        request='Use those four exported rows only'
        correct={'basis':'displayed_result','quote':'those four exported rows','changes_filters':False,'filter_change_quote':''}
        model=Mock();model.invoke.side_effect=[SimpleNamespace(content=json.dumps({**correct,'filter_change_quote':'four'})),
            SimpleNamespace(content=json.dumps(correct))]
        interpreter=SimpleNamespace(context=self.context,selection_model=model,
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)),model_recovery=None,diagnostics=Mock())
        result=read(interpreter,{'request_id':'repair','request_text':request},
            {'verified_previous':{'export_evidence':{'dataset_id':self.raw.id}}})
        self.assertTrue(result['current_result_only'])
        self.assertIn('Filter changes need their exact CURRENT restriction clause',model.invoke.call_args.args[0][-1].content)
        self.assertEqual(interpreter.diagnostics.emit.call_args_list[0].args[0],'goal_population_basis_rejected')
    def test_custom_input_grammar_excludes_derived_names_but_not_unseen_preview_columns(self):
        from core.analysis_agent.task_selection import schema_for,custom_input_names,validate,SCHEMA
        data={'tables':[{'columns':[{'name':'measurement'}],'column_count':2}],
              'observed_input_column_names':['measurement','comparison']}
        branch=next(b for b in schema_for(data)['anyOf'] if b['properties']['capabilities'].get('const')==['custom_analysis'])
        self.assertEqual(branch['properties']['output_columns']['items']['enum'],['measurement','comparison'])
        value={k:'' for k in SCHEMA['required']}
        value.update(mode='execute',capabilities=['custom_analysis'],source_reference='explicit',source_mentions=[],
            output_columns=['new_mean'],scalar_operations=[],chart_edit_fields=[])
        with self.assertRaises(ValueError):validate(value,'Calculate new_mean',data)
        value['output_columns']=['comparison'];validate(value,'Calculate new_mean',data)
        value['output_columns']=['comparison','comparison']
        self.assertEqual(validate(value,'Calculate new_mean',data)['output_columns'],['comparison'])
        self.assertEqual(custom_input_names({'tables':data['tables']}),[])

if __name__=='__main__':unittest.main()
