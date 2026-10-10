"""Historical display identity tests; scripted choices are not language scores."""
import json
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
from core.analysis_agent.output_memory import remember,available,bind
from core.analysis_agent.population_basis import read,validate
from core.analysis_agent.row_preview import evidence
from tests.test_llm_goal import GoalModel,goal
from tests import test_llm_goal as fixtures
from utils.analysis_datasets import DatasetStore,stored_dataset_digest
import pandas as pd


class HistoricalDisplayTests(unittest.TestCase):
    def setUp(self):
        self.store=DatasetStore()
        self.raw=self.store.register(pd.DataFrame({'metric':[1,2,3,100,200],'group':['A','A','B','B','C']}),
            source='lab.events',coverage='complete',predicate_known=True)
        self.context=SimpleNamespace(datasets=self.store,selected_dataset_id=self.raw.id,sql_dialect='mysql',reference_context=[{'table':'lab.events'}])
        self.first={'request_id':'first','status':'complete','request_text':'Show the first three rows',
            'goal':{'tasks':[{'capability':'row_preview'}]},'required_sources':['lab.events'],
            'table_preview_evidence':evidence(self.store,self.raw.id,3),'scope':{'conditions':[]}}
    def test_earlier_display_survives_intervening_scalar_and_another_preview(self):
        middle={'request_id':'scalar','status':'complete','goal':{'tasks':[{'capability':'calculation'}]},
                'output_references':remember(self.first)}
        second={**self.first,'request_id':'second','table_preview_evidence':evidence(self.store,self.raw.id,2),
                'output_references':remember(middle)}
        refs=remember(second)
        eligible=available(self.context,refs)
        self.assertEqual([r['reference_id'] for r in eligible],['first:0','second:0'])
        self.assertEqual([r['display_evidence']['rows'] for r in eligible],[3,2])
        bad=deepcopy(refs);bad[0]['display_evidence']['snapshot']='fabricated'
        self.assertEqual(len(available(self.context,bad)),1)
        self.assertEqual(remember({**middle,'status':'blocked','output_references':refs}),refs)
    def test_provider_population_can_select_earlier_display_after_scalar(self):
        refs=remember(self.first)
        request='Using only the first three rows shown, draw group frequency without reloading'
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps({'basis':'displayed_result',
            'quote':'first three rows shown','changes_filters':False,'filter_change_quote':'',
            'output_reference_id':'first:0'}))
        interpreter=SimpleNamespace(context=self.context,selection_model=model,
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)),model_recovery=None,diagnostics=Mock())
        result=read(interpreter,{'request_id':'third','request_text':request,'output_references':refs},
                    {'verified_previous':{'calculation':True},'selected_dataset':{'source':'lab.events'}})
        self.assertTrue(result['current_result_only'])
        self.assertEqual(result['output_reference']['display_evidence']['rows'],3)
        payload=json.loads(model.invoke.call_args.args[0][1].content)
        self.assertEqual(payload['eligible_output_references'][0]['reference_id'],'first:0')
        self.assertNotIn('raw_rows',payload['eligible_output_references'][0])
    def test_unavailable_display_is_not_permission_for_source_population(self):
        result=validate({'basis':'unavailable_result','quote':'those earlier rows',
            'changes_filters':False,'filter_change_quote':''},'Plot those earlier rows')
        self.assertEqual(result['population_basis'],'unavailable_result')
    def test_provider_grammar_rejects_paraphrased_current_reference(self):
        import jsonschema
        request='Chart only the earlier three rows, without querying the source again'
        value={'basis':'displayed_result','quote':request,'changes_filters':False,
               'filter_change_quote':'','output_reference_id':'first:0'}
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps(value))
        interpreter=SimpleNamespace(context=self.context,selection_model=model,
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)),model_recovery=None,diagnostics=Mock())
        grammars=[]
        def role(model,schema,tokens):
            grammars.append(deepcopy(schema));return model
        with patch('core.analysis_agent.model_roles.json_role',side_effect=role):
            read(interpreter,{'request_id':'quote','request_text':request,'output_references':remember(self.first)},
                 {'selected_dataset':{'source':'lab.events'}})
        jsonschema.validate(value,grammars[-1])
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({**value,'quote':'the three previously shown records'},grammars[-1])

    def test_expired_history_still_allows_semantic_unavailable_decision(self):
        refs=remember(self.first)
        self.store.metadata.clear()
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps({
            'basis':'unavailable_result','quote':'earlier three rows','changes_filters':False,'filter_change_quote':''}))
        interpreter=SimpleNamespace(context=self.context,selection_model=model,
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)),model_recovery=None,diagnostics=Mock())
        result=read(interpreter,{'request_id':'expired','request_text':'Plot earlier three rows',
                              'output_references':refs},{})
        self.assertEqual(result['population_basis'],'unavailable_result')
        model.invoke.assert_called_once()

    def test_unavailable_display_stops_graph_without_query_or_goal_generation(self):
        with tempfile.TemporaryDirectory() as root:
            model=GoalModel(goals=[goal('row_count')])
            runtime,raw=fixtures.LLMGoalTests().runtime(root,model)
            try:
                with patch('core.analysis_agent.task_selection.select',return_value={
                        'mode':'clarify','capabilities':[],'source_mentions':[],'source_reference':'previous_analysis','chart_kind':'',
                        'current_result_only':False,'population_basis':'unavailable_result'}):
                    result=runtime.submit('Use only the expired earlier table output')
                state=runtime.inspect()['recovery']
                self.assertEqual(state['goal']['mode'],'clarify')
                self.assertFalse(state['sent_calls'])
                self.assertEqual(model.goal_calls,0)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
            finally:runtime.close()

    def test_foreign_source_and_missing_columns_cannot_bind_display(self):
        reference=remember(self.first)[0]
        for source,columns in [('lab.other',['metric']),('lab.events',['missing'])]:
            state={'current_result_only':True,'required_sources':[source],'required_columns':columns,
                   'requested_output_reference':reference}
            with self.subTest(source=source,columns=columns),self.assertRaises(ValueError):bind(self.context,state)
        self.assertEqual(len(self.store.metadata),1)

    def test_prefix_is_materialized_locally_without_changing_original(self):
        digest=stored_dataset_digest(self.store,self.raw.id)
        current={'current_result_only':True,'required_sources':['lab.events'],'required_columns':['group'],
                 'requested_output_reference':remember(self.first)[0]}
        bind(self.context,current)
        derived=self.store.metadata[current['display_dataset_id']]
        self.assertEqual(derived.rows,3)
        self.assertEqual(derived.parent_id,self.raw.id)
        self.assertEqual(self.store.frames[derived.id]['group'].value_counts().to_dict(),{'A':2,'B':1})
        self.assertEqual(stored_dataset_digest(self.store,self.raw.id),digest)
        self.assertEqual(self.context.selected_dataset_id,self.raw.id)
    def test_real_graph_charts_earlier_preview_after_intervening_calculation(self):
        preview=goal('row_preview',{'limit':3})
        mean=goal('calculation',{'operations':['AVG']},columns=['reading']);mean['current_result_only']=True
        chart=goal('chart',{'kind':'histogram','axes':{'x':'reading'}},columns=['reading']);chart['current_result_only']=True
        with tempfile.TemporaryDirectory() as root:
            runtime,raw=fixtures.LLMGoalTests().runtime(root,GoalModel(goals=[preview,preview,mean,mean,chart,chart]))
            try:
                self.assertEqual(runtime.submit('Show first three rows')['status'],'answered')
                refs=remember(runtime.inspect()['recovery']);reference=refs[-1]
                selection={'mode':'execute','capabilities':['calculation'],'source_reference':'explicit',
                    'source_mentions':[],'chart_kind':'','output_columns':['reading'],'scalar_operations':['AVG'],
                    'current_result_only':True,'population_basis':'displayed_result','output_reference':reference}
                with patch('core.analysis_agent.task_selection.select',return_value=selection):
                    self.assertEqual(runtime.submit('Average only the first three rows')['status'],'answered')
                selection.update(capabilities=['chart'],chart_kind='histogram',scalar_operations=[],group_column='')
                with patch('core.analysis_agent.task_selection.select',return_value=selection):
                    result=runtime.submit('Draw only the first three rows again, no reload')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery'];card=runtime.artifacts[state['artifact_ids'][0]]
                self.assertEqual(runtime.datasets.metadata[card.dataset_id].rows,3)
                self.assertFalse(state.get('remote_submission'))
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
            finally:runtime.close()
