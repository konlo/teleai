"""Actual image/count and recovery contracts; fixtures are not language scores."""
import tempfile
import unittest
from types import SimpleNamespace
from langchain_core.messages import AIMessage
import json
from unittest.mock import patch
from tests.test_llm_goal import GoalModel, goal
from tests import test_llm_goal as goals
from utils.analysis_datasets import stored_dataset_digest


class CategoricalPresentationJourneyTests(unittest.TestCase):
    def test_entire_source_contract_cannot_be_restricted_by_an_auxiliary_scope_audit(self):
        from core.analysis_agent.population_basis import validate
        from unittest.mock import Mock
        initial=goal('chart',{'kind':'histogram','axes':{'x':'reading'}},columns=['reading'],
            conditions=[{'column':'reading','op':'ge','value':3}])
        whole=goal('chart',{'kind':'histogram','axes':{'x':'reading'}},columns=['reading'])
        whole['source_reference']='previous_analysis'
        selection={'mode':'execute','capabilities':['chart'],'source_reference':'previous_analysis',
            'source_mentions':[],'chart_kind':'histogram','output_columns':['reading'],'group_column':'',
            **validate({'basis':'unfiltered_source','quote':'entire source'},'Draw entire source')}
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,GoalModel(goals=[initial,initial,whole,whole]))
            try:
                self.assertEqual(r.submit('filtered reading distribution')['status'],'answered')
                audit=Mock(side_effect=AssertionError('typed entire-source contract is authoritative'))
                r.recovery.goal_interpreter.population_model=SimpleNamespace(invoke=audit)
                with patch('core.analysis_agent.task_selection.select',return_value=selection):
                    result=r.submit('Draw entire source')
                self.assertEqual(result['status'],'answered',result)
                current=r.inspect()['recovery'];card=r.artifacts[current['artifact_ids'][0]]
                self.assertFalse(current['scope']['conditions'])
                self.assertEqual(card.render_spec['total_count'],raw.rows)
                audit.assert_not_called()
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_presentation_edit_binds_verified_rows_across_population_reference_labels(self):
        initial=goal('chart',{'kind':'histogram','axes':{'x':'cohort'},'legend':False,
            'stacked':False,'palette':'default'},columns=['cohort'])
        initial['current_result_only']=True
        edit=goal('chart_adjust',{'legend':True},columns=[])
        model=GoalModel(goals=[initial,initial,edit,edit])
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,model)
            try:
                self.assertEqual(r.submit('selected original cohort distribution')['status'],'answered')
                previous=r.inspect()['recovery'];first=r.artifacts[previous['artifact_ids'][0]]
                result=r.submit('put a legend on this chart')
                self.assertEqual(result['status'],'answered',result)
                current=r.inspect()['recovery'];last=r.artifacts[current['artifact_ids'][0]]
                self.assertEqual(first.dataset_id,last.dataset_id)
                self.assertEqual(first.render_spec['counts'],last.render_spec['counts'])
                self.assertTrue(last.render_spec['legend'])
                self.assertEqual(current['scope'],previous['scope'])
            finally:r.close()

    def test_output_references_survive_scalar_result_and_failure_without_changing_subject(self):
        from core.analysis_agent.output_memory import remember
        inventory={'status':'complete','request_text':'available tables','required_sources':['information_schema.tables'],
            'goal':{'tasks':[{'capability':'table_list'}]}}
        scalar={'status':'complete','request_text':'count rows','required_sources':['lab.events'],
            'goal':{'tasks':[{'capability':'row_count'}]},'output_references':remember(inventory)}
        references=remember(scalar)
        self.assertEqual([r['capability'] for r in references],['table_list','row_count'])
        failed={**scalar,'status':'blocked','output_references':references}
        self.assertEqual(remember(failed),references)
        self.assertEqual(scalar['required_sources'],['lab.events'])
        self.assertNotIn('dataset_id',references[0])

    def test_display_adjustment_is_supported_without_reloading_or_losing_original(self):
        initial=goal('chart',{'kind':'histogram','axes':{'x':'cohort'},'legend':False,
            'stacked':False,'palette':'default'},columns=['cohort'])
        changes=[goal('chart_adjust',opts,columns=[]) for opts in
            ({'legend':True},{'palette':'high_contrast'},{'stacked':True})]
        model=GoalModel(goals=[p for p in [initial,*changes] for _ in range(2)])
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,model)
            try:
                digest=stored_dataset_digest(r.datasets,raw.id);cards=[]
                for text in ['cohort distribution','add legend','improve contrast','stack bars']:
                    result=r.submit(text)
                    self.assertEqual(result['status'],'answered',result)
                    current=r.inspect()['recovery'];cards.append(r.artifacts[current['artifact_ids'][0]])
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                    self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(len({c.dataset_id for c in cards}),1)
                self.assertTrue(cards[1].render_spec['legend'])
                self.assertEqual(cards[2].render_spec['palette'],'high_contrast')
                self.assertTrue(cards[3].render_spec['stacked'])
                self.assertEqual(len({c.image for c in cards}),4)
            finally:r.close()

    def test_selected_chart_subject_cannot_inherit_an_unrequested_group(self):
        from core.analysis_agent.goal_contract import response_schema
        from core.analysis_agent.goal_interpreter import GoalInterpreter
        from core.analysis_agent.task_selection import verify_goal
        selection={'mode':'execute','capabilities':['chart'],'chart_kind':'histogram',
            'source_reference':'previous_analysis','current_result_only':False,
            'source_mentions':[],'output_columns':['label'],'group_column':''}
        schema=response_schema(['chart'],'execute','histogram',['signal'])
        GoalInterpreter.bind_output_subject(schema,selection)
        self.assertEqual(schema['properties']['columns'],{'const':['label']})
        variants=schema['properties']['tasks']['items']['anyOf'][0]['properties']['options']['anyOf']
        self.assertEqual(len(variants),1)
        self.assertNotIn('category',variants[0]['properties'])
        wrong=goal('chart',{'kind':'histogram','axes':{'x':'signal'}},columns=['label'])
        wrong['source_reference']='previous_analysis'
        with self.assertRaises(ValueError):verify_goal(wrong,selection,SimpleNamespace())

    def test_group_subject_contract_uses_observed_database_types(self):
        from core.analysis_agent.task_selection import validate,schema_for
        data={'tables':[{'columns':[{'name':'signal','dtype':'double'},
                                    {'name':'cohort','dtype':'varchar'}]}]}
        selection={'mode':'execute','capabilities':['chart'],'source_reference':'explicit',
            'source_mentions':[],'chart_kind':'histogram','output_columns':['cohort'],'group_column':'signal'}
        with self.assertRaisesRegex(ValueError,'numeric measure'):validate(selection,'split by cohort',data)
        selection.update(output_columns=['signal'],group_column='cohort')
        self.assertEqual(validate(selection,'split by cohort',data)['output_columns'],['signal'])
        shapes=[s for s in schema_for(data)['anyOf'] if s['properties'].get('group_column',{}).get('enum')]
        self.assertEqual(len(shapes),1)
        self.assertEqual(shapes[0]['properties']['output_columns']['items']['enum'],['signal'])
        self.assertEqual(shapes[0]['properties']['group_column']['enum'],['cohort'])

    def test_source_alias_is_normalized_before_population_audit(self):
        from core.analysis_agent.goal_contract import compile_goal
        from core.analysis_agent.population_audit import audit
        scope={'conditions':[{'column':'reading','op':'ge','value':3}],'any_conditions':[]}
        plan=goal('chart',{'kind':'histogram','axes':{'x':'reading'}},sources=['observations'],
            columns=['reading'],conditions=scope['conditions'])
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,GoalModel(goals=[plan]))
            try:
                current={'request_id':'x','request_text':'add legend','confirmed_analysis':{
                    'required_sources':['lab.observations'],'scope':scope}}
                compiled=compile_goal(current,plan,r.context)
                self.assertEqual(compiled['goal']['sources'],['lab.observations'])
                i=r.recovery.goal_interpreter;i.model_recovery=None
                seen=[]
                def invoke(messages):
                    seen.append(json.loads(messages[1].content)['previous_requested_population'])
                    return AIMessage(content=json.dumps({'conditions':[],'any_conditions':[],
                        'change':'keep','evidence_quote':''}))
                i.population_model=SimpleNamespace(invoke=invoke)
                result=audit(i,current,{},compiled['goal'])
                self.assertEqual(seen,[scope])
                self.assertEqual(result['conditions'],scope['conditions'])
            finally:r.close()

    def test_self_grouped_histogram_is_rejected_before_tools_run(self):
        from core.analysis_agent.goal_contract import compile_goal
        plan=goal('chart',{'kind':'histogram','axes':{'x':'reading'},'category':'reading'},columns=['reading'])
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,GoalModel(goals=[plan]))
            try:
                with self.assertRaisesRegex(ValueError,'DISTINCT'):
                    compile_goal({},plan,r.context)
                self.assertFalse(r.artifacts)
            finally:r.close()

    def test_literal_subject_survives_a_subsequent_invalid_task_selection(self):
        from core.analysis_agent.task_selection import select,schema_for,validate
        schema=schema_for({})
        single=next(v for v in schema['anyOf'] if v['properties']['capabilities'].get('const')==['row_count'])
        self.assertEqual(single['properties']['chart_kind'],{'const':''})
        valid={'mode':'execute','capabilities':['row_count'],'source_reference':'explicit',
            'source_mentions':[],'chart_kind':'scatter'}
        self.assertEqual(validate(valid,'how many rows')['chart_kind'],'')
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,GoalModel(goals=[goal('row_count')]))
            try:
                i=r.recovery.goal_interpreter;i.model_recovery=None
                i.selection_model=SimpleNamespace(invoke=lambda messages:AIMessage(content=json.dumps({
                    'mode':'execute','capabilities':['chart'],'source_reference':'explicit',
                    'source_mentions':[{'name':'missing_observations','quote':'missing_observations'}],
                    'chart_kind':''})))
                current={'request_id':'bad-selection','request_text':'missing_observations row count'}
                with patch('core.analysis_agent.subject_identity.read',return_value=[{
                        'name':'missing_observations','quote':'missing_observations'}]),self.assertRaises(ValueError):
                    select(i,current,{'request':current['request_text']})
                self.assertEqual(current['requested_subject']['sources'],['missing_observations'])
                self.assertEqual(current['requested_subject']['request_id'],current['request_id'])
            finally:r.close()

    def test_frequency_style_changes_reuse_counts_preserve_original_and_render_actual_pngs(self):
        plans=[goal('chart',{'kind':'histogram','axes':{'x':'cohort'},'legend':True,
                'stacked':stacked,'palette':palette},columns=['cohort'])
            for palette,stacked in [('default',False),('high_contrast',False),('high_contrast',True)]]
        model=GoalModel(goals=[p for p in plans for _ in range(2)])
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,model)
            try:
                original=stored_dataset_digest(r.datasets,raw.id)
                cards=[]
                for text in ['cohort frequencies with legend','distinguish colors','one stacked bar']:
                    result=r.submit(text)
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery'];card=r.artifacts[state['artifact_ids'][0]]
                    self.assertTrue(card.image.startswith(b'\x89PNG'))
                    self.assertTrue(card.render_spec['legend'])
                    self.assertEqual(set(card.render_spec['legend_labels']),{'A','B'})
                    self.assertEqual(dict(zip(card.render_spec['labels'],card.render_spec['counts'])),{'A':10,'B':10})
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),original)
                    self.assertEqual(r.context.selected_dataset_id,raw.id)
                    cards.append(card)
                self.assertEqual(len({c.dataset_id for c in cards}),1)
                self.assertEqual(len({c.image for c in cards}),3)
                self.assertTrue(cards[-1].render_spec['stacked'])
                self.assertFalse(r.recovery._valid_card(cards[0],state,cards[0].dataset_id))
                self.assertTrue(r.recovery._valid_card(cards[-1],state,cards[-1].dataset_id))
            finally:r.close()

    def test_omitted_subject_grammar_is_bound_even_when_last_named_table_is_absent(self):
        from core.analysis_agent.goal_contract import response_schema
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,GoalModel(goals=[goal('metadata',{'kind':'columns'})]))
            try:
                selection={'capabilities':['metadata'],'source_reference':'previous_analysis','source_mentions':[]}
                schema=response_schema()
                r.recovery.goal_interpreter.bind_literal_schema(schema,selection,{
                    'requested_previous_subject':{'sources':['lab.missing']},
                    'verified_previous':{'required_sources':['lab.observations']}})
                self.assertEqual(schema['properties']['sources'],{'const':['lab.missing']})
                self.assertEqual(schema['properties']['source_reference'],{'const':'previous_analysis'})
            finally:r.close()

    def test_a_bad_population_audit_cannot_widen_a_reviewed_unchanged_style_request(self):
        from core.analysis_agent.population_audit import audit
        scope={'conditions':[{'column':'reading','op':'ge','value':3}],'any_conditions':[]}
        plan=goal('chart',{'kind':'histogram'},columns=['reading'],conditions=scope['conditions'])
        with tempfile.TemporaryDirectory() as root:
            r,raw=goals.LLMGoalTests().runtime(root,GoalModel(goals=[plan]))
            try:
                i=r.recovery.goal_interpreter;i.model_recovery=None
                calls=[]
                def invoke(messages):
                    calls.append(messages)
                    return AIMessage(content=json.dumps({'conditions':[],'any_conditions':[],
                        'change':'clear','evidence_quote':'an earlier request'}))
                i.population_model=SimpleNamespace(invoke=invoke)
                result=audit(i,{'request_id':'x','request_text':'add legend','confirmed_analysis':{
                    'required_sources':['lab.observations'],'scope':scope}}, {},plan)
                self.assertEqual(result['conditions'],scope['conditions'])
                self.assertEqual(len(calls),2)
            finally:r.close()


if __name__=='__main__':unittest.main()
