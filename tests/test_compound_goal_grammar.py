"""Compound obligations preserve scalar shape and bounded review terminates safely."""
import tempfile,unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator,ValidationError
from core.analysis_agent.goal_contract import response_schema
from core.analysis_agent.goal_interpreter import GoalInterpreter
from tests.test_llm_goal import GoalModel,goal
from tests import test_llm_goal as fixtures

def selection(operation='AVG'):
    return {'mode':'execute','capabilities':['calculation'],'source_reference':'selected_dataset',
        'source_mentions':[],'chart_kind':'','metadata_kind':'','chart_edit_fields':[],
        'output_columns':['reading'],'group_column':'','scalar_operations':[operation],
        'population_basis':'source_population','result_reference_quote':'','current_result_only':False}

class CompoundGoalGrammarTests(unittest.TestCase):
    def test_filters_cannot_become_groups_in_scalar_and_ungrouped_chart(self):
        contract=selection();contract.update(capabilities=['calculation','chart'],chart_kind='histogram')
        schema=response_schema();GoalInterpreter.bind_output_subject(schema,contract)
        options=next(b for b in schema['properties']['tasks']['items']['anyOf']
            if b['properties']['capability']=={'const':'calculation'})['properties']['options']
        Draft202012Validator(options).validate({'operations':['AVG'],'group_columns':[]})
        with self.assertRaises(ValidationError):
            Draft202012Validator(options).validate({'operations':['AVG'],'group_columns':['reading','cohort']})
        # A declared grouped chart must not erase legitimate scalar dimensions.
        contract['group_column']='cohort';schema=response_schema()
        GoalInterpreter.bind_output_subject(schema,contract)
        grouped=next(b for b in schema['properties']['tasks']['items']['anyOf']
            if b['properties']['capability']=={'const':'calculation'})['properties']['options']
        Draft202012Validator(grouped).validate({'operations':['AVG'],'group_columns':['cohort']})

    def test_reselection_at_budget_boundary_blocks_without_typeerror(self):
        good=goal('calculation',{'operations':['AVG']},columns=['reading'])
        bad=goal('calculation',{'operations':['AVG']},columns=['absent'])
        changed=goal('calculation',{'operations':['SUM']},columns=['reading'])
        for value in [good,bad,changed]:value['source_reference']='selected_dataset'
        model=GoalModel(goals=[good,bad,bad,changed])
        with tempfile.TemporaryDirectory() as root:
            r,raw=fixtures.LLMGoalTests().runtime(root,model)
            try:
                with patch('core.analysis_agent.task_selection.select',side_effect=[selection(),selection('SUM')]):
                    result=r.submit('그 값을 계산해줘')
                self.assertEqual(result['status'],'blocked',result)
                state=r.inspect()['recovery']
                self.assertEqual(model.goal_calls,4)
                self.assertEqual(state['stop_reason'],'goal_unverified')
                self.assertTrue(state['goal_contract_error'])
                self.assertFalse(state['evidence_ids'])
                self.assertEqual(len(r.datasets.metadata),1)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()
