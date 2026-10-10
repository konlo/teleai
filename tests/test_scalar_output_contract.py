"""Scalar meaning cannot drift into grouped or inferential outputs downstream."""
import unittest
from types import SimpleNamespace
from core.analysis_agent.task_selection import validate,verify_goal,schema_for
from core.analysis_agent.goal_contract import response_schema
from core.analysis_agent.goal_interpreter import GoalInterpreter
from tests.test_llm_goal import goal


class ScalarOutputContractTests(unittest.TestCase):
    def selection(self,operations):
        return {**validate(dict(mode='execute',capabilities=['calculation'],
            source_reference='explicit',source_mentions=[],chart_kind='',
            output_columns=['reading'],group_column='',scalar_operations=operations),
            'Total only, not a mean'), 'current_result_only':False}

    def test_scalar_operation_and_measure_cannot_change_after_selection(self):
        selection=self.selection(['SUM'])
        context=SimpleNamespace(reference_context=[],datasets=SimpleNamespace(metadata={}))
        good=goal('calculation',{'operations':['SUM']},columns=['reading'])
        verify_goal(good,selection,context)
        for operations,columns in [(['AVG'],['reading']),(['SUM','AVG'],['reading']),(['SUM'],['other'])]:
            with self.subTest(operations=operations,columns=columns),self.assertRaises(ValueError):
                verify_goal(goal('calculation',{'operations':operations},columns=columns),selection,context)
        grouped=goal('calculation',{'operations':['SUM'],'group_columns':['cohort']},columns=['reading','cohort'])
        verify_goal(grouped,selection,context)
        schema=response_schema(['calculation'],'execute')
        GoalInterpreter.bind_output_subject(schema,selection)
        branch=schema['properties']['tasks']['items']['anyOf'][0]
        self.assertEqual(branch['properties']['options']['properties']['operations'],{'const':['SUM']})
        self.assertNotIn('const',schema['properties']['columns'])
        GoalInterpreter.bind_population_contract(schema,{**selection,'unrestricted_measure':True})
        for field in ['conditions','any_conditions','measure_conditions']:
            self.assertEqual(schema['properties'][field],{'const':[]})
        self.assertEqual(schema['properties']['ratio'],{'const':None})

    def test_provider_requires_operations_only_for_calculation(self):
        for branch in schema_for({})['anyOf']:
            self.assertIn('scalar_operations',branch['required'])
            props=branch['properties'];caps=props['capabilities'].get('const')
            if caps==['calculation']:
                self.assertEqual(props['scalar_operations']['minItems'],1)
                self.assertEqual(props['output_columns']['minItems'],1)
            elif caps is not None or props['mode']['const']!='execute':
                self.assertEqual(props['scalar_operations'],{'const':[]})
        with self.assertRaises(ValueError):
            validate({**self.selection(['SUM']),'capabilities':['statistics'],
                'current_result_only':False},'Confidence interval')

    def test_legacy_contract_does_not_guess_an_operation(self):
        old=self.selection([])
        self.assertEqual(old['scalar_operations'],[])
        context=SimpleNamespace(reference_context=[],datasets=SimpleNamespace(metadata={}))
        verify_goal(goal('calculation',{'operations':['AVG']},columns=['reading']),old,context)


if __name__=='__main__':unittest.main()
