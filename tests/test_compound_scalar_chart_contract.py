"""A scalar plus a plot of the same measure keeps both output obligations."""
import unittest
from types import SimpleNamespace
from copy import deepcopy
from core.analysis_agent.task_selection import verify_goal,validate
from core.analysis_agent.goal_interpreter import GoalInterpreter
from core.analysis_agent.goal_contract import response_schema
from tests.test_llm_goal import goal

class CompoundScalarChartContractTests(unittest.TestCase):
    def test_same_measure_contract_binds_chart_axes_without_dropping_calculation(self):
        selection={**validate(dict(mode='execute',capabilities=['calculation','chart'],
            source_reference='explicit',source_mentions=[],chart_kind='histogram',
            output_columns=['reading'],group_column='',scalar_operations=['AVG']),
            'Calculate a mean and plot its histogram'), 'current_result_only':False}
        plan=goal('calculation',{'operations':['AVG'],'group_columns':[]},columns=['reading'])
        plan['source_reference']='explicit'
        plan['tasks'].append({'capability':'chart','options':{'kind':'histogram','axes':{'x':'reading'},'legend':False,'stacked':False,'palette':'default'}})
        context=SimpleNamespace(reference_context=[],datasets=SimpleNamespace(metadata={}))
        verify_goal(plan,selection,context)
        invalid=deepcopy(plan);invalid['columns'].append('cohort')
        invalid['tasks'][1]['options']['category']='cohort'
        with self.assertRaises(ValueError):verify_goal(invalid,selection,context)
        schema=response_schema(selection['capabilities'],'execute','histogram')
        GoalInterpreter.bind_output_subject(schema,selection)
        self.assertEqual(schema['properties']['columns'],{'const':['reading']})
        from jsonschema import Draft202012Validator
        validator=Draft202012Validator(schema)
        validator.validate(plan)
        omitted=deepcopy(plan);omitted['tasks']=omitted['tasks'][:1]
        self.assertTrue(list(validator.iter_errors(omitted)))
        duplicate=deepcopy(plan);duplicate['tasks']=[plan['tasks'][0],plan['tasks'][0]]
        self.assertTrue(list(validator.iter_errors(duplicate)))
        branches=schema['properties']['tasks']['items']['anyOf']
        calc=next(b for b in branches if b['properties']['capability']['const']=='calculation')
        chart=next(b for b in branches if b['properties']['capability']['const']=='chart')
        self.assertEqual(calc['properties']['options']['properties']['operations'],{'const':['AVG']})
        for v in chart['properties']['options']['anyOf']:
            self.assertEqual(v['properties']['axes'],{'const':{'x':'reading'}})
            self.assertNotIn('category',v['properties'])
