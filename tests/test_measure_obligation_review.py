"""The focused mathematical role corrects obligations without prose rules."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from core.analysis_agent.measure_selection import review


class MeasureReviewTests(unittest.TestCase):
    def call(self,value,selection):
        if value:value={'has_row_restriction':False,**value}
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps(value))
        interpreter=SimpleNamespace(selection_model=object(),model_recovery=None,diagnostics=Mock(),
            budget=SimpleNamespace(wrap_model_call=lambda req,handler:handler(req)))
        with patch('core.analysis_agent.measure_selection.json_role',return_value=model):
            result=review(interpreter,{'request_id':'r','request_text':'Within those rows, total only'},
                {'tables':[{'columns':[{'name':'reading'}]}]},selection)
        return result,model

    def test_scalar_reading_can_correct_preliminary_grouped_obligation(self):
        selection={'capabilities':['group_summary'],'output_columns':[],'scalar_operations':[],'source_reference':'explicit'}
        value={'capability':'calculation','operations':['SUM'],'measure_columns':['reading']}
        result,model=self.call(value,selection)
        self.assertEqual(result['capabilities'],['calculation'])
        self.assertEqual(result['scalar_operations'],['SUM'])
        self.assertEqual(result['output_columns'],['reading'])
        self.assertEqual(result['source_reference'],'explicit')
        self.assertTrue(result['unrestricted_measure'])
        payload=json.loads(model.invoke.call_args.args[0][-1].content)
        self.assertEqual(payload['request'],'Within those rows, total only')
        self.assertEqual(payload['observed_columns'],['reading'])

    def test_inferential_and_grouped_work_remain_distinct(self):
        for cap in ['statistics','group_summary','profile']:
            result,_=self.call({'capability':cap,'operations':[],'measure_columns':[]},
                {'capabilities':['calculation'],'scalar_operations':['AVG'],'output_columns':['reading']})
            self.assertEqual(result['capabilities'],[cap]);self.assertEqual(result['scalar_operations'],[])
        with self.assertRaises(ValueError):
            self.call({'capability':'statistics','operations':['SUM'],'measure_columns':['reading']},
                {'capabilities':['statistics']})

    def test_other_outputs_and_compound_work_do_not_gain_scalar_tasks(self):
        for caps in [['chart'],['calculation','chart']]:
            original={'capabilities':caps}
            result,model=self.call({},original)
            self.assertIs(result,original);model.invoke.assert_not_called()
        original={'capabilities':['metadata'],'metadata_kind':'dtypes'}
        result,_=self.call({'capability':'unchanged','operations':[],'measure_columns':[]},original)
        self.assertIs(result,original)
        result,_=self.call({'capability':'calculation','operations':['AVG'],'measure_columns':['reading']},original)
        self.assertEqual(result['metadata_kind'],'')
        conceptual={'mode':'explain','capabilities':[]}
        result,_=self.call({'capability':'unchanged','operations':[],'measure_columns':[]},conceptual)
        self.assertIs(result,conceptual)
        result,_=self.call({'capability':'calculation','operations':['AVG'],'measure_columns':['reading']},conceptual)
        self.assertEqual(result['mode'],'execute')
        result,_=self.call({'capability':'calculation','operations':['SUM'],
            'measure_columns':['reading'],'has_row_restriction':True},conceptual)
        self.assertFalse(result['unrestricted_measure'])


if __name__=='__main__':unittest.main()
