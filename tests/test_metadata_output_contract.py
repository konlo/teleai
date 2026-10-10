"""Schema proof must satisfy requested output, and types must appear in the UI."""
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest
from streamlit.testing.v1 import AppTest
from core.analysis_agent.task_selection import schema_for,validate,verify_goal
from core.analysis_agent.goal_contract import response_schema
from core.analysis_agent.goal_interpreter import GoalInterpreter
from core.analysis_agent.completion_renderers import render_metadata
from tests.test_llm_goal import goal


class MetadataOutputContractTests(unittest.TestCase):
    def selection(self,kind):
        return validate(dict(mode='execute',capabilities=['metadata'],source_reference='explicit',
            source_mentions=[],chart_kind='',output_columns=[],group_column='',metadata_kind=kind),
            'Show field names and database types')

    def test_metadata_subtype_cannot_be_dropped_by_detailed_planner(self):
        context=SimpleNamespace(reference_context=[],datasets=SimpleNamespace(metadata={}))
        selection={**self.selection('dtypes'),'current_result_only':False}
        with self.assertRaisesRegex(ValueError,'Metadata must preserve'):
            verify_goal(goal('metadata',{'kind':'columns'}),selection,context)
        verify_goal(goal('metadata',{'kind':'dtypes'}),selection,context)
        schema=response_schema(['metadata'],'execute')
        GoalInterpreter.bind_output_subject(schema,selection)
        branch=schema['properties']['tasks']['items']['anyOf'][0]
        self.assertEqual(branch['properties']['options']['properties']['kind'],{'const':'dtypes'})
        names={**selection,'metadata_kind':'columns'}
        verify_goal(goal('metadata',{'kind':'columns'}),names,context)
        with self.assertRaisesRegex(ValueError,'Metadata must preserve'):
            verify_goal(goal('metadata',{'kind':'dtypes'}),names,context)

    def test_provider_grammar_requires_subtype_only_on_metadata_branch(self):
        branches=schema_for({})['anyOf']
        for branch in branches:
            self.assertIn('metadata_kind',branch['required'])
            props=branch['properties'];caps=props['capabilities'].get('const')
            if caps==['metadata']:
                self.assertEqual(set(props['metadata_kind']['enum']),{'columns','dtypes','numeric_columns','categorical_columns'})
            elif caps is not None or props['mode']['const']!='execute':
                self.assertEqual(props['metadata_kind'],{'const':''})
        with self.assertRaises(ValueError):
            validate({**self.selection('dtypes'),'capabilities':['row_count']},'Count rows')

    def test_visible_type_table_survives_rerun_and_subject_change(self):
        evidence={'kind':'dtypes','table':'lab.alpha','schema':[{'name':'reading','dtype':'decimal(12,4)'},
            {'name':'recorded_at','dtype':'timestamp'}],'type_authority':'current_database_metadata'}
        text=render_metadata(None,{'metadata_evidence':evidence})
        script='import streamlit as st\nst.markdown('+repr(text)+')'
        app=AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        visible=app.markdown[0].value
        for fragment in ['lab.alpha','| reading | decimal(12,4) |','| recorded_at | timestamp |']:
            self.assertIn(fragment,visible)
        self.assertIn('DB 데이터 타입',visible)
        app.run();self.assertEqual(app.markdown[0].value,visible)
        other=deepcopy(evidence);other.update(table='lab.beta',schema=[{'name':'category','dtype':'varchar(64)'}])
        second=render_metadata(None,{'metadata_evidence':other})
        app=AppTest.from_string('import streamlit as st\nst.markdown('+repr(second)+')').run()
        self.assertIn('| category | varchar(64) |',app.markdown[0].value)
        self.assertNotIn('recorded_at',app.markdown[0].value)
        self.assertNotIn('lab.alpha',app.markdown[0].value)

if __name__=='__main__':unittest.main()
