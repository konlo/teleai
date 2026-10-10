"""Namespaces come from semantic literal roles, not arbitrary goal options."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from copy import deepcopy
import jsonschema
from core.analysis_agent.subject_identity import read
from core.analysis_agent.goal_interpreter import GoalInterpreter
from core.analysis_agent.goal_contract import response_schema
from core.analysis_agent.task_selection import verify_goal
from tests.test_llm_goal import goal


class InventoryNamespaceTests(unittest.TestCase):
    def subject(self,request,roles):
        model=Mock();model.invoke.return_value=SimpleNamespace(content=json.dumps({'candidate_roles':roles}))
        interpreter=SimpleNamespace(selection_model=object(),model_recovery=None,diagnostics=Mock(),
            budget=SimpleNamespace(wrap_model_call=lambda request,handler:handler(request)))
        data={'namespace':'observed_catalog'}
        with patch('core.analysis_agent.model_roles.json_role',return_value=model):
            tables=read(interpreter,{'request_id':'fixture','request_text':request},data)
        return tables,data['literal_inventory_scope']

    def test_generic_table_list_has_no_schema_and_new_namespace_is_not_a_table(self):
        cases=[('table list를 보여줘',{'table':'other','list':'other'},{'catalog':'','schema':''}),
               ('novel_schema 안의 table list',{'novel_schema':'requested_schema','table':'other','list':'other'},
                {'catalog':'','schema':'novel_schema'}),
               ('novel_catalog.novel_schema 안의 table list',
                {'novel_catalog.novel_schema':'requested_schema','table':'other','list':'other'},
                {'catalog':'novel_catalog','schema':'novel_schema'}),
               ('novel_catalog catalog의 table list',{'novel_catalog':'requested_catalog','catalog':'other','table':'other','list':'other'},
                {'catalog':'novel_catalog','schema':''})]
        for request,roles,expected in cases:
            with self.subTest(request=request):
                tables,scope=self.subject(request,roles)
                self.assertEqual(tables,[])
                self.assertEqual(scope,expected)
        tables,scope=self.subject('new_table row 10개',{'new_table':'requested_table','row':'other'})
        self.assertEqual(tables,[{'name':'new_table','quote':'new_table'}])
        self.assertEqual(scope,{'catalog':'','schema':''})

    def test_inventory_schema_and_local_validation_both_reject_unrequested_filters(self):
        selection={'mode':'execute','capabilities':['table_list'],'chart_kind':'',
            'source_reference':'explicit','source_mentions':[],'current_result_only':False,
            'inventory_scope':{'catalog':'','schema':''}}
        schema=response_schema(['table_list'],'execute')
        GoalInterpreter.bind_literal_schema(None,schema,selection,{})
        for options,valid in [({'catalog':'','schema':''},True),({'schema':'catalog'},False),
                              ({'catalog':'invented_catalog','schema':''},False)]:
            plan=goal('table_list',options,sources=[]);plan['source_reference']='explicit'
            with self.subTest(options=options):
                self.assertEqual(jsonschema.Draft202012Validator(schema).is_valid(plan),valid)
                if valid:verify_goal(plan,selection,SimpleNamespace())
                else:
                    with self.assertRaises(ValueError):verify_goal(plan,selection,SimpleNamespace())
        positive=deepcopy(selection);positive['inventory_scope']['schema']='newly_named_schema'
        plan=goal('table_list',{'schema':'newly_named_schema'},sources=[]);plan['source_reference']='explicit'
        verify_goal(plan,positive,SimpleNamespace())
