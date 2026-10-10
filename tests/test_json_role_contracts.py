"""Provider payload contracts and admission failures; no real Azure credentials."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
import httpx
import jsonschema
from langchain_core.messages import HumanMessage, SystemMessage
from openai import AzureOpenAI

from core.analysis_agent.json_contract import compact_schema, encoded, invoke_role
from core.analysis_agent.model_context import ModelContextBudgetMiddleware, ModelContextBudgetExceeded
from core.analysis_agent.model_roles import json_role
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.task_selection import SCHEMA, schema_for
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.support_report import summarize, brief
from tests.test_azure_databricks_service import AZURE
from tests.test_llm_goal import goal


def request_for(system='Return JSON',text='table list를 보여줘'):
    req=SimpleNamespace(state={},system_message=SystemMessage(content=system),
        messages=[HumanMessage(content=text)],tools=[])
    def override(**changes):
        result=SimpleNamespace(**{**vars(req),**changes});result.override=override;return result
    req.override=override
    return req


def azure_with_transport(transport):
    model=build_analysis_chat_model(RuntimePolicy(),environ=AZURE)
    sdk=AzureOpenAI(api_key=AZURE['AZURE_OPENAI_API_KEY'],azure_endpoint=AZURE['AZURE_OPENAI_ENDPOINT'],
        azure_deployment=AZURE['AZURE_OPENAI_DEPLOYMENT'],api_version=AZURE['AZURE_OPENAI_API_VERSION'],
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),max_retries=0)
    return model.model_copy(update={'root_client':sdk,'client':sdk.chat.completions})


def response(value):
    return httpx.Response(200,json={'id':'fixture-response','object':'chat.completion','created':0,
        'model':'fixture-model','choices':[{'index':0,'message':{'role':'assistant',
        'content':json.dumps(value,ensure_ascii=False)},'finish_reason':'stop'}]})


class JsonRoleContractTests(unittest.TestCase):
    def test_azure_wire_includes_exact_required_fields_without_schema_api_dependency(self):
        sent=[]
        def transport(request):
            payload=json.loads(request.content);sent.append(payload)
            return response({'answer':'fixture'})
        original=azure_with_transport(transport)
        schema={'type':'object','required':['answer'],'additionalProperties':False,
                'properties':{'answer':{'type':'string','enum':['fixture']}}}
        model=json_role(original,schema,128);req=request_for()
        recorded=[]
        class Budget:
            def wrap_model_call(self,request,handler):
                recorded.append(request);return handler(request)
        invoke_role(model,req,Budget())
        system=sent[0]['messages'][0]['content']
        delivered=json.loads(system.split('not evidence of user intent.\n')[1])
        self.assertEqual(delivered,schema)
        self.assertEqual(recorded[0].system_message.content,system)
        self.assertEqual(sent[0]['response_format'],{'type':'json_object'})
        self.assertNotIn('_telly_output_schema',sent[0])
        self.assertNotIn('JSON output contract:',req.system_message.content)
        self.assertFalse(getattr(original,'_telly_output_schema',None))
        self.assertEqual(sent[0]['max_tokens'],128)

    def test_contract_is_budgeted_before_network_and_never_silently_dropped(self):
        sent=[]
        model=json_role(azure_with_transport(lambda r:sent.append(r) or response({})),SCHEMA,512)
        with self.assertRaises(ModelContextBudgetExceeded):
            invoke_role(model,request_for(),ModelContextBudgetMiddleware(model,None,100))
        self.assertEqual(sent,[])

    def test_factoring_preserves_selection_shapes_and_invalid_cases(self):
        original=schema_for({});copy=deepcopy(original);compact=compact_schema(original)
        self.assertEqual(original,copy)
        self.assertLess(len(encoded(compact).encode()),len(encoded(original).encode())*0.6)
        valid={'mode':'execute','capabilities':['table_list'],'source_reference':'explicit',
               'source_mentions':[],'chart_kind':'','output_columns':[],'group_column':'',
               'metadata_kind':'','scalar_operations':[],'chart_edit_fields':[]}
        cases=[valid,{**valid,'extra':True},{**valid,'mode':'explain'},
               {**valid,'capabilities':['metadata'],'metadata_kind':'dtypes'},
               {**valid,'capabilities':['row_preview']},
               {**valid,'capabilities':['chart'],'chart_kind':'scatter','output_columns':['left','right']},
               {**valid,'capabilities':['chart'],'chart_kind':'scatter','output_columns':['left']},
               {**valid,'capabilities':['calculation'],'scalar_operations':['SUM'],'output_columns':['field']},
               {**valid,'capabilities':['calculation'],'scalar_operations':[]},
               {**valid,'mode':'explain','capabilities':[]},{**valid,'mode':'clarify','capabilities':[]}]
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(jsonschema.Draft202012Validator(original).is_valid(value),
                                 jsonschema.Draft202012Validator(compact).is_valid(value))
        self.assertTrue(jsonschema.Draft202012Validator(compact).is_valid(valid))
        invalid={**valid,'chart_kind':'scatter'}
        self.assertFalse(jsonschema.Draft202012Validator(compact).is_valid(invalid))

    def test_azure_inventory_goal_repair_then_real_graph_tool_result(self):
        # Simulate a provider that follows delivered JSON instructions, not an
        # oracle that supplies correct fields regardless of its actual input.
        prompt='table list를 보여줘';seen=[];selection_attempts=0;subject_attempts=0;validation_errors=[]
        plan=goal('table_list',{'catalog':'','schema':''},sources=[])
        plan['source_reference']='explicit'
        def transport(request):
            nonlocal selection_attempts,subject_attempts
            payload=json.loads(request.content);seen.append(payload)
            system=payload['messages'][0]['content']
            if 'JSON output contract:' not in system:
                return response({})
            contract=json.loads(system.split('not evidence of user intent.\n')[1])
            if 'subject_identity_v1' in system:
                subject_attempts+=1
                if subject_attempts==1:return response({'table_indexes':[]})
                value={'candidate_roles':{'table':'other','list':'other'}}
            elif 'goal_task_selection_v1' in system:
                selection_attempts+=1
                if selection_attempts==1:return response({'intent':'table_list'})
                value={'mode':'execute','capabilities':['table_list'],'source_reference':'explicit',
                       'source_mentions':[],'chart_kind':'','output_columns':[],'group_column':'',
                       'metadata_kind':'','scalar_operations':[],'chart_edit_fields':[]}
            else:
                value=deepcopy(plan)
                if 'decision_evidence is an OBJECT' in system:
                    value['decision_evidence']={'/tasks/0':{'origin':'request','quote':prompt,'reference':''}}
            try:jsonschema.validate(value,contract)
            except jsonschema.ValidationError as exc:
                validation_errors.append({'validator':exc.validator,'path':list(exc.path)});raise
            return response(value)
        model=azure_with_transport(transport);calls=[]
        def factory(datasets):
            def execute(envelope):
                from dataclasses import asdict
                import pandas as pd
                calls.append(envelope)
                frame=pd.DataFrame([{'table_schema':'lab','table_name':'unfamiliar_events','table_type':'BASE TABLE'}])
                d=datasets.register(frame,source=envelope['source'],query=envelope['query'],
                    coverage='complete',predicate_known=True)
                return {'status':'ready','dataset':asdict(d)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'fixture','inventory',model,source_namespace='catalog',
                connection_identity='fixture-db',remote_factory=factory,sql_dialect='databricks')
            try:
                result=runtime.submit(prompt)
                self.assertEqual(result['status'],'answered',(result,validation_errors,len(seen),summarize(runtime.diagnostics.path)))
                self.assertEqual(selection_attempts,2)
                self.assertEqual(subject_attempts,2)
                self.assertEqual(len(calls),1)
                self.assertIn('catalog.information_schema.tables',calls[0]['query'].replace('`',''))
                self.assertNotIn('WHERE',calls[0]['query'])
                self.assertIn('unfamiliar',result['text'])
                self.assertFalse(runtime.inspect()['recovery'].get('goal_interpretation_error'))
                self.assertIn('decision_evidence',encoded(json.loads(seen[-1]['messages'][0]['content'].split(
                    'not evidence of user intent.\n')[1])))
            finally:runtime.close()

    def test_unrepairable_contract_produces_findable_error_without_db_or_private_text(self):
        private='private_identifier secret-token-do-not-export'
        model=azure_with_transport(lambda r:response({'unexpected':private}))
        factory=MagicMock()
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'fixture','blocked',model,source_namespace='catalog',
                connection_identity='fixture-db',remote_factory=factory,sql_dialect='databricks')
            try:
                result=runtime.submit('table list를 보여줘')
                state=runtime.inspect()['recovery'];error_id=state['goal_error_id']
                self.assertEqual(state['failure_stage'],'goal_task_selection')
                self.assertIn(error_id,result['text'])
                factory.return_value.assert_not_called()
                report=summarize(runtime.diagnostics.path,error_id=error_id)
                self.assertTrue(report['found'])
                self.assertEqual(report['remote']['started'],0)
                self.assertEqual(report['errors'][-1]['stage'],'goal_task_selection')
                self.assertNotIn(private,encoded(report)+brief(report)+Path(runtime.diagnostics.path).read_text())
            finally:runtime.close()
