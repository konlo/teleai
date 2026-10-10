"""Company provider/SQL boundaries: fake Azure HTTP and fake SQL, no live keys."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from streamlit.testing.v1 import AppTest

from core.analysis_agent.databricks import ConnectionConfig, make_executor
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.model_roles import json_role
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.deployment_preflight import evaluate_deployment
from scripts.check_service_connections import check
from ui.analysis_preferences import saved_provider, save_provider
from tests.test_llm_goal import goal
from tests.test_row_preview import DATA, schema


AZURE = {'LLM_PROVIDER':'azure', 'AZURE_OPENAI_API_KEY':'fixture-azure-key',
         'AZURE_OPENAI_ENDPOINT':'https://azure.example.invalid/',
         'AZURE_OPENAI_DEPLOYMENT':'fixture-deployment',
         'AZURE_OPENAI_API_VERSION':'2024-02-15-preview'}
DATABASE = {'TELLY_DATA_BACKEND':'databricks','DATABRICKS_HOST':'workspace.example.invalid',
            'DATABRICKS_HTTP_PATH':'/sql/1.0/warehouses/fixture',
            'DATABRICKS_TOKEN':'fixture-db-token','DATABRICKS_CATALOG':'catalog',
            'DATABRICKS_SCHEMA':'lab'}


class AzureDatabricksServiceTests(unittest.TestCase):
    def test_legacy_azure_settings_select_azure_and_keep_api_contract(self):
        model = build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=25), environ=AZURE)
        self.assertIsInstance(model, AzureChatOpenAI)
        self.assertEqual(model.deployment_name, 'fixture-deployment')
        self.assertEqual(model.openai_api_version, AZURE['AZURE_OPENAI_API_VERSION'])
        self.assertEqual(model.request_timeout, 25)
        self.assertEqual(model.max_retries, 0)
        payload = model._get_request_payload([HumanMessage(content='fixture')])
        self.assertEqual(payload['max_tokens'], 4096)
        for key in ('temperature', 'top_p', 'seed', 'max_completion_tokens'):
            self.assertNotIn(key, payload)
        self.assertNotIn('fixture-azure-key', repr(model))
        modern = build_analysis_chat_model(RuntimePolicy(), environ={**AZURE,
            'TELLY_AZURE_TOKEN_PARAMETER':'max_completion_tokens'})
        modern_payload = modern._get_request_payload([HumanMessage(content='fixture')])
        self.assertEqual(modern_payload['max_completion_tokens'], 4096)
        self.assertNotIn('max_tokens', modern_payload)

    def test_missing_azure_config_cannot_fall_back_to_ollama_or_leak_key(self):
        for key in ('AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT','AZURE_OPENAI_DEPLOYMENT'):
            with self.subTest(key=key), self.assertRaises(ValueError) as error:
                build_analysis_chat_model(RuntimePolicy(), environ={**AZURE,key:''})
            self.assertIn(key, str(error.exception))
            self.assertNotIn('fixture-azure-key', str(error.exception))

    def test_azure_json_roles_retain_provider_and_bound_output(self):
        model = build_analysis_chat_model(RuntimePolicy(), environ=AZURE)
        role = json_role(model, {'type':'object'}, 128)
        payload = role._get_request_payload([HumanMessage(content='Return JSON')])
        self.assertEqual(role.deployment_name, model.deployment_name)
        self.assertEqual(payload['response_format'], {'type':'json_object'})
        self.assertEqual(payload['max_tokens'], 128)
        self.assertNotIn('response_format', model.model_kwargs)

    def test_preflight_accepts_azure_without_any_ollama_configuration(self):
        with tempfile.TemporaryDirectory() as storage:
            report = evaluate_deployment({**AZURE, **DATABASE, 'TELLY_V1_STORAGE':storage},
                profile='local-desktop', project_root=Path('/fixture/project'))
            self.assertTrue(report.ready, report.public())
            self.assertNotIn('ollama_configuration', [c.name for c in report.checks])
            self.assertEqual(next(c.status for c in report.checks if c.name=='azure_configuration'), 'pass')
            self.assertNotIn('fixture-azure-key', json.dumps(report.public()))
            self.assertNotIn('fixture-db-token', json.dumps(report.public()))

    def test_company_environment_overrides_saved_local_preference_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as root:
            save_provider(root, 'conversation', 'ollama')
            self.assertEqual(saved_provider(root,'conversation',environ=AZURE), 'azure')
            self.assertEqual(saved_provider(root,'conversation',environ={}), 'ollama')
            with patch.dict(os.environ, AZURE, clear=True):
                app = AppTest.from_string('from ui.analysis_preferences import render_provider_selector\n'
                    f'render_provider_selector({root!r}, "conversation")').run()
                self.assertFalse(app.exception)
                self.assertEqual(app.selectbox[0].value, 'azure')
                self.assertTrue(app.selectbox[0].disabled)
            self.assertEqual(saved_provider(root,'conversation',environ={}), 'ollama')

    def test_explicit_analysis_override_and_data_backend_are_independent(self):
        model = build_analysis_chat_model(RuntimePolicy(), environ={**AZURE, **DATABASE,
            'TELLY_ANALYSIS_MODEL_PROVIDER':'ollama'})
        from langchain_ollama import ChatOllama
        self.assertIsInstance(model, ChatOllama)
        self.assertEqual(ConnectionConfig.from_env(environ={**AZURE, **DATABASE}).catalog, 'catalog')

    def test_independent_probe_reports_session_and_execute_failure_without_model(self):
        for phase,fail in [('database_open_session', True), ('database_execute', False)]:
            connect = MagicMock()
            failure = RuntimeError('fixture-azure-key fixture-db-token private-error')
            failure.context = {'method':'OpenSession' if fail else 'ExecuteStatement', 'http-code':'400'}
            if fail:
                connect.side_effect = failure
            else:
                connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value.execute.side_effect = failure
            report = check({**AZURE, **DATABASE}, database=True, connect=connect)
            self.assertEqual(report['model_configuration'], 'PASS')
            self.assertEqual(report['database']['stage'], phase)
            self.assertEqual(report['database']['http_status'], 400)
            self.assertFalse(report['model_called'])
            connect.assert_called_once()
            for secret in ('fixture-azure-key','fixture-db-token','private-error'):
                self.assertNotIn(secret, json.dumps(report))
        connect = MagicMock()
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
        report = check({**AZURE, **DATABASE}, database=True, connect=connect)
        self.assertEqual(report['database']['status'], 'PASS')
        self.assertFalse(report['model_called'])
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value.execute.assert_called_once_with('SELECT 1')

    def test_missing_model_configuration_does_not_prevent_independent_database_probe(self):
        connect = MagicMock()
        connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
        report = check({**AZURE, **DATABASE, 'AZURE_OPENAI_API_KEY':''}, database=True, connect=connect)
        self.assertEqual(report['model_configuration'], 'FAIL')
        self.assertEqual(report['database']['status'], 'PASS')

    def test_actual_page_starts_azure_databricks_without_mysql_or_ollama(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                **AZURE, **DATABASE, 'TELLY_V1_STORAGE':root}, clear=True), \
                patch('dotenv.load_dotenv'), \
                patch('core.analysis_catalog.load_saved_reference_context', return_value=[]):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'main.py'),
                                   default_timeout=20).run()
            self.assertFalse(app.exception)
            runtime = app.session_state['v1_runtime']
            try:
                self.assertIsInstance(runtime.model, AzureChatOpenAI)
                self.assertEqual(runtime.sql_dialect, 'databricks')
                self.assertTrue(any(s.value=='azure' and s.disabled for s in app.selectbox))
                self.assertNotIn('mysql_eval', str(runtime.db.directory))
                connect = MagicMock()
                cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
                cursor.fetchone.return_value = (1,)
                button = next(b for b in app.button if b.label=='DB 연결만 확인')
                with patch('databricks.sql.connect', connect):
                    button.click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any('SELECT 1' in s.value for s in app.success))
                connect.assert_called_once()
                cursor.execute.assert_called_once_with('SELECT 1')
                # Redrawing the diagnostic result must not send another query.
                with patch('databricks.sql.connect', side_effect=AssertionError('unexpected requery')):
                    app.run()
                self.assertFalse(app.exception)
            finally:
                app.session_state['v1_runtime'].close()

    def test_azure_goal_http_to_databricks_executor_delivers_requested_rows(self):
        source = 'catalog.lab.observations'
        plan = goal('row_preview', {'limit':10}, sources=[source])
        requests = []
        def transport(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(request.url.host, 'azure.example.invalid')
            self.assertIn('/deployments/fixture-deployment/', request.url.path)
            self.assertEqual(payload['response_format'], {'type':'json_object'})
            response=plan
            if any('decision_evidence is an OBJECT' in str(m.get('content')) for m in payload['messages']):
                from core.analysis_agent.goal_grounding import required_paths
                response={**plan,'decision_evidence':{p:{'origin':'request',
                    'quote':'catalog.lab.observations table row 10개 보여줘','reference':''}
                    for p in required_paths(plan)}}
            if any('subject_identity_v1' in str(m.get('content')) for m in payload['messages']):
                response={'candidate_roles':{source:'requested_table','table':'other','row':'other'}}
            if any('population_basis_v1' in str(m.get('content')) for m in payload['messages']):
                body['choices'][0]['message']['content']=json.dumps({'basis':'source_population','quote':''})
                return httpx.Response(200,json=body)
            if any('goal_task_selection_v1' in str(m.get('content')) for m in payload['messages']):
                response={'mode':'execute','capabilities':['row_preview'],'source_reference':'explicit',
                    'source_mentions':[{'name':source,'quote':source}],
                    'chart_kind':''}
            return httpx.Response(200, json={'id':'fixture-response','object':'chat.completion',
                'created':0,'model':'fixture-model','choices':[{'index':0,
                    'message':{'role':'assistant','content':json.dumps(response)}, 'finish_reason':'stop'}]})
        client = httpx.Client(transport=httpx.MockTransport(transport))
        model = build_analysis_chat_model(RuntimePolicy(), environ=AZURE)
        # SDK clients are created with a fake transport, not patched model output.
        from openai import AzureOpenAI
        sdk = AzureOpenAI(api_key='fixture-azure-key', azure_endpoint=AZURE['AZURE_OPENAI_ENDPOINT'],
            azure_deployment='fixture-deployment', api_version=AZURE['AZURE_OPENAI_API_VERSION'],
            http_client=client, max_retries=0)
        model = model.model_copy(update={'root_client':sdk, 'client':sdk.chat.completions})
        connect = MagicMock()
        cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.description = [(c,) for c in DATA.columns]
        cursor.fetchmany.side_effect = [list(DATA.head(10).itertuples(index=False,name=None)), []]
        config = ConnectionConfig.from_env(environ=DATABASE)
        with tempfile.TemporaryDirectory() as root, patch('databricks.sql.connect', connect):
            runtime = GraphAnalysisRuntime(root,'fixture-owner','azure-journey',model,
                sql_dialect='databricks', source_namespace='catalog', connection_identity=config.identity(),
                reference_context_loader=lambda:schema(source),
                remote_factory=lambda datasets:make_executor(config,datasets))
            try:
                result = runtime.submit('catalog.lab.observations table row 10개 보여줘')
                self.assertEqual(result['status'], 'answered', result)
                self.assertEqual(runtime.inspect()['recovery']['table_preview_evidence']['rows'], 10)
                self.assertTrue(requests)
                connect.assert_called_once()
                self.assertIn('LIMIT 10', cursor.execute.call_args.args[0])
                self.assertEqual(runtime.context.sql_dialect, 'databricks')
            finally:
                runtime.close()
                client.close()


if __name__ == '__main__':
    unittest.main()
