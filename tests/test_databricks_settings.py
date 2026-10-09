"""Existing .env compatibility and credential forwarding without live secrets."""
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import pandas as pd

from app_io.databricks import DatabricksConfig
from core.analysis_agent.databricks import ConnectionConfig, make_executor
from core.analysis_agent.assets import AssetDB, PersistentDatasets
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.databricks_settings import normalize_hostname
from core.deployment_preflight import evaluate_deployment


SETTINGS = {
    'DATABRIckS_HOST': ' https://workspace.example.invalid/ ',
    'DATABRiCkS_HTTP_PATH': ' /sql/1.0/warehouses/fixture ',
    'DATABRiCkS_TOKEN': ' fixture-token ',
    'DATABRICKS_CATALOG': ' fixture_catalog ',
    'DATABRICKS_SCHEMA': ' fixture_schema ',
    'AZURE_STORAGE_ACCounT': 'optional-fixture-account',
    'BLOB_CONNECTION_STR': 'optional-fixture-connection',
}


class DatabricksSettingsTests(unittest.TestCase):
    def test_legacy_and_agent_parse_existing_names_identically_without_mutation(self):
        original = dict(SETTINGS)
        current = ConnectionConfig.from_env(environ=SETTINGS)
        current.validate()
        legacy = DatabricksConfig.from_env(environ=SETTINGS)
        for key in ('server_hostname', 'http_path', 'access_token', 'catalog', 'schema'):
            self.assertEqual(getattr(current, key), getattr(legacy, key))
        self.assertEqual(current.server_hostname, 'workspace.example.invalid')
        self.assertEqual(current.access_token, 'fixture-token')
        self.assertEqual(SETTINGS, original)
        self.assertNotIn('fixture-token', repr(current))
        self.assertNotIn('fixture-token', repr(legacy))

    def test_empty_primary_token_uses_existing_access_token_alias(self):
        config = ConnectionConfig.from_env(environ={
            'DATABRICKS_HOST':'workspace.example.invalid',
            'DATABRICKS_HTTP_PATH':'/sql/fixture',
            'DATABRICKS_TOKEN':'   ', 'Databricks_Access_Token':' alias-token '})
        config.validate()
        self.assertEqual(config.access_token, 'alias-token')

    def test_conflicting_case_variants_stop_without_disclosing_values(self):
        settings = {**SETTINGS, 'DATABRICKS_TOKEN':'another-sensitive-value'}
        with self.assertRaises(ValueError) as error:
            ConnectionConfig.from_env(environ=settings)
        self.assertIn('DATABRICKS_TOKEN', str(error.exception))
        self.assertNotIn('fixture-token', str(error.exception))
        self.assertNotIn('another-sensitive-value', str(error.exception))

    def test_blob_credentials_cannot_substitute_for_missing_sql_credentials(self):
        config = ConnectionConfig.from_env(environ={
            'AZURE_STORAGE_ACCOUNT':'sensitive-account',
            'BLOB_CONNECTION_STR':'sensitive-connection'})
        with self.assertRaises(ValueError) as error:
            config.validate()
        for name in ('DATABRICKS_HOST', 'DATABRICKS_HTTP_PATH', 'DATABRICKS_TOKEN'):
            self.assertIn(name, str(error.exception))
        self.assertNotIn('sensitive', str(error.exception))

    def test_invalid_workspace_url_is_rejected_without_echoing_url(self):
        for host in ('https://workspace.example.invalid/sql/warehouses/fixture',
                     'https://secret-user:secret-pass@workspace.example.invalid/',
                     'https://workspace.example.invalid/?token=secret-token',
                     'ftp://workspace.example.invalid/', 'workspace example.invalid'):
            with self.subTest(host=host), self.assertRaises(ValueError) as error:
                normalize_hostname(host)
            self.assertNotIn('secret-', str(error.exception))
        self.assertEqual(normalize_hostname('workspace.example.invalid/'),
                         'workspace.example.invalid')

    def test_model_serving_and_sql_share_token_alias_and_hostname_parsing(self):
        settings = {**SETTINGS, 'DATABRiCkS_TOKEN':' ',
                    'DATABRICKS_ACCESS_TOKEN':' alias-token '}
        model = build_analysis_chat_model(RuntimePolicy(), provider='databricks', environ=settings)
        self.assertEqual(str(model.openai_api_base),
                         'https://workspace.example.invalid/serving-endpoints')
        self.assertEqual(model.openai_api_key.get_secret_value(), 'alias-token')
        self.assertNotIn('alias-token', repr(model))

    def test_normalized_env_reaches_sql_connector_and_loads_bounded_result(self):
        config = ConnectionConfig.from_env(environ=SETTINGS)
        connect = MagicMock()
        cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.description = [('reading',)]
        cursor.fetchmany.side_effect = [[(2,), (4,)], []]
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'fixture-owner', 'fixture-conversation')
            datasets = PersistentDatasets(db)
            try:
                source = 'fixture_catalog.fixture_schema.observations'
                envelope = {'connection':config.identity(), 'source':source,
                            'query':'SELECT reading FROM '+source+' LIMIT 10'}
                with patch('databricks.sql.connect', connect):
                    result = make_executor(config, datasets)(envelope)
                connect.assert_called_once_with(server_hostname='workspace.example.invalid',
                    http_path='/sql/1.0/warehouses/fixture', access_token='fixture-token',
                    catalog='fixture_catalog', schema='fixture_schema')
                cursor.execute.assert_called_once_with(envelope['query'])
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(result['dataset']['rows'], 2)
                self.assertEqual(result['preview'], [{'reading':2}, {'reading':4}])
            finally:
                db.close()

    def test_connect_failure_does_not_retry_or_discard_existing_dataset(self):
        config = ConnectionConfig.from_env(environ=SETTINGS)
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'fixture-owner', 'fixture-conversation')
            datasets = PersistentDatasets(db)
            try:
                original = datasets.register(pd.DataFrame({'reading':[9]}),
                    source='fixture_catalog.fixture_schema.saved_observations')
                envelope = {'connection':config.identity(),
                    'source':'fixture_catalog.fixture_schema.observations',
                    'query':'SELECT reading FROM fixture_catalog.fixture_schema.observations LIMIT 10'}
                with patch('databricks.sql.connect', side_effect=TimeoutError) as connect:
                    with self.assertRaises(TimeoutError):
                        make_executor(config, datasets)(envelope)
                connect.assert_called_once()
                self.assertEqual(set(datasets.metadata), {original.id})
                self.assertEqual(datasets.frames[original.id]['reading'].tolist(), [9])
            finally:
                db.close()

    def test_preflight_uses_same_casing_and_rejects_conflicts_without_secrets(self):
        with tempfile.TemporaryDirectory() as storage:
            settings = {**SETTINGS, 'TELLY_V1_STORAGE':storage,
                        'OLLAMA_MODEL':'fixture', 'OLLAMA_BASE_URL':'http://localhost:11434'}
            for conflict in (False, True):
                with self.subTest(conflict=conflict):
                    environ = {**settings, **({'DATABRICKS_TOKEN':'other-private-token'} if conflict else {})}
                    report = evaluate_deployment(environ, profile='local-desktop',
                                                 project_root=Path('/fixture/project'))
                    check = next(c for c in report.checks if c.name=='databricks_configuration')
                    self.assertEqual(check.status, 'fail' if conflict else 'pass')
                    self.assertNotIn('fixture-token', str(report.public()))
                    self.assertNotIn('other-private-token', str(report.public()))


if __name__ == '__main__':
    unittest.main()
