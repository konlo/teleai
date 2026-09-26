"""Credential-safe, network-free checks for graph model selection."""
import unittest

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from core.analysis_agent.model_provider import (
    DatabricksChatModel, _databricks_schema, build_analysis_chat_model)
from core.analysis_agent.policy import RuntimePolicy


class AnalysisModelProviderTests(unittest.TestCase):
    def test_default_uses_existing_local_model(self):
        model = build_analysis_chat_model(RuntimePolicy(), environ={})
        self.assertIsInstance(model, ChatOllama)
        self.assertEqual(model.model, 'gemma4:e4b')

    def test_databricks_payload_is_compatible_and_secret_is_hidden(self):
        model = build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=21),
            provider='databricks', environ={
                'DATABRICKS_HOST':'workspace.example.com',
                'DATABRICKS_TOKEN':'test-secret-value',
                'TELLY_DATABRICKS_MODEL':'example-endpoint'})
        self.assertIsInstance(model, DatabricksChatModel)
        self.assertEqual(str(model.openai_api_base), 'https://workspace.example.com/serving-endpoints')
        self.assertNotIn('test-secret-value', repr(model))
        self.assertEqual(model.model_name, 'example-endpoint')
        payload = model._get_request_payload([HumanMessage(content='test')])
        self.assertEqual(payload['max_tokens'], 2048)
        self.assertNotIn('max_completion_tokens', payload)

    def test_missing_or_insecure_connection_is_rejected(self):
        for config in ({}, {'DATABRICKS_HOST':'https://workspace.example.com'},
                       {'DATABRICKS_HOST':'http://workspace.example.com',
                        'DATABRICKS_TOKEN':'test-secret-value'}):
            with self.subTest(config=sorted(config)), self.assertRaises(ValueError):
                build_analysis_chat_model(RuntimePolicy(),
                    provider='databricks', environ=config)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            build_analysis_chat_model(RuntimePolicy(), provider='unknown', environ={})

    def test_serving_schema_removes_unsupported_keywords_recursively(self):
        schema = {'type': 'object', 'properties': {
            'columns': {'type': 'array', 'uniqueItems': True,
                        'items': {'type': 'string', 'pattern': '^[a-z]+$'}},
            'value': {'oneOf': [{'type': 'array', 'items': {'type': 'number'}},
                                {'type': 'number'}]},
        }}
        normalized = _databricks_schema(schema)
        self.assertNotIn('uniqueItems', normalized['properties']['columns'])
        self.assertNotIn('pattern', normalized['properties']['columns']['items'])
        self.assertNotIn('oneOf', normalized['properties']['value'])
        self.assertEqual(normalized['properties']['value']['type'], 'number')


if __name__ == '__main__':
    unittest.main()
