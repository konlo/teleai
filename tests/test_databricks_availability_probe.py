"""Connectivity cannot be inferred from READY metadata or used as an agent score."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock

from scripts.probe_databricks_availability import error_result, probe, summarize
from tests.test_model_service_unavailable import error


class AvailabilityProbeTests(unittest.TestCase):
    def test_metadata_success_does_not_mask_inference_and_open_session_failure(self):
        config = SimpleNamespace(server_hostname='private-host', http_path='/private',
                                 access_token='secret-not-for-report')
        connect_error = RuntimeError('secret-not-for-report')
        connect_error.context = {'http-code': 400}
        connect = Mock(side_effect=connect_error)
        invoke = Mock(side_effect=error())
        result = probe(config, control=lambda: {'status': 'PASS', 'state': 'RUNNING'},
                       invoke_model=invoke, connect=connect)
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertFalse(result['temporary_confirmed'])
        self.assertEqual(result['checks']['sql']['stage'], 'OpenSession')
        self.assertEqual(result['checks']['model']['http_status'], 400)
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(connect.call_args.kwargs['_retry_stop_after_attempts_count'], 1)
        for secret in ('secret-not-for-report', 'private-host', 'foundation model endpoints'):
            self.assertNotIn(secret, json.dumps(result))

    def test_stopped_warehouse_can_autostart_and_real_calls_establish_availability(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (1,)
        config = SimpleNamespace(server_hostname='host', http_path='/sql', access_token='secret')
        result = probe(config, control=lambda: {'status': 'PASS', 'state': 'STOPPED'},
                       invoke_model=lambda: SimpleNamespace(content='OK'),
                       connect=Mock(return_value=connection))
        self.assertEqual(result['status'], 'AVAILABLE')
        self.assertEqual(result['scope'], 'connectivity_only_not_agent_quality_or_release')
        cursor.execute.assert_called_once_with('SELECT 1')
        self.assertEqual(cursor.fetchone.call_count, 1)
        cursor.fetchone.return_value = (2,)
        failed = probe(config, control=lambda: {},
                       invoke_model=lambda: SimpleNamespace(content='OK'),
                       connect=Mock(return_value=connection))
        self.assertEqual(failed['status'], 'BLOCKED')

    def test_missing_or_auth_failure_does_not_become_temporary_outage(self):
        self.assertEqual(summarize({})['status'], 'BLOCKED')
        for status in (401, 403, 400):
            result = error_result(error('Invalid configuration', status=status), 'model')
            self.assertIsNone(result['category'])
            self.assertFalse(result['temporary_confirmed'])
        e = RuntimeError('private')
        e.context = {'http-code': 'private'}
        self.assertIsNone(error_result(e, 'sql')['http_status'])


if __name__ == '__main__':
    unittest.main()
