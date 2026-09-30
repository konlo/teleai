"""An unavailable warehouse must never look successful to a CI shell caller."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import evaluate_remote_latest_live as live


class LiveEvaluatorExitTests(unittest.TestCase):
    def test_schema_connection_failure_reports_blocked_and_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'evaluation.json'
            argv=['eval','--source','fixture.telemetry.readings','--key','entity','--order','observed',
                  '--value','measure','--output',str(output)]
            config=SimpleNamespace(server_hostname='invalid.test',http_path='/fixture',access_token='test-secret',catalog='fixture',schema='telemetry')
            with patch('sys.argv',argv),patch.object(live.ConnectionConfig,'from_env',return_value=config),patch('databricks.sql.connect',side_effect=TimeoutError('must-not-be-logged:test-secret')),patch('builtins.print'):
                code=live.main()
            self.assertNotEqual(code,0)
            report=json.loads(output.read_text())
            self.assertEqual(report['status'],'BLOCKED')
            self.assertEqual(report['stage'],'schema')
            self.assertEqual(report['query_count'],1)
            self.assertNotIn('test-secret',output.read_text())

    def test_large_data_oracle_failure_returns_nonzero_even_when_agent_answers(self):
        from scripts import evaluate_large_latest as large
        recipe=json.loads((Path(__file__).parent/'fixtures/large_latest_recipe.json').read_text())
        recipe.update(rows=26_000,keys=260,payload_bytes=8,expected_counts={'C0':0,'C1':0,'C2':0})
        with tempfile.TemporaryDirectory() as root,patch('builtins.print'):
            large.seed(root,recipe)
            output=Path(root)/'evaluation.json'
            self.assertNotEqual(large.evaluate(root,recipe,output),0)
            report=json.loads(output.read_text())
            self.assertEqual(report['agent_status'],'answered')
            self.assertEqual(report['status'],'FAIL')
            self.assertTrue(report['raw_preserved'])
