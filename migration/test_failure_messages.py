import tempfile
import json
import unittest
from core.analysis_agent.failure_messages import remote_failure_message
from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.approvals import QueryNotSubmitted

class FailureMessageTests(unittest.TestCase):
    def test_403_is_not_user_rejection_and_unknown_is_not_unsubmitted(self):
        obs={'status':'unavailable','error_type':'QueryNotSubmitted','http_status':403}
        text=remote_failure_message([obs])
        self.assertIn('사용자 승인은 정상 처리',text)
        self.assertIn('SQL은 제출되지',text)
        self.assertIn('HTTP 403',text)
        self.assertNotIn('SQL은 제출되지',remote_failure_message([{**obs,'error_type':'TimeoutError'}]))
        self.assertIn('사용자가 조회를 취소',remote_failure_message([],True))

    def test_failure_keeps_run_id_and_status(self):
        with tempfile.TemporaryDirectory() as root:
            log=Diagnostics(root);log.run_id='current-run'
            log.failure(QueryNotSubmitted(403),stage='query_databricks')
            event=json.loads(log.path.read_text())
            self.assertEqual(event['run_id'],'current-run')
            self.assertEqual(event['http_status'],403)
