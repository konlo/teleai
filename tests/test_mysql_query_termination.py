"""Only server-confirmed execution failures may clear submission uncertainty."""
from pathlib import Path
from unittest.mock import patch
import os, tempfile, unittest
import mysql.connector
from core.analysis_agent.approvals import ApprovalLedger, QueryTerminated
from core.analysis_agent.mysql import MySQLConfig, make_executor
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_mysql_metadata_contract import NoInference
from datetime import datetime, timezone
import pandas as pd


class MySQLTerminationTests(unittest.TestCase):
    def test_timeout_setting_is_bounded_and_does_not_change_connection_identity(self):
        for seconds in ('0','601'):
            with patch.dict(os.environ,{'TELLY_MYSQL_QUERY_TIMEOUT_SECONDS':seconds}):
                with self.assertRaises(ValueError):MySQLConfig.from_env(Path.cwd())
        with patch.dict(os.environ,{'TELLY_MYSQL_QUERY_TIMEOUT_SECONDS':'360'}):
            self.assertEqual(MySQLConfig.from_env(Path.cwd()).query_timeout_seconds,360)

    def test_terminal_failure_is_failed_but_transport_disconnect_remains_unknown(self):
        for error,status in [(QueryTerminated(3024,'HY000'),'failed'),
                             (mysql.connector.DatabaseError(errno=2013,msg='Lost connection'),'unknown')]:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as root:
                ledger=ApprovalLedger(Path(root)/'ledger.sqlite',dialect='mysql')
                envelope=ledger.envelope('lab.events','SELECT COUNT(*) FROM lab.events','count','fixture')
                ledger.propose('call',envelope);ledger.authorize_automatic('call',envelope)
                def execute(_):raise error
                with self.assertRaises(type(error)):ledger.execute('call',envelope,execute)
                self.assertEqual(ledger.get('call')['status'],status)
                self.assertEqual(bool(ledger.uncertain()),status=='unknown')

    def test_runtime_reports_timeout_without_completion_or_automatic_resubmission(self):
        data=pd.DataFrame({'measurement':['A','B']})
        schema=[{'table':'lab.events','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'measurement','dtype':'object'}]}]
        calls=[]
        def factory(store):
            def execute(_):
                calls.append(1);raise QueryTerminated(3024,'HY000')
            return execute
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','timeout',NoInference(),sql_dialect='mysql',
                connection_identity='fixture',reference_context_loader=lambda:schema,remote_factory=factory,intent_mode='contract_fixture')
            try:
                result=r.submit('events measurement 분포를 보여줘')
                self.assertEqual(result['status'],'blocked',result)
                self.assertIn('조회 시간 제한',result['text'])
                self.assertEqual(len(calls),1)
                self.assertEqual(r.ledger.uncertain(),[])
                self.assertEqual(len(r.artifacts),0)
                self.assertIn('컬럼',r.submit('events 컬럼 목록 보여줘')['text'])
                self.assertEqual(len(calls),1)
            finally:r.close()
