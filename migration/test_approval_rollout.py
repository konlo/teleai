import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult,ChatGeneration
from migration.approval_ledger import ApprovalLedger
from migration.graph_runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel


class StatusModel(QuietModel):
    def _generate(self,messages,**kwargs):
        text='{"action":"status"}' if messages[0].content.startswith('사용자 메시지가 오직') else '상태 확인'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class LedgerTests(unittest.TestCase):
    def test_approval_exact_binding_and_replay_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=ApprovalLedger(Path(root)/'ledger.sqlite')
            envelope=ledger.envelope('fixture','SELECT 1','test','connection')
            ledger.propose('one',envelope);calls=[]
            with self.assertRaises(PermissionError):ledger.execute('one',envelope,lambda e:calls.append(e))
            ledger.decide('one',True)
            altered=dict(envelope,connection='changed')
            with self.assertRaises(PermissionError):ledger.execute('one',altered,lambda e:calls.append(e))
            result=ledger.execute('one',envelope,lambda e:calls.append(e) or {'answer':1})
            reopened=ApprovalLedger(Path(root)/'ledger.sqlite')
            self.assertEqual(reopened.execute('one',envelope,lambda e:self.fail('replayed')),result)
            self.assertEqual(len(calls),1)

    def test_process_crash_after_claim_never_resubmits(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'ledger.sqlite';ledger=ApprovalLedger(path)
            envelope=ledger.envelope('fixture','SELECT 1','test','connection')
            ledger.propose('one',envelope);ledger.decide('one',True)
            code='''
import sys,os,json
from migration.approval_ledger import ApprovalLedger
l=ApprovalLedger(sys.argv[1])
l.execute('one',json.loads(sys.argv[2]),lambda e:os._exit(23))
'''
            process=subprocess.run([sys.executable,'-c',code,str(path),json.dumps(envelope)],timeout=20)
            self.assertEqual(process.returncode,23)
            self.assertEqual(ledger.get('one')['status'],'submitting')
            with self.assertRaises(PermissionError):ledger.execute('one',envelope,lambda e:self.fail('resubmitted'))

    def test_proven_open_session_failure_is_not_unknown_submission(self):
        from databricks.sql.exc import RequestError
        from core.analysis_agent.databricks import ConnectionConfig,make_executor
        from utils.analysis_datasets import DatasetStore
        with tempfile.TemporaryDirectory() as root:
            ledger=ApprovalLedger(Path(root)/'ledger.sqlite')
            config=ConnectionConfig('test-host','test-path','test-token','','')
            envelope=ledger.envelope('fixture','SELECT 1','test',config.identity())
            ledger.propose('one',envelope);ledger.decide('one',True)
            with patch('core.analysis_agent.databricks.execute_approved',side_effect=RequestError(context={'method':'OpenSession','http-code':403})):
                with self.assertRaises(Exception):ledger.execute('one',envelope,make_executor(config,DatasetStore()))
            self.assertEqual(ledger.get('one')['status'],'failed')
            self.assertEqual(ledger.uncertain(),[])

    def test_exception_marks_unknown_and_requires_new_approval(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=ApprovalLedger(Path(root)/'ledger.sqlite')
            envelope=ledger.envelope('fixture','SELECT 1','test','connection')
            ledger.propose('one',envelope);ledger.decide('one',True)
            def fail(e):raise TimeoutError()
            with self.assertRaises(TimeoutError):ledger.execute('one',envelope,fail)
            self.assertEqual(ledger.get('one')['status'],'unknown')
            with self.assertRaises(PermissionError):ledger.execute('one',envelope,lambda e:self.fail('retried'))


class GraphApprovalTests(unittest.TestCase):
    def runtime(self,root,model,calls,connection='conn'):
        return GraphAnalysisRuntime(root,'owner','thread',model,connection_identity=connection,
            remote_factory=lambda d:lambda e:calls.append(e) or {'status':'ready','value':1})

    def test_reopen_status_question_approve_and_duplicate(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,StatusModel(),calls)
            pending=r.propose_query('fixture','SELECT 1','test')['requests'][0]
            self.assertEqual(calls,[]);r.close()
            r=self.runtime(root,StatusModel(),calls)
            self.assertEqual(r.submit('지금 어떤 승인을 기다려?')['status'],'awaiting_approval')
            self.assertEqual(r.inspect()['requests'][0]['id'],pending['id'])
            self.assertEqual(r.respond(pending['id'],approved=True)['status'],'answered')
            with self.assertRaises(PermissionError):r.respond(pending['id'],approved=True)
            self.assertEqual(len(calls),1);r.close()

    def test_explicit_chat_approval_and_rejection(self):
        for text,count in [('승인해줘',1),('취소해줘',0)]:
            with tempfile.TemporaryDirectory() as root:
                calls=[];r=self.runtime(root,QuietModel(),calls)
                r.propose_query('fixture','SELECT 1','test')
                self.assertEqual(r.submit(text)['status'],'answered')
                self.assertEqual(len(calls),count);r.close()

    def test_changed_request_invalidates_old_approval(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,QuietModel(),calls)
            pending=r.propose_query('fixture','SELECT 1','test')['requests'][0]
            r.submit('기간을 바꿔줘')
            self.assertEqual(r.ledger.get(pending['id'])['status'],'invalidated')
            with self.assertRaises(PermissionError):r.respond(pending['id'],approved=True)
            self.assertEqual(calls,[]);r.close()

    def test_chart_actions_finish_with_approval_middleware_enabled(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,QuietModel(),calls)
            fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
            info=r.datasets.register(pd.DataFrame(fixture['rows']),source=fixture['source'],coverage='complete',predicate_known=True)
            cards=r.recommend_charts(info.id)
            self.assertEqual(r.inspect()['state'],'idle')
            r.select_chart(cards['cards'][1]['id'])
            self.assertEqual(r.inspect()['state'],'idle')
            self.assertEqual(calls,[]);r.close()

    def test_decline_and_changed_connection_do_not_execute(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,QuietModel(),calls)
            pending=r.propose_query('fixture','SELECT 1','test')['requests'][0]
            r.close();r=self.runtime(root,QuietModel(),calls,connection='changed')
            with self.assertRaises(PermissionError):r.respond(pending['id'],approved=True)
            r.close();r=self.runtime(root,QuietModel(),calls)
            r.cancel(pending['id']);self.assertEqual(calls,[]);r.close()


class RolloutPageTests(unittest.TestCase):
    def test_new_page_example_propose_cancel_and_reopen(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root}),\
             patch('langchain_ollama.ChatOllama',return_value=QuietModel()),\
             patch('databricks.sql.connect') as connect:
            app=AppTest.from_file(str(Path('main.py').resolve()),default_timeout=20).run()
            self.assertEqual(len(app.exception),0)
            next(b for b in app.button if b.label=='예제 데이터로 시작').click().run()
            self.assertEqual(len(app.session_state.v1_runtime.datasets.metadata),1)
            app.text_input[0].set_value('catalog.schema.events').run()
            next(b for b in app.button if b.label=='데이터 불러오기 제안').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.session_state.v1_runtime.inspect()['requests']),1)
            next(b for b in app.button if b.label=='조회 취소').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.session_state.v1_runtime.inspect()['requests'],[])
            connect.assert_not_called()
            app.session_state.v1_runtime.close()

if __name__=='__main__':unittest.main()
