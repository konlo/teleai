"""Interrupted work cannot block or contaminate a new analysis request."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from core.analysis_agent.interruption import cancellation_update
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_chart_display_journey import CASE, DATA
from tests import test_chart_display_journey as journeys
from tests.test_mysql_metadata_contract import NoInference
from utils.analysis_datasets import stored_dataset_digest


class UnfinishedRequestBoundaryTests(unittest.TestCase):
    def make(self, root, calls):
        return journeys.ChartDisplayJourneyTests().make(root, calls=calls)

    def seed(self, runtime):
        journeys.ChartDisplayJourneyTests().seed(runtime)
        runtime.submit(CASE['preview_prompt'])
        runtime.submit(CASE['scatter_prompt'])

    def interrupt(self, runtime, calls=None):
        human = HumanMessage(id='failed-request', content='전체 데이터 관계를 분석해줘')
        current = dict(runtime.inspect()['recovery'])
        current.update(request_id=human.id, request_text=human.content, status='working',
            artifact_ids=[], scope={'conditions':[{'column':CASE['columns'][0], 'op':'>', 'value':999}],
                'any_conditions':[], 'unresolved':[], 'columns':[]})
        additions = [human]
        if calls:
            additions.append(AIMessage(content='', tool_calls=calls))
        runtime.agent.update_state(runtime.config, {'messages':additions, 'recovery':current},
            as_node='model' if calls else 'ObservedSummarizationMiddleware.before_model')
        runtime.model_attempts.failure(human.id, 60, 0)
        return human

    def test_end_preserves_assets_selection_budgets_and_survives_new_process(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]; r=self.make(root,calls)
            try:
                self.seed(r)
                digests={key:stored_dataset_digest(r.datasets,key) for key in r.datasets.metadata}
                images={key:r.artifacts[key].image for key in r.artifacts}
                selected=r.context.selected_dataset_id
                self.interrupt(r)
                self.assertEqual(r.abandon()['status'],'cancelled')
                self.assertFalse(r.agent.get_state(r.config).next)
                self.assertEqual(r.inspect()['recovery']['status'],'cancelled')
                self.assertEqual(r.context.selected_dataset_id,selected)
                self.assertEqual(digests,{key:stored_dataset_digest(r.datasets,key) for key in r.datasets.metadata})
                self.assertEqual(images,{key:r.artifacts[key].image for key in r.artifacts})
                self.assertEqual(r.model_attempts.get('failed-request')['failures'],1)
                self.assertEqual(len(calls),1)
                self.assertEqual(r.abandon()['status'],'idle')
            finally:r.close()
            code='''
import sys
from tests.test_chart_display_journey import ChartDisplayJourneyTests
r=ChartDisplayJourneyTests().make(sys.argv[1],calls=[])
assert r.inspect()['state']=='idle'
assert r.submit('테이블의 column 다시 보여줘')['status']=='answered'
assert r.inspect()['recovery']['model_calls']==0
assert r.model_attempts.get('failed-request')['failures']==1
r.close()
'''
            result=subprocess.run([sys.executable,'-c',code,root],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_new_request_replaces_failed_scope_but_identical_request_keeps_budget(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls)
            try:
                self.seed(r); human=self.interrupt(r)
                before=r.agent.get_state(r.config)
                result=r.submit(human.content)
                self.assertEqual(result['error_category'],'unfinished_request')
                self.assertEqual(r.agent.get_state(r.config).config,before.config)
                self.assertEqual(r.model_attempts.get(human.id)['failures'],1)
                result=r.submit(CASE['scatter_prompt'])
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.inspect()['recovery']['scope']['conditions'],[])
                self.assertEqual(r.inspect()['recovery']['model_calls'],0)
                self.assertEqual(len(calls),1)
                self.assertTrue(any(m.additional_kwargs.get('analysis_status')=='cancelled' for m in r.events()))
            finally:r.close()

    def test_pending_calls_closed_without_sql_or_local_execution(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls)
            try:
                self.seed(r)
                pending=[{'id':'never-remote','name':'query_databricks','args':{
                    'source':CASE['source'],'query':f'SELECT * FROM {CASE["source"]} LIMIT 10','reason':'fixture'}},
                    {'id':'never-local','name':'local_analysis_sql','args':{}}]
                r.ledger.propose('never-remote',r.ledger.envelope(**pending[0]['args'],connection='fixture'))
                self.interrupt(r,pending)
                r.abandon()
                observations={m.tool_call_id:m for m in r.events() if isinstance(m,ToolMessage)}
                for call in pending:
                    self.assertEqual(json.loads(observations[call['id']].content)['status'],'cancelled')
                    self.assertEqual(observations[call['id']].status,'error')
                self.assertEqual(r.ledger.get('never-remote')['status'],'invalidated')
                self.assertEqual(len(calls),1)
                self.assertEqual(r.submit('테이블의 column 다시 보여줘')['status'],'answered')
            finally:r.close()

    def test_real_model_timeout_can_be_followed_by_schema_request_without_retrying_model(self):
        from httpx import ReadTimeout
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls)
            try:
                self.seed(r)
                before=set(r.artifacts)
                with patch.object(r.recovery,'_next_local',return_value=None),patch.object(
                        NoInference,'_generate',side_effect=ReadTimeout('fixture timeout')) as inference:
                    result=r.submit('다른 분석 방법으로 전체 관계를 조사해줘')
                    self.assertEqual(result['status'],'incomplete',result)
                    self.assertGreater(inference.call_count,0)
                failed_id=r.inspect()['recovery']['request_id']
                attempts=r.model_attempts.get(failed_id)
                self.assertGreater(attempts['failures'],0)
                self.assertEqual(r.submit('테이블의 column 다시 보여줘')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['model_calls'],0)
                self.assertEqual(r.model_attempts.get(failed_id),attempts)
                self.assertEqual(set(r.artifacts),before)
                self.assertEqual(len(calls),1)
            finally:r.close()

    def test_active_uncertain_and_approval_states_cannot_be_abandoned(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.make(root,[])
            try:
                self.interrupt(r)
                with r._exclusive():
                    with self.assertRaises(RuntimeError):r.abandon()
                with patch.object(r,'_pending',return_value=[{'id':'manual'}]):
                    with self.assertRaises(ValueError):r.abandon()
                with patch.object(r.ledger,'uncertain',return_value=[{'id':'unconfirmed'}]):
                    with self.assertRaises(PermissionError):r.abandon()
                    with self.assertRaises(PermissionError):r.submit('새로운 요청')
                self.assertEqual(r.inspect()['state'],'incomplete')
            finally:r.close()

    def test_restart_finishes_durable_cancellation_without_running_old_nodes(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.make(root,calls)
            try:
                self.seed(r);self.interrupt(r)
                update=cancellation_update(r.agent.get_state(r.config).values,'user_cancelled')
                # Simulate process loss immediately after persisting the decision.
                r.agent.update_state(r.config,update,as_node='model')
                self.assertTrue(r.agent.get_state(r.config).next)
            finally:r.close()
            r=self.make(root,calls)
            try:
                self.assertEqual(r.inspect()['state'],'idle')
                self.assertEqual(r.submit('테이블의 column 다시 보여줘')['status'],'answered')
                self.assertEqual(len(calls),1)
            finally:r.close()

    def test_ui_end_button_removes_resume_and_accepts_next_request(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root}),patch(
                'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=NoInference()):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=30).run()
            r=app.session_state['v1_runtime']
            r.recovery.intent_mode = 'contract_fixture'  # UI execution/rendering fixture, not an intent score.
            try:
                from tests.test_row_preview import schema
                r.reference_context_loader=lambda:[{**schema(CASE['source'])[0],
                    'columns':[{'name':c,'dtype':str(DATA[c].dtype)} for c in DATA]}]
                r._refresh_reference_context()
                raw=r.datasets.register(DATA,source=CASE['source'],coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                self.assertEqual(r.submit(CASE['scatter_prompt'])['status'],'answered')
                self.interrupt(r);app.run()
                images=len(app.get('image'))
                next(b for b in app.button if b.label=='중단된 요청 종료하고 계속하기').click().run()
                self.assertFalse(app.exception)
                self.assertFalse(any(b.label=='미완료 분석 재개' for b in app.button))
                self.assertEqual(r.inspect()['state'],'idle')
                app.chat_input[0].set_value('테이블의 column 다시 보여줘').run()
                self.assertFalse(app.exception)
                self.assertEqual(r.inspect()['recovery']['status'],'complete')
                self.assertEqual(r.inspect()['recovery']['model_calls'],0)
                self.assertEqual(len(app.get('image')),images)
            finally:r.close()
