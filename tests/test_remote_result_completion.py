from core.analysis_agent.policy import RuntimePolicy
"""A synchronous query receipt must replace fictional deferred-result replies."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/remote_result_completion.json').read_text())


class DeferredReplyModel(QuietModel):
    sent: bool = False
    calls: int = 0
    propose: bool = True
    reply: str = FIXTURE['deferred_reply']

    def _generate(self, messages, **kwargs):
        self.calls += 1
        if self.propose and not self.sent:
            self.sent = True
            message = AIMessage(content='', tool_calls=[{
                'name': 'query_databricks', 'id': str(uuid4()),
                'args': {key: FIXTURE[key] for key in ('source', 'query', 'reason')}}])
        else:
            message = AIMessage(content=self.reply)
        return ChatResult(generations=[ChatGeneration(message=message)])


class RemoteResultCompletionTests(unittest.TestCase):
    def runtime(self, root, *, rows=None, error=None, propose=True, mismatched_result=False):
        executions = []
        def factory(datasets):
            def execute(envelope):
                executions.append(envelope['query'])
                if error:
                    raise error
                frame = pd.DataFrame(FIXTURE['rows'] if rows is None else rows,
                                     columns=['table_name', 'table_type'])
                info = datasets.register(frame, source=envelope['source'],
                                         query=envelope['query'] + (' LIMIT 1' if mismatched_result else ''),
                                         coverage='complete', predicate_known=True, snapshot='fixture-v1')
                return {'status': 'ready', 'dataset': asdict(info), 'preview': frame.head(10).to_dict('records')}
            return execute
        model = DeferredReplyModel(propose=propose)
        runtime = GraphAnalysisRuntime(root, 'owner', 'query-result', model,
            connection_identity='synthetic-connection', remote_factory=factory, policy=RuntimePolicy(require_remote_approval=True))
        self.addCleanup(runtime.close)
        return runtime, model, executions

    def test_completed_listing_is_rendered_from_receipt_not_model_promise(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root)
            proposal = runtime.submit(FIXTURE['prompt'])
            self.assertEqual(proposal['status'], 'awaiting_approval', proposal)
            self.assertEqual(executions, [])
            result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            self.assertEqual(result['status'], 'answered', result)
            self.assertEqual(executions, [FIXTURE['query']])
            self.assertIn('sample_events', result['text'].replace('\\_', '_'))
            self.assertIn('sample_summary', result['text'].replace('\\_', '_'))
            self.assertIn('VIEW', result['text'])
            self.assertNotIn('도착하면', result['text'])
            self.assertEqual(model.calls, 1)  # Result rendering does not ask the LLM to narrate it.
            self.assertEqual(runtime.ledger.get(proposal['requests'][0]['id'])['status'], 'completed')

    def test_unexecuted_promise_is_not_a_successful_answer(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root, propose=False)
            result = runtime.submit(FIXTURE['prompt'])
            self.assertNotEqual(result['status'], 'answered', result)
            self.assertNotIn('도착하면', result['text'])
            self.assertEqual(executions, [])
            self.assertLessEqual(model.calls, 3)

    def test_empty_result_is_completed_zero_rows_not_waiting(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root, rows=[])
            proposal = runtime.submit(FIXTURE['prompt'])
            result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            self.assertEqual(result['status'], 'answered', result)
            self.assertIn('0행', result['text'])
            self.assertNotIn('도착하면', result['text'])
            self.assertEqual(len(executions), 1)

    def test_rejected_query_never_runs_or_promises_future_results(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root)
            proposal = runtime.submit(FIXTURE['prompt'])
            result = runtime.respond(proposal['requests'][0]['id'], approved=False)
            self.assertEqual(result['status'], 'blocked', result)
            self.assertIn('취소', result['text'])
            self.assertNotIn('도착하면', result['text'])
            self.assertEqual(executions, [])

    def test_unknown_submission_stops_without_an_automatic_retry(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root, error=TimeoutError('synthetic timeout'))
            proposal = runtime.submit(FIXTURE['prompt'])
            result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            self.assertEqual(result['status'], 'blocked', result)
            self.assertIn('자동 재조회하지 않습니다', result['text'])
            self.assertEqual(runtime.ledger.get(proposal['requests'][0]['id'])['status'], 'unknown')
            self.assertEqual(len(executions), 1)
            self.assertNotIn('도착하면', result['text'])

    def test_proven_connection_failure_reports_no_submission(self):
        from core.analysis_agent.approvals import QueryNotSubmitted
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root, error=QueryNotSubmitted(403))
            proposal = runtime.submit(FIXTURE['prompt'])
            result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            self.assertEqual(result['status'], 'blocked', result)
            self.assertIn('403', result['text'])
            self.assertIn('SQL은 제출되지 않았', result['text'])
            self.assertEqual(len(executions), 1)

    def test_restart_and_duplicate_approval_do_not_resubmit(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root)
            proposal = runtime.submit(FIXTURE['prompt'])
            request = proposal['requests'][0]['id']
            runtime.respond(request, approved=True)
            runtime.close()
            reopened, _, again = self.runtime(root)
            self.assertEqual(reopened.ledger.get(request)['status'], 'completed')
            self.assertIn('sample_events', str(reopened.events()[-1].content).replace('\\_', '_'))
            with self.assertRaises(PermissionError):
                reopened.respond(request, approved=True)
            self.assertEqual(len(executions), 1)
            self.assertEqual(again, [])

    def test_reopen_repairs_old_deferred_reply_using_existing_receipt_only(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, _, executions = self.runtime(root)
            proposal = runtime.submit(FIXTURE['prompt'])
            runtime.respond(proposal['requests'][0]['id'], approved=True)
            last = runtime.agent.get_state(runtime.config).values['messages'][-1]
            old_reply = last.model_copy(update={'content':FIXTURE['deferred_reply']})
            runtime.agent.update_state(runtime.config, {'messages':[old_reply]},
                                       as_node='HumanInTheLoopMiddleware.after_model')
            runtime.transcript.record([old_reply])
            runtime.close()
            reopened, model, again = self.runtime(root)
            self.assertEqual(model.calls, 0)
            self.assertEqual(again, [])
            self.assertEqual(len(executions), 1)
            self.assertFalse(reopened.agent.get_state(reopened.config).next)
            self.assertIn('sample_events', str(reopened.events()[-1].content).replace('\\_', '_'))
            self.assertNotIn('도착하면', str(reopened.events()[-1].content))
            self.assertEqual(reopened.db.conn.execute('SELECT COUNT(*) FROM rejected_transcript').fetchone()[0], 1)

    def test_general_explanation_stays_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'explanation', QuietModel())
            try:
                result = runtime.submit('안녕')
                self.assertEqual(result['status'], 'answered', result)
            finally:
                runtime.close()

    def test_missing_execution_receipt_cannot_attest_existing_dataset(self):
        from core.analysis_agent.remote_completion import verified_receipt
        with tempfile.TemporaryDirectory() as root:
            runtime, _, _ = self.runtime(root)
            info = runtime.datasets.register(pd.DataFrame(FIXTURE['rows']),
                source=FIXTURE['source'], query=FIXTURE['query'], coverage='complete')
            call = {'id':'not-executed', 'args':{key:FIXTURE[key] for key in ('source','query','reason')}}
            observation = {'status':'ready', 'dataset':asdict(info)}
            self.assertIsNone(verified_receipt(runtime.ledger, runtime.context, call, observation))

    def test_completed_receipt_for_mismatched_result_is_not_published(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, _, executions = self.runtime(root, mismatched_result=True)
            proposal = runtime.submit(FIXTURE['prompt'])
            result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            self.assertEqual(result['status'], 'blocked', result)
            self.assertIn('실행 기록과 저장된 결과', result['text'])
            self.assertNotIn('sample_events', result['text'])
            self.assertEqual(len(executions), 1)

    def test_rendering_bound_does_not_claim_an_entire_large_listing(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [{'table_name':f'object_{i}', 'table_type':'VIEW'} for i in range(205)]
            runtime, _, executions = self.runtime(root, rows=rows)
            proposal = runtime.submit(FIXTURE['prompt'])
            result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            self.assertEqual(result['status'], 'answered', result)
            self.assertIn('200/205행', result['text'])
            self.assertNotIn('object\\_204', result['text'])
            self.assertEqual(len(executions), 1)

    def test_explicit_denial_of_async_capability_is_not_rejected(self):
        from core.analysis_agent.remote_completion import deferred_execution_claim
        self.assertFalse(deferred_execution_claim(
            '조회 결과가 도착하면 알려주는 자동 알림 기능은 지원하지 않습니다.'))
        self.assertTrue(deferred_execution_claim(FIXTURE['deferred_reply']))

    def test_explicit_catalog_columns_are_not_an_unspecified_statistic(self):
        from core.analysis_agent.operation_binding import candidate
        current = {'request_text':'table_name, table_type을 조회해서 실제 결과 목록을 보여줘.',
                   'required_columns':['table_name','table_type'],
                   'required_sources':[FIXTURE['source']], 'operations':[]}
        self.assertFalse(candidate(current))
        # Ordinary scalar requests must still use independent interpretation.
        self.assertTrue(candidate({'request_text':'reading 대표값을 구해줘',
                                   'required_columns':['reading'], 'operations':[]}))

    def test_catalog_query_plan_without_tool_is_not_completion(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, model, executions = self.runtime(root, propose=False)
            model.reply = '목록을 조회하겠습니다. 이 쿼리를 실행할까요?'
            runtime.context.reference_context[:] = [{'table':FIXTURE['source'],
                'columns':[{'name':c,'dtype':'string'} for c in ('table_name','table_type')]}]
            result = runtime.submit(FIXTURE['source'] + '에서 table_name, table_type 목록을 보여줘.')
            self.assertNotEqual(result['status'], 'answered', result)
            self.assertTrue(runtime.inspect()['recovery']['remote_result_requested'])
            self.assertFalse(runtime.inspect()['recovery'].get('operation_pending'))
            self.assertEqual(executions, [])

    def test_real_page_shows_listing_after_approval_button(self):
        import os
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        executions = []
        def factory(config, datasets, **kwargs):
            def execute(envelope):
                executions.append(envelope['query'])
                frame = pd.DataFrame(FIXTURE['rows'])
                info = datasets.register(frame, source=envelope['source'], query=envelope['query'],
                    coverage='complete', predicate_known=True, snapshot='ui-fixture')
                return {'status':'ready', 'dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'TELLY_V1_STORAGE':root, 'TELLY_REQUIRE_REMOTE_APPROVAL':'true'}), \
                patch('core.analysis_agent.model_provider.build_analysis_chat_model', return_value=DeferredReplyModel()), \
                patch('core.analysis_agent.databricks.make_executor', side_effect=factory):
            page = Path(__file__).resolve().parents[1] / 'ui/analysis_page.py'
            app = AppTest.from_file(str(page), default_timeout=20).run()
            app.chat_input[0].set_value(FIXTURE['prompt']).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(executions, [])
            approve = next(b for b in app.button if b.key and str(b.key).startswith('yes-'))
            approve.click().run()
            try:
                self.assertEqual(len(app.exception), 0)
                rendered = '\n'.join(item.value for item in app.markdown).replace('\\_', '_')
                self.assertIn('sample_events', rendered)
                self.assertIn('sample_summary', rendered)
                self.assertIn('VIEW', rendered)
                self.assertNotIn('도착하면', rendered)
                self.assertEqual(executions, [FIXTURE['query']])
            finally:
                app.session_state['v1_runtime'].close()


if __name__ == '__main__':
    unittest.main()
