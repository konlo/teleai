"""Changed approval requests become durable new turns without remote execution."""
import json
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.memory import QueuedRequestMiddleware, latest_user_request, memory_middleware
from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel


class ReplacementModel(QuietModel):
    seen_requests: list[str] = []

    def _generate(self, messages, **kwargs):
        if str(messages[0].content).startswith('사용자 메시지가 오직'):
            message = AIMessage(content='{"action":"change"}')
        else:
            current = latest_user_request(messages)
            self.seen_requests.append(current.content)
            message = AIMessage(content='', tool_calls=[{
                'name': 'query_databricks',
                'args': {'source': 'fixture', 'query': 'SELECT 2 AS updated_value',
                         'reason': '변경된 요청에 필요한 결과를 불러옵니다.'},
                'id': 'replacement-query',
            }])
        return ChatResult(generations=[ChatGeneration(message=message)])


class ReplacementTurnTests(unittest.TestCase):
    def runtime(self, root, model, calls):
        return GraphAnalysisRuntime(root, 'owner', 'replacement', model,
            connection_identity='test', remote_factory=lambda _: lambda envelope: calls.append(envelope))

    def test_change_is_a_new_request_with_new_approval_and_no_old_goal_model_call(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            model = ReplacementModel()
            runtime = self.runtime(root, model, calls)
            previous = runtime.propose_query('fixture', 'SELECT 1', '이전 자료를 불러와')['requests'][0]
            result = runtime.submit('아니 새 조건의 자료를 불러와')
            self.assertEqual(result['status'], 'awaiting_approval', result)
            self.assertEqual(result['requests'][0]['query'], 'SELECT 2 AS updated_value')
            self.assertEqual(runtime.ledger.get(previous['id'])['status'], 'invalidated')
            self.assertEqual(model.seen_requests, ['아니 새 조건의 자료를 불러와'])
            self.assertEqual(calls, [])
            with self.assertRaises(PermissionError):
                runtime.respond(previous['id'], approved=True)
            messages = runtime.agent.get_state(runtime.config).values['messages']
            latest = latest_user_request(messages)
            self.assertEqual(latest.content, '아니 새 조건의 자료를 불러와')
            old_rejection = next(i for i, m in enumerate(messages)
                                 if isinstance(m, ToolMessage) and 'rejected' in str(m.content).lower())
            self.assertLess(old_rejection, messages.index(latest))
            self.assertIn(latest.content, [m.content for m in runtime.events() if m.type == 'human'])
            runtime.close()

    def test_interruption_after_rejection_preserves_queued_request_on_reopen(self):
        original = QueuedRequestMiddleware.before_model

        def interrupt_queued(middleware, state, runtime):
            if state.get('queued_user_request'):
                raise KeyboardInterrupt('simulated process interruption')
            return original(middleware, state, runtime)

        with tempfile.TemporaryDirectory() as root:
            calls = []
            with patch.object(QueuedRequestMiddleware, 'before_model', interrupt_queued):
                runtime = self.runtime(root, ReplacementModel(), calls)
                previous = runtime.propose_query('fixture', 'SELECT 1', '이전 자료를 불러와')['requests'][0]
                with self.assertRaises(KeyboardInterrupt):
                    runtime.submit('새 조건의 자료를 불러와')
                self.assertTrue(runtime.agent.get_state(runtime.config).values['queued_user_request'])
                self.assertEqual(runtime.ledger.get(previous['id'])['status'], 'invalidated')
                runtime.close()
            model = ReplacementModel()
            runtime = self.runtime(root, model, calls)
            result = runtime.resume()
            self.assertEqual(result['status'], 'awaiting_approval', result)
            self.assertEqual(model.seen_requests, ['새 조건의 자료를 불러와'])
            self.assertEqual(calls, [])
            self.assertIsNone(runtime.agent.get_state(runtime.config).values['queued_user_request'])
            runtime.close()

    def test_model_and_summary_timings_do_not_record_user_text(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'timings', QuietModel())
            runtime.submit('private-request-value')
            text = runtime.diagnostics.path.read_text()
            self.assertNotIn('private-request-value', text)
            records = [json.loads(line) for line in text.splitlines()]
            for phase in ('summarization', 'model_call'):
                start = next(row for row in records if row['event'] == phase + '_started')
                end = next(row for row in records if row['event'] == phase + '_finished')
                self.assertEqual(start['span_id'], end['span_id'])
                self.assertEqual(start['run_id'], end['run_id'])
                self.assertGreaterEqual(end['elapsed_seconds'], 0)
                self.assertEqual(end['status'], 'ok')
            runtime.close()

    def test_actual_summary_is_timed_without_recording_history(self):
        from core.analysis_agent.diagnostics import Diagnostics
        with tempfile.TemporaryDirectory() as root:
            diagnostics = Diagnostics(root)
            middleware = memory_middleware(QuietModel(), trigger_tokens=10,
                                           keep_messages=2, diagnostics=diagnostics)
            messages = [HumanMessage(content='private-history-value ' * 50),
                        AIMessage(content='earlier answer'),
                        HumanMessage(content='현재 요청'), AIMessage(content='최근 답변')]
            result = middleware.before_model({'messages': messages}, None)
            self.assertTrue(result)
            text = diagnostics.path.read_text()
            self.assertNotIn('private-history-value', text)
            finished = json.loads(text.splitlines()[-1])
            self.assertTrue(finished['summarized'])
            self.assertGreaterEqual(finished['elapsed_seconds'], 0)

    def test_model_failure_timing_does_not_record_exception_payload(self):
        class FailingModel(QuietModel):
            def _generate(self, messages, **kwargs):
                raise TimeoutError('secret-model-exception')
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'model-failure', FailingModel())
            result = runtime.submit('요청 설명')
            self.assertEqual(result['status'], 'incomplete')
            text = runtime.diagnostics.path.read_text()
            self.assertNotIn('secret-model-exception', text)
            finished = next(json.loads(line) for line in text.splitlines()
                            if json.loads(line)['event'] == 'model_call_finished')
            self.assertEqual(finished['status'], 'error')
            self.assertEqual(finished['error_type'], 'TimeoutError')
            runtime.close()


if __name__ == '__main__':
    unittest.main()
