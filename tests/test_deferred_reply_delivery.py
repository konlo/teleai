"""Promises cannot leak through structured content or tool narration."""
import tempfile
import unittest

from langchain_core.messages import AIMessage
from core.analysis_agent.assets import AssetDB
from core.analysis_agent.memory import Transcript
from core.analysis_agent.remote_completion import deferred_execution_claim
from tests import test_remote_result_completion as remote_cases
FIXTURE = remote_cases.FIXTURE


class DeferredReplyDeliveryTests(unittest.TestCase):
    def test_structured_text_blocks_are_checked_but_reasoning_is_not_display_text(self):
        self.assertTrue(deferred_execution_claim(FIXTURE['reported_reply']))
        self.assertTrue(deferred_execution_claim([{'type':'text', 'text':FIXTURE['reported_reply']}]))
        self.assertFalse(deferred_execution_claim([{'type':'reasoning', 'text':FIXTURE['reported_reply']},
                                                 {'type':'text', 'text':'완료 알림 기능은 없습니다.'}]))

    def test_tool_narration_and_historical_unverified_promises_are_not_rendered(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'owner', 'delivery')
            try:
                transcript = Transcript(db)
                message = AIMessage(id='tool-plan', content=FIXTURE['reported_reply'], tool_calls=[{
                    'name':'query_databricks','id':'query-1','args':{}}])
                transcript.record([message])
                displayed = transcript.messages()[0]
                self.assertFalse(deferred_execution_claim(displayed.content))
                self.assertEqual(displayed.tool_calls, message.tool_calls)
                self.assertEqual(message.content, FIXTURE['reported_reply'])
                # Simulate a record written before the delivery guard existed.
                import json
                from langchain_core.messages import message_to_dict
                old = AIMessage(id='old-final',content=FIXTURE['reported_reply'])
                with db.conn:
                    db.conn.execute('INSERT INTO transcript VALUES (?,?)',
                                    (old.id,json.dumps(message_to_dict(old),ensure_ascii=False)))
                final = Transcript(db).messages()[-1]
                self.assertFalse(deferred_execution_claim(final.content))
                self.assertEqual(final.additional_kwargs['analysis_status'],'blocked')
                self.assertIn('예약되지 않았습니다', final.content)
                self.assertEqual(db.conn.execute('SELECT COUNT(*) FROM rejected_transcript').fetchone()[0],2)
            finally:
                db.close()

    def test_main_entrypoint_hides_tool_narration_and_renders_real_returned_listing(self):
        import os
        from dataclasses import asdict
        from pathlib import Path
        from unittest.mock import patch
        import pandas as pd
        from streamlit.testing.v1 import AppTest
        class NarratedModel(remote_cases.DeferredReplyModel):
            def _generate(self, messages, **kwargs):
                result = super()._generate(messages, **kwargs)
                message = result.generations[0].message
                if message.tool_calls:
                    message.content = [{'type':'text','text':FIXTURE['reported_reply']}]
                return result
        executions = []
        def factory(config, datasets, **kwargs):
            def execute(envelope):
                executions.append(envelope['query'])
                info = datasets.register(pd.DataFrame(FIXTURE['rows']), source=envelope['source'],
                    query=envelope['query'],coverage='complete',predicate_known=True)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(os.environ, {'TELLY_V1_STORAGE':root,'TELLY_REQUIRE_REMOTE_APPROVAL':'false',
                    'TELLY_DATA_BACKEND':'databricks','DATABRICKS_CATALOG':FIXTURE['source'].split('.')[0]}), \
                patch('core.analysis_agent.recovery.RecoveryMiddleware._next_local',return_value=None), \
                patch('core.analysis_agent.model_provider.build_analysis_chat_model', return_value=NarratedModel()), \
                patch('core.analysis_agent.databricks.make_executor', side_effect=factory):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'main.py'),default_timeout=20).run()
            app.session_state['v1_runtime'].recovery.intent_mode = 'contract_fixture'  # Render/receipt fixture only.
            app.chat_input[0].set_value(FIXTURE['prompt']).run()
            try:
                self.assertFalse(app.exception)
                rendered = '\n'.join(item.value for item in app.markdown).replace('\\_', '_')
                self.assertIn('sample_events', rendered)
                self.assertNotIn('도착하면', rendered)
                self.assertNotIn('승인해주셔서', rendered)
                self.assertEqual(executions,[FIXTURE['query']])
            finally:
                app.session_state['v1_runtime'].close()


class ExactGraphReplyTests(unittest.TestCase):
    def test_exact_reported_reply_string_and_blocks_cannot_complete_without_execution(self):
        for content in (FIXTURE['reported_reply'], [{'type':'text','text':FIXTURE['reported_reply']} ]):
            with self.subTest(content_type=type(content).__name__), tempfile.TemporaryDirectory() as root:
                runtime, model, executions = remote_cases.RemoteResultCompletionTests.runtime(self, root, propose=False)
                model.reply = content
                result = runtime.submit('승인했어')
                self.assertNotEqual(result['status'], 'answered', result)
                self.assertEqual(executions, [])
                self.assertFalse(any(deferred_execution_claim(m.content) for m in runtime.events()
                                     if isinstance(m, AIMessage)))
