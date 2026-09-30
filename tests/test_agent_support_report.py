import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.support_report import summarize


class AgentSupportReportTests(unittest.TestCase):
    def test_exception_id_selects_only_its_run_without_private_error_text(self):
        with tempfile.TemporaryDirectory() as folder:
            diagnostics=Diagnostics(folder)
            diagnostics.run_id='a'*32
            diagnostics.emit('run_started')
            try:
                raise ValueError('PRIVATE SQL TOKEN ROW PROMPT')
            except ValueError as exc:
                error_id=diagnostics.failure(exc,stage='query_databricks')
            diagnostics.emit('remote_query_finished',ledger_status='unknown',status='unavailable')
            diagnostics.emit('run_completed',status='incomplete')
            diagnostics.run_id='b'*32
            diagnostics.emit('run_started')
            diagnostics.emit('run_completed',status='answered')
            report=summarize(diagnostics.path,error_id=error_id)
            self.assertEqual(report['run_id'],'a'*32)
            self.assertEqual(report['errors'][0]['stage'],'query_databricks')
            self.assertTrue(report['errors'][0]['frames'])
            self.assertIn('REMOTE_SUBMISSION_UNCERTAIN',report['flags'])
            self.assertEqual(report['status'],'incomplete')
            self.assertNotIn('PRIVATE',diagnostics.path.read_text())
            self.assertNotIn('PRIVATE',json.dumps(report))
            self.assertEqual(summarize(diagnostics.path)['run_id'],'b'*32)
            self.assertFalse(summarize(diagnostics.path,error_id='c'*12)['found'])

    def test_semantic_failure_without_exception_and_rotated_partial_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'runtime.jsonl'
            entries=[{'event':'run_started','run_id':'a'*32},
                     {'event':'model_call_started','run_id':'a'*32},
                     {'event':'model_call_finished','run_id':'a'*32,'tool_call_count':0},
                     {'event':'unsupported_deferred_reply_rejected','run_id':'a'*32},
                     {'event':'completion_checked','run_id':'a'*32,'status':'blocked',
                      'missing_capabilities':['chart'],'reason':'missing_evidence',
                      'query':'PRIVATE SQL','prompt':'PRIVATE PROMPT','rows':['PRIVATE ROW']},
                     {'event':'run_completed','run_id':'a'*32,'status':'blocked'}]
            Path(str(path)+'.1').write_text('\n'.join(map(json.dumps,entries[:3]))+'\n')
            path.write_text('\n'.join(map(json.dumps,entries[3:]))+'\n{"unfinished":\n')
            report=summarize(path)
            self.assertEqual(report['status'],'blocked')
            self.assertEqual(report['model_calls'],1)
            self.assertEqual(report['completion']['missing_capabilities'],['chart'])
            self.assertEqual(report['log_integrity']['malformed_lines'],1)
            self.assertIn('DEFERRED_REPLY_BLOCKED',report['flags'])
            self.assertIn('NO_TOOL_EXECUTION_OBSERVED',report['flags'])
            self.assertNotIn('PRIVATE',json.dumps(report))

    def test_error_outside_a_run_and_incomplete_run_are_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            diagnostics=Diagnostics(folder)
            error=diagnostics.failure(OSError('private'),stage='chart_display')
            report=summarize(diagnostics.path,error_id=error)
            self.assertTrue(report['found'])
            self.assertIsNone(report['run_id'])
            self.assertEqual(report['status'],'not_recorded')
            self.assertIn('CHART_DISPLAY_FAILED',report['flags'])
            diagnostics.run_id='c'*32
            diagnostics.emit('run_started')
            diagnostics.emit('model_call_started')
            self.assertIn('NO_TERMINAL_EVENT',summarize(diagnostics.path)['flags'])

    def test_ui_shows_read_only_report_without_running_agent(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as folder:
            diagnostics=Diagnostics(folder)
            diagnostics.run_id='a'*32
            diagnostics.emit('run_started')
            diagnostics.emit('run_completed',status='blocked')
            def screen():
                import streamlit as st
                from ui.analysis_diagnostics import render_diagnostics
                render_diagnostics(st.session_state.runtime)
            runtime=Mock(diagnostics=diagnostics)
            app=AppTest.from_function(screen)
            app.session_state['runtime']=runtime
            app.run()
            app.button(key='diagnostic_summary').click().run()
            self.assertFalse(app.exception)
            self.assertIn('상태: blocked',app.code[0].value)
            runtime.submit.assert_not_called()
            runtime.resume.assert_not_called()
            app.text_input(key='diagnostic_reference').set_value('b'*12)
            app.button(key='diagnostic_summary').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)

    def test_real_runtime_logs_receipt_state_and_report_excludes_data(self):
        from tests import test_remote_result_completion as cases
        with tempfile.TemporaryDirectory() as folder:
            runtime,model,executions=cases.RemoteResultCompletionTests().runtime(folder)
            try:
                # Existing fixture helper starts in manual approval mode.
                result=runtime.submit(cases.FIXTURE['prompt'])
                if result['status']=='awaiting_approval':
                    runtime.respond(result['requests'][0]['id'],approved=True)
                report=summarize(runtime.diagnostics.path)
                self.assertTrue(report['found'])
                self.assertEqual(report['runtime']['route'],'langgraph-v1')
                self.assertTrue(report['runtime']['loaded_delivery_guard'])
                self.assertEqual(report['remote']['states'].get('completed'),1)
                self.assertEqual(len(executions),1)
                self.assertNotIn(cases.FIXTURE['query'],json.dumps(report))
                self.assertNotIn(cases.FIXTURE['source'],json.dumps(report))
            finally:
                runtime.close()

    def test_main_diagnostic_click_does_not_resume_pending_automatic_sql(self):
        import os
        from streamlit.testing.v1 import AppTest
        from tests.test_remote_result_completion import DeferredReplyModel
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{
                'TELLY_V1_STORAGE':folder,'TELLY_REQUIRE_REMOTE_APPROVAL':'false'}), patch(
                'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=DeferredReplyModel()), patch(
                'core.analysis_agent.databricks.make_executor',return_value=Mock()) as factory:
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'main.py'),default_timeout=20).run()
            self.assertFalse(app.exception)
            runtime=app.session_state['v1_runtime']
            try:
                pending={**runtime.inspect(),'requests':[{}],'state':'awaiting_approval'}
                with patch.object(runtime,'inspect',return_value=pending), patch.object(runtime,'resume') as resume:
                    app.text_input(key='diagnostic_reference').set_value('f'*12).run()
                    self.assertFalse(app.exception)
                    resume.assert_not_called()
                    app.button(key='diagnostic_summary').click().run()
                    self.assertFalse(app.exception)
                    resume.assert_not_called()
                    factory.return_value.assert_not_called()
            finally:
                runtime.close()


if __name__=='__main__':unittest.main()
