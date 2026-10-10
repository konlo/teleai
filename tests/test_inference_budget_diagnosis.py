"""Separate input size from call/time exhaustion without sending another inference."""
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from core.analysis_agent.assets import AssetDB
from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.model_recovery import (
    ModelAttemptBudgetExceeded, ModelAttemptLedger, ModelRecoveryMiddleware,
)
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.support_report import brief, summarize, public_attempt_budget


class InferenceBudgetDiagnosisTests(unittest.TestCase):
    def setUp(self):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        self.db=AssetDB(root.name,'owner','budget');self.addCleanup(self.db.close)
        self.ledger=ModelAttemptLedger(self.db)
        self.diagnostics=Diagnostics(self.db.directory)
        self.diagnostics.run_id='a'*32
        self.diagnostics.emit('run_started')
        self.middleware=ModelRecoveryMiddleware(self.ledger,self.diagnostics,RuntimePolicy())

    def test_valid_input_can_precede_auxiliary_call_limit_and_error_keeps_snapshot(self):
        self.diagnostics.emit('model_payload_budget',payload_bytes=4197,template_headroom=640,
                              input_budget_units=32000,within_budget=True)
        for _ in range(9):self.ledger.auxiliary_success('request',0.1)
        handler=Mock()
        with self.assertRaises(ModelAttemptBudgetExceeded) as caught:
            self.middleware.auxiliary_call({'request_id':'request'},handler)
        handler.assert_not_called()
        error_id=self.diagnostics.failure(caught.exception,stage='agent_stream')
        # Later model activity must not replace the budget attached to this error.
        self.diagnostics.emit('model_attempt_budget_exhausted',reason='time',calls_used=1)
        self.diagnostics.failure(ModelAttemptBudgetExceeded('later',budget={'reason':'time'}),
                                 stage='later_error')
        self.diagnostics.emit('run_completed',status='incomplete')
        report=summarize(self.diagnostics.path,error_id=error_id)
        self.assertTrue(report['input_budget']['within_budget'])
        budget=report['errors'][0]['attempt_budget']
        self.assertEqual(budget['reason'],'calls')
        self.assertEqual(budget['call_kind'],'auxiliary')
        self.assertEqual(budget['calls_used'],9)
        self.assertEqual(budget['call_limit'],9)
        self.assertEqual(budget['total_call_limit'],10)
        self.assertEqual(budget['reserved_calls'],1)
        self.assertEqual(budget['auxiliary_calls'],9)
        self.assertIn('원인=calls / 호출=9/9',brief(report))
        self.assertEqual(self.ledger.get('request')['aux_calls'],9)

    def test_elapsed_limit_is_distinct_from_call_limit_and_no_provider_failure_added(self):
        with self.assertRaises(ModelAttemptBudgetExceeded) as caught:
            self.middleware.auxiliary_call({'request_id':'request','model_seconds':181},Mock())
        budget=caught.exception.attempt_budget
        self.assertEqual(budget['reason'],'time')
        self.assertEqual(budget['calls_used'],0)
        self.assertGreaterEqual(budget['seconds_used'],181)
        self.assertEqual(budget['seconds_remaining'],0)
        self.assertEqual(self.ledger.get('request')['failures'],0)

    def test_retry_wait_exhaustion_reports_elapsed_and_provider_failures(self):
        from tests.test_model_recovery import rate_error
        clock=[0.]
        def wait(delay):clock[0]=101.
        middleware=ModelRecoveryMiddleware(self.ledger,self.diagnostics,
            replace(RuntimePolicy(),turn_slo_seconds=100,model_timeout_seconds=1),sleep=wait)
        handler=Mock(side_effect=rate_error())
        with patch('core.analysis_agent.model_recovery.time.monotonic',side_effect=lambda:clock[0]):
            with self.assertRaises(ModelAttemptBudgetExceeded) as caught:
                middleware.invoke({'request_id':'retry'},handler)
        handler.assert_called_once()
        self.assertEqual(caught.exception.attempt_budget['reason'],'time')
        self.assertEqual(caught.exception.attempt_budget['seconds_used'],101)
        self.assertEqual(caught.exception.attempt_budget['provider_failures'],1)
        self.assertEqual(caught.exception.attempt_budget['retries'],1)

    def test_both_limits_and_normal_call_keep_original_admission_rules(self):
        with self.assertRaises(ModelAttemptBudgetExceeded) as caught:
            self.middleware.invoke({'request_id':'both','model_calls':10,'model_seconds':181},Mock())
        self.assertEqual(caught.exception.attempt_budget['reason'],'calls_and_time')
        handler=Mock(return_value='ok')
        self.assertEqual(self.middleware.auxiliary_call({'request_id':'fresh'},handler),'ok')
        handler.assert_called_once()
        self.assertEqual(self.ledger.get('fresh')['aux_calls'],1)

    def test_public_budget_rejects_private_and_invalid_fields(self):
        clean=public_attempt_budget({'reason':'PRIVATE','call_kind':'PRIVATE',
            'calls_used':-1,'call_limit':'PRIVATE','prompt':'PRIVATE','query':'PRIVATE'})
        self.assertNotIn('PRIVATE',json.dumps(clean))
        self.assertIsNone(clean['reason']);self.assertIsNone(clean['calls_used'])

    def test_runtime_distinguishes_attempt_exhaustion_from_context_error(self):
        from tests.test_actual_agent_evaluation import EvaluationModel
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','runtime-budget',EvaluationModel(),
                                         intent_mode='contract_fixture')
            self.addCleanup(runtime.close)
            error=ModelAttemptBudgetExceeded('private',budget={'reason':'calls'})
            with patch.object(runtime,'_stream_with_local_recovery',side_effect=error):
                result=runtime.submit('검증 요청')
            self.assertEqual(result['error_type'],'ModelAttemptBudgetExceeded')
            self.assertIn('모델 호출 횟수 한도',result['text'])
            self.assertNotIn('입력과 도구 정의',result['text'])
            self.assertNotIn('재개로 다시 시도',result['text'])
            self.assertFalse(runtime.inspect()['uncertain_executions'])
