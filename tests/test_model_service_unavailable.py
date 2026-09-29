"""Structured Databricks HTTP400 outage retries inference only, never generic 400."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import httpx
from openai import BadRequestError
from core.analysis_agent.assets import AssetDB
from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.model_recovery import ModelAttemptLedger,ModelRecoveryMiddleware,transient_model_error
from core.analysis_agent.model_errors import model_error_category
from core.analysis_agent.policy import RuntimePolicy

MESSAGE='BAD_REQUEST: Cannot create or query foundation model endpoints, please try again later.'

def error(message=MESSAGE, code='BAD_REQUEST', status=400):
    return BadRequestError('private-token-not-for-logs',response=httpx.Response(status,
        request=httpx.Request('POST','https://example.invalid')),body={'error_code':code,'message':message})

class ProviderServiceUnavailableTests(unittest.TestCase):
    def test_exact_provider_outage_only(self):
        self.assertTrue(transient_model_error(error()))
        for item in [error('Invalid tool schema; please try again later.'),error(code='INVALID_ARGUMENT'),
                     error(status=401),error(status=403),ValueError(MESSAGE),error(MESSAGE+' user payload')]:
            self.assertFalse(transient_model_error(item))
        self.assertEqual(model_error_category(error()),'model_provider_temporarily_unavailable')

    def test_bounded_retry_and_restart_do_not_reset_allowance(self):
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','outage')
            try:
                ledger=ModelAttemptLedger(db);diag=Diagnostics(db.directory)
                retry=ModelRecoveryMiddleware(ledger,diag,RuntimePolicy(),sleep=Mock())
                request=SimpleNamespace(state={'recovery':{'request_id':'fixed','model_calls':0,'model_seconds':0}})
                handler=Mock(side_effect=error())
                with self.assertRaises(BadRequestError):retry.wrap_model_call(request,handler)
                self.assertEqual(handler.call_count,3)
                recreated=ModelRecoveryMiddleware(ModelAttemptLedger(db),diag,RuntimePolicy(),sleep=Mock())
                with self.assertRaises(BadRequestError):recreated.wrap_model_call(request,handler)
                self.assertEqual(handler.call_count,4)
                self.assertEqual(ledger.get('fixed')['retries'],2)
                diag.failure(error())
                log=diag.path.read_text()
                self.assertIn('model_provider_temporarily_unavailable',log)
                self.assertNotIn('private-token',log)
                self.assertNotIn('foundation model endpoints',log)
            finally:db.conn.close()

    def test_transient_then_success_continues_same_inference(self):
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','recovery')
            try:
                ledger=ModelAttemptLedger(db)
                middleware=ModelRecoveryMiddleware(ledger,Diagnostics(db.directory),RuntimePolicy(),sleep=Mock())
                request=SimpleNamespace(state={'recovery':{'request_id':'fixed','model_calls':0,'model_seconds':0}})
                handler=Mock(side_effect=[error(),'recovered response'])
                self.assertEqual(middleware.wrap_model_call(request,handler),'recovered response')
                self.assertEqual(ledger.get('fixed')['retries'],1)
            finally:db.conn.close()

    def test_graph_returns_specific_incomplete_and_preserves_raw_then_resumes(self):
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        from tests.test_model_recovery import IntermittentModel
        import pandas as pd
        from utils.analysis_datasets import stored_dataset_digest
        with tempfile.TemporaryDirectory() as root,patch('tests.test_model_recovery.rate_error',side_effect=error),patch('core.analysis_agent.model_recovery.time.sleep'):
            model=IntermittentModel(failures_left=3,calls=[{'name':'aggregate_dataset','args':{
                'dataset_id':'$fixture','aggregation':'mean','value_column':'measurement'}}])
            runtime=GraphAnalysisRuntime(root,'owner','outage-resume',model)
            try:
                raw=runtime.datasets.register(pd.DataFrame({'measurement':[2.,4.,9.]}),source='fixture.readings',coverage='complete',predicate_known=True)
                runtime.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
                digest=stored_dataset_digest(runtime.datasets,raw.id)
                with patch.object(runtime.recovery,'_next_local',return_value=None),patch.object(runtime.recovery,'resume_local_call',return_value=None):
                    failed=runtime.submit('measurement 평균을 알려줘')
                    self.assertEqual(failed['status'],'incomplete')
                    self.assertEqual(failed['error_category'],'model_provider_temporarily_unavailable')
                    self.assertIn('모델 공급자',failed['text'])
                    self.assertEqual(model.provider_attempts,3)
                    self.assertFalse(runtime.events()[-1].additional_kwargs.get('analysis_complete'))
                    result=runtime.resume()
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(runtime.inspect()['model_recovery']['retries'],2)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),digest)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                self.assertFalse(runtime.inspect()['requests'])
            finally:runtime.close()
