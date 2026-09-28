import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import RateLimitError, AuthenticationError
import pandas as pd

from core.analysis_agent.assets import AssetDB
from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.model_recovery import ModelAttemptLedger, ModelRecoveryMiddleware, ModelCoolingDown
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_actual_agent_evaluation import EvaluationModel


def rate_error(delay='0'):
    return RateLimitError('private-token-must-not-be-logged', response=httpx.Response(429,
        request=httpx.Request('POST','https://example.invalid'), headers={'retry-after':delay}),body=None)


class IntermittentModel(EvaluationModel):
    failures_left: int = 1
    provider_attempts: int = 0
    def _generate(self,*args,**kwargs):
        self.provider_attempts += 1
        if self.failures_left:
            self.failures_left -= 1
            raise rate_error()
        return super()._generate(*args,**kwargs)


class ModelRecoveryTests(unittest.TestCase):
    def setUp(self):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        self.db=AssetDB(root.name,'owner','retry');self.addCleanup(self.db.conn.close)
        self.ledger=ModelAttemptLedger(self.db);self.diagnostics=Diagnostics(self.db.directory)
        self.retry=ModelRecoveryMiddleware(self.ledger,self.diagnostics,RuntimePolicy(),sleep=Mock())
        self.request=SimpleNamespace(state={'recovery':{'request_id':'request','model_calls':0,'model_seconds':0}})

    def test_transient_retry_is_bounded_and_failure_is_not_logged_verbatim(self):
        handler=Mock(side_effect=[rate_error(),rate_error(),'response'])
        self.assertEqual(self.retry.wrap_model_call(self.request,handler),'response')
        self.assertEqual(handler.call_count,3)
        self.assertEqual(self.ledger.get('request')['retries'],2)
        state={'request_id':'request','model_calls':0,'model_seconds':0}
        self.ledger.sync(state);self.ledger.sync(state)
        self.assertEqual(state['model_calls'],2)
        self.assertNotIn('private-token',self.diagnostics.path.read_text())

    def test_auth_and_programming_errors_never_retry(self):
        auth=AuthenticationError('secret',response=httpx.Response(401,request=httpx.Request('POST','https://example.invalid')),body=None)
        for error in [auth, ValueError('bug')]:
            with self.subTest(error=type(error).__name__):
                handler=Mock(side_effect=error)
                with self.assertRaises(type(error)):self.retry.wrap_model_call(self.request,handler)
                self.assertEqual(handler.call_count,1)
        self.assertEqual(self.ledger.get('request')['retries'],0)

    def test_long_cooldown_survives_restart_and_never_sleeps_early(self):
        handler=Mock(side_effect=rate_error('120'))
        with self.assertRaises(RateLimitError):self.retry.wrap_model_call(self.request,handler)
        self.retry.sleep.assert_not_called()
        restarted=ModelAttemptLedger(self.db)
        middleware=ModelRecoveryMiddleware(restarted,self.diagnostics,RuntimePolicy(),sleep=Mock())
        with self.assertRaises(ModelCoolingDown):middleware.wrap_model_call(self.request,handler)
        self.assertEqual(handler.call_count,1)

    def test_restart_does_not_reset_automatic_retries_or_call_budget(self):
        handler=Mock(side_effect=rate_error())
        with self.assertRaises(RateLimitError):self.retry.wrap_model_call(self.request,handler)
        self.assertEqual(handler.call_count,3)
        middleware=ModelRecoveryMiddleware(ModelAttemptLedger(self.db),self.diagnostics,RuntimePolicy(),sleep=Mock())
        with self.assertRaises(RateLimitError):middleware.wrap_model_call(self.request,handler)
        self.assertEqual(handler.call_count,4)
        self.assertEqual(self.ledger.get('request')['retries'],2)
        self.request.state['recovery']['model_calls']=9
        self.request.state['recovery']['accounted_model_failures']=4
        with self.assertRaises(RateLimitError):middleware.wrap_model_call(self.request,handler)
        self.assertEqual(handler.call_count,5)

    def test_graph_recovers_inference_without_repeating_tools_or_losing_raw(self):
        with tempfile.TemporaryDirectory() as root, patch('core.analysis_agent.model_recovery.time.sleep'):
            model=IntermittentModel(calls=[{'name':'aggregate_dataset','args':{
                'dataset_id':'$fixture','aggregation':'mean','value_column':'measurement'}}])
            r=GraphAnalysisRuntime(root,'owner','graph',model)
            try:
                frame=pd.DataFrame({'measurement':[2.,4.,9.]})
                raw=r.datasets.register(frame,source='unfamiliar.measurements',coverage='complete',predicate_known=True)
                model.evaluation_dataset_id=raw.id;r.select_dataset(raw.id)
                with patch.object(r.recovery,'_next_local',return_value=None):
                    result=r.submit('measurement 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(state['model_calls'],2)
                self.assertEqual(state['model_retries'],1)
                self.assertEqual(model.provider_attempts,2)
                self.assertEqual(len(r.datasets.metadata),2)
                pd.testing.assert_frame_equal(r.datasets.frames[raw.id],frame)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()

    def test_low_time_budget_does_not_schedule_an_attempt_that_cannot_finish(self):
        from dataclasses import replace
        middleware=ModelRecoveryMiddleware(self.ledger,self.diagnostics,
            replace(RuntimePolicy(),turn_slo_seconds=30,model_timeout_seconds=60),sleep=Mock())
        handler=Mock(side_effect=rate_error())
        with self.assertRaises(RateLimitError):middleware.wrap_model_call(self.request,handler)
        self.assertEqual(handler.call_count,1)
        self.assertEqual(self.ledger.get('request')['retries'],0)
        middleware.sleep.assert_not_called()

    def test_semantic_interpretation_shares_retry_ledger_and_counts_each_attempt_once(self):
        from tests.test_analysis_semantic_binding import SemanticModel
        class FlakySemantic(SemanticModel):
            failed_once: bool = False
            def _generate(self,*args,**kwargs):
                if not self.failed_once:
                    self.failed_once=True
                    raise rate_error()
                return super()._generate(*args,**kwargs)
        plan={'uncertain':False,'operation':'MEDIAN','column':'measurement','request_span':'중위수'}
        with tempfile.TemporaryDirectory() as root, patch('core.analysis_agent.model_recovery.time.sleep'):
            r=GraphAnalysisRuntime(root,'owner','semantic-retry',FlakySemantic(replies=[plan,plan]))
            try:
                raw=r.datasets.register(pd.DataFrame({'measurement':[2.,4.,9.]}),source='arbitrary.trials',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                result=r.submit('measurement 중위수를 알려줘')
                state=r.inspect()['recovery']
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(state['model_calls'],3)
                self.assertEqual(state['model_retries'],1)
                self.assertEqual(r.inspect()['model_recovery']['failures'],1)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),4.)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()

    def test_failure_after_approved_query_never_reexecutes_that_query(self):
        from datetime import datetime, timezone
        class FailureAfterQuery(EvaluationModel):
            failed_once: bool = False
            def _generate(self,*args,**kwargs):
                if self.position == 1 and not self.failed_once:
                    self.failed_once=True
                    raise rate_error()
                return super()._generate(*args,**kwargs)
        model=FailureAfterQuery(calls=[
            {'name':'query_databricks','args':{'source':'custom.trials',
                'query':'SELECT measurement FROM custom.trials','reason':'Load requested measure'}},
            {'name':'aggregate_dataset','args':{'dataset_id':'$fixture','aggregation':'mean','value_column':'measurement'}}])
        executed=[]
        def factory(datasets):
            def execute(envelope):
                executed.append(envelope)
                raw=datasets.register(pd.DataFrame({'measurement':[2.,4.,9.]}),
                    source='custom.trials',query=envelope['query'],coverage='complete',predicate_known=True)
                model.evaluation_dataset_id=raw.id
                return {'status':'ready','dataset_id':raw.id}
            return execute
        refs=[{'table':'custom.trials','observed_at':datetime.now(timezone.utc).isoformat(),
               'columns':[{'name':'measurement','dtype':'double'}]}]
        with tempfile.TemporaryDirectory() as root,patch('core.analysis_agent.model_recovery.time.sleep'):
            r=GraphAnalysisRuntime(root,'owner','approved-retry',model,remote_factory=factory,
                connection_identity='synthetic-test-only',reference_context_loader=lambda:refs)
            try:
                with patch.object(r.recovery,'_next_local',return_value=None):
                    proposed=r.submit('measurement 평균을 알려줘')
                    self.assertEqual(proposed['status'],'awaiting_approval',proposed)
                    self.assertEqual(executed,[])
                    pending=r.inspect()['requests'][0]
                    result=r.respond(pending['id'],approved=True)
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(executed),1)
                self.assertEqual(r.inspect()['model_recovery']['retries'],1)
                self.assertEqual(r.inspect()['recovery']['model_calls'],3)
            finally:r.close()

    def test_resume_excludes_paused_wall_time_without_resetting_failed_attempts(self):
        with tempfile.TemporaryDirectory() as root, patch('core.analysis_agent.model_recovery.time.sleep'):
            model=IntermittentModel(failures_left=3,calls=[{'name':'aggregate_dataset','args':{
                'dataset_id':'$fixture','aggregation':'mean','value_column':'measurement'}}])
            r=GraphAnalysisRuntime(root,'owner','paused-retry',model)
            try:
                raw=r.datasets.register(pd.DataFrame({'measurement':[2.,4.,9.]}),source='custom.trials',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
                with patch.object(r.recovery,'_next_local',return_value=None),patch.object(r.recovery,'resume_local_call',return_value=None):
                    first=r.submit('measurement 평균을 알려줘')
                    self.assertEqual(first['status'],'incomplete',first)
                    saved=dict(r.agent.get_state(r.config).values['recovery'])
                    saved['model_started_at']=1.
                    r.agent.update_state(r.config,{'recovery':saved})
                    result=r.resume()
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.inspect()['recovery']['model_calls'],4)
                self.assertLess(r.inspect()['recovery']['model_seconds'],10)
                self.assertEqual(r.inspect()['model_recovery']['retries'],2)
            finally:r.close()
