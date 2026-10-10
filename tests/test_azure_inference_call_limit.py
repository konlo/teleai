"""Resolved Azure provider receives 50 calls; every other provider keeps 10."""
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import HumanMessage

from core.analysis_agent.memory import memory_middleware
from core.analysis_agent.model_provider import build_analysis_chat_model, inference_call_limit
from core.analysis_agent.model_recovery import ModelAttemptBudgetExceeded
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_azure_databricks_service import AZURE


class AzureInferenceCallLimitTests(unittest.TestCase):
    def models(self):
        policy=RuntimePolicy()
        return [('azure',build_analysis_chat_model(policy,environ=AZURE),50),
                ('ollama',build_analysis_chat_model(policy,environ={}),10),
                ('databricks',build_analysis_chat_model(policy,provider='databricks',environ={
                    'DATABRICKS_HOST':'workspace.example.invalid','DATABRICKS_TOKEN':'fixture'}),10)]

    def test_resolved_provider_controls_limit_independent_of_database(self):
        for provider,model,expected in self.models():
            for backend in ('mysql','databricks'):
                with self.subTest(provider=provider,backend=backend),tempfile.TemporaryDirectory() as root:
                    runtime=GraphAnalysisRuntime(root,'owner','limit',model,sql_dialect=backend)
                    try:
                        self.assertEqual(inference_call_limit(model),expected)
                        self.assertEqual(runtime.recovery.max_model_calls,expected)
                        self.assertEqual(runtime.model_recovery.max_calls,expected)
                        summary=memory_middleware(model,model_recovery=runtime.model_recovery)
                        self.assertEqual(summary.max_model_calls,expected)
                        self.assertEqual(runtime.policy.turn_slo_seconds,180)
                        self.assertEqual(runtime.model_recovery.max_retries,2)
                    finally:runtime.close()

    def test_azure_allows_more_than_nine_and_reserves_exactly_one_call(self):
        model=build_analysis_chat_model(RuntimePolicy(),environ=AZURE)
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','limit',model)
            try:
                current={'request_id':'request','model_calls':0,'model_seconds':0}
                handler=Mock(return_value='ok')
                for _ in range(49):
                    self.assertEqual(runtime.model_recovery.auxiliary_call(current,handler),'ok')
                with self.assertRaises(ModelAttemptBudgetExceeded) as caught:
                    runtime.model_recovery.auxiliary_call(current,handler)
                handler.assert_has_calls([unittest.mock.call()]*49)
                self.assertEqual(handler.call_count,49)
                self.assertEqual(caught.exception.attempt_budget['call_limit'],49)
                self.assertEqual(caught.exception.attempt_budget['total_call_limit'],50)
                self.assertEqual(caught.exception.attempt_budget['reserved_calls'],1)
                runtime.model_attempts.sync(current)
                # The remaining main-model call can still enter, then no more.
                self.assertEqual(runtime.model_recovery.invoke(current,handler),'ok')
                current['model_calls']+=1
                with self.assertRaises(ModelAttemptBudgetExceeded):
                    runtime.model_recovery.invoke(current,handler)
                self.assertEqual(handler.call_count,50)
            finally:runtime.close()

    def test_azure_summary_is_not_stopped_by_the_old_nine_call_boundary(self):
        model=build_analysis_chat_model(RuntimePolicy(),environ=AZURE)
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','summary',model)
            try:
                summary=memory_middleware(model,model_recovery=runtime.model_recovery)
                state={'messages':[HumanMessage(content='context')],
                       'recovery':{'request_id':'request','model_calls':10}}
                with patch.object(SummarizationMiddleware,'before_model',return_value=None) as parent:
                    summary.before_model(state,None)
                    parent.assert_called_once()
                state['recovery']['model_calls']=49
                with patch.object(SummarizationMiddleware,'before_model',return_value=None) as parent:
                    summary.before_model(state,None)
                    parent.assert_not_called()
            finally:runtime.close()

    def test_azure_time_limit_is_still_enforced_before_any_call(self):
        model=build_analysis_chat_model(RuntimePolicy(),environ=AZURE)
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','deadline',model)
            try:
                handler=Mock()
                with self.assertRaises(ModelAttemptBudgetExceeded) as caught:
                    runtime.model_recovery.auxiliary_call({
                        'request_id':'request','model_seconds':181},handler)
                self.assertEqual(caught.exception.attempt_budget['reason'],'time')
                self.assertEqual(caught.exception.attempt_budget['total_call_limit'],50)
                handler.assert_not_called()
            finally:runtime.close()

    def test_new_prompt_in_same_conversation_gets_independent_budget(self):
        from core.analysis_agent.goal_contract import pending_state
        model=build_analysis_chat_model(RuntimePolicy(),environ=AZURE)
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'same-user','same-conversation',model)
            try:
                first=HumanMessage(content='테이블 목록을 보여줘',id='first')
                previous=pending_state(first,{},runtime.context)
                for _ in range(49):runtime.model_attempts.auxiliary_success('first',0.)
                runtime.model_attempts.sync(previous)
                previous.update(status='complete',required_sources=['lab.observations'])
                second=pending_state(HumanMessage(content='컬럼을 보여줘',id='second'),
                                     previous,runtime.context)
                self.assertEqual(second['model_calls'],0)
                self.assertEqual(second['confirmed_analysis']['required_sources'],['lab.observations'])
                handler=Mock(return_value='ok')
                self.assertEqual(runtime.model_recovery.auxiliary_call(second,handler),'ok')
                self.assertEqual(runtime.model_attempts.get('first')['aux_calls'],49)
                self.assertEqual(runtime.model_attempts.get('second')['aux_calls'],1)
                # Reinterpreting the same unfinished ID cannot reset usage.
                previous.update(status='working')
                resumed=pending_state(first,previous,runtime.context)
                self.assertEqual(resumed['model_calls'],49)
                with self.assertRaises(ModelAttemptBudgetExceeded):
                    runtime.model_recovery.auxiliary_call(resumed,handler)
                handler.assert_called_once()
            finally:runtime.close()
