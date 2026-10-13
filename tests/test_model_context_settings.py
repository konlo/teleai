"""Model limits, role inheritance and language estimates without remote inference."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from langchain_core.messages import HumanMessage,SystemMessage
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.model_roles import json_role
from core.analysis_agent.model_context import ModelContextBudgetMiddleware,ModelContextBudgetExceeded,payload_bytes
from core.analysis_agent.context_settings import ollama_role_model
from core.analysis_agent.input_token_measurement import limits,measure
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.support_report import public_input_budget
from tests.test_azure_databricks_service import AZURE
from tests.test_json_role_contracts import request_for


class ModelContextSettingsTests(unittest.TestCase):
    def model(self,values):
        return build_analysis_chat_model(RuntimePolicy(),environ=values)

    def test_gemma_maximum_is_selected_without_requiring_azure_or_sql_settings(self):
        model=self.model({'LLM_PROVIDER':'ollama','OLLAMA_MODEL':'gemma4:e4b'})
        self.assertEqual(model.num_ctx,131072)
        self.assertEqual(model.num_predict,4096)
        role=json_role(model,{'type':'object'},512)
        self.assertEqual(limits(role,32000),(131072,512,130560))
        self.assertEqual(model.num_predict,4096)

    def test_other_ollama_models_do_not_inherit_gemma_maximum(self):
        model=self.model({'LLM_PROVIDER':'ollama','OLLAMA_MODEL':'unknown-model'})
        self.assertEqual(model.num_ctx,16384)
        explicit=self.model({'LLM_PROVIDER':'ollama','OLLAMA_MODEL':'unknown-model',
            'OLLAMA_CONTEXT_TOKENS':'32768','OLLAMA_MAX_OUTPUT_TOKENS':'2048'})
        self.assertEqual(limits(explicit,32000),(32768,2048,30720))
        role=ollama_role_model(self.model({'LLM_PROVIDER':'ollama'}),'unknown-model',{})
        self.assertEqual(role.num_ctx,16384)

    def test_invalid_and_over_model_settings_are_rejected_before_any_call(self):
        cases=[{'OLLAMA_CONTEXT_TOKENS':'0'},{'OLLAMA_CONTEXT_TOKENS':'131073'},
               {'OLLAMA_MAX_OUTPUT_TOKENS':'131072'}, {'OLLAMA_CONTEXT_TOKENS':'invalid'}]
        for values in cases:
            with self.subTest(values=values),self.assertRaises(ValueError):
                self.model({'LLM_PROVIDER':'ollama',**values})
        with self.assertRaises(ValueError):
            self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6','TELLY_AZURE_CONTEXT_TOKENS':'1050001'})
        with self.assertRaises(ValueError):
            self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6','TELLY_AZURE_MAX_OUTPUT_TOKENS':'128001'})

    def test_azure_gpt_profile_and_effective_json_role_reservation(self):
        model=self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6'})
        self.assertEqual(limits(model,32000),(1050000,4096,1045904))
        role=json_role(model,{'type':'object'},512)
        self.assertEqual(limits(role,32000),(1050000,512,1049488))
        payload=role._get_request_payload([HumanMessage(content='Return JSON')])
        self.assertEqual(payload['max_tokens'],512)
        for name in ('context_window_tokens','input_token_encoding'):
            self.assertNotIn(name,payload)
        smaller=self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6','TELLY_AZURE_CONTEXT_TOKENS':'32768'})
        self.assertEqual(smaller.context_window_tokens,32768)
        self.assertEqual(smaller.deployment_name,model.deployment_name)

    def test_opaque_azure_deployment_and_other_provider_do_not_get_gpt_maximum(self):
        legacy=self.model(AZURE)
        self.assertIsNone(legacy.context_window_tokens)
        self.assertEqual(limits(legacy,32000),(None,4096,32000))
        other=self.model({**AZURE,'AZURE_OPENAI_MODEL':'another-model',
            'TELLY_AZURE_CONTEXT_TOKENS':'32768'})
        self.assertEqual(other.context_window_tokens,32768)
        self.assertIsNone(other.input_token_encoding)

    def test_utf8_bytes_are_not_treated_as_gpt_tokens_and_current_goal_is_kept(self):
        model=self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6','TELLY_AZURE_CONTEXT_TOKENS':'16384'})
        request=request_for(text='한국어 조건을 반드시 유지하고 분석해 주세요. '*350)
        self.assertGreater(payload_bytes(request.system_message,request.messages,[]),12288)
        diagnostics=Mock()
        result=ModelContextBudgetMiddleware(model,diagnostics).wrap_model_call(request,lambda r:r)
        self.assertEqual(result.messages,request.messages)
        report=public_input_budget(diagnostics.emit.call_args.kwargs)
        self.assertEqual(report['measurement'],'tiktoken_estimate')
        self.assertLess(report['input_tokens_estimate']+report['template_headroom'],12288)
        self.assertEqual(report['input_budget_units'],12288)
        self.assertNotIn('한국어',json.dumps(report,ensure_ascii=False))

    def test_no_matching_tokenizer_falls_back_to_declared_conservative_bound(self):
        model=self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6'})
        with patch('core.analysis_agent.input_token_measurement.encoder',side_effect=ImportError):
            count,method=measure('한국어',model)
        self.assertEqual(count,len('한국어'.encode()))
        self.assertEqual(method,'conservative_utf8_token_upper_bound')

    def test_large_unsupported_input_is_still_stopped_before_handler(self):
        model=self.model({**AZURE,'AZURE_OPENAI_MODEL':'gpt-5.6','TELLY_AZURE_CONTEXT_TOKENS':'5000'})
        handler=Mock()
        request=request_for(text='한국어 분석 '*4000)
        with self.assertRaises(ModelContextBudgetExceeded):
            ModelContextBudgetMiddleware(model).wrap_model_call(request,handler)
        handler.assert_not_called()

    def test_same_previous_failure_sized_input_fits_gemma_maximum(self):
        model=self.model({'LLM_PROVIDER':'ollama'})
        diagnostics=Mock();request=request_for(text='x'*14048)
        result=ModelContextBudgetMiddleware(model,diagnostics).wrap_model_call(request,lambda r:r)
        self.assertEqual(result.messages,request.messages)
        report=public_input_budget(diagnostics.emit.call_args.kwargs)
        self.assertEqual(report['context_window'],131072)
        self.assertEqual(report['input_budget_units'],126976)
        self.assertEqual(report['measurement'],'conservative_utf8_token_upper_bound')
        self.assertTrue(report['within_budget'])


if __name__=='__main__':unittest.main()
