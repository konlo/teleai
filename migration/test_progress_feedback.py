import json
import unittest
from langchain_core.messages import ToolMessage
from core.analysis_agent.progress import tool_progress


class ProgressFeedbackTests(unittest.TestCase):
    def test_ollama_summary_copy_preserves_analysis_model_and_context_limits(self):
        from langchain_ollama import ChatOllama
        from core.analysis_agent.memory import memory_middleware
        model = ChatOllama(model='fixture-model', reasoning=True, num_ctx=16384, num_predict=4096)
        middleware = memory_middleware(model)
        self.assertTrue(model.reasoning)
        self.assertFalse(middleware.model.reasoning)
        self.assertEqual(middleware.model.num_ctx, model.num_ctx)
        self.assertEqual(middleware.model.num_predict, model.num_predict)

    def message(self, name, payload):
        return tool_progress(ToolMessage(name=name, content=json.dumps(payload), tool_call_id='test'))

    def test_failed_or_empty_chart_does_not_claim_image_generated(self):
        for status in ('needs_context', 'no_valid_chart', 'error', 'needs_data'):
            text = self.message('recommend_chart_images', {'status':status})
            self.assertIn('복구', text)
            self.assertNotIn('차트 생성 완료', text)
        self.assertNotIn('생성된 차트', self.message('render_histogram', {'status':'ready','cards':[]}))

    def test_ready_calculation_is_still_being_validated(self):
        text = self.message('local_analysis_sql', {'status':'ready'})
        self.assertIn('검증', text)
        self.assertNotIn('완료', text)

    def test_actual_chart_and_remote_load_have_distinct_progress(self):
        self.assertIn('생성된 차트', self.message('render_histogram', {'status':'ready','cards':[{'id':'png'}]}))
        self.assertIn('불러왔습니다', self.message('query_databricks', {'status':'ready'}))
        self.assertNotIn('불러왔습니다', self.message('query_databricks', {'status':'unavailable'}))

    def test_unknown_observation_never_implies_tool_success(self):
        self.assertEqual(self.message('query_databricks', []), '도구 결과를 확인하고 있습니다.')


if __name__ == '__main__': unittest.main()
