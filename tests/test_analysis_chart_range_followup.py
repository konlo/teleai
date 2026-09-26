"""A scoped chart follow-up must not replay an unfiltered cached image."""
import tempfile
import unittest

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.intent_scope import resolve_request_scope


class NoModelCall(BaseChatModel):
    @property
    def _llm_type(self):
        return 'no-model-chart-fixture'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        raise AssertionError('grounded local chart follow-up should not call the model')


class ChartRangeFollowupTests(unittest.TestCase):
    def test_web_wording_binds_interval_to_grounded_column(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'scope-only', NoModelCall())
            info = runtime.datasets.register(pd.DataFrame({'metric': [0, 1, 3, 5, 6, 100]}),
                source='catalog.schema.events', coverage='complete', predicate_known=True)
            runtime.select_dataset(info.id)
            scope = resolve_request_scope(
                '방금 metric 히스토그램에서 0부터 5까지 범위만 다시 보여줘. 현재 로딩된 6행 표본만 사용해.',
                runtime.context)
            self.assertEqual(scope['conditions'], [
                {'column': 'metric', 'op': 'ge', 'value': 0},
                {'column': 'metric', 'op': 'le', 'value': 5}])
            self.assertEqual(scope['unresolved'], [])
            runtime.close()

    def test_range_followup_creates_new_local_chart_and_preserves_root(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'range-followup', NoModelCall())
            frame = pd.DataFrame({'metric': [0, 1, 3, 5, 6, 100]})
            info = runtime.datasets.register(frame, source='catalog.schema.events',
                coverage='complete', predicate_known=True)
            runtime.select_dataset(info.id)
            first = runtime.submit('현재 로딩된 6행 표본의 metric 히스토그램을 보여줘')
            self.assertEqual(first['status'], 'answered', first)
            original_card_ids = set(runtime.artifacts)
            self.assertTrue(original_card_ids)

            result = runtime.submit(
                '방금 metric 히스토그램에서 0부터 5까지 범위만 다시 보여줘. 현재 로딩된 6행 표본만 사용해.')
            self.assertEqual(result['status'], 'answered', result)
            new_cards = [runtime.artifacts[key] for key in set(runtime.artifacts)-original_card_ids]
            self.assertEqual(len(new_cards), 1)
            chart_info = runtime.datasets.metadata[new_cards[0].dataset_id]
            self.assertEqual(chart_info.rows, 4)
            self.assertEqual(runtime.context.selected_dataset_id, info.id)
            pd.testing.assert_frame_equal(runtime.datasets.frames[info.id], frame)
            self.assertEqual(runtime.inspect()['requests'], [])

            count = runtime.submit(
                '현재 로딩된 6행 표본에서 metric이 0 이상 5 이하인 행은 몇 개인지 알려줘.')
            self.assertEqual(count['status'], 'answered', count)
            self.assertIn('건수: 4', count['text'])
            self.assertEqual(runtime.inspect()['requests'], [])
            pd.testing.assert_frame_equal(runtime.datasets.frames[info.id], frame)
            runtime.close()


if __name__ == '__main__':
    unittest.main()
