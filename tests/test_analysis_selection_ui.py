"""The Streamlit analysis page exposes the same persisted selection as runtime."""
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


class SelectionUiTests(unittest.TestCase):
    def test_selected_column_prefix_renders_grounded_preview_without_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {'TELLY_V1_STORAGE': directory}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] /
                                         'ui' / 'analysis_page.py'), default_timeout=20).run()
            next(button for button in app.button if button.label == '예제 데이터로 시작').click().run()
            runtime = app.session_state['v1_runtime']
            selected_before = runtime.db.selected_dataset_id()
            app.chat_input[0].set_value('value 컬럼의 앞 두 행만 보여줘. 보유 데이터만 사용해줘.').run()
            self.assertFalse(app.exception)
            self.assertEqual(runtime.inspect()['recovery']['model_calls'], 0)
            self.assertIn('앞 2개 값', app.session_state['v1_runtime'].events()[-1].content)
            self.assertEqual(runtime.db.selected_dataset_id(), selected_before)
            self.assertEqual(runtime.inspect()['requests'], [])
            runtime.close()

    def test_explicit_boxplot_tool_result_shows_actual_image_in_chat(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {'TELLY_V1_STORAGE': directory}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] /
                                         'ui' / 'analysis_page.py'), default_timeout=30).run()
            next(button for button in app.button if button.label == '예제 데이터로 시작').click().run()
            app.chat_input[0].set_value('value의 boxplot을 보여줘').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.image), 1,
                             'A completed chart response must render a visible image')
            runtime = app.session_state['v1_runtime']
            card = runtime.artifacts[runtime.inspect()['chart_ids'][0]]
            self.assertEqual(card.kind, 'boxplot')
            self.assertTrue(card.image.startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertEqual(runtime.inspect()['recovery']['model_calls'], 0)
            app.chat_input[0].set_value(
                'value 값들이 어느 구간에 얼마나 모여 있는지 그림으로 보여줘. 보유 데이터만 사용해줘.').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.image), 2)
            latest = runtime.artifacts[runtime.inspect()['chart_ids'][-1]]
            self.assertEqual(latest.kind, 'histogram')
            self.assertEqual(runtime.inspect()['recovery']['model_calls'], 0)
            runtime.close()

    def test_example_then_switch_active_dataset_without_remote_query(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {'TELLY_V1_STORAGE':directory}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] /
                                         'ui' / 'analysis_page.py'), default_timeout=20).run()
            self.assertFalse(app.exception)
            next(button for button in app.button if button.label == '예제 데이터로 시작').click().run()
            self.assertFalse(app.exception)
            runtime = app.session_state['v1_runtime']
            original = runtime.inspect()['selected_dataset']
            self.assertEqual(original['role'], 'root')
            self.assertEqual(runtime.datasets.frames.bytes, 0,
                             'Sidebar preview must not materialize the full dataset')
            second = runtime.datasets.register(pd.DataFrame({'metric':[2, 4, 8]}),
                source='fixture.second', coverage='complete', predicate_known=True)
            app.run()
            app.button(key='select-'+second.id).click().run()
            self.assertFalse(app.exception)
            self.assertEqual(runtime.inspect()['selected_dataset']['id'], second.id)
            self.assertEqual(runtime.inspect()['requests'], [])
            self.assertEqual(runtime.datasets.frames.bytes, 0)
            runtime.close()


if __name__ == '__main__':
    unittest.main()
