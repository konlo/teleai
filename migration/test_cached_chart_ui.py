"""The cached PNG must actually appear in the page, including repeated requests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from migration.test_completion_contracts import ScriptModel, SOURCE, COLUMN


class CachedChartUITests(unittest.TestCase):
    def test_repeated_prepared_chart_displays_without_duplicate_widget_keys(self):
        calls = [{'name':'prepare_histogram', 'args':{'source':SOURCE, 'column':COLUMN}} for _ in range(2)]
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'TELLY_V1_STORAGE':root}), \
                patch('langchain_ollama.ChatOllama', return_value=ScriptModel(calls=calls)), \
                patch('databricks.sql.connect') as connect:
            page = Path(__file__).resolve().parents[1] / 'ui' / 'analysis_page.py'
            app = AppTest.from_file(str(page), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            next(b for b in app.button if b.label == '예제 데이터로 시작').click().run()
            for _ in range(2):
                app.chat_input[0].set_value(f'{COLUMN} histogram').run()
                self.assertEqual(len(app.exception), 0)
            runtime = app.session_state['v1_runtime']
            self.assertEqual(runtime.inspect()['recovery']['status'], 'complete')
            self.assertEqual(len(runtime.artifacts), 1)
            self.assertEqual(len(app.get('image')), 2)
            self.assertEqual(len([b for b in app.button if b.label == '이 차트 선택']), 2)
            connect.assert_not_called()
            runtime.close()


if __name__ == '__main__': unittest.main()
