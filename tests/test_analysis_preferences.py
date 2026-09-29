"""Model billing/provider selection survives reload and stays conversation scoped."""
import sqlite3
from pathlib import Path
import tempfile
import unittest
from streamlit.testing.v1 import AppTest
from ui.analysis_preferences import saved_provider, save_provider


class AnalysisPreferencesTest(unittest.TestCase):
    def test_actual_widget_reload_switch_and_stale_tab_do_not_change_provider(self):
        with tempfile.TemporaryDirectory() as root:
            script = ('import streamlit as st\nfrom ui.analysis_preferences import render_provider_selector\n'
                      "st.session_state.setdefault('conversation', 'first')\n"
                      f"st.write(render_provider_selector({root!r}, st.session_state['conversation']))\n")
            app = AppTest.from_string(script).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox[0].value, 'ollama')
            app.selectbox[0].select('databricks').run()
            self.assertEqual(saved_provider(root, 'first'), 'databricks')
            fresh = AppTest.from_string(script).run()
            self.assertFalse(fresh.exception)
            self.assertEqual(fresh.selectbox[0].value, 'databricks')
            # A second conversation cannot inherit the paid model selection.
            fresh.session_state['conversation'] = 'second'
            fresh.run()
            self.assertEqual(fresh.selectbox[0].value, 'ollama')
            fresh.session_state['conversation'] = 'first'
            fresh.run()
            self.assertEqual(fresh.selectbox[0].value, 'databricks')
            fresh.selectbox[0].select('ollama').run()
            app.run()  # Existing tab holds old choice; mere render is not a change.
            self.assertEqual(saved_provider(root, 'first'), 'ollama')

    def test_invalid_saved_choice_falls_back_without_saving_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(saved_provider(root, 'first'), 'ollama')
            with sqlite3.connect(Path(root)/'conversations.sqlite') as db:
                db.execute('INSERT INTO conversation_preferences VALUES (?,?)', ('first', 'unknown'))
            self.assertEqual(saved_provider(root, 'first'), 'ollama')
            with self.assertRaises(ValueError):
                save_provider(root, 'first', 'unknown')
