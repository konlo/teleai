"""A retained selectbox cannot silently undo New conversation."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from tests.test_llm_goal import GoalModel,goal


class ConversationSelectorUITests(unittest.TestCase):
    def test_new_conversation_and_return_to_history_are_both_stable(self):
        backend=SimpleNamespace(name='mysql',label='fixture',dialect='mysql',namespace='lab',
            config=SimpleNamespace(identity=lambda:'fixture'),preflight=lambda:None,
            storage_root=lambda root:root,context_loader=lambda:[],executor_factory=lambda *a,**k:None)
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root}), \
                patch('core.analysis_agent.backends.load_data_backend',return_value=backend), \
                patch('core.analysis_agent.model_provider.build_analysis_chat_model',
                      return_value=GoalModel(goals=[goal('metadata',{'kind':'columns'})])):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=20).run()
            first=app.session_state['v1_conversation']
            app.button[next(i for i,b in enumerate(app.button) if b.label=='새 대화')].click().run()
            second=app.session_state['v1_conversation']
            self.assertNotEqual(first,second)
            app.run()
            self.assertEqual(app.session_state['v1_conversation'],second)
            app.selectbox(key='v1_conversation_selector').select(first).run()
            self.assertEqual(app.session_state['v1_conversation'],first)
            app.button[next(i for i,b in enumerate(app.button) if b.label=='새 대화')].click().run()
            third=app.session_state['v1_conversation']
            self.assertNotIn(third,[first,second])
            app.run()
            self.assertEqual(app.session_state['v1_conversation'],third)
            self.assertFalse(app.exception)
            app.session_state['v1_runtime'].close()
