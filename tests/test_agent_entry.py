"""Both public URLs must refuse an unsupported interpreter instead of opening legacy UI."""

from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class AgentEntryTests(unittest.TestCase):
    def test_unsupported_runtime_never_opens_legacy_chat(self):
        with patch('ui.agent_entry.metadata.version', return_value='0.3.27'):
            for page in ('main.py', 'pages/Telly.py'):
                with self.subTest(page=page):
                    app = AppTest.from_file(str(ROOT / page), default_timeout=20).run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.error)
                    self.assertIn('LangChain 0.3.27', app.error[0].value)
                    self.assertFalse(app.chat_input)


if __name__ == '__main__':
    unittest.main()
