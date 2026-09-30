"""The documented launcher must select the supported interpreter and requested port."""

from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.run_telly import main


class RunTellyTests(unittest.TestCase):
    def test_launcher_uses_pinned_runtime_on_requested_port(self):
        root = Path(__file__).resolve().parents[1]
        python = str(root / '.telly_runtime/v1-venv/bin/python')
        with patch('scripts.run_telly.port_in_use', return_value=False), \
                patch('scripts.run_telly.os.chdir') as chdir, \
                patch('scripts.run_telly.os.execv') as execv:
            main(['--port', '8501', '--server.headless=true'])
        chdir.assert_called_once_with(root)
        execv.assert_called_once_with(python, [python, '-m', 'streamlit', 'run', 'main.py',
            '--server.address=127.0.0.1', '--server.port=8501', '--server.headless=true'])

    def test_launcher_rejects_a_port_occupied_by_another_server(self):
        with patch('scripts.run_telly.port_in_use', return_value=True), \
                patch('scripts.run_telly.os.execv') as execv:
            with self.assertRaisesRegex(SystemExit, '이미 서버가 실행 중입니다'):
                main(['--port', '8501'])
        execv.assert_not_called()


if __name__ == '__main__':
    unittest.main()
