"""Storage/replay harness contracts; these are not LLM quality tests."""
import contextlib
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_prompt_scenario import main, replay
from test_set.prompt_scenarios import SCENARIO_ROOT, load_scenario, prompt_turns

ROOT = Path(__file__).resolve().parents[1]


class SavedPromptScenarioTests(unittest.TestCase):
    def test_exact_text_and_order_match_original_artifacts(self):
        sources = {
            'prompt_konlo_test_scenario_#1': ('2026-10-06_actual_prompt_replay', 31),
            'prompt_ai_test_scenario_#1': ('2026-10-06_ten_prompt_journey', 10),
        }
        for name, (folder, count) in sources.items():
            with self.subTest(name=name):
                source = json.loads((ROOT / 'docs/evaluation' / folder / 'plan.json').read_text())
                self.assertEqual(prompt_turns(name), [t['prompt'] for t in source[:count]])
                self.assertEqual(load_scenario(name)['prompt_count'], count)

    def test_changed_prompt_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in SCENARIO_ROOT.glob('*.json'):
                (root / path.name).write_bytes(path.read_bytes())
            name = 'prompt_ai_test_scenario_#1'
            path = root / (name + '.json')
            data = json.loads(path.read_text()); data['turns'][0]['prompt'] = 'changed'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_scenario(name, root)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                load_scenario('../outside', root)

    def test_list_show_export_do_not_open_live_runtime(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'scripts.run_prompt_scenario.open_runtime') as factory, contextlib.redirect_stdout(StringIO()):
            name = 'prompt_konlo_test_scenario_#1'
            self.assertEqual(main(['--list']), 0)
            self.assertEqual(main(['--scenario', name, '--show']), 0)
            path = Path(directory) / 'export.json'
            self.assertEqual(main(['--scenario', name, '--export', str(path)]), 0)
            self.assertEqual(json.loads(path.read_text()), load_scenario(name))
            with self.assertRaises(FileExistsError):
                main(['--scenario', name, '--export', str(path)])
            factory.assert_not_called()

    def test_replay_submits_exactly_once_and_continues_without_expectation_leak(self):
        scenario = load_scenario('prompt_ai_test_scenario_#1')
        seen = []

        class Runtime:
            diagnostics = SimpleNamespace(run_id='run', last_error_id=None)

            def submit(self, text):
                seen.append(text)
                if len(seen) == 2:
                    raise TimeoutError('do not dump this private exception')
                if len(seen) == 3:
                    return {'status': 'incomplete', 'text': 'unfinished'}
                return {'status': 'answered', 'text': 'not independently verified'}

            def inspect(self):
                return {'state': 'idle', 'recovery': {}}

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(StringIO()):
            path = Path(directory) / 'report.json'
            report = replay(scenario, Runtime(), path, backend='mysql', provider='ollama')
            self.assertEqual(seen, prompt_turns(scenario['name']))
            self.assertEqual(report['recorded_turns'], 10)
            self.assertEqual(report['answered_turns'], 8)
            self.assertEqual(report['grading'], 'NOT_GRADED')
            self.assertEqual(report['status'], 'INCOMPLETE_UNGRADED')
            self.assertEqual(report['turns'][1]['error_type'], 'TimeoutError')
            self.assertNotIn('private exception', path.read_text())
            self.assertNotIn('reference_expectation', path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                replay(scenario, Runtime(), path, backend='mysql', provider='ollama')
            self.assertEqual(len(seen), 10)

    def test_setup_failure_is_recorded_without_leaking_exception(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'scripts.run_prompt_scenario.open_runtime', side_effect=ValueError('private token')), \
                contextlib.redirect_stdout(StringIO()) as console:
            path = Path(directory) / 'setup.json'
            result = main(['--scenario', 'prompt_ai_test_scenario_#1', '--run',
                           '--backend', 'mysql', '--provider', 'ollama', '--output', str(path)])
            self.assertEqual(result, 1)
            self.assertEqual(json.loads(path.read_text())['status'], 'SETUP_FAILED')
            self.assertNotIn('private token', console.getvalue() + path.read_text())


if __name__ == '__main__':
    unittest.main()
