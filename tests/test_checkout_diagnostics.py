"""Diagnostics must distinguish disk evidence from running-process evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.diagnose_checkout import collect


class CheckoutDiagnosticsTests(unittest.TestCase):
    def test_no_git_or_dependencies_is_unknown_not_verified(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                'scripts.diagnose_checkout.package_version', return_value=None):
            report = collect(Path(folder), 'nonexistent')
        self.assertIsNone(report['expected_revision_in_history'])
        self.assertIsNone(report['tracked_changes'])
        self.assertFalse(report['running_server_verified'])
        self.assertIn('unavailable', report['telly_page_for_this_interpreter'])

    def test_commit_presence_does_not_hide_modified_source_or_expose_env(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def git(*args):
                return subprocess.check_output(['git', '-C', folder, *args], text=True).strip()
            git('init', '-q')
            (root / 'main.py').write_text('original')
            git('add', 'main.py')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '-qm', 'fixture')
            revision = git('rev-parse', 'HEAD')
            (root / '.env').write_text('SECRET=never-include-this')
            (root / 'main.py').write_text('modified')
            report = collect(root, revision)
        self.assertTrue(report['expected_revision_in_history'])
        self.assertTrue(report['tracked_changes'])
        self.assertFalse(report['running_server_verified'])
        self.assertNotIn('never-include-this', json.dumps(report))
        self.assertNotIn('.env', report['files_sha256'])

    def test_dependency_route_is_interpreter_specific(self):
        with tempfile.TemporaryDirectory() as folder:
            for version, route in [('0.3.0', 'ui/legacy_telly.py'),
                                   ('1.0.0', 'ui/analysis_page.py')]:
                with patch('scripts.diagnose_checkout.package_version', return_value=version):
                    self.assertEqual(collect(Path(folder))['telly_page_for_this_interpreter'], route)


if __name__ == '__main__':
    unittest.main()
