"""Real OS contention/release plus isolated Windows API error contracts."""
import errno
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.analysis_agent.file_lock import acquire_lock, conversation_lock, release_lock


class ConversationLockTests(unittest.TestCase):
    def test_same_process_contention_and_exception_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.lock'
            with self.assertRaisesRegex(ValueError, 'work failed'):
                with conversation_lock(path):
                    with self.assertRaises(BlockingIOError):
                        with conversation_lock(path):
                            self.fail('A second writer acquired the held lock')
                    with self.assertRaises(BlockingIOError):
                        with conversation_lock(path, shared=True):
                            self.fail('A snapshot acquired a writer-held lock')
                    raise ValueError('work failed')
            with conversation_lock(path):
                pass

    def test_other_process_is_excluded_and_can_acquire_after_release(self):
        code = '''
import sys
from core.analysis_agent.file_lock import conversation_lock
try:
 with conversation_lock(sys.argv[1]): print('ACQUIRED')
except BlockingIOError: print('BUSY')
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.lock'
            def child():
                result = subprocess.run([sys.executable, '-c', code, str(path)],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True,
                    text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout.strip()
            with conversation_lock(path):
                self.assertEqual(child(), 'BUSY')
            self.assertEqual(child(), 'ACQUIRED')

    def test_windows_uses_the_same_byte_for_lock_and_unlock(self):
        calls = []
        with tempfile.TemporaryFile('w+b') as stream:
            stub = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0,
                locking=lambda descriptor, mode, count: calls.append((descriptor, mode, count, stream.tell())))
            with patch('core.analysis_agent.file_lock.sys.platform', 'win32'), patch.dict(sys.modules, {'msvcrt':stub}):
                acquire_lock(stream, shared=True)
                stream.seek(20)
                release_lock(stream)
            self.assertEqual(calls, [(stream.fileno(), 2, 1, 0), (stream.fileno(), 0, 1, 0)])
            stream.seek(0)
            self.assertEqual(stream.read(), b'\0')

    def test_windows_contention_is_distinct_from_unexpected_io_failure(self):
        with tempfile.TemporaryFile('w+b') as stream:
            stream.write(b'existing lock'); stream.flush()
            for number, expected in [(errno.EACCES, BlockingIOError), (errno.EAGAIN, BlockingIOError),
                                     (errno.EIO, OSError)]:
                stub = SimpleNamespace(LK_NBLCK=2, locking=Mock(side_effect=OSError(number, 'fixture')))
                with self.subTest(errno=number), patch('core.analysis_agent.file_lock.sys.platform', 'win32'), \
                        patch.dict(sys.modules, {'msvcrt':stub}):
                    with self.assertRaises(expected) as caught:
                        acquire_lock(stream)
                    if number == errno.EIO:
                        self.assertNotIsInstance(caught.exception, BlockingIOError)
                stream.seek(0)
                self.assertEqual(stream.read(), b'existing lock')

    def test_import_does_not_require_unix_module(self):
        code = '''
import importlib.abc, sys
class NoUnixLock(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'fcntl': raise ModuleNotFoundError('fcntl unavailable')
sys.meta_path.insert(0, NoUnixLock())
import core.analysis_agent.file_lock
print('PASS')
'''
        result = subprocess.run([sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)
