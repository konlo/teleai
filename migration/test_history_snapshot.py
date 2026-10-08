"""History replay must not consume an approval or mutate the user's state."""
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.file_lock import conversation_lock
import tempfile
import unittest
from pathlib import Path

from core.analysis_agent.assets import AssetDB
from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel
from migration.test_approval_rollout import ChangeModel
from migration.test_recovery_journey import SOURCE
from scripts.check_cached_histogram import copy_runtime_snapshot


class HistorySnapshotTests(unittest.TestCase):
    def test_pending_history_is_copied_and_only_clone_approval_changes(self):
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as clone_root:
            calls = []
            factory = lambda _: lambda envelope: calls.append(envelope)
            original = GraphAnalysisRuntime(source_root, 'owner', 'original', QuietModel(),
                connection_identity='test', remote_factory=factory, policy=RuntimePolicy(require_remote_approval=True),intent_mode='contract_fixture')
            request = original.propose_query(SOURCE, 'SELECT 1', '검증용 조회')['requests'][0]
            initial_count = original.inspect()['message_count']
            target = AssetDB(clone_root, 'validation', 'cache')
            directory = target.directory
            target.close()
            copied = copy_runtime_snapshot(original.db.directory, directory)
            self.assertEqual(len(copied), 3)
            clone = GraphAnalysisRuntime(clone_root, 'validation', 'cache', ChangeModel(),
                connection_identity='test', remote_factory=factory, policy=RuntimePolicy(require_remote_approval=True),intent_mode='contract_fixture')
            self.assertEqual(clone.inspect()['message_count'], initial_count)
            self.assertEqual(clone.inspect()['state'], 'awaiting_approval')
            result = clone.submit('새 요청으로 바꿔서 설명해줘')
            self.assertEqual(result['status'], 'answered', result)
            self.assertEqual(clone.ledger.get(request['id'])['status'], 'invalidated')
            self.assertEqual(original.ledger.get(request['id'])['status'], 'proposed')
            self.assertEqual(original.inspect()['message_count'], initial_count)
            self.assertEqual(calls, [])
            clone.close()
            original.close()

    def test_running_source_is_not_snapshotted(self):
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as clone_root:
            original = GraphAnalysisRuntime(source_root, 'owner', 'original', QuietModel(),intent_mode='contract_fixture')
            original.submit('자료 설명')
            with conversation_lock(original.db.directory / 'runtime.lock'):
                with self.assertRaises(BlockingIOError):
                    copy_runtime_snapshot(original.db.directory, Path(clone_root))
            self.assertFalse(list(Path(clone_root).glob('*.sqlite')))
            original.close()


if __name__ == '__main__': unittest.main()
