import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_agent.assets import AssetDB, PersistentDatasets
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.storage_policy import storage_report
from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel


class OperationalPolicyTests(unittest.TestCase):
    def test_environment_policy_is_table_neutral_and_validated(self):
        with patch.dict('os.environ', {'TELLY_MAX_REMOTE_ROWS':'321',
                                       'TELLY_MAX_DATASET_COLUMNS':'7',
                                       'TELLY_RETENTION_DAYS':'5'}):
            policy=RuntimePolicy.from_env()
        self.assertEqual(policy.max_remote_rows,321)
        self.assertEqual(policy.max_dataset_columns,7)
        self.assertEqual(policy.retention_days,5)
        self.assertNotIn('table', json.dumps(policy.public()).lower())
        with patch.dict('os.environ', {'TELLY_MAX_REMOTE_ROWS':'0'}):
            with self.assertRaises(ValueError):RuntimePolicy.from_env()

    def test_column_and_memory_limits_fail_before_persisting(self):
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','conversation')
            columns=PersistentDatasets(db,max_columns=1,max_frame_bytes=1024)
            with self.assertRaises(ValueError):
                columns.register(pd.DataFrame({'a':[1],'b':[2]}),source='dynamic.table')
            memory=PersistentDatasets(db,max_columns=10,max_frame_bytes=1)
            with self.assertRaises(MemoryError):
                memory.register(pd.DataFrame({'arbitrary':['value']}),source='another.table')
            self.assertEqual(db.metadata('dataset'),{})
            db.close()

    def test_scope_quota_rejects_new_payload_and_keeps_existing_asset(self):
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','conversation',max_scope_bytes=100_000)
            db.put('first','chart',{'ok':True},b'png')
            with self.assertRaises(MemoryError):
                db.put('too-large','chart',{},b'x'*100_000)
            self.assertEqual(db.get('first','chart')[1],b'png')
            self.assertNotIn('too-large',db.metadata('chart'))
            db.close()

    def test_runtime_uses_deployment_turn_budget_between_model_calls(self):
        with tempfile.TemporaryDirectory() as root:
            policy = RuntimePolicy(turn_slo_seconds=17.5)
            runtime = GraphAnalysisRuntime(root, 'owner', 'budget-policy', QuietModel(), policy=policy)
            self.assertEqual(runtime.recovery.max_model_seconds, 17.5)
            runtime.close()

    def test_retention_report_is_read_only_and_marks_old_scope(self):
        import os,time
        with tempfile.TemporaryDirectory() as root:
            scope=Path(root)/('a'*64);scope.mkdir()
            payload=scope/'assets.sqlite';payload.write_bytes(b'kept')
            old=time.time()-40*86400
            os.utime(payload,(old,old));os.utime(scope,(old,old))
            report=storage_report(root,retention_days=30,scope_quota_bytes=3)
            self.assertEqual(report['expired_candidates'],1)
            self.assertEqual(report['over_quota_scopes'],1)
            self.assertFalse(report['destructive_action_performed'])
            self.assertEqual(payload.read_bytes(),b'kept')


if __name__=='__main__':unittest.main()
