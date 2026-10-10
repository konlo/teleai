"""Windows writable-handle contract and real storage failures, no OS spoofing."""
import errno
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from core.analysis_agent.assets import AssetDB, PersistentDatasets
from core.analysis_agent.approvals import ApprovalLedger
from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.support_report import summarize, brief
from core.analysis_databricks import execute_approved


@pytest.mark.parametrize('empty', [False, True])
def test_parquet_sync_uses_writable_handle_and_keeps_data(tmp_path, empty):
    db = AssetDB(tmp_path, 'owner', 'sync')
    store = PersistentDatasets(db)
    frame = pd.DataFrame({'metric': pd.Series([] if empty else [1,2], dtype='int64')})
    handles = {}
    original_open = Path.open
    real_fsync = __import__('os').fsync
    checks = []
    def tracked_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path.name.endswith('.staging.parquet'):
            handles[handle.fileno()] = handle
        return handle
    def windows_sync(fd):
        # Windows FlushFileBuffers requires GENERIC_WRITE. POSIX accepts rb.
        handle = handles[fd]
        checks.append(handle.writable())
        if not handle.writable():
            raise OSError(errno.EBADF, 'read-only flush handle')
        real_fsync(fd)
    try:
        with patch.object(Path, 'open', tracked_open), patch('core.analysis_agent.assets.os.fsync', windows_sync):
            info = store.register_batches([frame], columns=list(frame), source='fixture.generic', max_rows=10)
        assert checks == [True]
        pd.testing.assert_frame_equal(store.frames[info.id], frame)
        assert not list(db.directory.glob('*.staging.parquet'))
    finally:
        db.close()


def test_disk_sync_failure_preserves_original_and_does_not_requery(tmp_path):
    db = AssetDB(tmp_path, 'owner', 'failed-sync')
    store = PersistentDatasets(db)
    original = store.register(pd.DataFrame({'metric':[9]}), source='fixture.original')
    db.select_dataset(original.id)
    ledger = ApprovalLedger(tmp_path/'requests.sqlite')
    envelope = ledger.envelope('fixture.new', 'SELECT * FROM fixture.new LIMIT 10', 'fixture', 'connection')
    ledger.propose('call', envelope)
    ledger.authorize_automatic('call', envelope)
    config = SimpleNamespace(server_hostname='example.invalid', http_path='/sql/fixture',
                             access_token='secret', catalog='fixture', schema='lab')
    executions = []
    class Cursor:
        description = [('metric',)]
        rows = [(1,), (2,)]
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, query): executions.append(query)
        def fetchmany(self, size):
            rows, self.rows = self.rows, []
            return rows
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return Cursor()
    def executor(value):
        return execute_approved(SimpleNamespace(status='executing', **value), config, store,
                                connect=lambda **kwargs: Connection())
    diagnostics = Diagnostics(db.directory)
    diagnostics.run_id = 'c'*32
    diagnostics.emit('run_started')
    try:
        with patch('core.analysis_agent.assets.os.fsync', side_effect=OSError(errno.ENOSPC, 'private path token')):
            with pytest.raises(OSError) as caught:
                ledger.execute('call', envelope, executor)
        error_id = diagnostics.failure(caught.value, stage='query_databricks')
        diagnostics.emit('remote_query_finished', ledger_status=ledger.get('call')['status'])
        assert set(store.metadata) == {original.id}
        assert db.selected_dataset_id() == original.id
        assert not list(db.directory.glob('*.parquet'))
        assert ledger.uncertain() == [{'id':'call','status':'unknown'}]
        with pytest.raises(PermissionError):
            ledger.execute('call', envelope, executor)
        assert len(executions) == 1
        report = summarize(diagnostics.path, error_id=error_id)
        assert report['errors'][-1]['os_errno'] == errno.ENOSPC
        assert report['errors'][-1]['database_errno'] is None
        assert 'OS errno=28' in brief(report)
        assert 'private path token' not in json.dumps(report)
        assert 'private path token' not in diagnostics.path.read_text()
    finally:
        db.close()


def test_windows_error_numbers_are_safe_and_distinct_from_database_errno(tmp_path):
    diagnostics = Diagnostics(tmp_path)
    error = OSError(errno.EBADF, 'private filename and secret')
    error.winerror = 5
    error_id = diagnostics.failure(error, stage='query_databricks')
    report = summarize(diagnostics.path, error_id=error_id)
    last = report['errors'][-1]
    assert last['os_errno'] == errno.EBADF and last['winerror'] == 5
    assert last['database_errno'] is None
    assert 'WinError=5' in brief(report)
    assert 'private' not in diagnostics.path.read_text()
