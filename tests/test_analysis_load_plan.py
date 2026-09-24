"""Approved source/SQL binding and atomic candidate rejection, no network."""
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from core.analysis_agent.approvals import ApprovalLedger
from core.analysis_databricks import execute_approved
from core.analysis_load_plan import source_plan
from utils.analysis_datasets import DatasetStore


class LoadPlanTests(unittest.TestCase):
    def test_source_binding_accepts_qualified_single_source_and_declared_join(self):
        single = source_plan('events', 'SELECT amount FROM project.space.events')
        self.assertEqual(single.actual_tables, ('project.space.events',))
        self.assertEqual(single.grain, 'raw')
        self.assertEqual(single.expected_columns, ('amount',))
        joined = source_plan('project.space.a | project.space.b',
            'SELECT COUNT(*) AS n FROM project.space.a JOIN project.space.b ON a.id = b.id')
        self.assertEqual(joined.grain, 'aggregate')
        self.assertEqual(set(joined.actual_tables), {'project.space.a', 'project.space.b'})
        cte = source_plan('project.space.events',
            'WITH chosen AS (SELECT amount FROM project.space.events) SELECT SUM(amount) FROM chosen')
        self.assertEqual(cte.actual_tables, ('project.space.events',))

    def test_misleading_source_is_rejected_before_approval_record(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = ApprovalLedger(Path(root) / 'requests.sqlite')
            for source, query in (
                ('project.space.expected', 'SELECT * FROM project.space.other'),
                ('project.space.a', 'SELECT * FROM project.space.a JOIN project.space.b ON a.id=b.id'),
                ('project.space.a | project.space.b', 'SELECT * FROM project.space.a'),
            ):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    ledger.envelope(source, query, 'analysis', 'connection')
            with ledger.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM requests').fetchone()[0], 0)

    def test_invalid_result_candidate_never_replaces_stored_root(self):
        store = DatasetStore()
        original = pd.DataFrame({'amount': [3, 7]})
        root = store.register(original.copy(), source='project.space.events',
            coverage='complete', predicate_known=True)
        config = MagicMock()
        request = SimpleNamespace(status='executing', source='project.space.events',
            query='SELECT amount FROM project.space.events')
        for description, rows in (
            ([('amount',), ('amount',)], [(1, 2)]),
            ([('amount',)], [(1, 2)]),
            ([('wrong',)], [(1,)]),
            (None, [(1,)]),
        ):
            with self.subTest(description=description):
                connect = MagicMock()
                cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
                cursor.description = description
                cursor.fetchmany.side_effect = [rows, []]
                with self.assertRaises(ValueError):
                    execute_approved(request, config, store, connect=connect)
                self.assertEqual(set(store.metadata), {root.id})
                pd.testing.assert_frame_equal(store.frames[root.id], original)

    def test_direct_execution_rechecks_source_before_connection(self):
        store = DatasetStore()
        connect = MagicMock()
        request = SimpleNamespace(status='executing', source='project.space.expected',
            query='SELECT amount FROM project.space.other')
        with self.assertRaises(ValueError):
            execute_approved(request, MagicMock(), store, connect=connect)
        connect.assert_not_called()
        self.assertFalse(store.metadata)

    def test_short_fetch_batches_do_not_claim_complete_data_early(self):
        store = DatasetStore()
        connect = MagicMock()
        cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.description = [('amount',)]
        cursor.fetchmany.side_effect = [[(1,)], [(2,)], [(3,)], []]
        request = SimpleNamespace(status='executing', source='project.space.events',
            query='SELECT amount FROM project.space.events')
        result = execute_approved(request, MagicMock(), store, max_rows=2, connect=connect)
        self.assertEqual(result['dataset']['coverage'], 'truncated')
        self.assertEqual(result['dataset']['rows'], 2)
        self.assertEqual(store.frames[result['dataset']['id']]['amount'].tolist(), [1, 2])

    def test_large_batch_or_wide_schema_is_rejected_before_publication(self):
        for column_limit, byte_limit, columns, batches, expected_error in (
            (1, None, [('a',), ('b',)], [], ValueError),
            (None, 1024, [('payload',)], [[('x' * 2048,)]], MemoryError),
        ):
            with self.subTest(columns=columns, error=expected_error):
                store = DatasetStore()
                store.max_columns = column_limit
                store.max_frame_bytes = byte_limit
                connect = MagicMock()
                cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
                cursor.description = columns
                cursor.fetchmany.side_effect = [*batches, []]
                request = SimpleNamespace(status='executing', source='project.space.events',
                    query='SELECT * FROM project.space.events')
                with self.assertRaises(expected_error):
                    execute_approved(request, MagicMock(), store, connect=connect)
                self.assertFalse(store.metadata)


if __name__ == '__main__':
    unittest.main()
