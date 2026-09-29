"""Independent latest-row oracles and rejection boundaries for batched SQL."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import duckdb
from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel
from tests.test_latest_distribution import FIXTURE, KEY, VALUE, CLOCK, INGEST, source_frame
from utils.analysis_latest import latest_distribution
from utils.analysis_latest_sql import select_latest


class LatestSQLTests(unittest.TestCase):
    def check(self, frame, expected_error=None, order=None, keys=None):
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'sql-test', 'latest', QuietModel())
            try:
                info = r.datasets.register(frame, source=FIXTURE['source'],
                    coverage='complete', predicate_known=True)
                with patch.object(r.datasets.frames, 'project', side_effect=AssertionError('full projection')):
                    selected, error = select_latest(r.datasets, info, list(frame.columns), keys or [KEY], order or [CLOCK])
                self.assertEqual(len(r.datasets.metadata), 1)
                self.assertFalse(r.artifacts)
                self.assertFalse(r.datasets.frames.cache)
                if expected_error:
                    self.assertIsNone(selected)
                    self.assertEqual(error['error_code'], expected_error)
                else:
                    self.assertIsNone(error)
                return selected
            finally:
                r.close()

    def test_exact_latest_and_tie_policy(self):
        frame = source_frame()
        for seed in (4, 11):
            selected = self.check(frame.sample(frac=1, random_state=seed))
            self.assertEqual(dict(zip(selected[KEY], selected[VALUE])), FIXTURE['expected_latest'])
        winner = frame.sort_values(CLOCK).drop_duplicates(KEY, keep='last').iloc[[0]].copy()
        duplicate = pd.concat([frame, winner], ignore_index=True)
        self.check(duplicate, 'latest_order_tie')
        winner[INGEST] += pd.Timedelta(days=1)
        winner[VALUE] = 'replacement'
        selected = self.check(pd.concat([frame, winner]), order=[CLOCK, INGEST])
        self.assertIn('replacement', selected[VALUE].tolist())
        selected = self.check(frame, keys=[KEY, VALUE])
        expected = frame.sort_values(CLOCK).drop_duplicates([KEY, VALUE], keep='last')
        self.assertEqual(len(selected), len(expected))

    def test_null_type_output_and_metadata_limits_publish_nothing(self):
        for column in (KEY, CLOCK, VALUE):
            frame = source_frame()
            frame.loc[0, column] = None
            self.check(frame, 'latest_null_policy')
        frame = source_frame()
        frame[CLOCK] = frame[CLOCK].astype(str)
        self.check(frame, 'latest_order_type')
        with patch('utils.analysis_latest_sql.MAX_SELECTED_ROWS', 4):
            self.check(source_frame(), 'latest_output_limit')
        with patch('utils.analysis_latest_sql.MAX_SELECTED_BYTES', 1):
            self.check(source_frame(), 'latest_output_limit')
        for exception in (duckdb.OutOfMemoryException, duckdb.InterruptException):
            with patch('utils.analysis_latest_sql.duckdb.connect', side_effect=exception('injected')):
                self.check(source_frame(), 'latest_resource_limit')

    def test_renamed_schema_and_quoted_identifiers(self):
        mapping = {KEY: 'entity "key', CLOCK: 'event timestamp', VALUE: 'group', INGEST: 'row_sequence'}
        selected = self.check(source_frame().rename(columns=mapping),
            keys=[mapping[KEY]], order=[mapping[CLOCK]])
        self.assertEqual(dict(zip(selected[mapping[KEY]], selected[mapping[VALUE]])), FIXTURE['expected_latest'])

    def test_large_tool_uses_sql_and_survives_restart(self):
        # Older duplicates are harmless, but make the retained input exceed the SQL threshold.
        base = source_frame()
        old = base.iloc[[0]].copy()
        old[CLOCK] -= pd.Timedelta(days=100)
        frame = pd.concat([base, pd.concat([old] * 20_000)], ignore_index=True)
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'sql-test', 'large', QuietModel())
            info = r.datasets.register(frame, source=FIXTURE['source'], coverage='complete', predicate_known=True)
            r.select_dataset(info.id)
            try:
                original_project = r.datasets.frames.project
                def project(key, columns):
                    if key == info.id:
                        raise AssertionError('raw full projection')
                    return original_project(key, columns)
                with patch.object(r.datasets.frames, 'project', side_effect=project):
                    result = r.submit(FIXTURE['cases'][1]['prompt'])
                self.assertEqual(result['status'], 'answered', result)
                proof = r.inspect()['recovery']['latest_selection_evidence']
                self.assertEqual(proof['execution_mode'], 'bounded_local_sql')
                self.assertEqual(proof['selected_keys'], len(FIXTURE['expected_latest']))
                self.assertNotIn(info.id, r.datasets.frames.cache)
                selected_id = proof['dataset']['id']
            finally:
                r.close()
            r = GraphAnalysisRuntime(root, 'sql-test', 'large', QuietModel())
            try:
                selected = r.datasets.frames[selected_id]
                self.assertEqual(dict(zip(selected[KEY], selected[VALUE])), FIXTURE['expected_latest'])
                self.assertEqual(r.datasets.metadata[info.id].rows, len(frame))
                with self.assertRaises(ValueError):
                    select_latest(r.datasets, replace(r.datasets.metadata[info.id], rows=1),
                        [KEY, CLOCK, VALUE], [KEY], [CLOCK])
            finally:
                r.close()


if __name__ == '__main__':
    unittest.main()
