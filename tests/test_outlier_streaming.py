"""Outlier cohorts stream retained data without whole-frame reloads."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_outliers import select_outlier_rows


class OutlierStreamingTests(unittest.TestCase):
    def test_sparse_cohort_matches_independent_rows_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'owner', 'streamed-outliers')
            store = PersistentDatasets(db, budget=0, max_full_read_bytes=1024*1024)
            frame = pd.DataFrame({'reading': np.tile(np.arange(100, dtype=float), 300),
                                  'ordinal': np.arange(30000), 'label': ['payload']*30000})
            frame.loc[[100, 10000, 29000], 'reading'] = [-1000., 2000., 3000.]
            frame.loc[200, 'reading'] = np.nan
            parent = store.register_batches([frame], columns=list(frame), source='external.sensor',
                max_rows=len(frame), coverage='complete', snapshot='fixed:v1', predicate_known=True)
            db.select_dataset(parent.id)
            original = db.dataset_file(parent.id).read_bytes()
            q1, q3 = frame.reading.quantile([.25, .75]); delta = 1.5*(q3-q1)
            expected = frame.loc[(frame.reading < q1-delta) | (frame.reading > q3+delta)].reset_index(drop=True)
            with patch.object(FrameCache, '__getitem__', side_effect=AssertionError('whole-frame decode')):
                result = select_outlier_rows(store, parent.id, column='reading', method='iqr')
            self.assertEqual(result['status'], 'ready', result)
            child = result['dataset']['id']
            self.assertEqual(result['selection_summary']['execution_mode'], 'streamed_batches')
            self.assertEqual(result['selection_summary']['selected_rows'], 3)
            self.assertEqual(result['selection_summary']['data_sha256'], stored_dataset_digest(store, child))
            pd.testing.assert_frame_equal(store.frames.project(child, list(frame)), expected)
            self.assertEqual(db.dataset_file(parent.id).read_bytes(), original)
            self.assertEqual(db.selected_dataset_id(), parent.id)
            self.assertEqual(store.frames.bytes, 0)
            db.close()
            reopened = AssetDB(root, 'owner', 'streamed-outliers')
            try:
                restored = PersistentDatasets(reopened, budget=0)
                self.assertEqual(restored.metadata[child].parent_id, parent.id)
                self.assertEqual(restored.metadata[child].snapshot, 'fixed:v1')
                self.assertFalse(restored.metadata[child].predicate_known)
                pd.testing.assert_frame_equal(restored.frames.project(child, list(frame)), expected)
            finally:
                reopened.close()

    def test_mid_stream_failure_and_output_limit_do_not_publish_partial_cohorts(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'owner', 'stream-failure')
            try:
                store = PersistentDatasets(db, budget=0)
                frame = pd.DataFrame({'reading': np.arange(10000, dtype=float), 'payload': ['data']*10000})
                parent = store.register_batches([frame], columns=list(frame), source='external.other',
                    max_rows=len(frame), coverage='complete', predicate_known=True)
                db.select_dataset(parent.id)
                original = db.dataset_file(parent.id).read_bytes()
                real_batches = store.frames.batches
                def interrupted(*args, **kwargs):
                    batches = real_batches(*args, **kwargs)
                    try:
                        yield next(batches)
                        raise OSError('simulated disk read failure')
                    finally:
                        batches.close()
                with patch.object(store.frames, 'batches', side_effect=interrupted):
                    with self.assertRaises(OSError):
                        select_outlier_rows(store, parent.id, column='reading', method='iqr', selection='inliers')
                def truncated(*args, **kwargs):
                    batches = real_batches(*args, **kwargs)
                    try:
                        yield next(batches)
                    finally:
                        batches.close()
                with patch.object(store.frames, 'batches', side_effect=truncated):
                    with self.assertRaisesRegex(ValueError, '행 수'):
                        select_outlier_rows(store, parent.id, column='reading', method='iqr', selection='inliers')
                store.max_frame_bytes = 1024
                with self.assertRaises(MemoryError):
                    select_outlier_rows(store, parent.id, column='reading', method='iqr', selection='inliers')
                self.assertEqual(set(store.metadata), {parent.id})
                self.assertEqual(db.dataset_file(parent.id).read_bytes(), original)
                self.assertEqual(db.selected_dataset_id(), parent.id)
                self.assertFalse(list(db.directory.glob('.*.staging.parquet')))
                store.max_frame_bytes = None
                cohort = select_outlier_rows(store, parent.id, column='reading', method='iqr', selection='inliers')
                self.assertEqual(cohort['selection_summary']['selected_rows'], len(frame))
                self.assertEqual(cohort['selection_summary']['data_sha256'],
                                 stored_dataset_digest(store, cohort['dataset']['id']))
            finally:
                db.close()
