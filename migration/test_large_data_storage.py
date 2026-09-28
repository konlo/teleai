"""Storage ownership and lookup contracts, independent of model/remote services."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_agent.assets import AssetDB, PersistentDatasets, PersistentCharts
from scripts.check_large_data_storage import synthetic_frame
from utils.analysis_charts import ChartPreview


class StorageMemoryContractTests(unittest.TestCase):
    def fixture(self):
        return json.loads(Path('tests/fixtures/large_data_workload.json').read_text())

    def test_uncached_decode_is_returned_without_an_extra_copy(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'owner', 'conversation')
            store = PersistentDatasets(db, budget=0)
            frame = synthetic_frame(self.fixture(), 10)
            info = store.register(frame, source=self.fixture()['source'])
            with patch('core.analysis_agent.assets.pd.read_parquet', return_value=frame):
                # Identity verifies that an oversized read does not duplicate
                # its full allocation solely to return an uncached input.
                self.assertIs(store.frames[info.id], frame)
            first = store.frames[info.id]
            first.iloc[0, 0] = -999
            self.assertEqual(store.frames[info.id].iloc[0, 0], 0)
            self.assertEqual(store.frames.bytes, 0)
            db.close()

    def test_cached_decode_and_hits_are_isolated_from_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'owner', 'conversation')
            frame = synthetic_frame(self.fixture(), 10)
            store = PersistentDatasets(db, budget=0)
            first = store.register(frame, source=self.fixture()['source'])
            second = store.register(frame.copy(), source=self.fixture()['source'])
            restored_size = int(store.frames[first.id].memory_usage(index=True, deep=True).sum())
            store = PersistentDatasets(db, budget=restored_size)
            decoded = store.frames[first.id]
            self.assertIn(first.id, store.frames.cache)
            decoded.iloc[0, 0] = -999
            self.assertEqual(store.frames[first.id].iloc[0, 0], 0)
            store.frames[second.id]
            self.assertIn(second.id, store.frames.cache)
            self.assertNotIn(first.id, store.frames.cache)
            self.assertEqual(store.frames[first.id].iloc[0, 0], 0)
            self.assertLessEqual(store.frames.bytes, store.frames.budget)
            db.close()

    def test_chart_reference_check_never_loads_dataset_payload(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'owner', 'conversation')
            info = PersistentDatasets(db).register(synthetic_frame(self.fixture(), 10),
                                                  source=self.fixture()['source'])
            card = ChartPreview('chart', info.id, 'Synthetic', 'Storage test', 'line', (), 'test', b'png')
            with patch.object(db, 'get', side_effect=AssertionError('Payload lookup during reference check')):
                PersistentCharts(db)[card.id] = card
            self.assertEqual(PersistentCharts(db)[card.id], card)
            missing = ChartPreview('missing-chart', 'missing-dataset', 'Synthetic', 'Storage test', 'line', (), 'test', b'png')
            with self.assertRaises(KeyError):
                PersistentCharts(db)[missing.id] = missing
            self.assertNotIn('missing-chart', db.metadata('chart'))
            db.close()


if __name__ == '__main__': unittest.main()
