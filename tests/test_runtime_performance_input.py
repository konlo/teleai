"""Capacity sampling must work with current Parquet assets and arbitrary schemas."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.analysis_agent.assets import AssetDB, PersistentDatasets
from scripts.report_runtime_performance import (
    asset_fingerprint,
    benchmark_columns,
    local_benchmark,
    local_correlation_benchmark,
    read_dataset,
)


class RuntimePerformanceInputTests(unittest.TestCase):
    def test_file_backed_asset_is_bounded_and_preferred_to_legacy(self):
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, "owner", "capacity")
            try:
                store = PersistentDatasets(db)
                old = store.register(pd.DataFrame({"old_metric": [1, 2]}),
                    source="fixture.legacy", coverage="complete", predicate_known=True)
                source = pd.DataFrame({"측정값": range(50), "비교값": range(50, 100)})
                current = store.register_batches([source], columns=list(source),
                    source="fixture.renamed", max_rows=100,
                    coverage="complete", predicate_known=True)
                path = db.directory / "assets.sqlite"
                metadata, sample = read_dataset(path, max_rows=7)
                self.assertEqual(metadata["benchmark_asset_id"], current.id)
                self.assertEqual(metadata["persisted_rows"], 50)
                self.assertEqual(metadata["benchmark_sample_rows"], 7)
                self.assertEqual(sample["측정값"].tolist(), list(range(7)))
                self.assertEqual(benchmark_columns(sample), ["측정값", "비교값"])
                old_metadata, old_sample = read_dataset(path, dataset_id=old.id, max_rows=1)
                self.assertEqual(old_metadata["benchmark_asset_id"], old.id)
                self.assertEqual(old_sample["old_metric"].tolist(), [1])
                with self.assertRaises(ValueError):
                    read_dataset(path, dataset_id=old.id, max_blob_bytes=1)
                file_before = asset_fingerprint(path, current.id)
                blob_before = asset_fingerprint(path, old.id)
                self.assertEqual(file_before, asset_fingerprint(path, current.id))
                self.assertEqual(blob_before, asset_fingerprint(path, old.id))
                with (db.directory / f"{current.id}.parquet").open("ab") as stream:
                    stream.write(b"changed")
                self.assertNotEqual(file_before, asset_fingerprint(path, current.id))
            finally:
                db.close()

    def test_benchmark_uses_observed_numeric_columns_without_table_names(self):
        frame = pd.DataFrame({"metric_x": [1, 2, 3, 4, 5, 6],
                              "metric_y": [2, 3, 4, 5, 6, 7],
                              "segment": ["a", "b", "c", "a", "b", "c"]})
        metadata = {"source": "fixture.variable_schema", "benchmark_source": "benchmark.sample",
                    "persisted_rows": 6}
        charts = local_benchmark(frame, metadata, 5)
        self.assertTrue(charts["actual_png_each_run"])
        self.assertEqual(charts["model_calls"], 0)
        self.assertIn("metric_x", charts["prompt"])
        correlation = local_correlation_benchmark(frame, metadata, 2)
        self.assertEqual(correlation["model_calls"], 0)
        self.assertEqual(correlation["expected"], 1.0)
        self.assertIn("metric_y", correlation["prompt"])


if __name__ == "__main__":
    unittest.main()
