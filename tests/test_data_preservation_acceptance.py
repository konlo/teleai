"""Real storage/tool acceptance; no model or DB. Not full agent journey proof."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from core.analysis_agent.assets import AssetDB, PersistentDatasets, PersistentCharts
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from scripts.evaluate_analysis_agent import HistogramCapture
from utils.analysis_datasets import AnalysisNeed, Condition

FIXTURE = Path(__file__).parent / 'fixtures/data_preservation_v1.json'


def load_fixture(renamed=False):
    spec = json.loads(FIXTURE.read_text())
    frame = pd.DataFrame(spec['rows']).astype(spec['dtypes'])
    if renamed:
        mapping = spec['renamed']['columns']
        frame = frame.rename(columns=mapping)
        spec['source'] = spec['renamed']['source']
        spec['roles'] = {role: mapping[name] for role, name in spec['roles'].items()}
    return spec, frame


class DataPreservationAcceptanceTests(unittest.TestCase):
    def test_derive_transform_aggregate_chart_restart_keeps_root(self):
        for renamed in (False, True):
            with self.subTest(renamed=renamed), tempfile.TemporaryDirectory() as root:
                spec, expected = load_fixture(renamed)
                db = AssetDB(root, 'acceptance', 'journey')
                store = PersistentDatasets(db, budget=0)
                root_info = store.register(expected.copy(), source=spec['source'],
                    snapshot=spec['snapshot'], coverage='complete', predicate_known=True)
                self.assertEqual((root_info.role, root_info.root_id), ('root', root_info.id))
                measure, group = spec['roles']['measure'], spec['roles']['group']
                need = AnalysisNeed(spec['source'], tuple(expected.columns),
                    conditions=(Condition(group, 'eq', 'east'),))
                child = store.derive(root_info.id, need)
                self.assertEqual((child.role, child.root_id), ('derived', root_info.id))
                pd.testing.assert_frame_equal(store.frames[child.id],
                    expected.loc[expected[group].eq('east')])
                charts = PersistentCharts(db)
                context = AnalysisToolContext(store, charts, [],
                    lambda **_: self.fail('Remote proposal is forbidden in tool acceptance'))
                tools = {tool.name: tool.run for tool in build_analysis_tools(context)}
                # Mutation of a working frame must not modify the persisted root.
                working = store.frames[root_info.id]
                working[measure] = working[measure].abs()
                transformed = store.register(working, source=spec['source'],
                    parent_id=root_info.id, snapshot=spec['snapshot'],
                    coverage='complete', predicate_known=True)
                mean = tools['aggregate_dataset'](transformed.id, 'mean', value_column=measure)
                nonnull = [abs(row[spec['roles']['measure']]) for row in expected.to_dict('records')
                           if pd.notna(row[measure])]
                actual = store.frames[mean['dataset']['id']].iloc[0, 0]
                self.assertAlmostEqual(actual, sum(nonnull) / len(nonnull))
                with HistogramCapture() as capture:
                    chart = tools['render_chart_spec'](root_info.id, 'histogram', measure)
                values = expected[measure].dropna().tolist()
                observed = capture.histograms[-1]['distribution']
                self.assertEqual(dict(observed), {v: values.count(v) for v in set(values)})
                self.assertEqual(capture.histograms[-1]['rendered_total'], len(values))
                pd.testing.assert_frame_equal(store.frames[root_info.id], expected)
                root_metadata = store.metadata[root_info.id]
                chart_id = chart['cards'][0]['id']
                db.close()
                reopened = AssetDB(root, 'acceptance', 'journey')
                try:
                    restored = PersistentDatasets(reopened, budget=0)
                    pd.testing.assert_frame_equal(restored.frames[root_info.id], expected)
                    self.assertEqual(restored.metadata[root_info.id], root_metadata)
                    self.assertEqual(restored.metadata[child.id].parent_id, root_info.id)
                    self.assertEqual(restored.metadata[child.id].root_id, root_info.id)
                    self.assertTrue(PersistentCharts(reopened)[chart_id].image.startswith(b'\x89PNG'))
                finally:
                    reopened.close()

    def test_candidate_quota_failure_preserves_all_ready_assets(self):
        spec, frame = load_fixture()
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'acceptance', 'quota')
            try:
                store = PersistentDatasets(db, budget=0)
                info = store.register(frame.copy(), source=spec['source'],
                    coverage='complete', predicate_known=True)
                before = store.metadata
                # No space for a new asset; verify the actual persistence boundary.
                db.max_scope_bytes = sum(p.stat().st_size for p in db.directory.iterdir() if p.is_file())
                with self.assertRaises(MemoryError):
                    store.register(frame.copy(), source='fixture.candidate')
                self.assertEqual(store.metadata, before)
                pd.testing.assert_frame_equal(store.frames[info.id], frame)
            finally:
                db.close()

    def test_failed_transform_and_mutated_cache_read_do_not_destroy_root(self):
        spec, frame = load_fixture()
        with tempfile.TemporaryDirectory() as root:
            db = AssetDB(root, 'acceptance', 'bad-transform')
            try:
                store = PersistentDatasets(db)
                info = store.register(frame.copy(), source=spec['source'],
                    coverage='complete', predicate_known=True)
                for _ in range(2):  # cold read then cache hit
                    copy = store.frames[info.id]
                    copy.drop(columns=[spec['roles']['measure']], inplace=True)
                    pd.testing.assert_frame_equal(store.frames[info.id], frame)
                with self.assertRaises(ValueError):
                    store.derive(info.id, AnalysisNeed(spec['source'], ('missing_column',)))
                pd.testing.assert_frame_equal(store.frames[info.id], frame)
                self.assertEqual(set(store.metadata), {info.id})
            finally:
                db.close()


if __name__ == '__main__':
    unittest.main()
