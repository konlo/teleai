"""A follow-up can reuse only a provably sufficient local asset."""
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from core.analysis_agent.assets import AssetDB, PersistentDatasets
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import AnalysisNeed, Condition, DatasetStore, select_reusable_dataset


def catalog(store):
    propose = Mock()
    tools = {tool.name: tool.run for tool in build_analysis_tools(
        AnalysisToolContext(store, {}, [], propose))}
    return tools, propose


class DatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.frame = pd.DataFrame({'cohort': ['north', 'south', 'north'],
                                   'measure': [10, 20, 30]})
        self.root = self.store.register(self.frame.copy(), source='synthetic.events',
                                        snapshot='v1', coverage='complete', predicate_known=True)

    def test_filtered_child_widens_via_root_without_remote_or_root_mutation(self):
        child = self.store.derive(self.root.id, AnalysisNeed(
            self.root.source, ('cohort', 'measure'),
            conditions=(Condition('cohort', 'eq', 'north'),)))
        tools, propose = catalog(self.store)
        result = tools['local_analysis_sql'](child.id,
            'SELECT AVG(measure) AS average FROM data')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['selection_origin'], 'ancestor')
        self.assertEqual(result['selected_dataset_id'], self.root.id)
        self.assertEqual(result['preview'][0]['average'], 20)
        self.assertEqual(self.store.metadata[result['dataset']['id']].parent_id, self.root.id)
        pd.testing.assert_frame_equal(self.store.frames[self.root.id], self.frame)
        propose.assert_not_called()

    def test_aggregate_child_can_recover_raw_parent_but_explicit_result_stays_local(self):
        tools, propose = catalog(self.store)
        aggregate = tools['local_analysis_sql'](self.root.id,
            'SELECT AVG(measure) AS mean_measure FROM data')
        aggregate_id = aggregate['dataset']['id']
        raw = tools['use_dataset'](aggregate_id, ['measure'])
        self.assertEqual(raw['status'], 'ready')
        self.assertEqual(raw['dataset']['id'], self.root.id)
        self.assertEqual(raw['selection_origin'], 'ancestor')
        current = tools['use_dataset'](aggregate_id, ['mean_measure'], current_result_only=True)
        self.assertEqual(current['dataset']['id'], aggregate_id)
        self.assertEqual(current['selection_origin'], 'selected')
        sql = tools['local_analysis_sql'](aggregate_id,
            'SELECT SUM(measure) AS total FROM data')
        self.assertEqual(sql['preview'][0]['total'], 60)
        self.assertEqual(sql['selected_dataset_id'], self.root.id)
        ordered = tools['local_analysis_sql'](aggregate_id,
            'SELECT AVG(measure) AS average FROM data ORDER BY average')
        self.assertEqual(ordered['preview'][0]['average'], 20)
        self.assertEqual(ordered['selected_dataset_id'], self.root.id)
        propose.assert_not_called()

    def test_registry_fallback_requires_unique_matching_explicit_snapshot(self):
        tools, propose = catalog(self.store)
        subset = self.store.register(self.frame.iloc[:1].copy(),
            source=self.root.source, snapshot='v1', coverage='complete',
            predicate_known=True, conditions=(Condition('cohort', 'eq', 'north'),))
        selection = tools['use_dataset'](subset.id, ['measure'])
        self.assertEqual(selection['selected_dataset_id'], self.root.id)
        self.assertEqual(selection['selection_origin'], 'registry')
        second = self.store.register(self.frame.copy(), source=self.root.source,
            snapshot='v1', coverage='complete', predicate_known=True)
        ambiguous = tools['use_dataset'](subset.id, ['measure'])
        self.assertEqual(ambiguous['status'], 'needs_data')
        self.assertIn('여러 개', ambiguous['reason'])
        self.assertEqual(second.id in self.store.metadata, True)
        propose.assert_not_called()

    def test_mismatched_or_unknown_snapshot_never_selects_unrelated_root(self):
        for snapshot in ('v2', ''):
            with self.subTest(snapshot=snapshot):
                selected = self.store.register(self.frame.iloc[:1].copy(),
                    source=self.root.source, snapshot=snapshot,
                    coverage='complete', predicate_known=True,
                    conditions=(Condition('cohort', 'eq', 'north'),))
                tools, propose = catalog(self.store)
                self.assertEqual(tools['use_dataset'](selected.id, ['measure'])['status'],
                                 'needs_data')
                propose.assert_not_called()

    def test_incomplete_ancestor_and_explicit_current_result_do_not_widen(self):
        incomplete = self.store.register(self.frame.copy(), source='other.events',
            coverage='truncated', predicate_known=False)
        child = self.store.register(self.frame.iloc[:1].copy(), source='other.events',
            coverage='complete', predicate_known=True, parent_id=incomplete.id,
            conditions=(Condition('cohort', 'eq', 'north'),))
        tools, propose = catalog(self.store)
        self.assertEqual(tools['use_dataset'](child.id, ['measure'])['status'], 'needs_data')
        current = tools['local_analysis_sql'](child.id,
            'SELECT COUNT(*) AS n FROM data', current_result_only=True)
        self.assertEqual(current['status'], 'ready')
        self.assertEqual(current['selected_dataset_id'], child.id)
        self.assertEqual(current['preview'][0]['n'], 1)
        propose.assert_not_called()

    def test_metadata_only_selection_does_not_load_large_frames(self):
        child = self.store.derive(self.root.id, AnalysisNeed(self.root.source,
            ('measure',), conditions=(Condition('cohort', 'eq', 'north'),)))
        selection = select_reusable_dataset(self.store.metadata, child.id,
            AnalysisNeed(self.root.source, ('measure',)))
        self.assertEqual(selection.dataset_id, self.root.id)

    def test_persisted_lineage_is_reused_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            db = AssetDB(directory, 'owner', 'conversation')
            store = PersistentDatasets(db, budget=0)
            root = store.register(self.frame.copy(), source='synthetic.events',
                snapshot='v1', coverage='complete', predicate_known=True)
            child = store.derive(root.id, AnalysisNeed(root.source, ('cohort', 'measure'),
                conditions=(Condition('cohort', 'eq', 'north'),)))
            db.close()
            reopened = AssetDB(directory, 'owner', 'conversation')
            try:
                restored = PersistentDatasets(reopened, budget=0)
                tools, propose = catalog(restored)
                result = tools['local_analysis_sql'](child.id,
                    'SELECT SUM(measure) AS total FROM data')
                self.assertEqual(result['preview'][0]['total'], 60)
                self.assertEqual(result['selected_dataset_id'], root.id)
                pd.testing.assert_frame_equal(restored.frames[root.id], self.frame)
                propose.assert_not_called()
            finally:
                reopened.close()

    def test_histogram_never_guesses_between_independent_snapshots(self):
        second_frame = self.frame.assign(measure=[100, 200, 300])
        second = self.store.register(second_frame, source=self.root.source,
            snapshot='v2', coverage='complete', predicate_known=True)
        tools, propose = catalog(self.store)
        ambiguous = tools['prepare_histogram'](self.root.source, 'measure')
        self.assertEqual(ambiguous['status'], 'needs_context')
        self.assertEqual(set(ambiguous['candidate_dataset_ids']), {self.root.id, second.id})
        first_chart = tools['prepare_histogram'](self.root.source, 'measure', dataset_id=self.root.id)
        second_chart = tools['prepare_histogram'](self.root.source, 'measure', dataset_id=second.id)
        self.assertEqual(first_chart['status'], 'ready')
        self.assertEqual(second_chart['status'], 'ready')
        self.assertEqual(set(self.store.frames[first_chart['loaded_dataset']]['measure']),
                         set(self.frame['measure']))
        self.assertEqual(set(self.store.frames[second_chart['loaded_dataset']]['measure']),
                         set(second_frame['measure']))
        self.assertNotEqual(first_chart['cards'][0]['id'], second_chart['cards'][0]['id'])
        again = tools['prepare_histogram'](self.root.source, 'measure', dataset_id=self.root.id)
        self.assertEqual(again['cards'][0]['id'], first_chart['cards'][0]['id'])
        fresh = tools['prepare_histogram'](self.root.source, 'measure', fresh_source_required=True)
        self.assertEqual(fresh['status'], 'planned')
        propose.assert_not_called()

    def test_histogram_pin_controls_parent_recovery_and_current_subset(self):
        child = self.store.derive(self.root.id, AnalysisNeed(self.root.source,
            ('cohort', 'measure'), conditions=(Condition('cohort', 'eq', 'north'),)))
        tools, propose = catalog(self.store)
        whole = tools['prepare_histogram'](self.root.source, 'measure', dataset_id=child.id)
        self.assertEqual(whole['status'], 'ready')
        self.assertEqual(set(self.store.frames[whole['loaded_dataset']]['measure']),
                         set(self.frame['measure']))
        self.assertEqual(self.store.metadata[whole['loaded_dataset']].parent_id, self.root.id)
        subset = tools['prepare_histogram'](self.root.source, 'measure',
            dataset_id=child.id, current_result_only=True)
        self.assertEqual(subset['status'], 'ready')
        self.assertEqual(subset['loaded_dataset'], child.id)
        self.assertNotEqual(subset['cards'][0]['id'], whole['cards'][0]['id'])
        filtered = tools['prepare_histogram'](self.root.source, 'measure',
            where_sql='measure > 10', dataset_id=child.id, current_result_only=True)
        self.assertEqual(filtered['status'], 'no_valid_chart')
        self.assertEqual(self.store.frames[filtered['loaded_dataset']]['measure'].tolist(), [30])
        self.assertEqual(self.store.metadata[filtered['loaded_dataset']].parent_id, child.id)
        rendered = tools['prepare_histogram'](self.root.source, 'measure',
            where_sql='measure > 10', dataset_id=self.root.id, current_result_only=True)
        self.assertEqual(rendered['status'], 'ready')
        self.assertEqual(self.store.frames[rendered['loaded_dataset']]['measure'].tolist(), [20, 30])
        other = self.store.register(self.frame.copy(), source='other.events',
            coverage='complete', predicate_known=True)
        mismatch = tools['prepare_histogram'](self.root.source, 'measure', dataset_id=other.id)
        self.assertEqual(mismatch['status'], 'needs_context')
        propose.assert_not_called()


if __name__ == '__main__':
    unittest.main()
