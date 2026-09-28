"""Population, lineage and histogram reuse regression journeys; no remote I/O."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import pandas as pd
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from core.analysis_sql import validate_query
from utils.analysis_charts import histogram_from_counts
from utils.analysis_datasets import DatasetStore, Condition
from utils.analysis_provenance import query_coverage, raw_conditions

FIXTURE = json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
SOURCE = FIXTURE['source']
COLUMN = next(k for k, v in FIXTURE['rows'][0].items() if isinstance(v, (int, float)))


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.store = DatasetStore()
        self.context = AnalysisToolContext(self.store, {}, [], Mock())
        self.tools = {tool.name: tool.run for tool in build_analysis_tools(self.context)}
        self.frame = pd.DataFrame(FIXTURE['rows'])

    def raw(self, frame=None, **kwargs):
        return self.store.register(self.frame if frame is None else frame, source=SOURCE,
            coverage='complete', predicate_known=True, **kwargs)

    def test_sampling_offset_and_limit_are_not_complete(self):
        for suffix, coverage in [('TABLESAMPLE (10 PERCENT)', 'sampled'),
                                  ('LIMIT 100', 'unknown'), ('OFFSET 2', 'unknown')]:
            with self.subTest(suffix=suffix):
                tree = validate_query(f'SELECT * FROM {SOURCE} {suffix}')
                self.assertEqual(query_coverage(tree), coverage)
                self.assertIsNone(raw_conditions(tree))

    def test_filtered_cache_cannot_answer_unfiltered_population(self):
        condition = Condition(COLUMN, 'ge', 40)
        info = self.raw(self.frame[self.frame[COLUMN] >= 40], conditions=(condition,))
        outcome = self.tools['local_analysis_sql'](info.id, f'SELECT AVG({COLUMN}) FROM data')
        self.assertEqual(outcome['status'], 'needs_data')
        self.assertEqual(len(self.store.metadata), 1)

    def test_safe_filter_keeps_lineage_and_followup_can_narrow(self):
        info = self.raw()
        filtered = self.tools['local_analysis_sql'](info.id, f'SELECT * FROM data WHERE {COLUMN} >= 10')
        derived = self.store.metadata[filtered['dataset']['id']]
        self.assertTrue(derived.predicate_known)
        self.assertEqual(derived.conditions, (Condition(COLUMN, 'ge', 10),))
        requested = [asdict(Condition(COLUMN, 'ge', 20))]
        result = self.tools['local_analysis_sql'](derived.id, f'SELECT AVG({COLUMN}) AS result FROM data',
            requested_conditions=requested)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['preview'][0]['result'], self.frame.loc[self.frame[COLUMN] >= 20, COLUMN].mean())
        widened = self.tools['local_analysis_sql'](derived.id, f'SELECT AVG({COLUMN}) FROM data')
        self.assertEqual(widened['status'], 'ready')
        self.assertEqual(widened['selected_dataset_id'], info.id)
        self.assertEqual(widened['selection_origin'], 'ancestor')
        self.assertEqual(next(iter(widened['preview'][0].values())), self.frame[COLUMN].mean())

    def test_unknown_or_scope_is_not_reinterpreted_as_and(self):
        info = self.raw(self.frame[self.frame[COLUMN] >= 10], conditions=(Condition(COLUMN, 'ge', 10),))
        result = self.tools['local_analysis_sql'](info.id, f'SELECT * FROM data WHERE {COLUMN} < 5 OR {COLUMN} > 20')
        self.assertEqual(result['status'], 'needs_data')

    def test_projection_expression_does_not_claim_raw_lineage(self):
        info = self.raw()
        result = self.tools['local_analysis_sql'](info.id, f'SELECT {COLUMN} * 2 AS {COLUMN} FROM data')
        self.assertFalse(result['dataset']['predicate_known'])

    def test_missing_from_is_repaired_only_for_known_bound_columns(self):
        info = self.raw()
        result = self.tools['local_analysis_sql'](info.id, f'SELECT AVG({COLUMN}) AS average')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['preview'][0]['average'], self.frame[COLUMN].mean())
        self.assertEqual(result['applied_corrections'], ['added_local_data_from'])
        self.assertIn('FROM data', result['dataset']['query'])

    def test_count_star_without_from_counts_the_bound_dataframe(self):
        info = self.raw()
        for query in ('COUNT(*)', 'SELECT COUNT(*)', f'COUNT({COLUMN})'):
            with self.subTest(query=query):
                result = self.tools['local_analysis_sql'](info.id, query)
                self.assertEqual(result['status'], 'ready')
                self.assertEqual(next(iter(result['preview'][0].values())), len(self.frame))
                self.assertIn('FROM data', result['dataset']['query'])
                self.assertTrue(result['applied_corrections'])

    def test_exact_selected_source_name_is_safely_bound_to_local_data(self):
        info = self.raw()
        short_name = SOURCE.rsplit('.', 1)[-1]
        for query in (f'SELECT COUNT(*) AS n FROM {SOURCE}',
                      f'SELECT {short_name}.{COLUMN}, COUNT(*) AS n FROM {short_name} GROUP BY {short_name}.{COLUMN}'):
            with self.subTest(query=query):
                result = self.tools['local_analysis_sql'](info.id, query)
                self.assertEqual(result['status'], 'ready')
                self.assertIn('bound_source_table_to_local_data', result['applied_corrections'])
                self.assertIn('FROM data', result['dataset']['query'])

    def test_unrelated_source_and_join_are_never_rebound(self):
        info = self.raw()
        for query in ('SELECT COUNT(*) FROM another_table',
                      f'SELECT COUNT(*) FROM {SOURCE} JOIN data ON TRUE'):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.tools['local_analysis_sql'](info.id, query)

    def test_raw_numbers_and_sum_are_not_frequency(self):
        frame = self.frame[[COLUMN]].drop_duplicates().assign(weight=100)
        for query, grain, aggregation in [('', 'raw', ''),
                (f'SELECT {COLUMN}, SUM({COLUMN}) AS weight FROM {SOURCE} GROUP BY {COLUMN}', 'aggregate', 'sum')]:
            info = self.store.register(frame, source=SOURCE, coverage='complete',
                query=query, grain=grain, aggregation=aggregation)
            with self.assertRaises(ValueError): histogram_from_counts(self.store, info.id, COLUMN, 'weight')

    def test_raw_histogram_is_computed_once_then_same_png_reused(self):
        self.raw()
        first = self.tools['prepare_histogram'](SOURCE, COLUMN)
        self.assertEqual(first['status'], 'ready')
        count = len(self.store.metadata)
        second = self.tools['prepare_histogram'](SOURCE, COLUMN)
        self.assertEqual(second['loaded_dataset'], first['loaded_dataset'])
        self.assertEqual(second['cards'][0]['id'], first['cards'][0]['id'])
        self.assertEqual(len(self.store.metadata), count)
        self.context.propose_query.assert_not_called()

    def test_changed_filter_cannot_reuse_old_population_chart(self):
        self.raw()
        first = self.tools['prepare_histogram'](SOURCE, COLUMN, f'{COLUMN} >= 20')
        second = self.tools['prepare_histogram'](SOURCE, COLUMN, f'{COLUMN} >= 40')
        self.assertEqual(first['status'], 'ready')
        self.assertEqual(second['status'], 'ready')
        self.assertNotEqual(first['loaded_dataset'], second['loaded_dataset'])
        data = self.store.frames[second['loaded_dataset']]
        self.assertTrue((data[COLUMN] >= 40).all())

    def test_subset_only_histogram_proposes_loading_without_execution(self):
        self.raw(self.frame[self.frame[COLUMN] >= 20], conditions=(Condition(COLUMN, 'ge', 20),))
        result = self.tools['prepare_histogram'](SOURCE, COLUMN)
        self.assertEqual(result['status'], 'planned')
        self.assertFalse(self.context.artifacts)
        self.context.propose_query.assert_not_called()

    def test_stored_remote_frequency_is_reused_without_raw_rows(self):
        self.context.reference_context = [{'table': SOURCE, 'columns': [{'name': COLUMN}]}]
        plan = self.tools['prepare_histogram'](SOURCE, COLUMN)['histogram_plan']
        counts = self.frame.groupby(COLUMN).size().reset_index(name='__frequency')
        info = self.store.register(counts, source=SOURCE, coverage='complete', grain='aggregate',
            query=plan['query'], aggregation=plan['query'])
        result = self.tools['prepare_histogram'](SOURCE, COLUMN)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['loaded_dataset'], info.id)
        self.assertIn(str(len(self.frame)), result['cards'][0]['scope'])

    def test_unseen_string_condition_still_returns_legitimate_zero(self):
        category = next(k for k, v in FIXTURE['rows'][0].items() if isinstance(v, str))
        info = self.raw()
        result = self.tools['local_analysis_sql'](info.id, 'SELECT COUNT(*) AS n FROM data',
            requested_conditions=[{'column':category, 'op':'eq', 'value':'not-present-in-fixture'}])
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['preview'][0]['n'], 0)


if __name__ == '__main__': unittest.main()
