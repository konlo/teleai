"""Result schemas must never acquire authority over an unrelated base schema."""
from datetime import datetime, timezone
from types import SimpleNamespace
import tempfile
import unittest

import pandas as pd

from core.analysis_catalog import _is_full_schema_observation, resolve_table_context
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_agent_sql_recovery_boundaries import CapturingModel


class SchemaObservationAuthorityTests(unittest.TestCase):
    def test_table_address_is_not_a_measure_but_separate_column_and_literal_are(self):
        from langchain_core.messages import HumanMessage
        from core.analysis_agent.source_mentions import mask_source_mentions
        source = 'metrics.flag.events'
        saved = {'table': source, 'observed_at': datetime.now(timezone.utc).isoformat(),
            'columns': [{'name': 'measurement', 'dtype': 'DOUBLE'},
                        {'name': 'flag', 'dtype': 'INTEGER'}, {'name': 'label', 'dtype': 'STRING'}]}
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'test', 'source-mentions', CapturingModel(),
                reference_context_loader=lambda: [saved])
            try:
                for token in [source, '`metrics`.`flag`.`events`', '`metrics.flag.events`']:
                    text = token + '에서 measurement 평균을 구해줘'
                    state,_ = r.recovery._state({'messages': [HumanMessage(content=text,id=str(token))]})
                    self.assertEqual(state['required_columns'], ['measurement'])
                    self.assertNotIn('flag',state['scope']['columns'])
                text = source + "에서 flag = 1이고 label = 'metrics.flag.events'인 measurement 평균"
                state,_ = r.recovery._state({'messages': [HumanMessage(content=text,id='predicate')]})
                self.assertIn('flag',state['required_columns'])
                self.assertIn({'column':'flag','op':'eq','value':1},state['scope']['conditions'])
                self.assertIn({'column':'label','op':'eq','value':source},state['scope']['conditions'])
                self.assertEqual(len(mask_source_mentions(text,[source])),len(text))
                self.assertIn("'metrics.flag.events'",mask_source_mentions(text,[source]))
                self.assertEqual(mask_source_mentions('flag 평균',[source]),'flag 평균')
            finally:
                r.close()

    def test_only_unmodified_single_source_star_is_schema_authority(self):
        accepted = ['SELECT * FROM events', 'SELECT e.* FROM events e LIMIT 0',
                    'SELECT * FROM events WHERE reading > 0 LIMIT 5']
        rejected = ['SELECT COUNT(*) AS n FROM events',
                    'SELECT COUNT(*) AS n, AVG(reading) AS m FROM events',
                    'SELECT reading FROM events',
                    'SELECT * EXCEPT(reading) FROM events',
                    'SELECT * REPLACE (0 AS reading) FROM events',
                    'SELECT *, reading * 2 AS doubled FROM events',
                    'SELECT * FROM events JOIN other ON events.id = other.id',
                    'SELECT other.* FROM events',
                    'SELECT * FROM (SELECT reading FROM events)',
                    'WITH chosen AS (SELECT reading FROM events) SELECT * FROM chosen']
        for query in accepted + rejected:
            with self.subTest(query=query):
                self.assertEqual(_is_full_schema_observation(SimpleNamespace(
                    query=query, source='events', parent_id='')),
                    query in accepted)

    def test_aggregate_then_raw_query_preserves_schema_and_rejects_unknown_column(self):
        saved = {'table': 'events', 'observed_at': datetime.now(timezone.utc).isoformat(),
                 'columns': [{'name': 'reading', 'dtype': 'DOUBLE'},
                             {'name': 'event_id', 'dtype': 'INTEGER'}]}
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'test', 'schema-authority', CapturingModel(),
                reference_context_loader=lambda: [saved])
            try:
                aggregate = runtime.datasets.register(pd.DataFrame({'n': [3]}),
                    source='events', query='SELECT COUNT(*) AS n FROM events',
                    grain='aggregate', coverage='complete', predicate_known=True)
                runtime.select_dataset(aggregate.id)
                resolved = resolve_table_context([saved], runtime.datasets, 'events')
                self.assertEqual(resolved['authority'], 'saved_snapshot')
                self.assertEqual(resolved['table_context']['columns'], saved['columns'])
                for column, error in [('reading', None), ('missing', 'unknown_column')]:
                    feedback = runtime.recovery._proposal_preflight_error({
                        'name': 'query_databricks', 'args': {'source': 'events',
                            'query': f'SELECT {column} FROM events LIMIT 5'}})
                    self.assertEqual((feedback or {}).get('error_code'), error)
                self.assertEqual(runtime.context.selected_dataset_id, aggregate.id)
            finally:
                runtime.close()

    def test_aggregate_does_not_refresh_stale_schema_but_zero_row_star_does(self):
        from utils.analysis_datasets import DatasetStore
        store = DatasetStore()
        stale = {'table': 'events', 'observed_at': '2020-01-01T00:00:00Z',
                 'columns': [{'name': 'old', 'dtype': 'INTEGER'}]}
        store.register(pd.DataFrame({'n': [3]}), source='events',
            query='SELECT COUNT(*) AS n FROM events', grain='aggregate')
        self.assertEqual(resolve_table_context([stale], store, 'events')['status'], 'needs_refresh')
        store.register(pd.DataFrame({'reading': pd.Series(dtype='float64')}), source='events',
            query='SELECT * FROM events LIMIT 0')
        resolved = resolve_table_context([stale], store, 'events')
        self.assertEqual(resolved['status'], 'ready')
        self.assertEqual([c['name'] for c in resolved['table_context']['columns']], ['reading'])


if __name__ == '__main__':
    unittest.main()
