"""Independent remote SQL semantics with a local engine, not a live warehouse."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd
from sqlglot import parse_one, exp
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from migration.test_persistent_runtime import QuietModel
from tests.test_latest_distribution import FIXTURE, KEY, VALUE, CLOCK, INGEST, source_frame
from utils.analysis_remote_latest import prepare, FIELDS


def references():
    return [{'table': FIXTURE['source'], 'observed_at': datetime.now(timezone.utc).isoformat(),
        'columns': [{'name': c, 'dtype': str(source_frame()[c].dtype)} for c in FIXTURE['columns']]}]


def execute_fixture(query, frame):
    tree = parse_one(query, read='databricks')
    for table in tree.find_all(exp.Table):
        if table.catalog or table.db:
            table.replace(exp.to_table('data'))
    with duckdb.connect(config={'enable_external_access': False}) as conn:
        conn.register('data', frame)
        return conn.execute(tree.sql(dialect='duckdb')).df()


class RemoteLatestTests(unittest.TestCase):
    def runtime(self, root, frame=None, policy=None, error=None, mutate=None):
        executions = []
        def factory(datasets):
            def execute(envelope):
                executions.append(envelope['query'])
                if error:
                    raise error
                result = execute_fixture(envelope['query'], source_frame() if frame is None else frame)
                if mutate:
                    result = mutate(result)
                info = datasets.register(result, source=envelope['source'], query=envelope['query'],
                    coverage='complete', predicate_known=False, grain='aggregate', aggregation=envelope['query'], snapshot='fixture')
                return {'status': 'ready', 'dataset': asdict(info)}
            return execute
        r = GraphAnalysisRuntime(root, 'test', 'remote-latest', QuietModel(), connection_identity='fixture',
            remote_factory=factory, reference_context_loader=references, policy=policy)
        return r, executions

    def prompt(self):
        return json.loads((Path(__file__).parent/'fixtures/remote_latest_per_key.json').read_text())['prompt']

    def test_plan_execute_validate_chart_without_loading_raw(self):
        with tempfile.TemporaryDirectory() as root:
            r, calls = self.runtime(root)
            try:
                from utils.analysis_datasets import stored_dataset_digest
                original = r.datasets.register(source_frame(), source='fixture.other.events',
                    coverage='complete', predicate_known=True)
                r.select_dataset(original.id)
                digest = stored_dataset_digest(r.datasets, original.id)
                outcome = r.submit(self.prompt())
                self.assertEqual(outcome['status'], 'answered', outcome)
                state = r.inspect()['recovery']
                proof = state.get('latest_selection_evidence')
                self.assertTrue(proof, state)
                self.assertEqual({row[VALUE]: row[proof['count_column']] for row in proof['counts']}, FIXTURE['expected_counts'])
                self.assertEqual(proof['selected_keys'], len(FIXTURE['expected_latest']))
                self.assertEqual(proof['input_rows'], len(FIXTURE['rows']))
                self.assertEqual(len(calls), 1)
                self.assertEqual(r.datasets.metadata[proof['input_result_id']].rows, 4)
                self.assertTrue(state['artifact_ids'])
                self.assertFalse(r.inspect()['requests'])
                distribution_id = proof['distribution']['id']
                self.assertEqual(r.context.selected_dataset_id, original.id)
                self.assertEqual(stored_dataset_digest(r.datasets, original.id), digest)
            finally:
                r.close()
            r, calls = self.runtime(root)
            try:
                self.assertFalse(calls)
                self.assertEqual(r.datasets.metadata[distribution_id].grain, 'aggregate')
                self.assertEqual(r.inspect()['state'], 'idle')
                for _ in range(2):
                    repeated = r.submit(self.prompt())
                    self.assertEqual(repeated['status'], 'answered', repeated)
                    self.assertIn('재사용', repeated['text'])
                    self.assertFalse(calls)
                fresh = r.submit(self.prompt()+' DB에서 다시 조회해줘.')
                self.assertEqual(fresh['status'], 'answered', fresh)
                self.assertEqual(len(calls), 1)
            finally:
                r.close()

    def test_bad_data_never_publishes_chart(self):
        original = source_frame()
        newest = original.sort_values(CLOCK).drop_duplicates(KEY, keep='last').iloc[[0]]
        tie = pd.concat([original, newest], ignore_index=True)
        missing = original.copy(); missing.loc[0, KEY] = None
        for frame in (tie, missing, original.head(0)):
            with self.subTest(rows=len(frame)), tempfile.TemporaryDirectory() as root:
                r, calls = self.runtime(root, frame)
                try:
                    result = r.submit(self.prompt())
                    self.assertFalse(r.artifacts, result)
                    self.assertIsNone(r.inspect()['recovery'].get('latest_selection_evidence'))
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(r.events()[-1].additional_kwargs.get('analysis_complete'))
                finally:
                    r.close()

    def test_manual_approval_and_unknown_execution_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            r, calls = self.runtime(root, policy=RuntimePolicy(require_remote_approval=True))
            try:
                result = r.submit(self.prompt())
                self.assertEqual(result['status'], 'awaiting_approval', result)
                self.assertFalse(calls)
            finally:
                r.close()

            r, calls = self.runtime(root)
            try:
                result = r.resume()
                self.assertEqual(result['status'], 'answered', result)
                self.assertEqual(len(calls), 1)
            finally:
                r.close()
        with tempfile.TemporaryDirectory() as root:
            r, calls = self.runtime(root, error=TimeoutError('injected uncertain remote execution'))
            try:
                r.submit(self.prompt())
                self.assertEqual(len(calls), 1)
                self.assertTrue(r.ledger.uncertain())
                self.assertFalse(r.artifacts)
                with self.assertRaises(PermissionError):
                    r.resume()
                self.assertEqual(len(calls), 1)
            finally:
                r.close()

    def test_cached_only_request_and_wrong_query_are_not_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            r, calls = self.runtime(root)
            try:
                r.submit(FIXTURE['cases'][1]['prompt'])
                self.assertFalse(calls)
                r.submit(self.prompt())
                current = r.inspect()['recovery']
                self.assertFalse(r.recovery._proposed_scope_valid({'name': 'query_databricks',
                    'args': {'source': FIXTURE['source'], 'query': 'SELECT * FROM '+FIXTURE['source']}}, current))
                self.assertFalse(r.recovery._proposed_scope_valid({'name': 'query_databricks',
                    'args': {k: current['remote_latest_plan'][k] for k in ('source', 'query', 'reason')}}, current))
                from core.analysis_agent.latest_selection import accepted_remote
                proof = current['latest_selection_evidence']
                args = {k: current['latest_per_key_spec'][k] for k in ('source', 'key_columns', 'order_column', 'value_column')}
                args['result_dataset_id'] = proof['input_result_id']
                self.assertFalse(accepted_remote(r.context, r.artifacts, current['latest_per_key_spec'], args, proof, {}))
            finally:
                r.close()

    def test_stale_schema_refreshes_before_business_query(self):
        from unittest.mock import patch
        stale = references()
        stale[0]['observed_at'] = '2020-01-01T00:00:00+00:00'
        with tempfile.TemporaryDirectory() as root, patch(__name__+'.references', return_value=stale):
            r, calls = self.runtime(root)
            try:
                result = r.submit(self.prompt())
                self.assertEqual(result['status'], 'answered', result)
                self.assertTrue(r.inspect()['recovery'].get('latest_selection_evidence'), r.inspect())
                self.assertEqual(len(calls), 2)
                self.assertTrue(calls[0].endswith('LIMIT 0'))
                self.assertIn('DENSE_RANK', calls[1])
            finally:
                r.close()

    def test_catalog_numeric_types_and_renamed_quoted_columns(self):
        from unittest.mock import patch
        from utils.analysis_remote_latest import plan
        for dtype in ('LONG', 'SHORT', 'BYTE'):
            refs = references()
            next(c for c in refs[0]['columns'] if c['name'] == CLOCK)['dtype'] = dtype
            with tempfile.TemporaryDirectory() as root, patch(__name__+'.references', return_value=refs):
                r, calls = self.runtime(root)
                try:
                    self.assertEqual(plan(r.context, FIXTURE['source'], [KEY], CLOCK, VALUE)['status'], 'planned')
                    self.assertFalse(calls)
                finally:
                    r.close()
        rename = {KEY: 'entity key', CLOCK: 'order', VALUE: 'value ` label', INGEST: 'arrival time'}
        frame = source_frame().rename(columns=rename)
        refs = references()
        for column in refs[0]['columns']:
            column['name'] = rename[column['name']]
        with tempfile.TemporaryDirectory() as root, patch(__name__+'.references', return_value=refs):
            r, _ = self.runtime(root, frame)
            try:
                planned = plan(r.context, FIXTURE['source'], [rename[KEY]], rename[CLOCK], rename[VALUE])
                result = execute_fixture(planned['remote_latest_plan']['query'], frame)
                records = result[result['__kind'] == 1]
                self.assertEqual(dict(zip(records['__value'], records['__frequency'])), FIXTURE['expected_counts'])
            finally:
                r.close()

    def test_category_limit_and_corrupt_counts_never_complete(self):
        frame = source_frame().iloc[[0]*51].copy().reset_index(drop=True)
        frame[KEY] = [str(i) for i in range(51)]
        frame[VALUE] = [str(i) for i in range(51)]
        def corrupt(result):
            result.loc[result['__kind']==1, '__frequency'] += 1
            return result
        for data, mutation in ((frame, None), (None, corrupt)):
            with tempfile.TemporaryDirectory() as root:
                r, calls = self.runtime(root, data, mutate=mutation)
                try:
                    r.submit(self.prompt())
                    self.assertEqual(len(calls), 1)
                    self.assertFalse(r.artifacts)
                    self.assertIsNone(r.inspect()['recovery'].get('latest_selection_evidence'))
                finally:
                    r.close()
