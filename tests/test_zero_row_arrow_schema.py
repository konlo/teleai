"""Connector-provided zero-row Arrow schemas must survive persistence."""
from types import SimpleNamespace
import tempfile
import unittest
import pyarrow as pa
from core.analysis_databricks import execute_approved
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_catalog import resolve_table_context
from migration.test_persistent_runtime import QuietModel


class Cursor:
    def __init__(self, table):
        self.table = table
        self.description = [(name,) for name in table.column_names]
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def execute(self, query): self.query = query
    def fetchmany_arrow(self, count): return self.table
    def fetchmany(self, count): raise AssertionError('Do not discard the typed schema')
    def cursor(self): return self


class ZeroRowArrowTests(unittest.TestCase):
    def test_actual_types_persist_and_no_old_dtype_is_invented(self):
        table = pa.table({'sequence': pa.array([], type=pa.int64()),
            'clock': pa.array([], type=pa.timestamp('us', tz='UTC')),
            'amount': pa.array([], type=pa.decimal128(20, 2))})
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'test', 'schema', QuietModel())
            try:
                config = SimpleNamespace(server_hostname='', http_path='', access_token='', catalog='', schema='')
                request = SimpleNamespace(status='executing', source='fixture.events', query='SELECT * FROM fixture.events LIMIT 0')
                result = execute_approved(request, config, r.datasets, connect=lambda **kwargs: Cursor(table))
                info = r.datasets.metadata[result['dataset']['id']]
                self.assertEqual(info.rows, 0)
                columns = resolve_table_context([], r.datasets, request.source)['table_context']['columns']
                self.assertIn('int64', columns[0]['dtype'])
                self.assertIn('datetime', columns[1]['dtype'])
                self.assertIn('decimal', columns[2]['dtype'])
            finally:
                r.close()

    def test_nonempty_schema_probe_is_rejected_without_publishing(self):
        table = pa.table({'value': [1]})
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'test', 'schema-invalid', QuietModel())
            try:
                config = SimpleNamespace(server_hostname='', http_path='', access_token='', catalog='', schema='')
                request = SimpleNamespace(status='executing', source='fixture.events', query='SELECT * FROM fixture.events LIMIT 0')
                with self.assertRaises(ValueError):
                    execute_approved(request, config, r.datasets, connect=lambda **kwargs: Cursor(table))
                self.assertFalse(r.datasets.metadata)
            finally:
                r.close()
