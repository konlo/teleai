import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from core.analysis_databricks import execute_approved
from core.analysis_sql import local_query, validate_query
from utils.analysis_datasets import DatasetStore


class QueryTests(unittest.TestCase):
    def test_local_or_and_aggregation_use_real_rows(self):
        frame = pd.DataFrame({"value": [1, 15, 30]})
        result, truncated, _ = local_query(frame,
            "SELECT avg(value) AS mean FROM data WHERE value < 10 OR value > 20")
        self.assertEqual(result.iloc[0]['mean'], 15.5)
        self.assertFalse(truncated)

    def test_filtered_raw_query_retains_reusable_scope_without_flattening_or(self):
        from utils.analysis_provenance import raw_conditions
        conditions=raw_conditions(validate_query("SELECT * FROM events WHERE segment = 'A' AND value >= 10"))
        self.assertIsNotNone(conditions)
        self.assertEqual([(c.column,c.op,c.value) for c in conditions],[('segment','eq','A'),('value','ge',10)])
        self.assertIsNone(raw_conditions(validate_query("SELECT * FROM events WHERE value < 10 OR value > 20")))
        self.assertIsNone(raw_conditions(validate_query("SELECT AVG(value) FROM events WHERE segment = 'A'")))

    def test_write_and_multiple_statements_are_rejected(self):
        for sql in ["DROP TABLE data", "SELECT 1; SELECT 2", "DELETE FROM data"]:
            with self.assertRaises(ValueError):
                validate_query(sql)

    def test_external_local_scan_is_rejected(self):
        with self.assertRaises(Exception):
            local_query(pd.DataFrame({"x": [1]}), "SELECT * FROM read_csv_auto('/etc/passwd')")

    def test_remote_execution_is_bounded_and_only_after_approval(self):
        request = SimpleNamespace(status="proposed", source="events", query="SELECT x FROM events")
        connect = MagicMock()
        cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchmany.return_value = [(1,), (2,), (3,)]
        cursor.description = [("x",)]
        config = MagicMock()
        store = DatasetStore()
        with self.assertRaises(PermissionError):
            execute_approved(request, config, store, connect=connect)
        connect.assert_not_called()
        request.status = "executing"
        result = execute_approved(request, config, store, max_rows=2, connect=connect)
        cursor.fetchmany.assert_called_once_with(3)
        cursor.execute.assert_called_once_with("SELECT x FROM events")
        self.assertEqual(result['dataset']['coverage'], 'truncated')
        self.assertEqual(result['dataset']['rows'], 2)
        self.assertTrue(result['dataset']['snapshot'])


if __name__ == '__main__':
    unittest.main()
