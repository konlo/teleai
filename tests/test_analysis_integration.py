import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from core.analysis_approval import ApprovalQueue
from core.analysis_databricks import execute_approved
from core.analysis_sql import local_query, validate_query
from utils.analysis_datasets import DatasetStore
from core.analysis_model import OllamaAnalysisModel


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
        queue = ApprovalQueue("test")
        request = queue.propose(source="events", query="SELECT x FROM events", reason="needed", goal="analyze")
        connect = MagicMock()
        cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchmany.return_value = [(1,), (2,), (3,)]
        cursor.description = [("x",)]
        config = MagicMock()
        store = DatasetStore()
        with self.assertRaises(PermissionError):
            execute_approved(request, config, store, connect=connect)
        connect.assert_not_called()
        queue.approve(request.id)
        result = queue.execute(request.id, lambda r: execute_approved(r, config, store, max_rows=2, connect=connect))
        cursor.fetchmany.assert_called_once_with(3)
        cursor.execute.assert_called_once_with("SELECT x FROM events")
        self.assertEqual(result['dataset']['coverage'], 'truncated')
        self.assertEqual(result['dataset']['rows'], 2)


class PageTests(unittest.TestCase):
    def test_page_proposal_and_cancel_do_not_connect(self):
        with patch('databricks.sql.connect') as connect:
            app = AppTest.from_file('pages/Telly.py', default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            app.text_input[0].set_value('catalog.schema.events').run()
            next(b for b in app.button if b.label == '데이터 불러오기 제안').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state['analysis_session'].approvals.pending()), 1)
            next(b for b in app.button if b.label == '조회 취소').click().run()
            self.assertEqual(len(app.session_state['analysis_session'].approvals.pending()), 0)
            connect.assert_not_called()


class ModelProtocolTests(unittest.TestCase):
    def test_ollama_receives_tools_and_observations_as_native_messages(self):
        import json
        from io import BytesIO
        sent = []
        def respond(request, timeout):
            sent.append(json.loads(request.data))
            return BytesIO(json.dumps({"message": {"content": "", "tool_calls": [{
                "function": {"name": "inspect_dataset", "arguments": {"dataset_id": "d"}}}]}}).encode())
        with patch('urllib.request.urlopen', side_effect=respond):
            result = OllamaAnalysisModel('test-model')([
                {"role": "user", "content": "이전 결과를 살펴봐"},
                {"role": "tool", "name": "list_analysis_context", "tool_call_id": "t", "content": '{"dataset":"d"}'}],
                [{"name": "inspect_dataset", "description": "inspect", "parameters": {"type": "object"}}])
        self.assertEqual(sent[0]['messages'][1]['tool_name'], 'list_analysis_context')
        self.assertEqual(sent[0]['tools'][0]['type'], 'function')
        self.assertEqual(result['tool_calls'][0]['arguments'], {'dataset_id': 'd'})


if __name__ == '__main__':
    unittest.main()
