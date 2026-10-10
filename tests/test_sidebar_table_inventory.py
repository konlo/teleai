"""Inventory UI must preserve analysis data and avoid model/agent execution."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.table_inventory import fetch_table_inventory
from core.analysis_agent.databricks import ConnectionConfig
from tests.test_data_backend_isolation import CONNECTION
from tests.test_mysql_metadata_contract import NoInference


class Cursor:
    def __init__(self, rows, fail=False):
        self.rows, self.fail, self.calls = rows, fail, []
        self.closed = False
    def execute(self, query, parameters):
        self.calls.append((query, parameters))
        if self.fail:
            raise RuntimeError('private-secret-must-not-be-logged')
    def fetchmany(self, limit):
        assert limit == 201
        return self.rows[:limit]
    def close(self):
        self.closed = True


class Connection:
    def __init__(self, cursor):
        self.value, self.closed = cursor, False
    def cursor(self):
        return self.value
    def close(self):
        self.closed = True


def db_backend(schema='lab'):
    return SimpleNamespace(name='databricks', config=ConnectionConfig(
        'example.invalid', '/sql/fixture', 'fixture-secret', 'eval_catalog', schema))


def test_inventory_parameterizes_scope_and_closes_resources(tmp_path):
    cursor = Cursor([('eval_catalog', "lab' OR 1=1 --", 'new_table', 'VIEW')])
    connection = Connection(cursor)
    frame, report = fetch_table_inventory(db_backend("lab' OR 1=1 --"), Diagnostics(tmp_path),
                                         connect=lambda **kw: connection)
    assert report['status'] == 'PASS' and report['rows'] == 1
    query, params = cursor.calls[0]
    assert "OR 1=1" not in query and params == {'schema': "lab' OR 1=1 --"}
    assert cursor.closed and connection.closed
    assert frame.iloc[0]['table_type'] == 'VIEW'
    assert 'fixture-secret' not in (tmp_path/'runtime.jsonl').read_text()


def test_mysql_inventory_limit_is_explicit_and_no_other_backend(tmp_path):
    cursor = Cursor([('new_database', f't{i:03}', 'BASE TABLE') for i in range(201)])
    connection = Connection(cursor)
    backend = SimpleNamespace(name='mysql', config=SimpleNamespace(database='new_database',
                                                                 connect=lambda: connection))
    frame, report = fetch_table_inventory(backend, Diagnostics(tmp_path))
    assert report['truncated'] and len(frame) == 200
    assert cursor.calls[0][1] == ('new_database',)
    assert 'information_schema.tables' in cursor.calls[0][0]
    assert connection.closed


def test_failure_reports_actual_stage_and_safe_error_id(tmp_path):
    cursor = Cursor([], fail=True)
    connection = Connection(cursor)
    frame, report = fetch_table_inventory(db_backend(), Diagnostics(tmp_path),
                                         connect=lambda **kw: connection)
    assert frame is None and report['status'] == 'FAIL'
    assert report['stage'] == 'inventory_execute'
    assert len(report['error_id']) == 12
    logs = (tmp_path/'runtime.jsonl').read_text()
    assert report['error_id'] in logs and 'private-secret' not in logs
    assert cursor.closed and connection.closed


def test_sidebar_button_does_not_call_agent_or_replace_original(tmp_path):
    cursor = Cursor([('eval_catalog','lab','fresh_table','MANAGED')])
    connection = Connection(cursor)
    with patch.dict('os.environ', {**CONNECTION, 'TELLY_V1_STORAGE': str(tmp_path)}), \
         patch('core.analysis_agent.model_provider.build_analysis_chat_model', return_value=NoInference()), \
         patch('core.analysis_catalog.load_saved_reference_context', return_value=[]), \
         patch('core.analysis_agent.databricks.make_executor') as executor, \
         patch('databricks.sql.connect', return_value=connection):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'main.py'),
                               default_timeout=20).run()
        assert not app.exception
        buttons = [b.label for b in app.sidebar.button]
        assert buttons.index('DB 연결만 확인') < buttons.index('테이블 목록 조회')
        runtime = app.session_state['v1_runtime']
        original = runtime.datasets.register(pd.DataFrame({'reading':[2,3]}),
                                            source='eval_catalog.lab.original')
        runtime.select_dataset(original.id)
        before = runtime.inspect()
        messages = runtime.events()
        app.button(key='v1_table_inventory').click().run()
        assert not app.exception
        assert len(cursor.calls) == 1
        assert 'fresh_table' in app.sidebar.dataframe[0].value['table_name'].tolist()
        assert runtime.inspect() == before
        assert runtime.events() == messages
        assert set(runtime.datasets.metadata) == {original.id}
        executor.assert_called_once()  # Runtime constructed; its executor never runs.
        app.run()
        assert not app.exception and len(cursor.calls) == 1  # No query on ordinary rerun.
        cursor.rows = []
        app.button(key='v1_table_inventory').click().run()
        assert not app.exception and len(cursor.calls) == 2
        assert all('table_name' not in item.value.columns for item in app.sidebar.dataframe)
        # The original dataset preview remains; the old inventory disappears.
        assert app.sidebar.info
        cursor.fail = True
        app.button(key='v1_table_inventory').click().run()
        assert not app.exception and len(cursor.calls) == 3
        assert app.sidebar.error
        assert 'inventory_execute' in '\n'.join(x.value for x in app.sidebar.caption)
        assert all('table_name' not in item.value.columns for item in app.sidebar.dataframe)
        assert runtime.inspect() == before and runtime.events() == messages
        runtime.close()
