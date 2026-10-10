"""Bounded, read-only inventory without model calls or analysis-state mutation."""
from datetime import datetime, timezone
import time
from uuid import uuid4

import pandas as pd
from core.analysis_sql import validate_query


def fetch_table_inventory(backend, diagnostics, *, connect=None, limit=200):
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('테이블 목록 한도는 1~200입니다.')
    run_id = uuid4().hex
    started = time.monotonic()
    stage = 'inventory_configuration'
    report = {'run_id': run_id, 'backend': backend.name, 'limit': limit,
              'model_called': False, 'observed_at': datetime.now(timezone.utc).isoformat()}
    diagnostics.emit('table_inventory_started', **report)
    try:
        config = backend.config
        if backend.name == 'databricks':
            config.validate()
            if not config.catalog:
                raise ValueError('DATABRICKS_CATALOG가 필요합니다.')
            catalog = '`' + config.catalog.replace('`', '``') + '`'
            columns = ['table_catalog', 'table_schema', 'table_name', 'table_type']
            query = f'SELECT {", ".join(columns)} FROM {catalog}.information_schema.tables'
            parameters = {}
            if config.schema:
                query += ' WHERE table_schema = :schema'
                parameters['schema'] = config.schema
            query += f' ORDER BY table_catalog, table_schema, table_name LIMIT {limit + 1}'
            validate_query(query, dialect='databricks')
            report['scope'] = config.catalog + ('.' + config.schema if config.schema else ' · 전체 schema')
            stage = 'inventory_driver'
            if connect is None:
                from databricks import sql
                connect = sql.connect
            stage = 'inventory_open_session'
            connection = connect(server_hostname=config.server_hostname, http_path=config.http_path,
                                 access_token=config.access_token, catalog=config.catalog,
                                 schema=config.schema or None)
        elif backend.name == 'mysql':
            columns = ['table_schema', 'table_name', 'table_type']
            query = ('SELECT table_schema, table_name, table_type FROM information_schema.tables '
                     f'WHERE table_schema = %s ORDER BY table_name LIMIT {limit + 1}')
            validate_query(query.replace('%s', "'scope'"), dialect='mysql')
            parameters = (config.database,)
            report['scope'] = config.database
            stage = 'inventory_open_session'
            connection = (connect or config.connect)()
        else:
            raise ValueError('지원하지 않는 데이터 backend입니다.')
        try:
            stage = 'inventory_cursor'
            cursor = connection.cursor()
            try:
                stage = 'inventory_execute'
                cursor.execute(query, parameters)
                stage = 'inventory_fetch'
                rows = cursor.fetchmany(limit + 1)
                frame = pd.DataFrame(rows[:limit], columns=columns)
            finally:
                cursor.close()
        finally:
            connection.close()
        report.update(status='PASS', rows=len(frame), truncated=len(rows) > limit)
    except Exception as exc:
        error_id = diagnostics.failure(exc, run_id=run_id, stage=stage)
        frame = None
        report.update(status='FAIL', error_id=error_id, error_type=type(exc).__name__)
    report.update(stage=stage, elapsed_seconds=round(time.monotonic()-started, 3))
    diagnostics.emit('table_inventory_finished', **report)
    return frame, report
