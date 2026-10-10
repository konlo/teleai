"""Read-only MySQL evaluation backend for the shared analysis runtime."""
from __future__ import annotations

from configparser import ConfigParser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import mysql.connector
import pandas as pd
from mysql.connector.constants import FieldType
from sqlglot import exp

from core.analysis_agent.approvals import QueryNotSubmitted, QueryTerminated, QueryRejected
from core.analysis_load_plan import source_plan
from core.analysis_sql import validate_query
from utils.analysis_provenance import query_coverage, raw_conditions


_IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


@dataclass(frozen=True)
class MySQLConfig:
    option_file: Path
    database: str
    query_timeout_seconds: int = 120

    @classmethod
    def from_env(cls, root: Path):
        value = os.getenv('TELLY_MYSQL_OPTION_FILE', '.telly_runtime/mysql_eval/reader.cnf')
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        database = os.getenv('TELLY_MYSQL_DATABASE', 'teleai_default')
        if not _IDENTIFIER.fullmatch(database):
            raise ValueError('MySQL database 이름이 유효하지 않습니다.')
        timeout=int(os.getenv('TELLY_MYSQL_QUERY_TIMEOUT_SECONDS', '120'))
        if not 1 <= timeout <= 600:
            raise ValueError('MySQL 조회 실행 제한은 1~600초여야 합니다.')
        return cls(path.resolve(), database, timeout)

    def credentials(self):
        if not self.option_file.is_file():
            raise ValueError('MySQL 읽기 전용 접속 설정 파일이 없습니다.')
        if self.option_file.stat().st_mode & 0o077:
            raise ValueError('MySQL 접속 설정 파일 권한은 0600이어야 합니다.')
        options = ConfigParser(interpolation=None)
        if not options.read(self.option_file) or 'client' not in options:
            raise ValueError('MySQL 접속 설정 파일의 [client] 항목이 없습니다.')
        section = options['client']
        if not section.get('user') or not section.get('password'):
            raise ValueError('MySQL 접속 설정에 user/password가 없습니다.')
        return dict(user=section['user'], password=section['password'],
                    unix_socket=section.get('socket', '/tmp/mysql.sock'))

    def identity(self):
        credentials = self.credentials()
        encoded = json.dumps([self.database, credentials], sort_keys=True).encode()
        return sha256(b'mysql-eval-v1:' + encoded).hexdigest()

    def connect(self):
        connection = mysql.connector.connect(**self.credentials(), database=self.database,
            charset='utf8mb4', collation='utf8mb4_bin', connection_timeout=10,
            read_timeout=max(150,self.query_timeout_seconds+30), autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'SET SESSION MAX_EXECUTION_TIME = {self.query_timeout_seconds*1000}')
            return connection
        except BaseException:
            connection.close()
            raise


def _validate_sources(query, database):
    tree = validate_query(query, dialect='mysql')
    for table in tree.find_all(exp.Table):
        if table.catalog or (table.db and table.db.casefold() not in
                             {database.casefold(), 'information_schema'}):
            raise ValueError('선택한 MySQL 데이터베이스 밖의 테이블은 조회할 수 없습니다.')
        if not table.db and table.name.casefold() in {'mysql', 'sys', 'performance_schema'}:
            raise ValueError('시스템 데이터베이스는 조회할 수 없습니다.')
    return tree


def _empty_schema(columns, description):
    integer = {FieldType.TINY, FieldType.SHORT, FieldType.LONG, FieldType.LONGLONG,
               FieldType.INT24, FieldType.YEAR}
    floating = {FieldType.FLOAT, FieldType.DOUBLE, FieldType.DECIMAL, FieldType.NEWDECIMAL}
    date_like = {FieldType.DATE, FieldType.DATETIME, FieldType.TIMESTAMP}
    types = []
    for item in description:
        code = item[1]
        types.append('Int64' if code in integer else
                     'float64' if code in floating else
                     'datetime64[ns]' if code in date_like else 'string')
    return pd.DataFrame({name: pd.Series(dtype=dtype)
                         for name, dtype in zip(columns, types)})


def make_executor(config: MySQLConfig, datasets, *, max_rows=100_000, max_coordinate_rows=None):
    from core.analysis_agent.query_control import QueryControl
    control=QueryControl()
    def execute(envelope):
        if envelope['connection'] != config.identity():
            raise PermissionError('MySQL 연결 설정이 변경되어 기존 조회를 실행할 수 없습니다.')
        if max_rows < 1:
            raise ValueError('조회 행 한도는 1 이상이어야 합니다.')
        query = envelope['query']
        tree = _validate_sources(query, config.database)
        from core.analysis_agent.source_scatter import fetch_limit
        row_limit=fetch_limit(query,'mysql',max_rows,max_coordinate_rows)
        if row_limit<1:raise ValueError('좌표 조회 한도는 1 이상이어야 합니다.')
        plan = source_plan(envelope['source'], query, dialect='mysql')
        try:
            connection = config.connect()
        except mysql.connector.Error as exc:
            raise QueryNotSubmitted() from exc
        try:
            cursor = connection.cursor(buffered=False)
            def cancel():
                # The connection ID comes from this runtime's own active session.
                identifier=connection.connection_id
                if type(identifier) is not int or identifier<=0:raise ValueError('Invalid active session ID')
                controller=config.connect()
                try:
                    target=controller.cursor()
                    try:target.execute('KILL QUERY '+str(identifier))
                    finally:target.close()
                finally:controller.close()
            control.start(cancel if type(getattr(connection,'connection_id',None)) is int else None)
            try:
                cursor.execute(query)
                control.validate_result()
                description = cursor.description
                if description is None:
                    raise ValueError('조회 결과의 컬럼 정보를 확인할 수 없습니다.')
                columns = [item[0] for item in description]
                if not columns or len(columns) != len(set(columns)) or any(not name for name in columns):
                    raise ValueError('조회 결과 컬럼 이름이 유효하지 않거나 중복되었습니다.')
                if (plan.expected_columns and
                        tuple(x.casefold() for x in columns) !=
                        tuple(x.casefold() for x in plan.expected_columns)):
                    raise ValueError('조회 결과 컬럼이 SQL의 출력 컬럼과 다릅니다.')
                if len(columns) > datasets.max_columns:
                    raise ValueError('조회 결과의 컬럼 수가 운영 한도를 초과합니다.')
                limit = tree.args.get('limit')
                schema_probe = bool(limit and isinstance(limit.expression, exp.Literal)
                                    and limit.expression.is_int and int(limit.expression.this) == 0)
                observed = 0
                estimated = 0
                preview = []

                def batches():
                    nonlocal observed, estimated
                    if schema_probe:
                        yield _empty_schema(columns, description)
                        return
                    while observed <= row_limit:
                        rows = cursor.fetchmany(min(1024, row_limit + 1 - observed))
                        control.validate_result()
                        if not rows:
                            break
                        frame = pd.DataFrame.from_records(rows, columns=columns)
                        estimated += int(frame.memory_usage(index=True, deep=True).sum())
                        if estimated > datasets.max_frame_bytes:
                            raise MemoryError('조회 결과가 데이터 메모리 한도를 초과해 발행하지 않았습니다.')
                        available = row_limit - observed
                        observed += len(rows)
                        if available > 0:
                            result = frame.head(available)
                            if len(preview) < 10:
                                preview.extend(result.head(10-len(preview)).to_dict(orient='records'))
                            yield result

                conditions = raw_conditions(tree)
                provenance = dict(query=query, predicate_known=conditions is not None,
                    conditions=conditions or (), grain=plan.grain,
                    aggregation=tree.sql(dialect='mysql') if plan.grain == 'aggregate' else '',
                    snapshot=datetime.now(timezone.utc).isoformat())
                info = datasets.register_batches(batches(), columns=columns,
                    source=' | '.join(plan.actual_tables) or envelope['source'],
                    max_rows=row_limit,
                    final_provenance=lambda: {'coverage': query_coverage(
                        tree, truncated=observed > row_limit)}, **provenance)
                return {'status':'ready', 'dataset':asdict(info), 'preview':preview[:10]}
            finally:
                # An intentionally bounded fetch can leave server rows unread.
                # Closing the connection below discards them without draining
                # a potentially multi-million-row result into memory.
                try:
                    cursor.close()
                except mysql.connector.InternalError:
                    pass
        except mysql.connector.DatabaseError as exc:
            if exc.errno in {3024,1317}:
                raise QueryTerminated(exc.errno,exc.sqlstate) from exc
            if exc.errno in {1064,1149,1054,1146,1305}:
                raise QueryRejected(exc.errno,exc.sqlstate) from exc
            raise
        finally:
            connection.close()
            control.finish()
    def probe():
        try:
            connection=config.connect()
            try:
                cursor=connection.cursor()
                try:
                    cursor.execute('SELECT 1');ok=tuple(cursor.fetchone() or ())==(1,)
                finally:cursor.close()
            finally:connection.close()
            return {'status':'ready' if ok else 'unavailable','backend':'mysql','probe_sql':'SELECT 1'}
        except Exception as exc:
            return {'status':'unavailable','backend':'mysql','error_type':type(exc).__name__,'error_code':'database_probe_failed','retryable':False}
    execute.probe=probe
    execute.control=control
    return execute


def reference_context(config: MySQLConfig):
    """Observe current tables and columns; never embed dataset schema in code."""
    with config.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute('SELECT table_name, column_name, data_type '
                'FROM information_schema.columns WHERE table_schema = %s '
                'ORDER BY table_name, ordinal_position', (config.database,))
            rows = cursor.fetchall()
    observed = datetime.now(timezone.utc).isoformat()
    tables = {}
    for table, column, dtype in rows:
        tables.setdefault(table, []).append({'name':column, 'dtype':dtype})
    return [{'table':f'{config.database}.{name}', 'columns':columns,
             'training_status':'observed_schema', 'observed_at':observed}
            for name, columns in tables.items()]
