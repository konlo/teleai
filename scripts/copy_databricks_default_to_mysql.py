"""Copy selected Databricks tables into an isolated local MySQL evaluation DB.

The source is read with one Arrow cursor per table. A new staging table is
published only after an order-independent full-row digest and count match.
No application credentials or data rows are written into the repository.
"""
from __future__ import annotations

import argparse
from configparser import ConfigParser
from datetime import date, datetime, timezone
from decimal import Decimal
from hashlib import blake2b
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from databricks import sql as databricks_sql
from dotenv import load_dotenv
import mysql.connector
import pyarrow as pa

from core.analysis_agent.databricks import ConnectionConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES = ('error_test', 'titanic', 'ncr_ride', 'bank_loan', 'stormtrooper')
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
MODULUS = 1 << 128


def identifier(name: str) -> str:
    if not NAME.fullmatch(name) or len(name) > 64:
        raise ValueError(f'Unsupported table/database identifier: {name!r}')
    return '`' + name + '`'


def column_identifier(name: str) -> str:
    if not name or len(name) > 64 or '\x00' in name:
        raise ValueError('Invalid source column name')
    return '`' + name.replace('`', '``') + '`'


def mysql_type(arrow_type) -> str:
    if pa.types.is_integer(arrow_type):
        return 'BIGINT'
    if pa.types.is_floating(arrow_type):
        return 'DOUBLE'
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return 'LONGTEXT'
    if pa.types.is_date(arrow_type):
        return 'DATE'
    if pa.types.is_timestamp(arrow_type):
        return 'DATETIME(6)'
    if pa.types.is_boolean(arrow_type):
        return 'BOOLEAN'
    if pa.types.is_decimal(arrow_type) and arrow_type.precision <= 65:
        return f'DECIMAL({arrow_type.precision},{arrow_type.scale})'
    raise ValueError(f'No lossless MySQL mapping for {arrow_type}')


def portable_value(value):
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite float cannot be silently copied into MySQL')
    return value


def encoded_value(value):
    if value is None:
        return ['null']
    if isinstance(value, datetime):
        return ['timestamp_utc', portable_value(value).isoformat(timespec='microseconds')]
    if isinstance(value, date):
        return ['date', value.isoformat()]
    if isinstance(value, bool):
        return ['boolean', value]
    if isinstance(value, int):
        return ['integer', value]
    if isinstance(value, float):
        return ['double', repr(portable_value(value))]
    if isinstance(value, Decimal):
        return ['decimal', format(value, 'f')]
    if isinstance(value, str):
        return ['string', value]
    raise ValueError(f'Unsupported value type: {type(value).__name__}')


class MultisetDigest:
    def __init__(self):
        self.count = 0
        self.total = 0
        self.xor = 0

    def add(self, row):
        canonical = json.dumps([encoded_value(value) for value in row],
            ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        value = int.from_bytes(blake2b(canonical, digest_size=16).digest(), 'big')
        self.count += 1
        self.total = (self.total + value) % MODULUS
        self.xor ^= value

    def public(self):
        return {'rows': self.count, 'sum128': f'{self.total:032x}',
                'xor128': f'{self.xor:032x}'}


def mysql_connection(option_file: Path, database: str):
    options = ConfigParser()
    if not options.read(option_file) or 'client' not in options:
        raise ValueError('Local MySQL credential file is missing')
    client = options['client']
    return mysql.connector.connect(user=client['user'], password=client['password'],
        unix_socket=client.get('socket', '/tmp/mysql.sock'), database=database,
        charset='utf8mb4', collation='utf8mb4_bin', connection_timeout=10,
        autocommit=False)


def source_connection(config):
    if not all((config.server_hostname, config.http_path, config.access_token)):
        raise ValueError('Databricks connection is not configured')
    return databricks_sql.connect(server_hostname=config.server_hostname,
        http_path=config.http_path, access_token=config.access_token,
        catalog=config.catalog, schema=config.schema)


def source_count(connection, source):
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT COUNT(*) FROM {source}')
        return int(cursor.fetchone()[0])


def source_schema(connection, source):
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT * FROM {source} LIMIT 0')
        arrow = cursor.fetchmany_arrow(1)
        if arrow.num_rows:
            raise ValueError('Schema probe unexpectedly returned data')
        return arrow.schema


def target_digest(connection, table, columns, batch_size):
    digest = MultisetDigest()
    cursor = connection.cursor(buffered=False)
    try:
        cursor.execute('SELECT ' + ', '.join(map(column_identifier, columns))
                       + ' FROM ' + identifier(table))
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            for row in batch:
                digest.add(row)
    finally:
        cursor.close()
    return digest


def copy_table(source_db, target_db, *, catalog, source_schema_name,
               target_database, table, batch_size, progress_rows):
    source = '.'.join(identifier(name) for name in (catalog, source_schema_name, table))
    started = time.monotonic()
    expected_rows = source_count(source_db, source)
    schema = source_schema(source_db, source)
    columns = list(schema.names)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f'Invalid source columns for {table}')
    types = [mysql_type(field.type) for field in schema]
    stage = f'_teleai_stage_{table}_{uuid4().hex[:8]}'
    with target_db.cursor() as cursor:
        cursor.execute('SELECT COUNT(*) FROM information_schema.tables '
            'WHERE table_schema = DATABASE() AND table_name = %s', (table,))
        if cursor.fetchone()[0]:
            raise ValueError(f'{table} already exists; refusing to overwrite verified data')
        definitions = ', '.join(f'{column_identifier(name)} {kind} NULL'
                                for name, kind in zip(columns, types))
        cursor.execute(f'CREATE TABLE {identifier(stage)} ({definitions}) ENGINE=InnoDB '
                       'DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin')
    target_db.commit()
    digest = MultisetDigest()
    next_progress = progress_rows
    try:
        insert = (f'INSERT INTO {identifier(stage)} ('
                  + ', '.join(map(column_identifier, columns)) + ') VALUES ('
                  + ', '.join(['%s'] * len(columns)) + ')')
        with source_db.cursor() as source_cursor, target_db.cursor() as target_cursor:
            source_cursor.execute(f'SELECT * FROM {source}')
            while True:
                arrow = source_cursor.fetchmany_arrow(batch_size)
                if arrow.num_rows == 0:
                    break
                for chunk in arrow.to_batches(max_chunksize=1000):
                    rows = []
                    for record in chunk.to_pylist():
                        row = tuple(portable_value(record[name]) for name in columns)
                        digest.add(row)
                        rows.append(row)
                    target_cursor.executemany(insert, rows)
                target_db.commit()
                if digest.count >= next_progress:
                    print(f'{table}: copied {digest.count:,} / {expected_rows:,} rows', flush=True)
                    next_progress = digest.count + progress_rows
        if digest.count != expected_rows:
            raise ValueError(f'{table}: source count changed or rows were lost during copy')
        observed = target_digest(target_db, stage, columns, batch_size)
        if observed.public() != digest.public():
            raise ValueError(f'{table}: full-row multiset digest mismatch; staging table not published')
        after_rows = source_count(source_db, source)
        if after_rows != expected_rows:
            raise ValueError(f'{table}: source row count changed during migration')
        with target_db.cursor() as cursor:
            cursor.execute(f'RENAME TABLE {identifier(stage)} TO {identifier(table)}')
        target_db.commit()
        report = {
            'table': table, 'source': f'{catalog}.{source_schema_name}.{table}',
            'target': f'{target_database}.{table}', 'source_rows_before': expected_rows,
            'source_rows_after': after_rows, 'target_rows': observed.count,
            'columns': [{'name': name, 'source_type': str(field.type), 'target_type': kind}
                        for name, field, kind in zip(columns, schema, types)],
            'digest': digest.public(), 'seconds': round(time.monotonic() - started, 3),
            'timestamp_policy': 'UTC timestamp stored as DATETIME(6)',
        }
        print(f'{table}: verified {observed.count:,} rows, published in {report["seconds"]}s',
              flush=True)
        return report
    except BaseException:
        target_db.rollback()
        with target_db.cursor() as cursor:
            cursor.execute(f'DROP TABLE IF EXISTS {identifier(stage)}')
        target_db.commit()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', nargs='+', default=list(DEFAULT_TABLES))
    parser.add_argument('--catalog', default='workspace')
    parser.add_argument('--source-schema', default='default')
    parser.add_argument('--target-database', default='teleai_default')
    parser.add_argument('--batch-size', type=int, default=20_000)
    parser.add_argument('--progress-rows', type=int, default=100_000)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100_000 or args.progress_rows < 1:
        parser.error('batch size must be 1..100000 and progress rows positive')
    for name in [args.catalog, args.source_schema, args.target_database, *args.tables]:
        identifier(name)
    load_dotenv(ROOT / '.env')
    config = ConnectionConfig.from_env()
    credentials = ROOT / '.telly_runtime/mysql_eval/root.cnf'
    report_file = ROOT / '.telly_runtime/mysql_eval/migration_report.json'
    reports = json.loads(report_file.read_text()) if report_file.exists() else {}
    with source_connection(config) as source_db, mysql_connection(credentials, args.target_database) as target_db:
        for table in args.tables:
            reports[table] = copy_table(source_db, target_db,
                catalog=args.catalog, source_schema_name=args.source_schema,
                target_database=args.target_database, table=table, batch_size=args.batch_size,
                progress_rows=args.progress_rows)
            temporary = report_file.with_suffix('.tmp')
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                json.dump(reports, output, ensure_ascii=False, indent=2)
            temporary.chmod(0o600)
            temporary.replace(report_file)
    print('Completed tables: ' + ', '.join(args.tables))


if __name__ == '__main__':
    main()
