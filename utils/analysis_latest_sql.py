"""Bounded local SQL selection over stored Arrow batches; no remote queries."""
from itertools import chain
from tempfile import TemporaryDirectory
from threading import Timer

import duckdb
import pyarrow as pa

MAX_SELECTED_ROWS = 100_000
MAX_SELECTED_BYTES = 64 * 1024 * 1024
QUERY_TIMEOUT_SECONDS = 30


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _issue(code, message):
    return None, {'status': 'needs_context', 'error_code': code, 'message': message}


def select_latest(store, info, columns, keys, order):
    batches = iter(store.frames.batches(info.id, columns, expected_rows=info.rows))
    try:
        first = next(batches, None)
        if first is None:
            return _issue('latest_empty', '조건에 해당하는 행이 없습니다.')
        # Ordering text as timestamps needs an explicit normalization policy;
        # numeric and native date/time columns have unambiguous SQL ordering.
        for column in order:
            dtype = first.schema.field(column).type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
                    or pa.types.is_decimal(dtype) or pa.types.is_timestamp(dtype)
                    or pa.types.is_date(dtype)):
                return _issue('latest_order_type', '저장된 정렬 컬럼을 시간 또는 수치 자료형으로 변환한 뒤 선택해주세요.')
        if any(pa.types.is_nested(field.type) for field in first.schema):
            return _issue('latest_column_type', '키·정렬·분포 컬럼은 단일 값 자료형이어야 합니다.')
        names = ', '.join(map(_quote, columns))
        partition = ', '.join(map(_quote, keys))
        ordering = ', '.join(_quote(c) + ' DESC' for c in order)
        with TemporaryDirectory(prefix='telly-latest-sql-') as temporary:
            with duckdb.connect(config={'enable_external_access': False, 'threads': 2,
                    'memory_limit': '128MB', 'temp_directory': temporary,
                    'max_temp_directory_size': '512MB', 'preserve_insertion_order': False}) as conn:
                timer = Timer(QUERY_TIMEOUT_SECONDS, conn.interrupt)
                timer.daemon = True
                timer.start()
                try:
                    reader = pa.RecordBatchReader.from_batches(first.schema, chain([first], batches))
                    conn.register('input_batches', reader)
                    conn.execute('CREATE TEMP TABLE data AS SELECT ' + names + ' FROM input_batches')
                    null_test = ' OR '.join(_quote(c) + ' IS NULL' for c in columns)
                    # NaN is a missing value in the pandas path; preserve parity.
                    floats = [f.name for f in first.schema if pa.types.is_floating(f.type)]
                    if floats:
                        null_test += ' OR ' + ' OR '.join('isnan(' + _quote(c) + ')' for c in floats)
                    if conn.execute('SELECT EXISTS(SELECT 1 FROM data WHERE ' + null_test + ')').fetchone()[0]:
                        return _issue('latest_null_policy', '키·정렬·분포 컬럼에 결측값이 있습니다. 결측값 처리 기준을 알려주세요.')
                    # DENSE_RANK retains tied winners so ties cannot be hidden by LIMIT/ROW_NUMBER.
                    query = ('SELECT ' + names + ' FROM data QUALIFY DENSE_RANK() OVER (PARTITION BY '
                             + partition + ' ORDER BY ' + ordering + ') = 1 LIMIT '
                             + str(MAX_SELECTED_ROWS + 1))
                    result = conn.execute(query).to_arrow_reader(1024)
                    parts, rows, byte_count = [], 0, 0
                    for batch in result:
                        rows += batch.num_rows
                        byte_count += batch.nbytes
                        if rows > MAX_SELECTED_ROWS or byte_count > MAX_SELECTED_BYTES:
                            return _issue('latest_output_limit', '최신행 결과가 로컬 처리 한도를 넘습니다. 기간·대상 범위를 좁히거나 원격 집계가 필요합니다. 표본으로 대체하지 않았습니다.')
                        parts.append(batch)
                    selected = pa.Table.from_batches(parts, schema=result.schema).to_pandas()
                    if selected.duplicated(keys, keep=False).any():
                        return _issue('latest_order_tie', '같은 키에 최신 정렬값이 동일한 행이 여러 개입니다. 추가 정렬 컬럼을 알려주세요.')
                    return selected, None
                finally:
                    timer.cancel()
                    timer.join()
    except (duckdb.OutOfMemoryException, duckdb.InterruptException):
        return _issue('latest_resource_limit', '최신행 선택이 메모리 또는 시간 한도에 도달했습니다. 원본은 보존했으며 범위를 좁히거나 원격 집계가 필요합니다.')
    finally:
        batches.close()
