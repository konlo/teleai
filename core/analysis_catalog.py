"""Dynamic table discovery metadata; table-specific facts stay outside code."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError


DEFAULT_CONTEXT_MAX_AGE_SECONDS = 24 * 60 * 60


def _source_key(value):
    from utils.analysis_provenance import single_table, table_identity
    try:
        table = single_table(parse_one('SELECT * FROM ' + str(value), read='databricks'))
        return table_identity(table).casefold() if table is not None else str(value).casefold()
    except (TypeError, ValueError, SqlglotError):
        return str(value).strip().casefold()


def _observed_at(item):
    return item.get('observed_at') or item.get('trained_at')


def table_context_freshness(item, *, now=None, max_age_seconds=None):
    """Classify metadata without pretending a cached profile is live schema."""
    stamp = _observed_at(item)
    if not stamp:
        return 'unknown'
    try:
        parsed = datetime.fromisoformat(str(stamp).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return 'unknown'
    if max_age_seconds is None:
        try:
            max_age_seconds = int(os.getenv(
                'TELLY_TABLE_CONTEXT_MAX_AGE_SECONDS', DEFAULT_CONTEXT_MAX_AGE_SECONDS))
        except ValueError:
            max_age_seconds = DEFAULT_CONTEXT_MAX_AGE_SECONDS
    current = now or datetime.now(timezone.utc)
    return 'fresh' if (current - parsed).total_seconds() <= max(0, max_age_seconds) else 'stale'


def schema_fingerprint(columns):
    normalized = [(str(column.get('name', '')).casefold(), str(column.get('dtype', '')).casefold())
                  for column in columns if isinstance(column, dict) and column.get('name')]
    return sha256(json.dumps(normalized, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def enrich_reference_context(item):
    enriched = dict(item)
    enriched['observed_at'] = _observed_at(enriched)
    enriched['freshness'] = table_context_freshness(enriched)
    enriched['schema_fingerprint'] = schema_fingerprint(enriched.get('columns', []))
    return enriched


def discovered_reference_context(datasets, *, dialect='databricks'):
    """Discover names from bounded, fresh remote inventory; never infer columns."""
    from utils.analysis_datasets import full_read_preflight, project_dataset
    from utils.analysis_provenance import single_table, table_identity
    if dialect not in {'databricks', 'mysql'}:
        return []
    found = {}
    for info in datasets.metadata.values():
        source = _source_key(info.source)
        if (not source.endswith('information_schema.tables') or info.parent_id
                or not 0 < info.rows <= 200 or not info.query
                or table_context_freshness({'observed_at':info.snapshot}) != 'fresh'):
            continue
        try:
            tree = parse_one(info.query, read=dialect)
            table = single_table(tree)
            limit = tree.args.get('limit')
            if (table is None or _source_key(table_identity(table)) != source
                    or not limit or not limit.expression.is_int
                    or not 1 <= int(limit.expression.this) <= 200):
                continue
            fields = {str(column).casefold():column for column in info.columns}
            names = ['table_schema', 'table_name']
            if dialect == 'databricks':
                names.insert(0, 'table_catalog')
            projections = {}
            for expression in tree.expressions:
                column = expression.this if isinstance(expression, exp.Alias) else expression
                if isinstance(column, exp.Column):
                    projections[expression.alias_or_name.casefold()] = column.name.casefold()
            if any(projections.get(name) != name for name in names):
                continue
            if not all(name in fields for name in names) or full_read_preflight(datasets, [info.id]):
                continue
            frame = project_dataset(datasets, info.id, [fields[name] for name in names])
            for values in frame.itertuples(index=False, name=None):
                if not all(isinstance(value, str) and value and len(value) <= 255
                           and not any(char in value for char in "`.\\;'\"\x00\n\r")
                           for value in values):
                    continue
                if dialect == 'databricks' and values[0].casefold() != source.split('.')[0]:
                    continue
                name = '.'.join(values)
                found[_source_key(name)] = {'table':name, 'columns':[],
                    'training_status':'discovered_name', 'observed_at':info.snapshot}
        except (TypeError, ValueError, KeyError, OSError, SqlglotError):
            continue
    return list(found.values())


def full_schema_source(query, *, dialect='databricks'):
    """Return the base table only for a sole, unmodified identity wildcard.

    COUNT(*), wildcard EXCEPT/REPLACE, joined schemas and computed projections
    describe a result, not the full source schema. Never give them authority
    to remove or invent columns in the table context.
    """
    from utils.analysis_provenance import single_table
    try:
        tree = parse_one(query, read=dialect)
    except (TypeError, ValueError, SqlglotError):
        return None
    table = single_table(tree)
    if table is None or len(tree.expressions) != 1:
        return None
    projection = tree.expressions[0]
    if isinstance(projection, exp.Column):
        if projection.table.casefold() != table.alias_or_name.casefold():
            return None
        projection = projection.this
    if not isinstance(projection, exp.Star) or any(projection.args.values()):
        return None
    return table


def _is_full_schema_observation(info):
    """Only a matching remote identity projection can refresh source schema."""
    from utils.analysis_provenance import table_identity
    if getattr(info, 'parent_id', ''):
        return False
    table = full_schema_source(info.query)
    return bool(table is not None and _source_key(table_identity(table)) == _source_key(info.source))


def _is_zero_row_schema_probe(query):
    if full_schema_source(query) is None:
        return False
    try:
        tree = parse_one(query, read='databricks')
    except (TypeError, ValueError, SqlglotError):
        return False
    limit = tree.args.get('limit') if isinstance(tree, exp.Select) else None
    return bool(limit and isinstance(limit.expression, exp.Literal)
                and limit.expression.is_int and int(limit.expression.this) == 0)


def _quoted_table(table):
    parts = [part.strip('` ') for part in str(table).split('.') if part.strip('` ')]
    if not 1 <= len(parts) <= 3:
        raise ValueError('테이블 이름을 확인해주세요.')
    return '.'.join('`' + part.replace('`', '``') + '`' for part in parts)


def resolve_table_context(reference_context, datasets, table):
    """Prefer an approved full-schema observation over a cached profile."""
    requested = _source_key(table)
    known_sources = {_source_key(item.get('table', '')) for item in reference_context}
    known_sources.update(_source_key(info.source) for info in datasets.metadata.values())
    if requested in known_sources:
        wanted = requested
    else:
        short_matches = {source for source in known_sources
                         if source.rsplit('.', 1)[-1] == requested.rsplit('.', 1)[-1]}
        if len(short_matches) != 1:
            return {'status':'needs_context',
                    'message':'저장된 테이블 정보가 없거나 짧은 이름이 모호합니다. catalog.schema.table 전체 이름을 지정하세요.'}
        wanted = next(iter(short_matches))
    matches = [enrich_reference_context(item) for item in reference_context
               if _source_key(item.get('table', '')) == wanted]
    if len(matches) > 1:
        return {'status':'needs_context',
                'message':'같은 이름의 테이블 정보가 여러 개입니다. catalog.schema.table 전체 이름을 지정하세요.'}
    saved = matches[0] if matches else None
    observations = [info for info in datasets.metadata.values()
                    if _source_key(info.source) == wanted and _is_full_schema_observation(info)]
    if observations:
        latest = observations[-1]
        schema_only = _is_zero_row_schema_probe(latest.query)
        saved_columns = {column.get('name'):column for column in (saved or {}).get('columns', [])
                         if isinstance(column, dict) and column.get('name')}
        # SQL types and dataframe storage types are distinct evidence. Attach
        # current DB types only for an actual metadata observation made after
        # this result, with exactly the same column identities. Cached training
        # profiles and mismatched/stale schemas cannot supply SQL type claims.
        database_types = False
        if (saved and saved.get('training_status') in {'observed_schema','approved_column_metadata'}
                and saved.get('freshness') == 'fresh'
                and set(saved_columns) == set(latest.columns)):
            try:
                db_stamp=datetime.fromisoformat(str(_observed_at(saved)).replace('Z','+00:00'))
                result_stamp=datetime.fromisoformat(str(latest.snapshot).replace('Z','+00:00'))
                database_types=db_stamp >= result_stamp
            except (ValueError,TypeError):pass
        try:
            inspection = datasets.inspect(latest.id) if hasattr(datasets, 'inspect') else {}
            actual_dtypes = (inspection['dtypes'] if inspection else
                            {str(name):str(dtype) for name, dtype in datasets.frames[latest.id].dtypes.items()})
            storage_dtypes = inspection.get('storage_dtypes', {})
        except (KeyError, OSError, ValueError, TypeError):
            actual_dtypes = {}
            storage_dtypes = {}
        columns = []
        for name in latest.columns:
            column = dict(saved_columns.get(name, {'name':name, 'dtype':''}))
            column['name'] = name
            # Never carry an old dtype across an approved schema observation.
            # An empty pandas/Parquet frame may report a generic object dtype
            # even when the SQL column is numeric. Preserve only explicit type
            # evidence supplied by the result schema.
            observed_dtype = str(actual_dtypes.get(name, '') or '')
            # Empty pandas object columns cannot prove a type, but a typed
            # Parquet footer can. Do not discard the connector's Arrow string
            # schema merely because its pandas equivalent is object.
            if schema_only and observed_dtype.casefold() in {'', 'object', 'unknown', 'null', 'none'}:
                storage_dtype = str(storage_dtypes.get(name, '') or '')
                if storage_dtype.casefold() not in {'', 'null', 'unknown', 'none'}:
                    observed_dtype = storage_dtype
            column['dtype'] = ('' if schema_only and observed_dtype.casefold()
                               in {'', 'object', 'unknown', 'null', 'none'}
                               else observed_dtype)
            column.pop('database_dtype',None)
            if database_types and saved_columns[name].get('dtype'):
                column['database_dtype']=str(saved_columns[name]['dtype'])
            columns.append(column)
        current = {**(saved or {}), 'table':latest.source, 'columns':columns,
                   'training_status':'runtime_schema', 'freshness':'current_loaded_schema',
                   'observed_at':latest.snapshot or None,
                   'schema_fingerprint':schema_fingerprint(columns),
                   'dataset_id':latest.id}
        previous = (saved or {}).get('schema_fingerprint') or schema_fingerprint((saved or {}).get('columns', []))
        if database_types:
            # Matching fresh SQL metadata and a loaded object dtype merely use
            # different type systems; that is not evidence of schema drift.
            schema_changed = False
        elif schema_only and any(not column['dtype'] for column in columns):
            old_names = [str(column.get('name', '')).casefold()
                         for column in (saved or {}).get('columns', [])]
            new_names = [str(name).casefold() for name in latest.columns]
            schema_changed = bool(saved and old_names != new_names)
        else:
            schema_changed = bool(saved and previous != current['schema_fingerprint'])
        return {'status':'ready', 'table_context':current,
                'schema_changed':schema_changed,
                'authority':'approved_select_star_result',
                'database_type_authority': ('current_database_metadata' if database_types else None),
                'scope':('실행된 0행 조회의 실제 컬럼명입니다. 일부 데이터 타입은 확인되지 않았습니다.'
                         if schema_only and any(not column['dtype'] for column in columns) else
                         '실행된 0행 결과 스키마의 실제 컬럼명과 데이터 타입입니다.'
                         if schema_only else
                         '로딩된 SELECT * 결과의 실제 컬럼입니다. 해당 결과의 생성 시점 스키마를 나타냅니다.')}
    if saved is None:
        return {'status':'needs_context',
                'message':'저장되거나 조회로 확인된 테이블 정보가 없습니다. 정확한 테이블명을 확인한 뒤 읽기 전용 스키마 조회로 확인하세요.'}
    names_only = saved.get('training_status') == 'discovered_name'
    if saved['freshness'] == 'stale' or names_only:
        try:
            refresh_query = 'SELECT * FROM ' + _quoted_table(saved.get('table', table)) + ' LIMIT 0'
        except ValueError:
            refresh_query = ''
        return {'status':'needs_refresh', 'table_context':saved,
                'refresh_query':refresh_query,
                'message':('조회로 테이블 이름만 확인했습니다. 컬럼을 추측하지 말고 0행 스키마 조회로 확인하세요.'
                           if names_only else '저장된 스키마 스냅샷이 오래되어 현재 컬럼이라고 보장할 수 없습니다. 이 목록으로 새 분석 SQL을 만들지 말고 읽기 전용 스키마 조회로 갱신하세요.'),
                'scope':'과거 스키마 스냅샷이며 현재 테이블 구조를 보장하지 않습니다.'}
    return {'status':'ready', 'table_context':saved, 'schema_changed':False,
            'authority':'saved_snapshot',
            'scope':'저장된 스키마 스냅샷입니다. observed_at과 freshness를 함께 확인해야 합니다.'}


def grounded_reference_columns(item, datasets):
    """Stale aliases are usable only for columns proven by a loaded result."""
    enriched = enrich_reference_context(item)
    if enriched['freshness'] != 'stale':
        return enriched.get('columns', [])
    identity = _source_key(enriched.get('table', ''))
    actual = {column for info in datasets.metadata.values()
              if _source_key(info.source) == identity for column in info.columns}
    return [column for column in enriched.get('columns', [])
            if isinstance(column, dict) and column.get('name') in actual]


def load_saved_reference_context(storage_dir):
    """Load the same bounded external table context for the page and evals."""
    import json
    from pathlib import Path
    from utils.table_context import load_saved_table_context
    result = []
    for path in sorted((Path(storage_dir) / 'contexts').glob('*.json')):
        try:
            raw = json.loads(path.read_text())
            saved = load_saved_table_context(raw['table_fqn'], storage_dir=storage_dir)
            if saved:
                result.append(enrich_reference_context({'table':saved.table_fqn, 'training_status':saved.training_status,
                    'trained_at':saved.trained_at, 'observed_at':saved.trained_at,
                    'columns':[{'name':c.name, 'dtype':c.dtype, 'aliases':c.aliases, 'description':c.description,
                                'top_values':c.top_values[:10]} for c in saved.columns]}))
        except (OSError, ValueError, KeyError):
            continue
    return result


def compact_catalog(catalog):
    return {**catalog, 'available_tables': [
        {'table': item.get('table'), 'training_status': item.get('training_status'),
         # A stale snapshot can identify a candidate table, but its old column
         # count must not be presented as the current database schema.
         'column_count': (len(item.get('columns', []))
                          if table_context_freshness(item) == 'fresh' else None),
         'freshness': enrich_reference_context(item).get('freshness'),
         'observed_at': _observed_at(item),
         'schema_fingerprint': item.get('schema_fingerprint') or schema_fingerprint(item.get('columns', []))}
        for item in catalog.get('available_tables', [])]}
