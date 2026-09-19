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


def _is_full_schema_observation(info):
    """A SELECT * result proves the returned schema, even when it has zero rows."""
    if not info.query:
        return False
    try:
        tree = parse_one(info.query, read='databricks')
    except (TypeError, ValueError, SqlglotError):
        return False
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return False
    return any(isinstance(node, exp.Star) for expression in tree.expressions
               for node in expression.walk())


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
        saved_columns = {column.get('name'):column for column in (saved or {}).get('columns', [])
                         if isinstance(column, dict) and column.get('name')}
        try:
            frame = datasets.frames[latest.id]
            actual_dtypes = {str(name):str(dtype) for name, dtype in frame.dtypes.items()}
        except (KeyError, OSError, ValueError, TypeError):
            actual_dtypes = {}
        columns = []
        for name in latest.columns:
            column = dict(saved_columns.get(name, {'name':name, 'dtype':''}))
            column['name'] = name
            # Never carry an old dtype across an approved schema observation.
            column['dtype'] = actual_dtypes.get(name, '')
            columns.append(column)
        current = {**(saved or {}), 'table':latest.source, 'columns':columns,
                   'training_status':'runtime_schema', 'freshness':'current_loaded_schema',
                   'observed_at':latest.snapshot or None,
                   'schema_fingerprint':schema_fingerprint(columns),
                   'dataset_id':latest.id}
        previous = (saved or {}).get('schema_fingerprint') or schema_fingerprint((saved or {}).get('columns', []))
        return {'status':'ready', 'table_context':current,
                'schema_changed':bool(saved and previous != current['schema_fingerprint']),
                'authority':'approved_select_star_result',
                'scope':'승인 후 로딩된 SELECT * 결과의 실제 컬럼입니다. 해당 결과의 생성 시점 스키마를 나타냅니다.'}
    if saved is None:
        return {'status':'needs_context',
                'message':'저장되거나 승인 후 확인된 테이블 정보가 없습니다. 정확한 테이블명을 확인한 뒤 스키마 조회 승인을 받아야 합니다.'}
    if saved['freshness'] == 'stale':
        try:
            refresh_query = 'SELECT * FROM ' + _quoted_table(saved.get('table', table)) + ' LIMIT 0'
        except ValueError:
            refresh_query = ''
        return {'status':'needs_refresh', 'table_context':saved,
                'refresh_query':refresh_query,
                'message':'저장된 스키마 스냅샷이 오래되어 현재 컬럼이라고 보장할 수 없습니다. 이 목록으로 새 분석 SQL을 만들지 말고 스키마 조회 승인을 받아 갱신하세요.',
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
                    'columns':[{'name':c.name, 'dtype':c.dtype, 'aliases':c.aliases,
                                'top_values':c.top_values[:10]} for c in saved.columns]}))
        except (OSError, ValueError, KeyError):
            continue
    return result


def compact_catalog(catalog):
    return {**catalog, 'available_tables': [
        {'table': item.get('table'), 'training_status': item.get('training_status'),
         'column_count': len(item.get('columns', [])),
         'freshness': enrich_reference_context(item).get('freshness'),
         'observed_at': _observed_at(item),
         'schema_fingerprint': item.get('schema_fingerprint') or schema_fingerprint(item.get('columns', []))}
        for item in catalog.get('available_tables', [])]}
