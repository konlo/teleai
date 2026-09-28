"""Discover declared database relationships without inferring keys from names.

Catalog constraints describe a join candidate, not proof of row completeness,
business intent, or actual cardinality. Execution always uses the SQL gate.
"""
from collections import defaultdict
from copy import deepcopy
import json
from hashlib import sha256
from numbers import Integral

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from core.analysis_catalog import _source_key, _quoted_table, table_context_freshness
from utils.analysis_datasets import project_dataset, full_read_preflight


FIELDS = ('constraint_name', 'source_column', 'target_catalog', 'target_schema',
          'target_table', 'target_column', 'ordinal_position')


def _clean_keys(keys):
    if not isinstance(keys, list) or len(keys) > 64:
        return None
    result = []
    for key in keys:
        if not isinstance(key, dict):
            return None
        columns, targets = key.get('columns'), key.get('target_columns')
        if (not isinstance(columns, list) or not isinstance(targets, list)
                or not 1 <= len(columns) == len(targets) <= 64
                or not all(isinstance(v, str) and v and len(v) <= 256
                           for v in [key.get('target_table'), *columns, *targets])
                or len(set(columns)) != len(columns) or len(set(targets)) != len(targets)):
            return None
        result.append({'name': str(key.get('name', ''))[:256], 'columns': columns,
                       'target_table': key['target_table'], 'target_columns': targets})
    # Never hide extra candidate keys by truncating: ambiguity must be retained.
    return result if len(json.dumps(result, ensure_ascii=False)) <= 16000 else None


def relationship_plan(table):
    parts = _source_key(table).split('.')
    if len(parts) != 3 or any(not p or any(ch in p for ch in "`\"'\\\n\r;") for p in parts):
        return {'status': 'needs_context', 'message': 'catalog.schema.table 전체 이름이 필요합니다.'}
    catalog, schema, name = parts
    rc = catalog + '.information_schema.referential_constraints'
    ku = catalog + '.information_schema.key_column_usage'
    query = (
        'SELECT r.constraint_name, f.column_name AS source_column, '
        'p.table_catalog AS target_catalog, p.table_schema AS target_schema, '
        'p.table_name AS target_table, p.column_name AS target_column, f.ordinal_position '
        f'FROM {_quoted_table(rc)} r JOIN {_quoted_table(ku)} f '
        'ON r.constraint_catalog = f.constraint_catalog '
        'AND r.constraint_schema = f.constraint_schema AND r.constraint_name = f.constraint_name '
        f'JOIN {_quoted_table(ku)} p ON r.unique_constraint_catalog = p.constraint_catalog '
        'AND r.unique_constraint_schema = p.constraint_schema '
        'AND r.unique_constraint_name = p.constraint_name '
        'AND f.position_in_unique_constraint = p.ordinal_position '
        f"WHERE f.table_catalog = '{catalog}' AND f.table_schema = '{schema}' "
        f"AND f.table_name = '{name}' ORDER BY r.constraint_name, f.ordinal_position LIMIT 65")
    return {'status': 'planned', 'metadata_plan': {'source': ' | '.join(sorted([rc, ku])),
        'query': query, 'reason': '선택 테이블에 선언된 외래 키와 참조 키를 확인합니다.'},
        'target_table': _source_key(table), 'scope': '같은 catalog의 참조 키 관계 metadata 최대 65행. 다른 catalog 참조와 원본 행은 포함하지 않습니다.',
        'user_action': '정확한 SQL을 query_databricks로 제안하고 승인 후 실행하세요.'}


def stored_relationships(datasets, table):
    plan = relationship_plan(table)
    if plan['status'] != 'planned':
        return None
    matches = [d for d in datasets.metadata.values()
               if d.query == plan['metadata_plan']['query']
               and sorted(_source_key(s) for s in d.source.split(' | '))
               == sorted(plan['metadata_plan']['source'].split(' | '))]
    if not matches:
        return None
    info = matches[-1]
    if (not 0 <= info.rows < 65 or set(info.columns) != set(FIELDS)
            or table_context_freshness({'observed_at': info.snapshot}) != 'fresh'
            or full_read_preflight(datasets, [info.id])):
        return None
    grouped = defaultdict(list)
    for row in project_dataset(datasets, info.id, list(FIELDS)).to_dict('records'):
        if not all(isinstance(row[k], str) and row[k] and len(row[k]) <= 256 for k in FIELDS[:-1]):
            return None
        grouped[row['constraint_name']].append(row)
    keys = []
    for name, rows in grouped.items():
        if any(isinstance(r['ordinal_position'], bool)
               or not isinstance(r['ordinal_position'], Integral) for r in rows):
            return None
        if [r['ordinal_position'] for r in rows] != list(range(1, len(rows) + 1)):
            return None
        targets = {'.'.join(r[k] for k in ('target_catalog', 'target_schema', 'target_table')) for r in rows}
        if len(targets) != 1:
            return None
        keys.append({'name': name, 'columns': [r['source_column'] for r in rows],
                     'target_table': targets.pop(), 'target_columns': [r['target_column'] for r in rows]})
    keys = _clean_keys(keys)
    if keys is None:
        return None
    return {'table': _source_key(table), 'foreign_keys': keys, 'observed_at': info.snapshot,
            'relationship_authority': 'database_catalog', 'definition_dataset_id': info.id}


def relationship_catalog(context):
    """Only fresh, bounded external observations can supply relationship facts."""
    contexts = defaultdict(list)
    for item in context.reference_context:
        contexts[_source_key(item.get('table', ''))].append(item)
    accepted = {}
    for table, entries in contexts.items():
        if len(entries) != 1 or table_context_freshness(entries[0]) != 'fresh':
            continue
        item = deepcopy(entries[0])
        observed = stored_relationships(context.datasets, table)
        if observed:
            item.update(observed)
        if item.get('relationship_authority') != 'database_catalog':
            continue
        keys = _clean_keys(item.get('foreign_keys', []))
        if keys is None:
            continue
        item['foreign_keys'] = keys
        accepted[table] = item
    return accepted


def inspect_relationships(context, table):
    table = _source_key(table)
    known = {_source_key(c.get('table', '')) for c in context.reference_context}
    known.update(_source_key(d.source) for d in context.datasets.metadata.values())
    if table not in known:
        return {'status': 'needs_context', 'message': '먼저 실제 테이블 이름과 스키마를 확인하세요.'}
    item = stored_relationships(context.datasets, table) or relationship_catalog(context).get(table)
    if item is not None:
        keys = item.get('foreign_keys', [])
        return {'status': 'ready', 'table': table, 'foreign_keys': keys,
            'observed_at': item['observed_at'],
            'version': sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest(),
            'scope': 'DB에 선언된 관계입니다. 실제 행의 유일성·누락·업무 역할을 보증하지 않습니다.',
            'message': '여러 키가 같은 테이블을 가리키면 역할을 확인하세요. 새 SQL은 별도 승인이 필요합니다.',
            'next_steps': [
                '사용자가 지정한 테이블·조건의 실제 컬럼을 inspect_table_context에서 확인하세요.',
                '양쪽 raw dataset이 이미 로딩되어 있으면 보유 dataset ID로 로컬 조인을 검토하세요.',
                '일부 테이블이 로딩되지 않은 통계 요청은 연결된 SQL 엔진에서 JOIN/집계 SELECT를 query_databricks로 승인 요청할 수 있습니다. 전체 raw 로딩은 필수가 아닙니다.',
                '명시된 컬럼·리터럴 조건과 최신 스키마로 의미가 확정됐으면 추가 설명이나 catalog 탐색 없이 계산 계획으로 진행하세요.'
            ]}
    return relationship_plan(table)


def metadata_join_scope(query, context, requested, *, dialect, required_sources=None):
    """Ground a small unambiguous inner-join tree in fresh catalog constraints.

    Never replaces explicit user edges. Repeated table roles, outer joins,
    derived sources, non-key ON filters and alternative FKs remain unsupported.
    The returned scope still has to pass the existing exact predicate validator.
    """
    if requested.get('join_edges') or requested.get('unresolved'):
        return None
    try:
        tree = parse_one(query, read=dialect)
        if (not isinstance(tree, exp.Select) or tree.args.get('with_')
                or len(list(tree.find_all(exp.Select))) != 1):
            return None
        source = tree.args.get('from_')
        tables = [source.this] if source else []
        joins = tree.args.get('joins') or []
        tables += [j.this for j in joins]
        if not 2 <= len(tables) <= 4 or not all(isinstance(t, exp.Table) for t in tables):
            return None
        # Use table parts, independent of alias spelling or quoting.
        aliases = {t.alias_or_name: _source_key('.'.join(p.name for p in t.parts)) for t in tables}
        if len(aliases) != len(tables) or len(set(aliases.values())) != len(tables):
            return None
        if required_sources is not None and (
                len(required_sources) < 2
                or {_source_key(source) for source in required_sources} != set(aliases.values())):
            return None
        catalog = relationship_catalog(context)
        schemas = {}
        for item in context.reference_context:
            key = _source_key(item.get('table', ''))
            if key in aliases.values() and table_context_freshness(item) == 'fresh':
                if key in schemas:
                    return None
                schemas[key] = {c['name'] for c in item.get('columns', [])}
        if set(schemas) != set(aliases.values()):
            return None
        edges = []
        seen = {tables[0].alias_or_name}
        for join in joins:
            if join.side or join.kind not in ('', 'INNER') or not join.args.get('on'):
                return None
            terms = list(join.args['on'].flatten()) if isinstance(join.args['on'], exp.And) else [join.args['on']]
            pairs = []
            for term in terms:
                if not isinstance(term, exp.EQ) or not all(isinstance(n, exp.Column) for n in (term.this, term.expression)):
                    return None
                left, right = term.this, term.expression
                if left.table == join.this.alias_or_name:
                    left, right = right, left
                if left.table not in seen or right.table != join.this.alias_or_name:
                    return None
                if left.name not in schemas[aliases[left.table]] or right.name not in schemas[aliases[right.table]]:
                    return None
                pairs.append((left, right))
            if not pairs or len({(a.table, b.table) for a, b in pairs}) != 1:
                return None
            a, b = pairs[0]
            ta, tb = aliases[a.table], aliases[b.table]
            possibilities = []
            for origin, target in ((ta, tb), (tb, ta)):
                for fk in catalog.get(origin, {}).get('foreign_keys', []):
                    if _source_key(fk.get('target_table', '')) == target:
                        cols, refs = fk.get('columns', []), fk.get('target_columns', [])
                        if not cols or len(cols) != len(refs) or len(set(cols)) != len(cols) or len(set(refs)) != len(refs):
                            return None
                        possibilities.append({(c, r) if origin == ta else (r, c) for c, r in zip(cols, refs)})
            if len(possibilities) != 1 or {(a.name, b.name) for a, b in pairs} != possibilities[0]:
                return None
            edges.extend(sorted([a.table + '.' + a.name, b.table + '.' + b.name]) for a, b in pairs)
            seen.add(join.this.alias_or_name)
        augmented = deepcopy(requested)
        augmented['join_edges'] = edges
        for field in ('conditions', 'any_conditions', 'measure_conditions'):
            for condition in augmented.get(field, []):
                column = condition['column']
                if '.' not in column:
                    owners = [alias for alias, table in aliases.items() if column in schemas[table]]
                    if len(owners) != 1:
                        return None
                    condition['column'] = owners[0] + '.' + column
        return augmented
    except (ValueError, TypeError, KeyError, AttributeError, SqlglotError):
        return None
