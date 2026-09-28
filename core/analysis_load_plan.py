"""Table-neutral checks binding an approved load to its actual SQL sources."""
from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from core.analysis_sql import validate_query
from utils.analysis_provenance import table_identity


@dataclass(frozen=True)
class SourcePlan:
    declared_source: str
    actual_tables: tuple[str, ...]
    grain: str
    bounded_result: bool
    expected_columns: tuple[str, ...]


def _parts(identity: str) -> tuple[str, ...]:
    return tuple(part.strip().strip('`"').casefold() for part in identity.split('.') if part.strip())


def _compatible(declared: str, actual: str) -> bool:
    wanted, observed = _parts(declared), _parts(actual)
    return bool(wanted and len(wanted) <= len(observed)
                and observed[-len(wanted):] == wanted)


def _actual_tables(tree) -> tuple[str, ...]:
    cte_names = {cte.alias.casefold() for cte in tree.find_all(exp.CTE) if cte.alias}
    return tuple(dict.fromkeys(
        table_identity(table) for table in tree.find_all(exp.Table)
        if not (not table.db and not table.catalog and table.name.casefold() in cte_names)))


def query_sources(query: str, *, dialect='databricks') -> tuple[str, ...]:
    """Report physical SQL sources without trusting a model-supplied label."""
    return _actual_tables(validate_query(query, dialect=dialect))


def source_plan(source: str, query: str, *, dialect='databricks') -> SourcePlan:
    """Reject misleading source labels before a query can enter approval.

    A short name may bind one fully qualified table, but multi-table queries
    must enumerate every source using `` | ``. No table/column names are fixed
    in code; the query AST supplies the actual source list.
    """
    tree = validate_query(query, dialect=dialect)
    actual = _actual_tables(tree)
    declared = tuple(part.strip() for part in source.split('|') if part.strip())
    if not declared:
        raise ValueError('조회 출처를 명시해야 합니다.')
    if actual:
        if len(declared) != len(actual):
            raise ValueError('승인 대상 출처와 SQL의 테이블 수가 다릅니다.')
        remaining = list(actual)
        for item in declared:
            match = next((table for table in remaining if _compatible(item, table)), None)
            if match is None:
                raise ValueError('승인 대상 출처와 SQL의 실제 테이블이 다릅니다.')
            remaining.remove(match)
    aggregated = bool(tree.args.get('group') or tree.find(exp.AggFunc))
    expected_columns = ()
    if isinstance(tree, exp.Select):
        projected = []
        for item in tree.expressions:
            expression = item.this if isinstance(item, exp.Alias) else item
            if isinstance(expression, exp.Star) or (isinstance(expression, exp.Column)
                    and isinstance(expression.this, exp.Star)):
                projected = []
                break
            name = item.alias_or_name
            if not name or name == '*' or not isinstance(item, (exp.Column, exp.Alias)):
                projected = []
                break
            projected.append(name)
        expected_columns = tuple(projected)
    return SourcePlan(source, actual, 'aggregate' if aggregated else 'raw',
                      tree.args.get('limit') is not None, expected_columns)
