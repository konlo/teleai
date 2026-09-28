"""Bounded, schema-neutral checks before a remote SQL approval proposal."""
from __future__ import annotations

from sqlglot import exp

from core.analysis_catalog import table_context_freshness
from core.analysis_sql import validate_query


def known_column_error(query: str, reference_context: list[dict], *, dialect='databricks'):
    """Return a concrete error for columns disproved by current table metadata.

    This is intentionally not a full SQL type checker. Qualified references
    are checked against their bound table, even inside a CTE. Unqualified
    references are checked only for a single base table, because SELECT aliases
    and nested scopes can otherwise be mistaken for source columns. Missing or
    stale metadata never becomes fabricated proof that a column exists.
    """
    if not reference_context:
        return None
    tree = validate_query(query, dialect=dialect)
    cte_names = {cte.alias.casefold() for cte in tree.find_all(exp.CTE) if cte.alias}
    tables = [table for table in tree.find_all(exp.Table)
              if not (not table.db and not table.catalog
                      and table.name.casefold() in cte_names)]
    if not tables:
        return None

    def source_name(table):
        return '.'.join(part for part in (table.catalog, table.db, table.name) if part).casefold()

    def context_for(table):
        source = source_name(table)
        exact = [item for item in reference_context
                 if str(item.get('table', '')).strip('`"').casefold() == source]
        matches = exact or [item for item in reference_context
                            if str(item.get('table', '')).strip('`"').casefold().split('.')[-1]
                            == table.name.casefold()]
        return matches[0] if len(matches) == 1 else None

    aliases = {}
    for table in tables:
        alias = table.alias_or_name.casefold()
        if alias in aliases:
            return None  # Nested scopes may reuse aliases; do not guess binding.
        aliases[alias] = context_for(table)

    simple_unqualified = (len(aliases) == 1 and not cte_names
                          and len(list(tree.find_all(exp.Select))) == 1)
    projection_aliases = {alias.alias.casefold() for alias in tree.find_all(exp.Alias)
                          if alias.alias}

    for column in tree.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            continue
        if column.table:
            context = aliases.get(column.table.casefold())
        elif (simple_unqualified and
              not (column.name.casefold() in projection_aliases
                   and column.find_ancestor(exp.Order, exp.Group))):
            context = next(iter(aliases.values()))
        else:
            continue
        if context is None or (context.get('freshness') != 'current_loaded_schema'
                              and table_context_freshness(context) != 'fresh'):
            continue
        names = {str(item.get('name', '')).casefold() for item in context.get('columns', [])}
        if column.name.casefold() not in names:
            return {'error_code':'unknown_column',
                    'message':f"{column.sql(dialect=dialect)} is absent from the observed schema of {context.get('table')}",
                    'available_columns':sorted(names)[:64]}
    return None
