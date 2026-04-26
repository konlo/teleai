from __future__ import annotations

from typing import Any, List


TABLE_TRAINING_TOP_VALUE_LIMIT = 5


def quote_identifier(identifier: str) -> str:
    return f"`{str(identifier).replace('`', '``')}`"


def quote_sql_literal(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_bulk_profile_stats_sql(
    table_ref: str,
    column_names: List[str],
    *,
    include_min_max: bool = True,
) -> tuple[str, dict[str, dict[str, str]]]:
    select_parts = ["COUNT(*) AS `__row_count`"]
    aliases: dict[str, dict[str, str]] = {}
    for idx, column_name in enumerate(column_names):
        column_ref = quote_identifier(column_name)
        prefix = f"c{idx}"
        column_aliases = {
            "null_count": f"{prefix}_null_count",
            "distinct_count": f"{prefix}_distinct_count",
        }
        select_parts.append(
            f"SUM(CASE WHEN {column_ref} IS NULL THEN 1 ELSE 0 END) "
            f"AS `{column_aliases['null_count']}`"
        )
        select_parts.append(f"COUNT(DISTINCT {column_ref}) AS `{column_aliases['distinct_count']}`")
        if include_min_max:
            column_aliases["min_value"] = f"{prefix}_min_value"
            column_aliases["max_value"] = f"{prefix}_max_value"
            select_parts.append(f"MIN({column_ref}) AS `{column_aliases['min_value']}`")
            select_parts.append(f"MAX({column_ref}) AS `{column_aliases['max_value']}`")
        aliases[column_name] = column_aliases
    sql = "SELECT\n  " + ",\n  ".join(select_parts) + f"\nFROM {table_ref}"
    return sql, aliases


def build_bulk_top_values_sql(
    table_ref: str,
    column_names: List[str],
    *,
    limit: int = TABLE_TRAINING_TOP_VALUE_LIMIT,
) -> str:
    value_count_queries = []
    for column_name in column_names:
        column_ref = quote_identifier(column_name)
        value_count_queries.append(
            "SELECT "
            f"{quote_sql_literal(column_name)} AS column_name, "
            f"CAST({column_ref} AS STRING) AS value, "
            "COUNT(*) AS stat_count "
            f"FROM {table_ref} "
            f"WHERE {column_ref} IS NOT NULL "
            f"GROUP BY {column_ref}"
        )
    union_sql = "\nUNION ALL\n".join(value_count_queries)
    return (
        "WITH value_counts AS (\n"
        f"{union_sql}\n"
        "), ranked AS (\n"
        "  SELECT column_name, value, stat_count,\n"
        "         ROW_NUMBER() OVER (PARTITION BY column_name ORDER BY stat_count DESC, value ASC) AS rn\n"
        "  FROM value_counts\n"
        ")\n"
        "SELECT column_name, value, stat_count\n"
        "FROM ranked\n"
        f"WHERE rn <= {int(limit)}\n"
        "ORDER BY column_name, stat_count DESC, value ASC"
    )


__all__ = [
    "TABLE_TRAINING_TOP_VALUE_LIMIT",
    "build_bulk_profile_stats_sql",
    "build_bulk_top_values_sql",
    "quote_identifier",
    "quote_sql_literal",
]
