"""
Utility helpers to talk to Databricks SQL Warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from databricks import sql as databricks_sql
except ImportError:  # pragma: no cover - handled at runtime
    databricks_sql = None


class DatabricksConnectorError(RuntimeError):
    """Raised when the Databricks SQL connector is not available."""


def _normalize_hostname(host: str) -> str:
    value = (host or "").strip()
    if value.startswith("https://"):
        value = value[len("https://") :]
    elif value.startswith("http://"):
        value = value[len("http://") :]
    return value.strip("/")


@dataclass
class DatabricksConfig:
    server_hostname: str
    http_path: str
    access_token: str
    catalog: Optional[str] = None
    schema: Optional[str] = None

    def __post_init__(self) -> None:
        self.server_hostname = _normalize_hostname(self.server_hostname)

    def table_identifier(self, table: str) -> str:
        """Compose a catalog.schema.table reference for the given table."""
        if table.strip() == "":
            raise ValueError("Table name must not be empty.")
        if "." in table:
            parts = [part.strip() for part in table.split(".") if part.strip()]
            if len(parts) < 1:
                raise ValueError(f"Invalid table identifier: {table}")
            return ".".join(_quote_identifier(part) for part in parts)
        return ".".join(
            _quote_identifier(part)
            for part in (self.catalog, self.schema, table)
            if part
        )

    def namespace_reference(self) -> Optional[str]:
        """Return catalog.schema (or schema) reference if available."""
        parts = [self.catalog, self.schema]
        parts = [p for p in parts if p]
        if not parts:
            return None
        return ".".join(_quote_identifier(part) for part in parts)


def connector_available() -> bool:
    """Return True when the databricks-sql-connector package is installed."""
    return databricks_sql is not None


def _ensure_connector() -> None:
    if databricks_sql is None:
        raise DatabricksConnectorError(
            "databricks-sql-connector is not installed. "
            "Install it with `pip install databricks-sql-connector`."
        )


def _quote_identifier(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


def _collect_rows(cursor) -> pd.DataFrame:
    rows = cursor.fetchall()
    columns = [col[0] for col in (cursor.description or [])]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def run_query(query: str, config: DatabricksConfig) -> pd.DataFrame:
    """Execute an arbitrary SQL statement and return a DataFrame."""
    _ensure_connector()
    with databricks_sql.connect(
        server_hostname=config.server_hostname,
        http_path=config.http_path,
        access_token=config.access_token,
        catalog=config.catalog,
        schema=config.schema,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return _collect_rows(cursor)




def list_catalogs(config: DatabricksConfig) -> pd.DataFrame:
    """Return available catalogs in the workspace."""
    _ensure_connector()
    cfg = DatabricksConfig(
        server_hostname=config.server_hostname,
        http_path=config.http_path,
        access_token=config.access_token,
    )
    catalogs = run_query("SHOW CATALOGS", cfg)
    if catalogs.empty:
        return catalogs
    rename_map = {"catalog": "name", "catalog_name": "name"}
    return catalogs.rename(columns=rename_map)


def list_schemas(config: DatabricksConfig, catalog: Optional[str] = None) -> pd.DataFrame:
    """Return schemas within a catalog."""
    _ensure_connector()
    cfg = DatabricksConfig(
        server_hostname=config.server_hostname,
        http_path=config.http_path,
        access_token=config.access_token,
        catalog=catalog,
    )
    statement = "SHOW SCHEMAS"
    if catalog:
        statement += f" IN {_quote_identifier(catalog)}"
    schemas = run_query(statement, cfg)
    if schemas.empty:
        return schemas
    rename_map = {"schemaName": "name", "databaseName": "name", "schema_name": "name"}
    return schemas.rename(columns=rename_map)
def list_tables(
    config: DatabricksConfig,
    like: Optional[str] = None,
) -> pd.DataFrame:
    """List tables within the configured catalog/schema."""
    namespace = config.namespace_reference()
    statement = "SHOW TABLES"
    if namespace:
        statement += f" IN {namespace}"
    if like:
        pattern = like.replace("'", "''")
        statement += f" LIKE '{pattern}'"
    tables = run_query(statement, config)
    if tables.empty:
        return tables
    # Normalise columns (connector returns namespace/tableName/isTemporary)
    rename_map = {
        "tableName": "table",
        "database": "schema",
        "namespace": "schema",
    }
    tables = tables.rename(columns=rename_map)
    if "schema" not in tables.columns:
        tables["schema"] = config.schema or ""
    table_col = "table" if "table" in tables.columns else tables.columns[0]
    tables["table"] = tables[table_col]
    tables["full_name"] = tables.apply(
        lambda row: ".".join(
            part for part in (config.catalog, row.get("schema") or "", row["table"]) if part
        ),
        axis=1,
    )
    return tables


def load_table(
    table: str,
    config: DatabricksConfig,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load a full table (optionally with LIMIT) into a DataFrame."""
    identifier = config.table_identifier(table)
    limit_clause = f" LIMIT {int(limit)}" if limit and int(limit) > 0 else ""
    query = f"SELECT * FROM {identifier}{limit_clause}"
    return run_query(query, config)


__all__ = [
    "DatabricksConfig",
    "DatabricksConnectorError",
    "connector_available",
    "list_catalogs",
    "list_schemas",
    "run_query",
    "list_tables",
    "load_table",
]


def list_catalogs(config: DatabricksConfig) -> pd.DataFrame:
    """Return available catalogs in the workspace."""
    _ensure_connector()
    cfg = DatabricksConfig(
        server_hostname=config.server_hostname,
        http_path=config.http_path,
        access_token=config.access_token,
        catalog=None,
        schema=None,
    )
    catalogs = run_query("SHOW CATALOGS", cfg)
    if catalogs.empty:
        return catalogs
    rename_map = {"catalog": "name", "catalog_name": "name"}
    return catalogs.rename(columns=rename_map)
