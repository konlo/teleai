"""
High-level helpers for interacting with Databricks SQL Warehouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from app_io.databricks import (
    DatabricksConfig,
    DatabricksConnectorError,
    list_catalogs as _list_catalogs,
    list_schemas as _list_schemas,
    connector_available as _connector_available,
    list_tables as _list_tables,
    load_table as _load_table,
    run_query as _run_query,
)


@dataclass
class DatabricksCredentials:
    """User-supplied credentials used to build a DatabricksConfig."""

    server_hostname: str
    http_path: str
    access_token: str
    catalog: Optional[str] = None
    schema: Optional[str] = None

    def to_config(self) -> DatabricksConfig:
        return DatabricksConfig(
            server_hostname=self.server_hostname.strip(),
            http_path=self.http_path.strip(),
            access_token=self.access_token.strip(),
            catalog=(self.catalog or "").strip() or None,
            schema=(self.schema or "").strip() or None,
        )


def connector_available() -> bool:
    """Return True when the Databricks SQL connector is installed."""
    return _connector_available()


def list_catalogs(creds: DatabricksCredentials) -> pd.DataFrame:
    """Return available catalogs for the configured workspace."""
    cfg = creds.to_config()
    cfg.catalog = None
    cfg.schema = None
    return _list_catalogs(cfg)


def list_schemas(creds: DatabricksCredentials, catalog: str) -> pd.DataFrame:
    """Return schemas for a given catalog."""
    cfg = creds.to_config()
    return _list_schemas(cfg, catalog)


def list_tables(
    creds: DatabricksCredentials,
    like: Optional[str] = None,
) -> pd.DataFrame:
    """Return a DataFrame describing available tables."""
    return _list_tables(creds.to_config(), like=like)


def load_table(
    table: str,
    creds: DatabricksCredentials,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load the given table into a DataFrame."""
    return _load_table(table, creds.to_config(), limit=limit)


def run_sql(
    query: str,
    creds: DatabricksCredentials,
) -> pd.DataFrame:
    """Execute an arbitrary SQL query and return a DataFrame."""
    return _run_query(query, creds.to_config())


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


__all__ = [
    "DatabricksCredentials",
    "DatabricksConnectorError",
    "connector_available",
    "list_catalogs",
    "list_schemas",
    "list_tables",
    "load_table",
    "run_sql",
    "to_csv_bytes",
]
