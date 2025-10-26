import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.dataload.databricks_sql_loader import (
    DatabricksConnectorError,
    DatabricksCredentials,
    list_catalogs as databricks_list_catalogs,
    list_schemas as databricks_list_schemas,
    connector_available as databricks_connector_available,
    list_tables as databricks_list_tables,
    load_table as databricks_load_table,
)


load_dotenv()

DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "/Users/najongseong/dataset")
DFB_DEFAULT_NAME = "telemetry_raw.csv"
SUPPORTED_EXTENSIONS = (".csv", ".parquet")
TIME_COLUMN_CANDIDATES = [
    "datetime",
    "timestamp",
    "ts",
    "time",
    "event_time",
    "eventtime",
    "date",
    "created_at",
]


def parse_int(val: Any, default: int) -> int:
    """Robust conversion to int that tolerates blanks and string numbers."""
    if val is None:
        return default
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float):
        try:
            return int(val)
        except Exception:
            return default
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return default
        try:
            return int(float(s))
        except Exception:
            return default
    return default


def parse_float(val: Any, default: float) -> float:
    """Robust conversion to float that tolerates blanks and string numbers."""
    if val is None:
        return default
    if isinstance(val, (int, float, np.integer, np.floating)):
        try:
            return float(val)
        except Exception:
            return default
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return default
        try:
            return float(s)
        except Exception:
            return default
    return default


def resolve_time_column(df: Optional[pd.DataFrame], preferred: str) -> Optional[str]:
    """Return a best-effort time column name matching preferred or common aliases."""
    if df is None or not isinstance(df, pd.DataFrame) or not preferred:
        return None
    if preferred in df.columns:
        return preferred
    lower_map = {c.lower(): c for c in df.columns}
    pref_lower = preferred.lower()
    if pref_lower in lower_map:
        return lower_map[pref_lower]
    for alias in TIME_COLUMN_CANDIDATES:
        if alias == preferred:
            continue
        if alias in df.columns:
            return alias
        alias_lower = alias.lower()
        if alias_lower in lower_map:
            return lower_map[alias_lower]
    return None


def read_table(path: str) -> pd.DataFrame:
    """Load CSV or Parquet based on the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {ext}")


def ensure_session_state() -> None:
    """Populate Streamlit session_state with defaults required by the app."""
    defaults: List[Tuple[str, Any]] = [
        ("DATA_DIR", DEFAULT_DATA_DIR),
        ("df_A_data", None),
        ("df_A_name", "No Data"),
        ("csv_path", os.path.join(DEFAULT_DATA_DIR, "stormtrooper.csv")),
        ("df_B_data", None),
        ("df_B_name", "No Data"),
        ("csv_b_path", ""),
        ("explanation_lang", "English"),
        ("df_A_signature", ""),
        ("df_B_signature", ""),
        (
            "databricks_config",
            {
                "server_hostname": os.getenv("DATABRICKS_SERVER_HOSTNAME", ""),
                "http_path": os.getenv("DATABRICKS_HTTP_PATH", ""),
                "access_token": os.getenv("DATABRICKS_TOKEN", os.getenv("DATABRICKS_ACCESS_TOKEN", "")),
                "catalog": os.getenv("DATABRICKS_CATALOG", ""),
                "schema": os.getenv("DATABRICKS_SCHEMA", ""),
            },
        ),
        ("databricks_table_filter", ""),
        ("databricks_table_options", []),
        ("databricks_tables_last", None),
        ("databricks_limit", 0),
        ("databricks_catalogs", None),
        ("databricks_catalog_options", []),
        ("databricks_selected_catalog", os.getenv("DATABRICKS_CATALOG", "")),
        ("databricks_schema_options", []),
        ("databricks_selected_schema", os.getenv("DATABRICKS_SCHEMA", "")),
        ("databricks_schemas_last", None),
    ]
    for key, default in defaults:
        if key not in st.session_state:
            st.session_state[key] = default


def load_df_a(path: str, display_name: str) -> Tuple[bool, str]:
    """Load df_A from disk and update session state."""
    try:
        new_df = read_table(path)
        st.session_state["df_A_data"] = new_df
        st.session_state["df_A_name"] = display_name
        st.session_state["csv_path"] = path
        return True, f"Loaded file: {display_name} (Shape: {new_df.shape})"
    except Exception as exc:
        st.session_state["df_A_data"] = None
        st.session_state["df_A_name"] = "Load Failed"
        st.session_state["csv_path"] = ""
        return False, f"Failed to load data: {path}\n{exc}"


def load_df_b(path: str, display_name: str) -> Tuple[bool, str]:
    """Load df_B from disk and update session state."""
    try:
        new_df = read_table(path)
        st.session_state["df_B_data"] = new_df
        st.session_state["df_B_name"] = display_name
        st.session_state["csv_b_path"] = path
        return True, f"Loaded file (df_B): {display_name} (Shape: {new_df.shape})"
    except Exception as exc:
        st.session_state["df_B_data"] = None
        st.session_state["df_B_name"] = "Load Failed"
        st.session_state["csv_b_path"] = ""
        return False, f"Failed to load df_B: {path}\n{exc}"


def update_databricks_config(**kwargs: Any) -> None:
    """Persist Databricks connection details into session state."""
    ensure_session_state()
    cfg = st.session_state.get("databricks_config", {})
    cfg.update({k: (v or "") for k, v in kwargs.items()})
    st.session_state["databricks_config"] = cfg


def get_databricks_credentials() -> DatabricksCredentials:
    """Build DatabricksCredentials from stored session configuration."""
    ensure_session_state()
    cfg = st.session_state.get("databricks_config", {})
    return DatabricksCredentials(
        server_hostname=cfg.get("server_hostname", ""),
        http_path=cfg.get("http_path", ""),
        access_token=cfg.get("access_token", ""),
        catalog=cfg.get("catalog", "") or None,
        schema=cfg.get("schema", "") or None,
    )




def list_databricks_schemas_in_session(catalog: str) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """List schemas for a catalog and cache the result."""
    if not databricks_connector_available():
        return False, None, "databricks-sql-connector is not installed."
    if not catalog:
        return False, None, "Catalog를 먼저 선택해주세요."
    try:
        creds = get_databricks_credentials()
        schemas = databricks_list_schemas(creds, catalog)
        options = []
        if schemas is not None and not schemas.empty:
            if "name" in schemas.columns:
                options = schemas["name"].tolist()
            else:
                options = schemas.iloc[:, 0].tolist()
        st.session_state["databricks_schema_options"] = options
        st.session_state["databricks_schemas_last"] = schemas
        if options and not st.session_state.get("databricks_selected_schema"):
            st.session_state["databricks_selected_schema"] = options[0]
        if schemas is None or schemas.empty:
            return True, schemas, f"No schemas found in catalog '{catalog}'."
        return True, schemas, f"Loaded {len(options)} schemas."
    except DatabricksConnectorError as exc:  # pragma: no cover
        return False, None, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, None, f"Failed to list schemas: {exc}"
def list_databricks_tables_in_session(pattern: str = "") -> Tuple[bool, Optional[pd.DataFrame], str]:
    """List tables and cache the result in session state."""
    if not databricks_connector_available():
        return False, None, "databricks-sql-connector is not installed."
    try:
        creds = get_databricks_credentials()
        selected_catalog = st.session_state.get("databricks_selected_catalog", "")
        selected_schema = st.session_state.get("databricks_selected_schema", "")
        if selected_catalog:
            creds.catalog = selected_catalog
        if selected_schema:
            creds.schema = selected_schema
        else:
            creds.schema = None
        df = databricks_list_tables(creds, like=pattern or None)
        if df.empty:
            table_options = []
        elif "full_name" in df.columns:
            table_options = df["full_name"].tolist()
        elif "name" in df.columns:
            table_options = df["name"].tolist()
        elif "table" in df.columns:
            table_options = df["table"].tolist()
        else:
            table_options = df.iloc[:, 0].astype(str).tolist()
        st.session_state["databricks_table_options"] = table_options
        st.session_state["databricks_tables_last"] = df
        if df.empty:
            return True, df, "No tables found for the current catalog/schema."
        return True, df, f"Loaded {len(table_options)} tables from Databricks."
    except DatabricksConnectorError as exc:  # pragma: no cover - runtime guard
        return False, None, str(exc)
    except Exception as exc:  # pragma: no cover - external dependency errors
        return False, None, f"Failed to list tables: {exc}"


def list_databricks_catalogs_in_session() -> Tuple[bool, Optional[pd.DataFrame], str]:
    """List catalogs and cache the result in session state."""
    if not databricks_connector_available():
        return False, None, "databricks-sql-connector is not installed."
    try:
        creds = get_databricks_credentials()
        catalogs = databricks_list_catalogs(creds)
        st.session_state["databricks_catalogs"] = catalogs
        options = []
        if catalogs is not None and not catalogs.empty:
            if "name" in catalogs.columns:
                options = catalogs["name"].tolist()
            else:
                options = catalogs.iloc[:, 0].tolist()
        st.session_state["databricks_catalog_options"] = options
        if options and not st.session_state.get("databricks_selected_catalog"):
            st.session_state["databricks_selected_catalog"] = options[0]
        if catalogs is None or catalogs.empty:
            return True, catalogs, "No catalogs found in this workspace."
        return True, catalogs, f"Loaded {len(options)} catalogs."
    except DatabricksConnectorError as exc:  # pragma: no cover
        return False, None, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, None, f"Failed to list catalogs: {exc}"


def load_df_from_databricks(
    table: str,
    target: str = "A",
    limit: Optional[int] = None,
) -> Tuple[bool, str]:
    """Load a Databricks table into df_A or df_B session slots."""
    if not table:
        return False, "Table name must not be empty."
    if target not in {"A", "B"}:
        return False, "Target must be 'A' or 'B'."
    if not databricks_connector_available():
        return False, "databricks-sql-connector is not installed."
    try:
        creds = get_databricks_credentials()
        selected_catalog = st.session_state.get("databricks_selected_catalog", "")
        selected_schema = st.session_state.get("databricks_selected_schema", "")
        if selected_catalog:
            creds.catalog = selected_catalog
        if selected_schema:
            creds.schema = selected_schema
        else:
            creds.schema = None
        df = databricks_load_table(table, creds, limit=limit)
    except DatabricksConnectorError as exc:  # pragma: no cover
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, f"Failed to load Databricks table: {exc}"

    name_key = "df_A_name" if target == "A" else "df_B_name"
    data_key = "df_A_data" if target == "A" else "df_B_data"
    path_key = "csv_path" if target == "A" else "csv_b_path"

    st.session_state[data_key] = df
    st.session_state[name_key] = f"{table} (Databricks)"
    st.session_state[path_key] = f"databricks://{table}"

    return True, f"Loaded Databricks table '{table}' into df_{target}."


def dataframe_signature(df: Optional[pd.DataFrame], path: str) -> str:
    """Create a simple signature string for change detection."""
    if df is None:
        return "none"
    rows, cols = df.shape
    return f"{path}|{rows}x{cols}"


__all__ = [
    "DEFAULT_DATA_DIR",
    "DFB_DEFAULT_NAME",
    "SUPPORTED_EXTENSIONS",
    "TIME_COLUMN_CANDIDATES",
    "parse_int",
    "parse_float",
    "resolve_time_column",
    "read_table",
    "ensure_session_state",
    "load_df_a",
    "load_df_b",
    "dataframe_signature",
    "update_databricks_config",
    "get_databricks_credentials",
    "list_databricks_catalogs_in_session",
    "list_databricks_schemas_in_session",
    "list_databricks_tables_in_session",
    "load_df_from_databricks",
    "databricks_connector_available",
]
