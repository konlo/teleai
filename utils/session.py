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
    run_sql as databricks_run_sql,
)


load_dotenv()

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/Users/najongseong/dataset")
DFB_DEFAULT_NAME = "telemetry_raw.csv"
SUPPORTED_EXTENSIONS = (".csv", ".parquet")
DEFAULT_SQL_LIMIT_MIN = 1
DEFAULT_SQL_LIMIT_MAX = 10_000_000
_DEFAULT_SQL_LIMIT = 2000
SESSION_SQL_LIMIT_KEY = "sql_limit"
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
def get_default_sql_limit() -> int:
    """Return the session-scoped SQL limit with a global fallback."""
    if SESSION_SQL_LIMIT_KEY in st.session_state:
        candidate = st.session_state[SESSION_SQL_LIMIT_KEY]
        coerced = parse_int(candidate, _DEFAULT_SQL_LIMIT)
        if DEFAULT_SQL_LIMIT_MIN <= coerced <= DEFAULT_SQL_LIMIT_MAX:
            return coerced
    return _DEFAULT_SQL_LIMIT


def set_default_sql_limit(value: int) -> int:
    """Persist the SQL limit to the active session after validation."""
    if not isinstance(value, int):
        raise TypeError("LIMIT 값은 정수여야 합니다.")
    if not (DEFAULT_SQL_LIMIT_MIN <= value <= DEFAULT_SQL_LIMIT_MAX):
        raise ValueError(
            f"LIMIT 값은 {DEFAULT_SQL_LIMIT_MIN} 이상 {DEFAULT_SQL_LIMIT_MAX} 이하의 정수여야 합니다."
        )
    st.session_state[SESSION_SQL_LIMIT_KEY] = value
    return value


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
    catalog_env = os.environ.get("DATABRICKS_CATALOG", "")
    schema_env = os.environ.get("DATABRICKS_SCHEMA", "")
    catalog_default = catalog_env or "hive_metastore"
    schema_default = schema_env or "default"
    defaults: List[Tuple[str, Any]] = [
        ("DATA_DIR", DEFAULT_DATA_DIR),
        ("df_A_data", None),
        ("df_A_name", "No Data"),
        ("csv_path", os.path.join(DEFAULT_DATA_DIR, "stormtrooper.csv")),
        ("df_B_data", None),
        ("df_B_name", "No Data"),
        ("csv_b_path", ""),
        ("df_init_data", None),
        ("explanation_lang", "한국어"),
        ("df_A_initial", None),
        ("df_A_signature", ""),
        ("df_B_signature", ""),
        (
            "databricks_config",
            {
                "server_hostname": os.environ.get("DATABRICKS_HOST", ""),
                "http_path": os.environ.get("DATABRICKS_HTTP_PATH", ""),
                "access_token": os.environ.get("DATABRICKS_TOKEN", os.environ.get("DATABRICKS_ACCESS_TOKEN", "")),
                "catalog": catalog_default,
                "schema": schema_default,
            },
        ),
        ("databricks_table_filter", ""),
        ("databricks_table_options", []),
        ("databricks_tables_last", None),
        ("databricks_limit", 0),
        ("databricks_catalogs", None),
        ("databricks_catalog_options", []),
        ("databricks_selected_catalog", catalog_default),
        ("databricks_schema_options", []),
        ("databricks_selected_schema", schema_default),
        ("databricks_selected_table", ""),
        ("databricks_schemas_last", None),
        ("databricks_table_input", ""),
        ("databricks_sql_query", ""),
        ("databricks_last_preview_message", ""),
        ("databricks_last_preview_table", ""),
        ("databricks_column_source_table", ""),
        ("databricks_column_options", []),
        ("last_agent_mode", "SQL Builder"),
        ("last_sql_statement", ""),
        ("last_sql_label", "SQL Query"),
        ("last_sql_table", ""),
        ("databricks_selected_column", ""),
        (SESSION_SQL_LIMIT_KEY, _DEFAULT_SQL_LIMIT),
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
        pattern_clean = (pattern or "").strip()
        df = databricks_list_tables(creds, like=pattern_clean or None)
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
        current_selected = st.session_state.get("databricks_selected_table", "")
        if table_options:
            if pattern_clean:
                # Agent-driven listing shouldn't override the current table selection.
                pass
            else:
                if current_selected not in table_options:
                    current_selected = table_options[0]
                st.session_state["databricks_selected_table"] = current_selected
                st.session_state["databricks_table_input"] = current_selected
                update_databricks_namespace_from_table(current_selected)
        else:
            if not pattern_clean:
                st.session_state["databricks_selected_table"] = ""
        if df.empty:
            if not pattern_clean:
                st.session_state["databricks_last_preview_table"] = ""
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
    st.session_state["databricks_last_preview_table"] = table

    st.session_state["databricks_last_preview_message"] = (
        f"{table} – {len(df)} rows loaded into df_{target}."
    )

    if target == "A":
        st.session_state["databricks_column_source_table"] = table
        st.session_state["databricks_column_options"] = [str(col) for col in df.columns]

    return True, f"Loaded Databricks table '{table}' into df_{target}."


def update_databricks_namespace_from_table(table: str) -> None:
    """Update selected catalog/schema based on a fully qualified table."""
    ensure_session_state()
    parts = [part.strip() for part in (table or "").split(".") if part.strip()]
    if not parts:
        return
    catalog = parts[0]
    schema = parts[1] if len(parts) > 1 else ""
    if catalog:
        st.session_state["databricks_selected_catalog"] = catalog
    if schema:
        st.session_state["databricks_selected_schema"] = schema


def generate_select_all_query(table: str) -> str:
    """Return a canonical SELECT statement for the given table with a LIMIT."""
    table_clean = (table or "").strip()
    if not table_clean:
        raise ValueError("Table name must not be empty.")
    update_databricks_namespace_from_table(table_clean)
    return f"SELECT * FROM {table_clean} LIMIT {get_default_sql_limit()}"


def load_preview_from_databricks_query(
    table: str,
    query: Optional[str] = None,
    *,
    target: str = "A",
    limit: int = 10,
) -> Tuple[bool, str]:
    """Execute a Databricks SQL preview and store the full result (display shows head)."""
    ensure_session_state()
    if target not in {"A", "B"}:
        return False, "Target must be 'A' or 'B'."
    if not databricks_connector_available():
        return False, (
            "databricks-sql-connector is not installed. "
            "Install it with `pip install databricks-sql-connector`."
        )

    table_clean = (table or "").strip()
    if not table_clean:
        return False, "Table name must not be empty."

    base_query = (query or "").strip()
    if not base_query:
        try:
            base_query = generate_select_all_query(table_clean)
        except ValueError as exc:
            return False, str(exc)

    try:
        update_databricks_namespace_from_table(table_clean)
        creds = get_databricks_credentials()
        selected_catalog = st.session_state.get("databricks_selected_catalog", "")
        selected_schema = st.session_state.get("databricks_selected_schema", "")
        if selected_catalog:
            creds.catalog = selected_catalog
        if selected_schema:
            creds.schema = selected_schema
        else:
            creds.schema = None
        if (not creds.catalog or not creds.schema) and "." in table_clean:
            parts = [part.strip() for part in table_clean.split(".") if part.strip()]
            if len(parts) >= 3:
                if not creds.catalog:
                    creds.catalog = parts[0]
                if not creds.schema:
                    creds.schema = parts[1]
        df = databricks_run_sql(base_query, creds)
    except DatabricksConnectorError as exc:  # pragma: no cover
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, f"Databricks SQL 실행에 실패했습니다: {exc}"

    name_key = "df_A_name" if target == "A" else "df_B_name"
    data_key = "df_A_data" if target == "A" else "df_B_data"
    path_key = "csv_path" if target == "A" else "csv_b_path"

    st.session_state[data_key] = df
    st.session_state[name_key] = f"{table_clean} (preview)"
    st.session_state[path_key] = f"databricks://{table_clean}"
    st.session_state["databricks_table_input"] = table_clean
    st.session_state["databricks_sql_query"] = base_query
    st.session_state["databricks_last_preview_message"] = (
        f"{table_clean} – {len(df)} rows loaded."
    )
    st.session_state["databricks_last_preview_table"] = table_clean

    return True, f"Loaded data from '{table_clean}'."


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
    "DEFAULT_SQL_LIMIT_MIN",
    "DEFAULT_SQL_LIMIT_MAX",
    "TIME_COLUMN_CANDIDATES",
    "get_default_sql_limit",
    "set_default_sql_limit",
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
    "generate_select_all_query",
    "load_preview_from_databricks_query",
    "update_databricks_namespace_from_table",
]
