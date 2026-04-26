from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd


TABLE_CONTEXT_VERSION = 1
TABLE_CONTEXT_DIR = Path(".telly_table_context")
TABLE_CONTEXTS_DIR = TABLE_CONTEXT_DIR / "contexts"
TABLE_CONTEXT_OVERRIDES_DIR = TABLE_CONTEXT_DIR / "overrides"
TABLE_CONTEXT_MANIFEST = TABLE_CONTEXT_DIR / "manifest.json"
INTERNAL_CONTEXT_MARKER = "\n\n[중요 컨텍스트]"


@dataclass(frozen=True)
class ColumnContext:
    name: str
    dtype: str = ""
    semantic_type: str = "unknown"
    nullable: Optional[bool] = None
    null_count: Optional[int] = None
    distinct_count: Optional[int] = None
    top_values: list[dict[str, Any]] = field(default_factory=list)
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TableContext:
    table_fqn: str
    catalog: str = ""
    schema: str = ""
    table: str = ""
    columns: list[ColumnContext] = field(default_factory=list)
    row_count: Optional[int] = None
    training_status: str = "none"
    source: str = "preview"
    trained_at: Optional[str] = None
    version: int = TABLE_CONTEXT_VERSION


def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def split_table_name(table_fqn: str, *, catalog: str = "", schema: str = "") -> tuple[str, str, str]:
    parts = [part.strip("` ").strip() for part in (table_fqn or "").split(".") if part.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return catalog or "", parts[0], parts[1]
    if len(parts) == 1:
        return catalog or "", schema or "", parts[0]
    return catalog or "", schema or "", ""


def table_context_hash(table_fqn: str) -> str:
    normalized = (table_fqn or "").strip().lower().encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()


def table_context_path(table_fqn: str, *, storage_dir: Optional[Any] = None) -> Path:
    base = Path(storage_dir) if storage_dir is not None else TABLE_CONTEXT_DIR
    return base / "contexts" / f"{table_context_hash(table_fqn)}.json"


def table_context_override_path(table_fqn: str, *, storage_dir: Optional[Any] = None) -> Path:
    base = Path(storage_dir) if storage_dir is not None else TABLE_CONTEXT_DIR
    return base / "overrides" / f"{table_context_hash(table_fqn)}.json"


def manifest_path(*, storage_dir: Optional[Any] = None) -> Path:
    base = Path(storage_dir) if storage_dir is not None else TABLE_CONTEXT_DIR
    return base / "manifest.json"


def strip_internal_prompt_context(prompt: str) -> str:
    return (prompt or "").split(INTERNAL_CONTEXT_MARKER, 1)[0]


def _safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return str(value)


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    seen: dict[str, str] = {}
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen[key] = text
    return list(seen.values())


def generate_column_aliases(column: Any) -> list[str]:
    """Generate table-agnostic alias candidates from column metadata only."""

    column_context = coerce_column_context(column)
    name = column_context.name.strip()
    if not name:
        return []

    variants: list[str] = []
    normalized_words = re.sub(r"[_\-.]+", " ", name).strip()
    if normalized_words and normalized_words != name:
        variants.append(normalized_words)
    compact = re.sub(r"[_\-\.\s]+", "", name).strip()
    if compact and compact != name:
        variants.append(compact)
    camel_words = re.sub(r"(?<!^)([A-Z])", r" \1", name).strip()
    if camel_words and camel_words != name:
        variants.append(camel_words)
    lower = name.lower()
    if lower != name:
        variants.append(lower)

    return [alias for alias in _dedupe_strings(variants) if alias.lower() != name.lower()]


def infer_semantic_type(dtype: Any, *, distinct_count: Optional[int] = None) -> str:
    dtype_text = str(dtype or "").lower()
    if any(token in dtype_text for token in ("bool", "boolean")):
        return "boolean"
    if any(token in dtype_text for token in ("datetime", "date", "timestamp", "time")):
        return "datetime"
    if any(token in dtype_text for token in ("int", "bigint", "smallint", "double", "float", "decimal", "numeric", "long")):
        return "numeric"
    if distinct_count is not None and distinct_count <= 50:
        return "categorical"
    if any(token in dtype_text for token in ("char", "string", "object", "varchar")):
        return "categorical"
    return "unknown"


def coerce_column_context(value: Any) -> ColumnContext:
    if isinstance(value, ColumnContext):
        return value
    if isinstance(value, Mapping):
        return ColumnContext(
            name=str(value.get("name", "")),
            dtype=str(value.get("dtype", "")),
            semantic_type=str(value.get("semantic_type", "unknown") or "unknown"),
            nullable=value.get("nullable"),
            null_count=value.get("null_count"),
            distinct_count=value.get("distinct_count"),
            top_values=list(value.get("top_values") or []),
            min_value=value.get("min_value"),
            max_value=value.get("max_value"),
            aliases=[str(item) for item in value.get("aliases", [])],
        )
    return ColumnContext(name=str(value))


def coerce_table_context(value: Any) -> Optional[TableContext]:
    if value is None:
        return None
    if isinstance(value, TableContext):
        return value
    if isinstance(value, Mapping):
        return TableContext(
            table_fqn=str(value.get("table_fqn", "")),
            catalog=str(value.get("catalog", "")),
            schema=str(value.get("schema", "")),
            table=str(value.get("table", "")),
            columns=[coerce_column_context(item) for item in value.get("columns", [])],
            row_count=value.get("row_count"),
            training_status=str(value.get("training_status", "none") or "none"),
            source=str(value.get("source", "preview") or "preview"),
            trained_at=value.get("trained_at"),
            version=int(value.get("version", TABLE_CONTEXT_VERSION) or TABLE_CONTEXT_VERSION),
        )
    return None


def table_context_to_dict(context: TableContext) -> dict[str, Any]:
    return asdict(context)


def table_context_from_dict(payload: Mapping[str, Any]) -> TableContext:
    context = coerce_table_context(payload)
    if context is None:
        raise ValueError("Invalid table context payload.")
    return context


def build_schema_only_context(
    table_fqn: str,
    df: Optional[pd.DataFrame] = None,
    *,
    columns: Optional[Iterable[Any]] = None,
    dtypes: Optional[Mapping[str, Any]] = None,
    catalog: str = "",
    schema: str = "",
) -> TableContext:
    catalog_name, schema_name, table_name = split_table_name(table_fqn, catalog=catalog, schema=schema)
    if isinstance(df, pd.DataFrame):
        column_names = [str(column) for column in df.columns]
        dtype_map = {str(column): str(dtype) for column, dtype in df.dtypes.items()}
    else:
        column_names = [str(column) for column in (columns or [])]
        dtype_map = {str(key): str(value) for key, value in (dtypes or {}).items()}

    column_contexts = [
        ColumnContext(
            name=column,
            dtype=dtype_map.get(column, ""),
            semantic_type=infer_semantic_type(dtype_map.get(column, "")),
        )
        for column in column_names
    ]
    return TableContext(
        table_fqn=(table_fqn or "").strip(),
        catalog=catalog_name,
        schema=schema_name,
        table=table_name,
        columns=column_contexts,
        training_status="schema_only" if column_contexts else "none",
        source="preview",
        version=TABLE_CONTEXT_VERSION,
    )


def build_trained_context(
    base_context: TableContext,
    *,
    row_count: Optional[int],
    column_profiles: Mapping[str, Mapping[str, Any]],
) -> TableContext:
    columns: list[ColumnContext] = []
    for column in base_context.columns:
        profile = column_profiles.get(column.name, {})
        distinct_count = profile.get("distinct_count")
        automatic_aliases = generate_column_aliases(column)
        columns.append(
            ColumnContext(
                name=column.name,
                dtype=column.dtype,
                semantic_type=infer_semantic_type(column.dtype, distinct_count=distinct_count),
                nullable=profile.get("nullable"),
                null_count=profile.get("null_count"),
                distinct_count=distinct_count,
                top_values=list(profile.get("top_values") or []),
                min_value=_safe_json_value(profile.get("min_value")),
                max_value=_safe_json_value(profile.get("max_value")),
                aliases=_dedupe_strings([*column.aliases, *automatic_aliases]),
            )
        )
    return TableContext(
        table_fqn=base_context.table_fqn,
        catalog=base_context.catalog,
        schema=base_context.schema,
        table=base_context.table,
        columns=columns,
        row_count=row_count,
        training_status="trained",
        source="databricks_profile",
        trained_at=utc_now_iso(),
        version=TABLE_CONTEXT_VERSION,
    )


def save_table_context(context: TableContext, *, storage_dir: Optional[Any] = None) -> Path:
    path = table_context_path(context.table_fqn, storage_dir=storage_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table_context_to_dict(context), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_file = manifest_path(storage_dir=storage_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    except Exception:
        manifest = {}
    manifest.setdefault("version", TABLE_CONTEXT_VERSION)
    manifest.setdefault("tables", {})
    manifest["tables"][context.table_fqn] = {
        "path": str(path),
        "hash": table_context_hash(context.table_fqn),
        "training_status": context.training_status,
        "trained_at": context.trained_at,
        "column_count": len(context.columns),
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def default_table_context_override_payload(context: Any) -> dict[str, Any]:
    coerced = coerce_table_context(context)
    if coerced is None:
        raise ValueError("Invalid table context.")
    return {
        "version": TABLE_CONTEXT_VERSION,
        "table_fqn": coerced.table_fqn,
        "description": "Manual aliases for TableContext column resolution. Edit aliases only; do not store raw rows.",
        "columns": {
            column.name: {
                "aliases": list(column.aliases),
            }
            for column in coerced.columns
            if column.name
        },
    }


def ensure_table_context_override_file(context: Any, *, storage_dir: Optional[Any] = None) -> Path:
    coerced = coerce_table_context(context)
    if coerced is None:
        raise ValueError("Invalid table context.")
    path = table_context_override_path(coerced.table_fqn, storage_dir=storage_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = default_table_context_override_payload(coerced)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_table_context_overrides(table_fqn: str, *, storage_dir: Optional[Any] = None) -> dict[str, list[str]]:
    path = table_context_override_path(table_fqn, storage_dir=storage_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    columns_payload = payload.get("columns", {}) if isinstance(payload, Mapping) else {}
    aliases_by_column: dict[str, list[str]] = {}
    if isinstance(columns_payload, Mapping):
        for column_name, value in columns_payload.items():
            if isinstance(value, Mapping):
                aliases = value.get("aliases", [])
            else:
                aliases = value
            if isinstance(aliases, list):
                aliases_by_column[str(column_name)] = _dedupe_strings(aliases)
    return aliases_by_column


def apply_table_context_overrides(
    context: Any,
    *,
    storage_dir: Optional[Any] = None,
    overrides: Optional[Mapping[str, Iterable[Any]]] = None,
) -> TableContext:
    coerced = coerce_table_context(context)
    if coerced is None:
        raise ValueError("Invalid table context.")
    aliases_by_column = (
        {str(key): _dedupe_strings(value) for key, value in overrides.items()}
        if overrides is not None
        else load_table_context_overrides(coerced.table_fqn, storage_dir=storage_dir)
    )
    columns = []
    for column in coerced.columns:
        manual_aliases = aliases_by_column.get(column.name, [])
        columns.append(
            ColumnContext(
                name=column.name,
                dtype=column.dtype,
                semantic_type=column.semantic_type,
                nullable=column.nullable,
                null_count=column.null_count,
                distinct_count=column.distinct_count,
                top_values=list(column.top_values),
                min_value=column.min_value,
                max_value=column.max_value,
                aliases=_dedupe_strings([*manual_aliases, *column.aliases]),
            )
        )
    return TableContext(
        table_fqn=coerced.table_fqn,
        catalog=coerced.catalog,
        schema=coerced.schema,
        table=coerced.table,
        columns=columns,
        row_count=coerced.row_count,
        training_status=coerced.training_status,
        source=coerced.source,
        trained_at=coerced.trained_at,
        version=coerced.version,
    )


def load_saved_table_context(table_fqn: str, *, storage_dir: Optional[Any] = None) -> Optional[TableContext]:
    path = table_context_path(table_fqn, storage_dir=storage_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return apply_table_context_overrides(
            table_context_from_dict(payload),
            storage_dir=storage_dir,
        )
    except Exception:
        return None


def load_table_context_for_selection(
    table_fqn: str,
    *,
    preview_df: Optional[pd.DataFrame] = None,
    catalog: str = "",
    schema: str = "",
    storage_dir: Optional[Any] = None,
) -> TableContext:
    saved = load_saved_table_context(table_fqn, storage_dir=storage_dir)
    if saved is not None:
        return saved
    return build_schema_only_context(
        table_fqn,
        preview_df,
        catalog=catalog,
        schema=schema,
    )


def get_column_names(context: Any) -> list[str]:
    coerced = coerce_table_context(context)
    if coerced is None:
        return []
    return [column.name for column in coerced.columns if column.name]


def _identifier_position(text: str, column: str) -> int:
    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(column.lower())}(?![A-Za-z0-9_])", text)
    return match.start() if match else -1


def resolve_column_from_prompt(
    prompt: str,
    context: Any,
    *,
    excluded_columns: Optional[Sequence[str]] = None,
) -> Optional[str]:
    coerced = coerce_table_context(context)
    if coerced is None:
        return None
    lowered = strip_internal_prompt_context(prompt).lower()
    excluded = {str(column).lower() for column in (excluded_columns or [])}
    matches: list[tuple[int, str]] = []
    for column in coerced.columns:
        name = column.name
        if not name or name.lower() in excluded:
            continue
        candidates = [name, *column.aliases]
        positions = [_identifier_position(lowered, candidate) for candidate in candidates if candidate]
        valid_positions = [position for position in positions if position >= 0]
        if valid_positions:
            matches.append((min(valid_positions), name))
    matches.sort()
    return matches[0][1] if matches else None


def table_context_summary(context: Any, *, max_columns: int = 30) -> str:
    coerced = coerce_table_context(context)
    if coerced is None or not coerced.table_fqn:
        return ""
    lines = [
        f"Current selected table: {coerced.table_fqn}",
        f"Table context status: {coerced.training_status}",
    ]
    if coerced.row_count is not None:
        lines.append(f"Approx row count: {coerced.row_count}")
    lines.append("Columns:")
    for column in coerced.columns[:max_columns]:
        details = [column.semantic_type]
        if column.dtype:
            details.append(f"dtype={column.dtype}")
        if column.distinct_count is not None:
            details.append(f"distinct={column.distinct_count}")
        if column.null_count is not None:
            details.append(f"nulls={column.null_count}")
        if column.top_values:
            top_values = ", ".join(str(item.get("value", "")) for item in column.top_values[:3])
            details.append(f"top_values=[{top_values}]")
        if column.min_value is not None or column.max_value is not None:
            details.append(f"range=[{column.min_value}, {column.max_value}]")
        lines.append(f"- {column.name}: " + ", ".join(item for item in details if item))
    remaining = len(coerced.columns) - max_columns
    if remaining > 0:
        lines.append(f"- ... {remaining} more columns")
    return "\n".join(lines)


def contains_raw_sample_rows(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key).lower() in {"sample", "samples", "sample_rows", "rows", "data"}:
                return True
            if contains_raw_sample_rows(value):
                return True
    elif isinstance(payload, list):
        return any(contains_raw_sample_rows(item) for item in payload)
    return False


__all__ = [
    "ColumnContext",
    "INTERNAL_CONTEXT_MARKER",
    "TABLE_CONTEXT_DIR",
    "TABLE_CONTEXT_MANIFEST",
    "TABLE_CONTEXT_OVERRIDES_DIR",
    "TABLE_CONTEXT_VERSION",
    "TABLE_CONTEXTS_DIR",
    "TableContext",
    "apply_table_context_overrides",
    "build_schema_only_context",
    "build_trained_context",
    "coerce_table_context",
    "contains_raw_sample_rows",
    "default_table_context_override_payload",
    "ensure_table_context_override_file",
    "generate_column_aliases",
    "get_column_names",
    "infer_semantic_type",
    "load_saved_table_context",
    "load_table_context_for_selection",
    "load_table_context_overrides",
    "manifest_path",
    "resolve_column_from_prompt",
    "save_table_context",
    "split_table_name",
    "strip_internal_prompt_context",
    "table_context_hash",
    "table_context_override_path",
    "table_context_path",
    "table_context_summary",
    "table_context_to_dict",
]
