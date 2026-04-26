from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import uuid4

import pandas as pd


TRACE_DIR = Path(".telly_runtime")
TRACE_JSONL = TRACE_DIR / "telly_trace.jsonl"
LATEST_TRACE_JSON = TRACE_DIR / "latest_trace.json"
TRACE_SESSION_KEY = "runtime_trace_current"
SENSITIVE_KEYWORDS = ("token", "api_key", "apikey", "password", "secret")
MAX_STRING_LENGTH = 4000
DEFAULT_SAMPLE_ROWS = 5

_CURRENT_TRACE: Optional[Dict[str, Any]] = None


def _utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(keyword in lowered for keyword in SENSITIVE_KEYWORDS)


def _streamlit_state():
    try:
        import streamlit as st

        return st.session_state
    except Exception:
        return None


def _get_active_trace() -> Optional[Dict[str, Any]]:
    state = _streamlit_state()
    if state is not None:
        try:
            trace = state.get(TRACE_SESSION_KEY)
            if isinstance(trace, dict):
                return trace
        except Exception:
            pass
    return _CURRENT_TRACE


def _set_active_trace(trace: Optional[Dict[str, Any]]) -> None:
    global _CURRENT_TRACE
    _CURRENT_TRACE = trace
    state = _streamlit_state()
    if state is not None:
        try:
            if trace is None:
                state.pop(TRACE_SESSION_KEY, None)
            else:
                state[TRACE_SESSION_KEY] = trace
        except Exception:
            pass


def _storage_paths(storage_dir: Optional[Any] = None) -> tuple[Path, Path, Path]:
    base = Path(storage_dir) if storage_dir is not None else TRACE_DIR
    return base, base / TRACE_JSONL.name, base / LATEST_TRACE_JSON.name


def sanitize_for_trace(value: Any, *, key: str = "", sample_rows: int = DEFAULT_SAMPLE_ROWS) -> Any:
    if key and _is_sensitive_key(key):
        return "[REDACTED]"

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + "...[truncated]"
        return value
    if isinstance(value, pd.DataFrame):
        sample = value.head(sample_rows).to_dict(orient="records") if sample_rows > 0 else []
        return {
            "kind": "dataframe",
            "row_count": len(value),
            "columns": [str(column) for column in value.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in value.dtypes.items()},
            "sample": sanitize_for_trace(sample, sample_rows=0),
        }
    if isinstance(value, pd.Index):
        return [str(item) for item in value]
    if is_dataclass(value):
        return sanitize_for_trace(asdict(value), sample_rows=sample_rows)
    if hasattr(value, "model_dump"):
        try:
            return sanitize_for_trace(value.model_dump(), sample_rows=sample_rows)
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_for_trace(item_value, key=str(item_key), sample_rows=sample_rows)
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_trace(item, sample_rows=sample_rows) for item in value]

    return sanitize_for_trace(str(value), key=key, sample_rows=sample_rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _persist_event(trace: Mapping[str, Any], event: Mapping[str, Any]) -> None:
    storage_dir = trace.get("storage_dir")
    _, jsonl_path, latest_path = _storage_paths(storage_dir)
    _append_jsonl(jsonl_path, event)
    public_trace = {key: value for key, value in trace.items() if key != "storage_dir"}
    _write_json(latest_path, public_trace)


def _base_context_from_trace(trace: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "trace_id": trace.get("trace_id"),
        "conversation_id": trace.get("conversation_id"),
        "turn_id": trace.get("turn_id"),
        "run_id": trace.get("run_id"),
    }


def build_trace_event(context: Mapping[str, Any], event_type: str, **fields: Any) -> Dict[str, Any]:
    event = {
        "event_ts": _utc_now(),
        "event_type": event_type,
        "event_seq": int(context.get("event_seq", 0) or 0),
        "trace_id": context.get("trace_id"),
        "conversation_id": context.get("conversation_id"),
        "turn_id": context.get("turn_id"),
        "run_id": context.get("run_id"),
    }
    event.update(sanitize_for_trace(fields))
    return event


def start_turn_trace(
    *,
    conversation_id: str,
    turn_id: int,
    run_id: str,
    user_message: str,
    storage_dir: Optional[Any] = None,
    **fields: Any,
) -> Dict[str, Any]:
    trace_id = f"{conversation_id}:{turn_id}:{run_id}" if conversation_id else str(uuid4())
    trace: Dict[str, Any] = {
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "run_id": run_id,
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "storage_dir": str(storage_dir) if storage_dir is not None else None,
        "summary": sanitize_for_trace(
            {
                "user_message": user_message,
                **fields,
            }
        ),
        "events": [],
    }
    _set_active_trace(trace)
    record_trace_event("turn_start", user_message=user_message, **fields)
    return trace


def record_trace_event(event_type: str, **fields: Any) -> Optional[Dict[str, Any]]:
    trace = _get_active_trace()
    if not isinstance(trace, dict):
        return None

    event_seq = len(trace.setdefault("events", [])) + 1
    context = {**_base_context_from_trace(trace), "event_seq": event_seq}
    event = build_trace_event(context, event_type, **fields)
    trace["events"].append(event)
    trace["summary"].update(_summary_fields_for_event(event_type, event))
    _set_active_trace(trace)

    try:
        _persist_event(trace, event)
    except Exception:
        pass
    return event


def _summary_fields_for_event(event_type: str, event: Mapping[str, Any]) -> Dict[str, Any]:
    if event_type == "command_detected":
        return {
            "command_name": event.get("command_name"),
            "command_prefix": event.get("command_prefix"),
        }
    if event_type == "keyword_forced_sql":
        return {
            "matched_keywords": event.get("matched_keywords", []),
            "keyword_forced_sql": event.get("keyword_forced_sql", False),
        }
    if event_type == "router_result":
        return {
            "intent_type": event.get("intent_type"),
            "suggested_agents": event.get("suggested_agents"),
            "llm_router_suggested_chaining": event.get("llm_router_suggested_chaining"),
        }
    if event_type in {"agent_start", "agent_result"}:
        return {"agent_mode": event.get("agent_mode")}
    if event_type == "sql_execution":
        return {
            "sql": event.get("sql"),
            "sql_execution_status": event.get("status"),
            "df_A_state": event.get("df_A_state"),
        }
    if event_type == "data_readiness":
        return {
            "data_readiness_decision": event.get("decision"),
            "missing_columns": event.get("missing_columns"),
            "df_A_state": event.get("df_A_state"),
        }
    if event_type == "table_training_result":
        return {
            "table_training_status": event.get("status"),
            "table_training_message": event.get("message"),
            "active_table_context": event.get("active_table_context"),
        }
    if event_type == "chain_check":
        return {
            "chain_triggered": event.get("chain_triggered"),
            "last_sql_status": event.get("last_sql_status"),
            "llm_router_suggested_chaining": event.get("llm_router_suggested_chaining"),
        }
    if event_type in {"eda_validation", "turn_end"}:
        return {
            key: event.get(key)
            for key in ("error", "final_status", "assistant_response", "python_execution_status")
            if key in event
        }
    return {}


def finish_turn_trace(final_status: str = "completed", **fields: Any) -> Optional[Dict[str, Any]]:
    trace = _get_active_trace()
    if not isinstance(trace, dict):
        return None
    record_trace_event("turn_end", final_status=final_status, **fields)
    trace = _get_active_trace()
    if not isinstance(trace, dict):
        return None
    trace["status"] = final_status
    trace["finished_at"] = _utc_now()
    trace["summary"].update(sanitize_for_trace({"final_status": final_status, **fields}))

    try:
        _, _, latest_path = _storage_paths(trace.get("storage_dir"))
        public_trace = {key: value for key, value in trace.items() if key != "storage_dir"}
        _write_json(latest_path, public_trace)
    except Exception:
        pass
    return trace


def snapshot_session_state(session_state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if session_state is None:
        session_state = _streamlit_state() or {}

    keys = (
        "last_sql_status",
        "llm_router_suggested_chaining",
        "auto_eda_pending",
        "command_prefix",
        "last_agent_mode",
        "last_sql_statement",
        "last_sql_table",
        "df_A_state",
        "df_A_data",
        "active_table_context",
        "active_table_context_table",
        "databricks_selected_table",
        "databricks_table_input",
    )
    return {
        key: sanitize_for_trace(session_state.get(key))
        for key in keys
        if hasattr(session_state, "get") and key in session_state
    }


def read_recent_traces(limit: int = 20, *, storage_dir: Optional[Any] = None) -> List[Dict[str, Any]]:
    _, jsonl_path, _ = _storage_paths(storage_dir)
    if not jsonl_path.exists():
        return []
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, Any]] = []
    for line in lines[-max(1, int(limit)):]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def read_latest_trace(*, storage_dir: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    _, _, latest_path = _storage_paths(storage_dir)
    if not latest_path.exists():
        return None
    try:
        return json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def runtime_trace_fixture_for_keyword_sql() -> Dict[str, Any]:
    context = {
        "trace_id": "fixture",
        "conversation_id": "conv",
        "turn_id": 1,
        "run_id": "run",
    }
    return {
        "turn_start": build_trace_event(
            {**context, "event_seq": 1},
            "turn_start",
            user_message="housing이 yes 인사람들의 loan 분포를 그려줘",
        ),
        "keyword_forced_sql": build_trace_event(
            {**context, "event_seq": 2},
            "keyword_forced_sql",
            matched_keywords=["분포"],
            keyword_forced_sql=True,
            actual_command_prefix=False,
        ),
        "chain_check": build_trace_event(
            {**context, "event_seq": 3},
            "chain_check",
            handled_command=False,
            last_sql_status="success",
            llm_router_suggested_chaining=False,
            chain_triggered=False,
        ),
    }


__all__ = [
    "LATEST_TRACE_JSON",
    "TRACE_DIR",
    "TRACE_JSONL",
    "build_trace_event",
    "finish_turn_trace",
    "read_latest_trace",
    "read_recent_traces",
    "record_trace_event",
    "runtime_trace_fixture_for_keyword_sql",
    "sanitize_for_trace",
    "snapshot_session_state",
    "start_turn_trace",
]
