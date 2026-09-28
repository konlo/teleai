"""Evaluate SQL proposals from the real TeleAI graph on public Spider2-Lite SQLite tasks.

This is a SQL-proposal adapter, not the full interactive product journey. It
never executes a remote query or gives gold answers to the agent. Submit the
saved SQL files to Spider2's unmodified official evaluator for accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SQLITE_PROPOSAL_INSTRUCTIONS = """You are evaluating a public SQLite SQL task.
Use the runtime table context to inspect the current schema. The task document is
reference data, separate from the user's question. Propose one read-only SQLite
SELECT through query_databricks; that legacy tool name only stages a proposal
behind approval in this evaluation and does not execute Databricks SQL.
The source argument must list each physical SQL table, separated by ` | `;
exclude CTE names and never put the tool name in source.
Use only columns, data types and functions supported by the observed SQLite
schema. Do not claim a computed answer before the query has been evaluated.
SQLite does not supply Databricks spatial functions by default. Check the
bounded type examples: a POINT stored as `(longitude,latitude)` text needs
numeric extraction from that text, and a JSON object needs its observed key.
Keep every filter, OR condition, join role and requested unit from the question.
Inspect declared foreign keys with inspect_table_relationships before joining.
Do not infer a relationship from similar names or silently pick an ambiguous role.
If the schema is insufficient, inspect it; if the request cannot be grounded,
report the missing information instead of inventing results.
"""


def task_manifest(spider_root: Path) -> dict:
    path = spider_root / "spider2-lite" / "spider2-lite.jsonl"
    return {item["instance_id"]: item for line in path.read_text().splitlines()
            if (item := json.loads(line))["instance_id"].startswith("local")}


def database_path(spider_root: Path, task: dict) -> Path:
    path = spider_root / "spider2-lite" / "resource" / "databases" / f"{task['db']}.sqlite"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Official SQLite database missing or empty: {path}")
    return path


def schema_context(path: Path) -> list[dict]:
    """Read public SQLite schema and bounded complex-type encodings, never gold."""
    contexts = []
    stamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for name in names:
            escaped = name.replace('"', '""')
            columns = []
            for row in db.execute(f'PRAGMA table_info("{escaped}")'):
                column = {"name": row[1], "dtype": row[2] or "unknown"}
                # Public benchmark types such as JSONB and POINT do not tell
                # the model how SQLite actually encodes a value. Inspect two
                # bounded, non-null column values, never entire source rows.
                if any(label in str(row[2]).casefold() for label in ('json', 'point')):
                    quoted = str(row[1]).replace('"', '""')
                    values = db.execute(
                        f'SELECT "{quoted}" FROM "{escaped}" '
                        f'WHERE "{quoted}" IS NOT NULL LIMIT 2').fetchall()
                    column['top_values'] = [str(value[0])[:160] for value in values]
                columns.append(column)
            foreign_keys = {}
            for row in db.execute(f'PRAGMA foreign_key_list("{escaped}")'):
                key = foreign_keys.setdefault(row[0], {'name': str(row[0]), 'columns': [],
                    'target_table': row[2], 'target_columns': []})
                key['columns'].append((row[1], row[3]))
                key['target_columns'].append((row[1], row[4]))
            for key in foreign_keys.values():
                key['columns'] = [value for _, value in sorted(key['columns'])]
                key['target_columns'] = [value for _, value in sorted(key['target_columns'])]
                if any(value is None for value in key['target_columns']):
                    target = key['target_table'].replace('"', '""')
                    primary = sorted((row[5], row[1]) for row in db.execute(f'PRAGMA table_info("{target}")') if row[5])
                    key['target_columns'] = [value for _, value in primary]
            contexts.append({"table": name, "training_status": "runtime_schema",
                             "observed_at": stamp, "columns": columns,
                             "relationship_authority": "database_catalog",
                             "foreign_keys": list(foreign_keys.values()),
                             "source": "read-only public Spider2 SQLite schema and bounded type examples"})
    return contexts


def task_document(spider_root: Path, task: dict) -> str:
    """Expose task-supplied public documentation, never benchmark gold answers."""
    name = task.get("external_knowledge")
    if not name:
        return ""
    if Path(name).name != name:
        raise ValueError("Invalid external knowledge filename")
    path = spider_root / "spider2-lite" / "resource" / "documents" / name
    if not path.is_file():
        raise FileNotFoundError(f"Official task document missing: {path}")
    return path.read_text()


def check_sqlite_candidate(path: Path, query: str, *, timeout_seconds=5.0) -> dict:
    """Execute only a public, read-only benchmark SELECT with a VM time bound.

    SQLite EXPLAIN alone does not detect missing runtime functions such as ST_Y.
    This probe is diagnostic; official Spider EX remains the correctness scorer.
    """
    from core.analysis_sql import validate_query
    try:
        validate_query(query, dialect="sqlite")
    except Exception as exc:
        return {"status": "sql_runtime_error", "error_type": type(exc).__name__,
                "error": str(exc)[:300]}
    started = time.monotonic()
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_progress_handler(
                lambda: int(time.monotonic() - started >= timeout_seconds), 1000)
            db.execute(query).fetchone()
    except sqlite3.Error as exc:
        message = str(exc)
        status = ("sql_timeout" if "interrupted" in message.lower()
                  and time.monotonic()-started >= timeout_seconds else
                  "dialect_error" if "no such function" in message.lower() else
                  "sql_runtime_error")
        return {"status": status, "error_type": type(exc).__name__,
                "error": message[:300],
                "elapsed_seconds": round(time.monotonic()-started, 3)}
    return {"status": "executable", "elapsed_seconds": round(time.monotonic()-started, 3)}


def evaluate_task(spider_root: Path, task: dict, model, predictions: Path,
                  *, benchmark_instruction=False) -> dict:
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    from langchain_core.messages import AIMessage, ToolMessage

    case_id = task["instance_id"]
    base = {"id": case_id, "db": task["db"], "benchmark": "Spider2-Lite SQLite",
            "mode": "TeleAI SQL proposal; no automatic approval or remote SQL execution; read-only public SQLite probe",
            "prompt_mode": "sql_instructed" if benchmark_instruction else "original_question"}
    try:
        path = database_path(spider_root, task)
        contexts = schema_context(path)
        supplied_document = task_document(spider_root, task)
    except (FileNotFoundError, sqlite3.DatabaseError, ValueError) as exc:
        return {**base, "status": "BLOCKED_INFRA", "error_type": type(exc).__name__}
    started = time.monotonic()
    remote_calls = []

    def forbidden_remote(_datasets):
        def execute(envelope):
            remote_calls.append(envelope)
            raise AssertionError("SQL proposal test must stop before approval")
        return execute

    with tempfile.TemporaryDirectory(prefix=f"teleai-spider-{case_id}-") as temp:
        try:
            runtime = GraphAnalysisRuntime(temp, "evaluation", case_id, model,
                connection_identity="public-spider2-sqlite-proposal-only",
                remote_factory=forbidden_remote,
                reference_context_loader=lambda: contexts,
                reference_document=supplied_document, sql_dialect="sqlite",
                proposal_validator=lambda query: check_sqlite_candidate(path, query),
                tool_allowlist=({'inspect_table_context', 'inspect_table_relationships', 'query_databricks'}
                                if benchmark_instruction else None),
                agent_instructions=(SQLITE_PROPOSAL_INSTRUCTIONS
                                    if benchmark_instruction else None))
        except Exception as exc:
            return {**base, "status": "FAIL", "error_type": type(exc).__name__,
                    "stage": "runtime_setup", "remote_executions": 0}
        try:
            outcome = runtime.submit(task["question"])
            requests = outcome.get("requests") or []
            events = runtime.events()
            calls = [call["name"] for message in events
                     if isinstance(message, AIMessage) for call in message.tool_calls]
            if remote_calls:
                return {**base, "status": "FAIL", "reason": "Unapproved SQL execution attempted",
                        "remote_executions": len(remote_calls), "tools": calls}
            if outcome.get("status") != "awaiting_approval" or len(requests) != 1:
                recovery = runtime.inspect().get("recovery") or {}
                if benchmark_instruction and outcome.get("status") == "answered":
                    from core.analysis_sql import validate_query
                    candidate = str(outcome.get("text") or "").strip()
                    if candidate.startswith("```sql") and candidate.endswith("```"):
                        candidate = candidate[6:-3].strip()
                    try:
                        validate_query(candidate, dialect="sqlite")
                    except Exception:
                        pass
                    else:
                        probe = check_sqlite_candidate(path, candidate)
                        if probe["status"] != "executable":
                            return {**base, "status": "SQL_INVALID", "probe": probe,
                                    "agent_status": outcome["status"],
                                    "remote_executions": 0,
                                    "elapsed_seconds": round(time.monotonic()-started, 3)}
                        predictions.mkdir(parents=True, exist_ok=True)
                        path = predictions / f"{case_id}.sql"
                        path.write_text(candidate + "\n")
                        return {**base, "status": "SQL_PROPOSED_TEXT",
                                "agent_status": outcome["status"], "prediction": str(path.resolve()),
                                "tools": calls, "model_calls": recovery.get("model_calls"),
                                "elapsed_seconds": round(time.monotonic()-started, 3),
                                "remote_executions": 0}
                # The public benchmark contains no private user data. Keep
                # rejected SQL proposals and structured tool errors so a
                # missing proposal can be diagnosed without a model replay.
                drafts = [call.get("args", {}) for message in events
                          if isinstance(message, AIMessage) for call in message.tool_calls
                          if call.get("name") == "query_databricks"]
                draft_probe = (check_sqlite_candidate(path, drafts[-1]["query"])
                               if drafts and isinstance(drafts[-1].get("query"), str) else None)
                observations = []
                for message in events:
                    if not isinstance(message, ToolMessage):
                        continue
                    try:
                        result = json.loads(message.content)
                    except (ValueError, TypeError):
                        result = {}
                    observations.append({"tool": message.name,
                                         "status": result.get("status"),
                                         "error_code": result.get("error_code")})
                failure_causes = []
                if draft_probe and draft_probe['status'] != 'executable':
                    failure_causes.append(draft_probe['status'])
                if recovery.get('scope_error'):
                    failure_causes.append(recovery['scope_error'])
                if recovery.get('proposal_error'):
                    failure_causes.append(recovery['proposal_error'])
                if outcome.get('error_type'):
                    error_type = str(outcome['error_type'])
                    failure_causes.append('model_timeout' if 'Timeout' in error_type
                                          else error_type)
                if not failure_causes:
                    failure_causes.append('model_no_output' if not drafts else 'agent_incomplete')
                return {**base, "status": "NO_SQL_PROPOSAL", "agent_status": outcome.get("status"),
                        "tools": calls, "model_calls": recovery.get("model_calls"),
                        "recovery_attempts": recovery.get("attempts"),
                        "sql_drafts": drafts, "observations": observations,
                        "draft_probe": draft_probe,
                        "failure_causes": list(dict.fromkeys(failure_causes)),
                        "remote_executions": 0,
                        "error_type": outcome.get("error_type"),
                        "elapsed_seconds": round(time.monotonic()-started, 3)}
            request = requests[0]
            sql = request["query"].strip()
            from core.analysis_load_plan import source_plan
            source_plan(request["source"], sql, dialect="sqlite")
            probe = check_sqlite_candidate(path, sql)
            if probe["status"] != "executable":
                return {**base, "status": "SQL_INVALID", "probe": probe,
                        "agent_status": outcome["status"], "tools": calls,
                        "remote_executions": 0,
                        "elapsed_seconds": round(time.monotonic()-started, 3)}
            predictions.mkdir(parents=True, exist_ok=True)
            (predictions / f"{case_id}.sql").write_text(sql + "\n")
            return {**base, "status": "SQL_PROPOSED", "agent_status": outcome["status"],
                    "prediction": str((predictions / f"{case_id}.sql").resolve()),
                    "probe": probe,
                    "tools": calls, "schema_tables": len(contexts),
                    "elapsed_seconds": round(time.monotonic()-started, 3),
                    "remote_executions": 0}
        except Exception as exc:
            return {**base, "status": "FAIL", "error_type": type(exc).__name__,
                    "elapsed_seconds": round(time.monotonic()-started, 3),
                    "remote_executions": len(remote_calls)}
        finally:
            runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spider-root", required=True, type=Path)
    parser.add_argument("--id", required=True, action="append")
    parser.add_argument("--provider", choices=("ollama", "databricks"), default="ollama")
    parser.add_argument("--model", default=None,
                        help="Model or serving endpoint name; defaults to the selected provider's configured model")
    parser.add_argument("--model-timeout-seconds", type=float, default=60.0,
                        help="Per-model HTTP timeout; 60 matches the current product setting")
    parser.add_argument("--benchmark-instruction", action="store_true",
                        help="Add a generic SQL/tool instruction, without table or gold leakage")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.model_timeout_seconds <= 0:
        parser.error("--model-timeout-seconds must be positive")
    from core.analysis_agent.model_provider import build_analysis_chat_model
    from core.analysis_agent.policy import RuntimePolicy
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    manifest = task_manifest(args.spider_root)
    unknown = set(args.id) - manifest.keys()
    if unknown:
        parser.error(f"Unknown official SQLite IDs: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    model_config = dict(os.environ)
    if args.model:
        model_config['OLLAMA_MODEL' if args.provider == 'ollama'
                     else 'TELLY_DATABRICKS_MODEL'] = args.model
    model = build_analysis_chat_model(
        RuntimePolicy(model_timeout_seconds=args.model_timeout_seconds),
        provider=args.provider, environ=model_config)
    selected_model = (model_config.get('OLLAMA_MODEL', 'gemma4:e4b')
                      if args.provider == 'ollama' else
                      model_config.get('TELLY_DATABRICKS_MODEL', 'databricks-qwen3-next-80b-a3b-instruct'))
    results = []
    for case_id in args.id:
        result = evaluate_task(args.spider_root, manifest[case_id], model,
                               args.output_dir / "predictions",
                               benchmark_instruction=args.benchmark_instruction)
        results.append(result)
        (args.output_dir / "proposals.json").write_text(json.dumps({"provider":args.provider,
            "model": selected_model,
            "model_timeout_seconds": args.model_timeout_seconds,
            "results": results}, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"id": case_id, "status": result["status"],
                          "elapsed_seconds": result.get("elapsed_seconds")}), flush=True)
    return 0 if all(item["status"] in {"SQL_PROPOSED", "SQL_PROPOSED_TEXT"}
                    for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
