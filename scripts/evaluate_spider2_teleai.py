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
    """Read the live public SQLite schema without looking at benchmark gold."""
    contexts = []
    stamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        for name in names:
            escaped = name.replace('"', '""')
            columns = [{"name": row[1], "dtype": row[2] or "unknown"}
                       for row in db.execute(f'PRAGMA table_info("{escaped}")')]
            contexts.append({"table": name, "training_status": "runtime_schema",
                             "observed_at": stamp, "columns": columns,
                             "source": "read-only public Spider2 SQLite schema"})
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


def evaluate_task(spider_root: Path, task: dict, model, predictions: Path,
                  *, benchmark_instruction=False) -> dict:
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    from langchain_core.messages import AIMessage, ToolMessage

    case_id = task["instance_id"]
    base = {"id": case_id, "db": task["db"], "benchmark": "Spider2-Lite SQLite",
            "mode": "TeleAI SQL proposal; no automatic approval or SQL execution",
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
                reference_context_loader=lambda: contexts)
        except Exception as exc:
            return {**base, "status": "FAIL", "error_type": type(exc).__name__,
                    "stage": "runtime_setup", "remote_executions": 0}
        try:
            prompt = task["question"]
            if benchmark_instruction:
                prompt = ("For this public SQLite benchmark, use the available schema "
                          "to propose a read-only SQL query through the approval-gated "
                          "query tool. Do not claim the answer before execution. "
                          "Question: " + prompt)
            if supplied_document:
                prompt += "\n\nTask-supplied reference document:\n" + supplied_document
            outcome = runtime.submit(prompt)
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
                        validate_query(candidate)
                    except Exception:
                        pass
                    else:
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
                return {**base, "status": "NO_SQL_PROPOSAL", "agent_status": outcome.get("status"),
                        "tools": calls, "model_calls": recovery.get("model_calls"),
                        "recovery_attempts": recovery.get("attempts"),
                        "sql_drafts": drafts, "observations": observations,
                        "remote_executions": 0,
                        "error_type": outcome.get("error_type"),
                        "elapsed_seconds": round(time.monotonic()-started, 3)}
            request = requests[0]
            sql = request["query"].strip()
            from core.analysis_load_plan import source_plan
            source_plan(request["source"], sql)
            predictions.mkdir(parents=True, exist_ok=True)
            (predictions / f"{case_id}.sql").write_text(sql + "\n")
            return {**base, "status": "SQL_PROPOSED", "agent_status": outcome["status"],
                    "prediction": str((predictions / f"{case_id}.sql").resolve()),
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
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma4:e4b"))
    parser.add_argument("--model-timeout-seconds", type=float, default=60.0,
                        help="Per-model HTTP timeout; 60 matches the current product setting")
    parser.add_argument("--benchmark-instruction", action="store_true",
                        help="Add a generic SQL/tool instruction, without table or gold leakage")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.model_timeout_seconds <= 0:
        parser.error("--model-timeout-seconds must be positive")
    from langchain_ollama import ChatOllama
    manifest = task_manifest(args.spider_root)
    unknown = set(args.id) - manifest.keys()
    if unknown:
        parser.error(f"Unknown official SQLite IDs: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    model = ChatOllama(model=args.model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        reasoning=True, temperature=0, num_ctx=16384, num_predict=4096,
        client_kwargs={"timeout": args.model_timeout_seconds})
    results = []
    for case_id in args.id:
        result = evaluate_task(args.spider_root, manifest[case_id], model,
                               args.output_dir / "predictions",
                               benchmark_instruction=args.benchmark_instruction)
        results.append(result)
        (args.output_dir / "proposals.json").write_text(json.dumps({"model": args.model,
            "model_timeout_seconds": args.model_timeout_seconds,
            "results": results}, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"id": case_id, "status": result["status"],
                          "elapsed_seconds": result.get("elapsed_seconds")}), flush=True)
    return 0 if all(item["status"] in {"SQL_PROPOSED", "SQL_PROPOSED_TEXT"}
                    for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
