#!/usr/bin/env python3
"""Evaluate the production graph's bounded join against an independent pandas oracle."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime


CUSTOMERS = ROOT / "tests/analysis_benchmark_100/customer_analytics.csv"
TRANSACTIONS = ROOT / "tests/analysis_benchmark_100/transaction_history.csv"


class ForbiddenModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "join-evaluation-forbids-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("deterministic join evaluation must not call a model")


def _digest(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values("tx_id").reset_index(drop=True)
    columns = json.dumps(list(normalized.columns), ensure_ascii=False).encode()
    values = pd.util.hash_pandas_object(normalized, index=False).values.tobytes()
    return sha256(columns + values).hexdigest()


def evaluate() -> dict:
    customers = pd.read_csv(CUSTOMERS)
    transactions = pd.read_csv(TRANSACTIONS)
    reference = customers.merge(transactions, on="customer_id", how="inner")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-join-eval-") as root:
        runtime = GraphAnalysisRuntime(root, "evaluation", "benchmark-join", ForbiddenModel())
        left = runtime.datasets.register(
            customers, source="customer_analytics", coverage="complete",
            predicate_known=True, snapshot="fixture:customer_analytics")
        right = runtime.datasets.register(
            transactions, source="transaction_history", coverage="complete",
            predicate_known=True, snapshot="fixture:transaction_history")
        try:
            outcome = runtime.submit(
                "customer_analytics와 transaction_history를 customer_id로 내부 조인해서 "
                "조인 결과 행 개수와 컬럼 수를 보여줘")
            observations = []
            for message in runtime.events():
                if isinstance(message, ToolMessage) and message.name == "join_datasets":
                    observations.append(json.loads(message.content))
            joined = [info for info in runtime.datasets.metadata.values() if info.grain == "joined"]
            actual = runtime.datasets.frames[joined[-1].id] if joined else pd.DataFrame()
            same_values = False
            if list(actual.columns) == list(reference.columns) and len(actual) == len(reference):
                try:
                    pd.testing.assert_frame_equal(
                        actual.sort_values("tx_id").reset_index(drop=True),
                        reference.sort_values("tx_id").reset_index(drop=True),
                        check_dtype=False,
                    )
                    same_values = True
                except AssertionError:
                    pass
            recovery = runtime.inspect().get("recovery", {})
            passed = bool(
                outcome.get("status") == "answered"
                and len(observations) == 1
                and len(joined) == 1
                and joined[0].parent_ids == (left.id, right.id)
                and observations[0].get("join_summary", {}).get("relationship") == "one_to_many"
                and observations[0].get("join_summary", {}).get("expected_rows") == len(reference)
                and observations[0].get("join_summary", {}).get("actual_rows") == len(reference)
                and same_values
                and recovery.get("model_calls") == 0
            )
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "PASS" if passed else "FAIL",
                "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
                "prompt": "customer_analytics와 transaction_history를 customer_id로 내부 조인해서 조인 결과 행 개수와 컬럼 수를 보여줘",
                "data_scope": {
                    "left_fixture": str(CUSTOMERS.relative_to(ROOT)),
                    "right_fixture": str(TRANSACTIONS.relative_to(ROOT)),
                    "left_rows": len(customers),
                    "right_rows": len(transactions),
                },
                "oracle": {
                    "implementation": "pandas.DataFrame.merge independent of production DuckDB join",
                    "rows": len(reference),
                    "columns": len(reference.columns),
                    "data_sha256": _digest(reference),
                },
                "evidence": {
                    "outcome": outcome.get("status"),
                    "tool_observations": len(observations),
                    "join_summary": observations[0].get("join_summary") if observations else None,
                    "dataset": asdict(joined[0]) if joined else None,
                    "actual_data_sha256": _digest(actual) if same_values else None,
                    "same_values": same_values,
                    "model_calls": recovery.get("model_calls"),
                    "remote_executions": 0,
                },
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "limitations": [
                    "Local fixture evaluation; no Databricks or browser execution",
                    "Validates an unambiguous inner join journey; many-to-many and output-limit rejection are unit contracts",
                ],
            }
        finally:
            runtime.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/actual_agent_evaluation_join_2026-09-19.json")
    args = parser.parse_args(argv)
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "evidence": report["evidence"]}, ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
