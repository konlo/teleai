#!/usr/bin/env python3
"""Evaluate bounded pivot recovery against independent pandas oracles."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from langchain_core.messages import ToolMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_agent import (
    evaluate_case,
    fixture_reference_context,
    load_frames,
    load_grading,
    load_specs,
)
from scripts.evaluate_analysis_statistics import ForbiddenModel


PIVOT_CASES = (
    "L1_062", "L1_063", "L1_064",
    "L2_021", "L2_022", "L2_023", "L2_024", "L2_025",
    "L2_027", "L2_028", "L2_029", "L2_031", "L2_034",
)
SOURCE = "evaluation.runtime_matrix"


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            " | ".join(str(part) for part in item if str(part) not in {"", "None"})
            for item in result.columns.to_flat_index()
        ]
    else:
        result.columns = [str(column) for column in result.columns]
    return result


def arbitrary_schema_case(root: str) -> dict:
    frame = pd.DataFrame({
        "segment_code": ["west", "east", "west", "north", "east", "north"],
        "channel_code": ["web", "store", "store", "web", "web", "store"],
        "metric_value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    expected = _flatten(pd.crosstab(frame["segment_code"], frame["channel_code"]))
    runtime = GraphAnalysisRuntime(root, "evaluation", "pivot-arbitrary", ForbiddenModel())
    source_info = runtime.datasets.register(
        frame.copy(), source=SOURCE, coverage="complete", predicate_known=True,
        snapshot="fixture:pivot-arbitrary")
    runtime.context.reference_context[:] = [fixture_reference_context(SOURCE, frame)]
    try:
        outcome = runtime.submit(
            "segment_code를 행 축, channel_code를 열 축으로 개수(count) 피벗 테이블을 만들어줘")
        observations = [json.loads(message.content) for message in runtime.events()
                        if isinstance(message, ToolMessage) and message.name == "pivot_dataset"]
        recovery = runtime.inspect().get("recovery", {})
        observation = observations[-1] if observations else {}
        dataset_id = observation.get("dataset", {}).get("id")
        actual = runtime.datasets.frames.get(dataset_id)
        info = runtime.datasets.metadata.get(dataset_id)
        exact = actual is not None and actual.equals(expected)
        lineage = bool(info and info.parent_id == source_info.id)
        digest = bool(observation.get("pivot_result", {}).get("data_sha256"))
        evidence_before = recovery.get("pivot_evidence")
    finally:
        runtime.close()

    reopened = GraphAnalysisRuntime(root, "evaluation", "pivot-arbitrary", ForbiddenModel())
    try:
        restart_evidence = reopened.inspect().get("recovery", {}).get("pivot_evidence")
    finally:
        reopened.close()
    passed = (
        outcome.get("status") == "answered"
        and len(observations) == 1
        and recovery.get("model_calls") == 0
        and exact and lineage and digest
        and restart_evidence == evidence_before
    )
    return {
        "name": "arbitrary-schema production pivot",
        "status": "PASS" if passed else "FAIL",
        "tools": {"pivot_dataset": len(observations)},
        "model_calls": recovery.get("model_calls"),
        "remote_executions": 0,
        "frame_matches": exact,
        "lineage_matches": lineage,
        "digest_present": digest,
        "restart_evidence_matches": restart_evidence == evidence_before,
        "expected": expected.to_dict(orient="records"),
        "actual": actual.to_dict(orient="records") if actual is not None else None,
    }


def evaluate() -> dict:
    started = time.monotonic()
    specs = {item["id"]: item for item in load_specs()}
    grading = load_grading()
    frames = load_frames()
    with tempfile.TemporaryDirectory(prefix="telly-pivot-eval-") as root:
        arbitrary = arbitrary_schema_case(root)
    references = []
    for case_id in PIVOT_CASES:
        result = evaluate_case(
            specs[case_id], grading[case_id], ForbiddenModel(), frames=frames)
        references.append({
            "name": f"unchanged {case_id} prompt",
            "reference_id": case_id,
            "status": result["status"],
            "tools": result.get("tools", {}),
            "model_calls": result.get("runtime_metadata", {}).get("recovery_model_calls"),
            "remote_executions": result.get("remote_executions"),
            "evidence": result.get("evidence", {}),
        })
    cases = [arbitrary, *references]
    passed = all(case["status"] == "PASS" for case in cases)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
        "oracle": "independent pandas crosstab/pivot_table executed from unchanged reference code",
        "cases": cases,
        "passed": sum(case["status"] == "PASS" for case in cases),
        "total": len(cases),
        "primary_grading_cases": len(references),
        "primary_grading_total": 87,
        "model_calls": sum(case.get("model_calls") or 0 for case in cases),
        "remote_executions": sum(case.get("remote_executions") or 0 for case in cases),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Local fixtures only; no Databricks or browser execution",
            "Thirteen unchanged pivot prompts raise independent grading coverage to 87/200",
            "Multi-measure summaries and binned pivot requests remain outside this tool contract",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/actual_agent_evaluation_pivots_2026-09-24.json")
    args = parser.parse_args(argv)
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "passed", "total", "primary_grading_cases",
        "model_calls", "remote_executions")}, ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
