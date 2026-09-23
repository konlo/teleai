#!/usr/bin/env python3
"""Evaluate bounded, non-mutating winsorization with no model or remote calls."""
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

import numpy as np
import pandas as pd
from langchain_core.messages import ToolMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_agent import evaluate_case, load_frames, load_grading, load_specs
from scripts.evaluate_analysis_statistics import ForbiddenModel


SOURCE = "evaluation.runtime_sensor_values"


def arbitrary_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "metric_value": [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, np.nan],
        "payload": list("abcdefgh"),
    })


def run_arbitrary(root: str) -> dict:
    frame = arbitrary_frame()
    before = frame.copy(deep=True)
    runtime = GraphAnalysisRuntime(root, "evaluation", "winsor-arbitrary", ForbiddenModel())
    info = runtime.datasets.register(
        frame.copy(), source=SOURCE, coverage="complete", predicate_known=True,
        snapshot="fixture:winsorization")
    try:
        outcome = runtime.submit(
            "metric_value의 극단치 왜곡을 줄이도록 상하위 10% 윈저화(Clipping)를 적용하고 "
            "원본 평균과 보정 평균을 비교해줘")
        observations = []
        for message in runtime.events():
            if isinstance(message, ToolMessage) and message.name == "winsorize_numeric":
                observations.append(json.loads(message.content))
        recovery = runtime.inspect().get("recovery", {})
        result = observations[-1]["winsorization_result"] if observations else {}
        clean = before["metric_value"].dropna().astype(float)
        lower, upper = clean.quantile([0.1, 0.9])
        clipped = clean.clip(lower=lower, upper=upper)
        expected = {
            "lower": float(lower), "upper": float(upper),
            "original_mean": float(clean.mean()),
            "winsorized_mean": float(clipped.mean()),
            "lower_count": int(clean.lt(lower).sum()),
            "upper_count": int(clean.gt(upper).sum()),
        }
        actual = {
            "lower": result.get("thresholds", {}).get("lower"),
            "upper": result.get("thresholds", {}).get("upper"),
            "original_mean": result.get("original", {}).get("mean"),
            "winsorized_mean": result.get("winsorized", {}).get("mean"),
            "lower_count": result.get("clipped_counts", {}).get("lower"),
            "upper_count": result.get("clipped_counts", {}).get("upper"),
        }
        numeric = ("lower", "upper", "original_mean", "winsorized_mean")
        passed = (
            outcome.get("status") == "answered"
            and len(observations) == 1
            and recovery.get("model_calls") == 0
            and all(np.isclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
                    for key in numeric)
            and actual["lower_count"] == expected["lower_count"]
            and actual["upper_count"] == expected["upper_count"]
            and runtime.datasets.frames[info.id].equals(before)
        )
        return {
            "name": "arbitrary-schema winsorization",
            "status": "PASS" if passed else "FAIL",
            "tools": {"winsorize_numeric": len(observations)},
            "model_calls": recovery.get("model_calls"),
            "expected": expected,
            "actual": actual,
            "source_unchanged": runtime.datasets.frames[info.id].equals(before),
        }
    finally:
        runtime.close()


def evaluate() -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-winsor-eval-") as root:
        arbitrary = run_arbitrary(root)
    specs = {item["id"]: item for item in load_specs()}
    grading = load_grading()
    result = evaluate_case(
        specs["L2_096"], grading["L2_096"], ForbiddenModel(), frames=load_frames())
    reference = {
        "name": "unchanged L2_096 prompt",
        "reference_id": "L2_096",
        "status": result["status"],
        "tools": result.get("tools", {}),
        "model_calls": result.get("runtime_metadata", {}).get("recovery_model_calls"),
        "remote_executions": result.get("remote_executions"),
        "evidence": result.get("evidence", {}),
    }
    cases = [arbitrary, reference]
    passed = all(case["status"] == "PASS" for case in cases)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
        "cases": cases,
        "passed": sum(case["status"] == "PASS" for case in cases),
        "total": len(cases),
        "model_calls": sum(case.get("model_calls") or 0 for case in cases),
        "remote_executions": sum(case.get("remote_executions") or 0 for case in cases),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Local fixtures; no Databricks or browser execution",
            "The tool returns aggregate evidence and intentionally does not materialize a transformed dataset",
            "L2_096 increases primary independent grading coverage to 74/200 together with L2_089",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/actual_agent_evaluation_winsorization_2026-09-23.json")
    args = parser.parse_args(argv)
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "passed", "total", "model_calls", "remote_executions")},
                     ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
