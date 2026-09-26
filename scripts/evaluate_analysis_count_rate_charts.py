#!/usr/bin/env python3
"""Evaluate bounded count/rate composite charts with no model or remote calls."""
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
from scripts.evaluate_analysis_agent import evaluate_case, load_frames, load_grading, load_specs
from scripts.evaluate_analysis_statistics import ForbiddenModel


SOURCE = "evaluation.runtime_campaign_events"


def arbitrary_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "period_code": ["jan", "jan", "feb", "feb", "feb", "mar"],
        "result_flag": ["pass", "fail", "pass", "fail", None, "pass"],
        "payload": [4, 8, 15, 16, 23, 42],
    })


def expected_points(frame: pd.DataFrame) -> list[dict]:
    grouped = frame.dropna(subset=["period_code"]).groupby(
        "period_code", sort=False, observed=True)["result_flag"]
    result = grouped.agg(row_count="size", denominator_count="count").reset_index()
    success = grouped.apply(lambda values: int(values.eq("pass").sum())).reset_index(
        name="success_count")
    result = result.merge(success, on="period_code", validate="one_to_one")
    result["rate_percent"] = 100.0 * result["success_count"] / result["denominator_count"]
    order = {"jan": 1, "feb": 2, "mar": 3}
    result = result.sort_values("period_code", key=lambda values: values.map(order))
    return [{
        "group": row["period_code"],
        "row_count": int(row["row_count"]),
        "denominator_count": int(row["denominator_count"]),
        "success_count": int(row["success_count"]),
        "rate_percent": float(row["rate_percent"]),
    } for _, row in result.iterrows()]


def run_arbitrary(root: str) -> dict:
    frame = arbitrary_frame()
    runtime = GraphAnalysisRuntime(root, "evaluation", "count-rate-arbitrary", ForbiddenModel())
    runtime.datasets.register(
        frame, source=SOURCE, coverage="complete", predicate_known=True,
        snapshot="fixture:count-rate")
    try:
        outcome = runtime.submit(
            "period_code별 전체 건수(막대)와 result_flag='pass' 성공률(선)을 "
            "이중 축(Dual Y-axis)으로 시각화해줘")
        observations = []
        for message in runtime.events():
            if isinstance(message, ToolMessage) and message.name == "render_count_rate_chart":
                observations.append(json.loads(message.content))
        recovery = runtime.inspect().get("recovery", {})
        actual = observations[-1]["render_summary"]["points"] if observations else []
        card = (runtime.artifacts[observations[-1]["cards"][0]["id"]]
                if observations else None)
        expected = expected_points(frame)
        passed = (
            outcome.get("status") == "answered"
            and len(observations) == 1
            and recovery.get("model_calls") == 0
            and actual == expected
            and card is not None and card.kind == "dual_axis"
            and card.image.startswith(b"\x89PNG\r\n\x1a\n")
        )
        return {
            "name": "arbitrary-schema count/rate dual axis",
            "status": "PASS" if passed else "FAIL",
            "tools": {"render_count_rate_chart": len(observations)},
            "model_calls": recovery.get("model_calls"),
            "expected": expected,
            "actual": actual,
        }
    finally:
        runtime.close()


def evaluate() -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-count-rate-eval-") as root:
        arbitrary = run_arbitrary(root)
    specs = {item["id"]: item for item in load_specs()}
    grading = load_grading()
    frames = load_frames()
    cases = [arbitrary]
    for reference_id in ("L2_061", "L2_064", "L2_065", "L2_066", "L2_089"):
        result = evaluate_case(
            specs[reference_id], grading[reference_id], ForbiddenModel(), frames=frames)
        cases.append({
            "name": f"unchanged {reference_id} prompt",
            "reference_id": reference_id,
            "status": result["status"],
            "tools": result.get("tools", {}),
            "model_calls": result.get("runtime_metadata", {}).get("recovery_model_calls"),
            "remote_executions": result.get("remote_executions"),
            "evidence": result.get("evidence", {}),
        })
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
            "L2_061, L2_064, L2_065, L2_066 and L2_089 are covered by the count/rate contract",
            "General 2x2 dashboards and arbitrary multi-measure panels remain ungraded",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/actual_agent_evaluation_count_rate_charts_2026-09-23.json")
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
