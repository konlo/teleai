#!/usr/bin/env python3
"""Evaluate production outlier recovery against unchanged Level 2 references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_analysis_agent import evaluate_case, load_frames, load_grading, load_specs
from scripts.evaluate_analysis_statistics import ForbiddenModel


CASE_IDS = ["L2_036", "L2_039", "L2_048"]


def evaluate() -> dict:
    specs = {spec["id"]: spec for spec in load_specs()}
    grading = load_grading()
    frames = load_frames()
    started = time.monotonic()
    results = [
        evaluate_case(specs[case_id], grading[case_id], ForbiddenModel(), frames=frames)
        for case_id in CASE_IDS
    ]
    passed = all(
        result.get("status") == "PASS"
        and result.get("tools") == {"detect_outliers": 1}
        and result.get("runtime_metadata", {}).get("recovery_model_calls") == 0
        and result.get("remote_executions") == 0
        for result in results
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
        "case_ids": CASE_IDS,
        "passed": sum(result.get("status") == "PASS" for result in results),
        "total": len(results),
        "model_calls": sum(
            result.get("runtime_metadata", {}).get("recovery_model_calls", 0)
            for result in results),
        "remote_executions": sum(result.get("remote_executions", 0) for result in results),
        "results": results,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Local fixture evaluation; no Databricks or browser execution",
            "Only direct single-column IQR/Z-score detection journeys are graded here",
            "Outlier-row profiling, winsorization, and transformation remain ungraded",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/actual_agent_evaluation_outliers_2026-09-21.json")
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
