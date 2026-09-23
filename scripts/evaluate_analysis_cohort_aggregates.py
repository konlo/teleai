#!/usr/bin/env python3
"""Evaluate lineage-safe cohort aggregates against independent pandas oracles."""
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
from scripts.evaluate_analysis_agent import (evaluate_case, fixture_reference_context, load_frames,
                                             load_grading, load_specs)
from scripts.evaluate_analysis_statistics import ForbiddenModel


SOURCE = "evaluation.runtime_observations"


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "anomaly_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 110, 120],
        "segment_code": ["north", "south", "east", "north", "south", "east",
                         "north", "south", "east", "north", "north", "south"],
        "score_value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
    })


def independent_cohorts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    q1, q3 = frame["anomaly_value"].quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    return (
        frame.loc[frame["anomaly_value"] > upper].copy(),
        frame.loc[frame["anomaly_value"] <= upper].copy(),
        float(upper),
    )


def run_case(root: str, conversation: str, prompt: str, frame: pd.DataFrame) -> dict:
    runtime = GraphAnalysisRuntime(root, "evaluation", conversation, ForbiddenModel())
    parent = runtime.datasets.register(
        frame.copy(), source=SOURCE, coverage="complete", predicate_known=True,
        snapshot="fixture:cohort-aggregate")
    runtime.context.reference_context[:] = [fixture_reference_context(SOURCE, frame)]
    try:
        outcome = runtime.submit(prompt)
        observations: dict[str, list[dict]] = {}
        for message in runtime.events():
            if isinstance(message, ToolMessage) and message.name in {
                    "select_outlier_rows", "aggregate_dataset"}:
                observations.setdefault(message.name, []).append(json.loads(message.content))
        recovery = runtime.inspect().get("recovery", {})
        evidence = recovery.get("outlier_aggregate_evidence", {})
        frames = {
            key: runtime.datasets.frames[item["dataset"]["id"]].copy()
            for key, item in evidence.items()
        }
        lineage = {
            key: {
                "aggregate_parent": runtime.datasets.metadata[item["dataset"]["id"]].parent_id,
                "cohort_parent": runtime.datasets.metadata[
                    runtime.datasets.metadata[item["dataset"]["id"]].parent_id].parent_id,
            }
            for key, item in evidence.items()
        }
        return {
            "outcome": outcome,
            "recovery": recovery,
            "observations": observations,
            "frames": frames,
            "lineage": lineage,
            "parent_id": parent.id,
        }
    finally:
        runtime.close()


def evaluate() -> dict:
    frame = fixture()
    outliers, inliers, upper = independent_cohorts(frame)
    expected_overall = pd.DataFrame({"mean_score_value": [outliers["score_value"].mean()]})
    expected_top = (outliers.groupby("segment_code", sort=False).size().rename("count").reset_index()
                    .sort_values("count", ascending=False, kind="mergesort").head(2)
                    .reset_index(drop=True))
    expected_grouped = (inliers.groupby("segment_code", sort=False)["anomaly_value"].mean()
                        .rename("mean_anomaly_value").reset_index()
                        .sort_values("mean_anomaly_value", ascending=False, kind="mergesort")
                        .reset_index(drop=True))
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-cohort-aggregate-eval-") as root:
        top = run_case(
            root, "top-frequency",
            f"{SOURCE}의 anomaly_value IQR 상한 1.5*IQR 초과 이상치들의 "
            "score_value 평균과 주요 segment_code TOP 2를 알려줘",
            frame,
        )
        grouped = run_case(
            root, "grouped-mean",
            "anomaly_value IQR 상한 이상치를 제외한 일반 행의 "
            "segment_code별 anomaly_value 평균을 보여줘",
            frame,
        )
    specs = {item["id"]: item for item in load_specs()}
    grading = load_grading()
    benchmark = evaluate_case(
        specs["L2_037"], grading["L2_037"], ForbiddenModel(), frames=load_frames())

    top_matches = (
        top["frames"].get("overall") is not None
        and top["frames"].get("grouped") is not None
        and top["frames"]["overall"].equals(expected_overall)
        and top["frames"]["grouped"].equals(expected_top)
    )
    grouped_matches = (
        grouped["frames"].get("grouped") is not None
        and grouped["frames"]["grouped"].equals(expected_grouped)
    )
    top_lineage = all(
        item["aggregate_parent"] == top["recovery"].get("outlier_dataset")
        and item["cohort_parent"] == top["parent_id"]
        for item in top["lineage"].values()
    )
    grouped_lineage = all(
        item["aggregate_parent"] == grouped["recovery"].get("outlier_dataset")
        and item["cohort_parent"] == grouped["parent_id"]
        for item in grouped["lineage"].values()
    )
    cases = [
        {
            "name": "outlier overall mean plus grouped top frequency",
            "status": "PASS" if (
                top["outcome"].get("status") == "answered"
                and len(top["observations"].get("select_outlier_rows", [])) == 1
                and len(top["observations"].get("aggregate_dataset", [])) == 2
                and top["recovery"].get("model_calls") == 0
                and top_matches and top_lineage
            ) else "FAIL",
            "tools": {name: len(items) for name, items in top["observations"].items()},
            "model_calls": top["recovery"].get("model_calls"),
            "frame_matches": top_matches,
            "lineage_matches": top_lineage,
        },
        {
            "name": "inlier grouped mean",
            "status": "PASS" if (
                grouped["outcome"].get("status") == "answered"
                and len(grouped["observations"].get("select_outlier_rows", [])) == 1
                and len(grouped["observations"].get("aggregate_dataset", [])) == 1
                and grouped["recovery"].get("model_calls") == 0
                and grouped_matches and grouped_lineage
            ) else "FAIL",
            "tools": {name: len(items) for name, items in grouped["observations"].items()},
            "model_calls": grouped["recovery"].get("model_calls"),
            "frame_matches": grouped_matches,
            "lineage_matches": grouped_lineage,
        },
        {
            "name": "unchanged L2_037 bank_loan prompt",
            "status": benchmark["status"],
            "tools": benchmark.get("tools", {}),
            "model_calls": benchmark.get("runtime_metadata", {}).get("recovery_model_calls"),
            "frame_matches": benchmark["status"] == "PASS",
            "lineage_matches": benchmark["status"] == "PASS",
            "reference_id": "L2_037",
            "evidence": benchmark.get("evidence", {}),
        },
    ]
    passed = all(case["status"] == "PASS" for case in cases)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
        "data_scope": {"source": SOURCE, "rows": len(frame), "columns": list(frame.columns)},
        "oracle": {
            "implementation": "independent pandas quantile/filter/groupby/mean/size/sort/head",
            "iqr_upper": upper,
            "outlier_rows": len(outliers),
            "inlier_rows": len(inliers),
            "overall": expected_overall.to_dict(orient="records"),
            "top_groups": expected_top.to_dict(orient="records"),
            "grouped_mean": expected_grouped.to_dict(orient="records"),
        },
        "cases": cases,
        "passed": sum(case["status"] == "PASS" for case in cases),
        "total": len(cases),
        "model_calls": sum(case["model_calls"] or 0 for case in cases),
        "remote_executions": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Local synthetic fixture; no Databricks or browser execution",
            "L2_037 increases primary independent grading coverage to 67/200; the two arbitrary-schema cases are supplementary",
            "Parent-versus-cohort side-by-side comparison remains outside these cases",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/actual_agent_evaluation_cohort_aggregates_2026-09-23.json")
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
