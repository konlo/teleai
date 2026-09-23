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
                    "select_outlier_rows", "aggregate_dataset", "compare_group_aggregates"}:
                observations.setdefault(message.name, []).append(json.loads(message.content))
        recovery = runtime.inspect().get("recovery", {})
        evidence = recovery.get("outlier_aggregate_evidence", {})
        frames = {
            key: runtime.datasets.frames[item["dataset"]["id"]].copy()
            for key, item in evidence.items()
        }
        lineage = {}
        for key, item in evidence.items():
            info = runtime.datasets.metadata[item["dataset"]["id"]]
            parents = tuple(info.parent_ids) or ((info.parent_id,) if info.parent_id else ())
            cohort_id = parents[-1] if parents else ""
            lineage[key] = {
                "aggregate_parent": info.parent_id,
                "aggregate_parents": list(parents),
                "cohort_parent": (runtime.datasets.metadata[cohort_id].parent_id
                                  if cohort_id in runtime.datasets.metadata else ""),
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
    baseline_mean = (frame.groupby("segment_code", sort=False)["anomaly_value"].mean()
                     .rename("baseline_mean_anomaly_value").reset_index())
    cohort_mean = (inliers.groupby("segment_code", sort=False)["anomaly_value"].mean()
                   .rename("cohort_mean_anomaly_value").reset_index())
    expected_comparison = baseline_mean.merge(
        cohort_mean, on="segment_code", how="outer", validate="one_to_one")
    expected_comparison["difference"] = (
        expected_comparison["cohort_mean_anomaly_value"]
        - expected_comparison["baseline_mean_anomaly_value"])
    expected_comparison["percent_change"] = (
        expected_comparison["difference"]
        / expected_comparison["baseline_mean_anomaly_value"].abs() * 100)
    expected_comparison = expected_comparison.sort_values("segment_code").reset_index(drop=True)
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
        compared = run_case(
            root, "grouped-comparison",
            "anomaly_value IQR 상한 이상치를 제외한 일반 행의 segment_code별 "
            "anomaly_value 평균을 계산하고 이상치 포함 전후 평균을 비교해줘",
            frame,
        )
    specs = {item["id"]: item for item in load_specs()}
    grading = load_grading()
    benchmark = evaluate_case(
        specs["L2_037"], grading["L2_037"], ForbiddenModel(), frames=load_frames())
    comparison_benchmark = evaluate_case(
        specs["L2_038"], grading["L2_038"], ForbiddenModel(), frames=load_frames())

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
    comparison_matches = (
        compared["frames"].get("comparison") is not None
        and compared["frames"]["comparison"].equals(expected_comparison)
    )
    comparison_lineage = all(
        item["aggregate_parents"] == [
            compared["parent_id"], compared["recovery"].get("outlier_dataset")]
        and item["cohort_parent"] == compared["parent_id"]
        for item in compared["lineage"].values()
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
        {
            "name": "parent-versus-inlier grouped mean",
            "status": "PASS" if (
                compared["outcome"].get("status") == "answered"
                and len(compared["observations"].get("select_outlier_rows", [])) == 1
                and len(compared["observations"].get("compare_group_aggregates", [])) == 1
                and compared["recovery"].get("model_calls") == 0
                and comparison_matches and comparison_lineage
            ) else "FAIL",
            "tools": {name: len(items) for name, items in compared["observations"].items()},
            "model_calls": compared["recovery"].get("model_calls"),
            "frame_matches": comparison_matches,
            "lineage_matches": comparison_lineage,
        },
        {
            "name": "unchanged L2_038 bank_loan prompt",
            "status": comparison_benchmark["status"],
            "tools": comparison_benchmark.get("tools", {}),
            "model_calls": comparison_benchmark.get("runtime_metadata", {}).get("recovery_model_calls"),
            "frame_matches": comparison_benchmark["status"] == "PASS",
            "lineage_matches": comparison_benchmark["status"] == "PASS",
            "reference_id": "L2_038",
            "evidence": comparison_benchmark.get("evidence", {}),
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
            "grouped_comparison": expected_comparison.to_dict(orient="records"),
        },
        "cases": cases,
        "passed": sum(case["status"] == "PASS" for case in cases),
        "total": len(cases),
        "model_calls": sum(case["model_calls"] or 0 for case in cases),
        "remote_executions": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Local synthetic fixture; no Databricks or browser execution",
            "L2_037 and L2_038 increase primary independent grading coverage to 68/200; three arbitrary-schema cases are supplementary",
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
