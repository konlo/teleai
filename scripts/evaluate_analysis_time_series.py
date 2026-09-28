#!/usr/bin/env python3
"""Evaluate production datetime preparation against an independent pandas oracle."""
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
from scripts.evaluate_analysis_statistics import ForbiddenModel


SOURCE = "evaluation.runtime_events"
PROMPT = (
    "보유 데이터에서 occurred_at 날짜를 기준으로 cohort_key별 일별 metric_amount 합계를 "
    "빈 날짜는 0으로 채워 선 그래프로 보여줘. 시간대는 Asia/Seoul"
)


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "occurred_at": [
            "2026-01-01 01:00", "2026-01-01 01:00", "2026-01-01 03:00",
            "2026-01-03 01:00", "2026-01-03 02:00",
        ],
        "cohort_key": ["alpha", "alpha", "beta", "alpha", "beta"],
        "metric_amount": [2, 4, 11, 7, 13],
    })


def independent_oracle(frame: pd.DataFrame) -> pd.DataFrame:
    reference = frame.copy()
    reference["occurred_at"] = pd.to_datetime(
        reference["occurred_at"], format="%Y-%m-%d %H:%M").dt.tz_localize("Asia/Seoul")
    reference["occurred_at"] = reference["occurred_at"].dt.floor("D")
    grouped = (reference.groupby(["occurred_at", "cohort_key"], sort=True)["metric_amount"]
               .sum().reset_index())
    groups = sorted(grouped["cohort_key"].unique())
    dates = pd.date_range(grouped["occurred_at"].min(), grouped["occurred_at"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [dates, groups], names=["occurred_at", "cohort_key"]).to_frame(index=False)
    return (grid.merge(grouped, on=["occurred_at", "cohort_key"], how="left")
            .fillna({"metric_amount": 0.0}).sort_values(["occurred_at", "cohort_key"])
            .reset_index(drop=True))


def evaluate() -> dict:
    frame = fixture()
    expected = independent_oracle(frame)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-timeseries-eval-") as root:
        runtime = GraphAnalysisRuntime(root, "evaluation", "datetime-series", ForbiddenModel())
        runtime.datasets.register(
            frame, source=SOURCE, coverage="complete", predicate_known=True,
            snapshot="fixture:datetime-series")
        try:
            outcome = runtime.submit(PROMPT)
            observations = {}
            for message in runtime.events():
                if isinstance(message, ToolMessage) and message.name in {
                        "prepare_time_series", "render_chart_spec"}:
                    observations.setdefault(message.name, []).append(json.loads(message.content))
            recovery = runtime.inspect().get("recovery", {})
            child_id = recovery.get("time_series_dataset")
            actual = runtime.datasets.frames.get(child_id)
            frame_matches = actual is not None and actual.equals(expected)
            chart_ids = runtime.inspect().get("chart_ids", [])
            png_valid = bool(chart_ids and runtime.artifacts[chart_ids[0]].image.startswith(
                b"\x89PNG\r\n\x1a\n"))
            chart = observations.get("render_chart_spec", [{}])[0]
            points = chart.get("render_summary", {}).get("points", [])
            point_tuples = [(point.get("x"), point.get("category"), point.get("value"))
                            for point in points]
            expected_plot = expected.sort_values(["cohort_key", "occurred_at"])
            expected_tuples = [
                (row.occurred_at.isoformat(), row.cohort_key, float(row.metric_amount))
                for row in expected_plot.itertuples(index=False)
            ]
            passed = bool(
                outcome.get("status") == "answered"
                and len(observations.get("prepare_time_series", [])) == 1
                and len(observations.get("render_chart_spec", [])) == 1
                and frame_matches and png_valid and point_tuples == expected_tuples
                and recovery.get("model_calls") == 0
            )
            prepared = observations.get("prepare_time_series", [{}])[0].get(
                "time_series_result", {})
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "PASS" if passed else "FAIL",
                "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
                "prompt": PROMPT,
                "data_scope": {"source": SOURCE, "rows": len(frame), "columns": list(frame.columns)},
                "oracle": {
                    "implementation": "independent pandas parse/floor/groupby/grid/merge",
                    "rows": len(expected),
                    "points": expected_tuples,
                },
                "evidence": {
                    "outcome": outcome.get("status"),
                    "tools": {name: len(items) for name, items in observations.items()},
                    "model_calls": recovery.get("model_calls"),
                    "remote_executions": 0,
                    "frame_matches": frame_matches,
                    "points_match": point_tuples == expected_tuples,
                    "png_valid": png_valid,
                    "preparation": prepared,
                },
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "limitations": [
                    "Local synthetic fixture; no Databricks or browser execution",
                    "This separate datetime benchmark does not increase the 66/200 primary grading score",
                    "Dual-axis and multi-panel rendering remain outside this case",
                ],
            }
        finally:
            runtime.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "docs/actual_agent_evaluation_time_series_2026-09-23.json")
    args = parser.parse_args(argv)
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "evidence": report["evidence"]},
                     ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
