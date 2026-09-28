#!/usr/bin/env python3
"""Evaluate production statistical recovery against an independent SciPy oracle."""
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
from scipy import stats
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_agent import evaluate_case, load_frames, load_grading, load_specs


FIXTURE = ROOT / "test_set/data/bank_loan.csv"
PROMPT = "bank_loan에서 y 그룹별 balance 차이에 대해 독립표본 t-검정을 수행하고 p-value를 보여줘"


class ForbiddenModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "statistics-evaluation-forbids-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("unambiguous statistical evaluation must not call a model")


def evaluate() -> dict:
    frame = pd.read_csv(FIXTURE)
    levels = list(pd.unique(frame["y"].dropna()))
    first = frame.loc[frame["y"] == levels[0], "balance"].dropna().to_numpy(dtype=float)
    second = frame.loc[frame["y"] == levels[1], "balance"].dropna().to_numpy(dtype=float)
    reference = stats.ttest_ind(first, second, equal_var=False)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="telly-statistics-eval-") as root:
        runtime = GraphAnalysisRuntime(root, "evaluation", "benchmark-statistics", ForbiddenModel())
        runtime.datasets.register(
            frame,
            source="bank_loan",
            coverage="complete",
            predicate_known=True,
            snapshot="fixture:bank_loan",
        )
        try:
            outcome = runtime.submit(PROMPT)
            observations = []
            for message in runtime.events():
                if isinstance(message, ToolMessage) and message.name == "statistical_test":
                    observations.append(json.loads(message.content))
            result = observations[0].get("test_result", {}) if observations else {}
            recovery = runtime.inspect().get("recovery", {})
            statistic_matches = abs(float(result.get("statistic", float("inf"))) - float(reference.statistic)) < 1e-12
            p_value_matches = abs(float(result.get("p_value", float("inf"))) - float(reference.pvalue)) < 1e-12
            benchmark_passed = bool(
                outcome.get("status") == "answered"
                and len(observations) == 1
                and result.get("kind") == "independent_t"
                and result.get("sample", {}).get("complete_rows") == len(first) + len(second)
                and result.get("sample", {}).get("dropped_rows") == int(frame[["balance", "y"]].isna().any(axis=1).sum())
                and statistic_matches
                and p_value_matches
                and result.get("effect_size", {}).get("name") == "hedges_g"
                and len(result.get("confidence_intervals", [])) == 1
                and recovery.get("model_calls") == 0
            )
            specs = {spec["id"]: spec for spec in load_specs()}
            grading = load_grading()
            frames = load_frames()
            case_ids = [f"L2_{number:03d}" for number in range(51, 61)]
            case_results = [
                evaluate_case(specs[case_id], grading[case_id], ForbiddenModel(), frames=frames)
                for case_id in case_ids
            ]
            suite_passed = all(
                result.get("status") == "PASS"
                and result.get("tools") == {"statistical_test": 1}
                and result.get("runtime_metadata", {}).get("recovery_model_calls") == 0
                and result.get("remote_executions") == 0
                for result in case_results
            )
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "status": "PASS" if benchmark_passed and suite_passed else "FAIL",
                "runtime": "core.analysis_agent.runtime.GraphAnalysisRuntime",
                "prompt": PROMPT,
                "data_scope": {
                    "fixture": str(FIXTURE.relative_to(ROOT)),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "groups": [str(level) for level in levels],
                },
                "oracle": {
                    "implementation": "scipy.stats.ttest_ind(equal_var=False) independent of production recovery",
                    "statistic": float(reference.statistic),
                    "p_value": float(reference.pvalue),
                    "sample_sizes": [len(first), len(second)],
                },
                "evidence": {
                    "outcome": outcome.get("status"),
                    "tool_observations": len(observations),
                    "test_result": result or None,
                    "statistic_matches": statistic_matches,
                    "p_value_matches": p_value_matches,
                    "model_calls": recovery.get("model_calls"),
                    "remote_executions": 0,
                },
                "level2_statistical_suite": {
                    "case_ids": case_ids,
                    "passed": sum(result.get("status") == "PASS" for result in case_results),
                    "total": len(case_results),
                    "model_calls": sum(
                        result.get("runtime_metadata", {}).get("recovery_model_calls", 0)
                        for result in case_results),
                    "remote_executions": sum(result.get("remote_executions", 0)
                                             for result in case_results),
                    "results": case_results,
                },
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "limitations": [
                    "Local fixture evaluation; no Databricks or browser execution",
                    "The ten statistical journeys use the unchanged Level 2 prompts and checked-in SciPy reference code",
                    "Observation independence and paired-study validity cannot be inferred from dataframe values alone",
                ],
            }
        finally:
            runtime.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/actual_agent_evaluation_statistics_2026-09-19.json")
    args = parser.parse_args(argv)
    report = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "evidence": report["evidence"]}, ensure_ascii=False))
    print(f"Report: {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
