#!/usr/bin/env python3
"""Run the production agent loop's fault-injection and recovery contracts.

These are deterministic graph/tool tests. They prove that the controller can
reject, recover, persist, or stop safely for the named failure. They do not
measure a natural-language model's general understanding.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "docs" / "agentic_recovery_evaluation.json"

CASES = [
    ("A3_001", "bad SQL observation is repaired without restart",
     "migration.test_completion_contracts.CompletionTests.test_bad_sql_observation_allows_corrected_sql_without_restart"),
    ("A3_002", "wrong local population is rejected before execution and corrected",
     "migration.test_completion_contracts.CompletionTests.test_wrong_month_result_is_rejected_then_correct_scope_completes"),
    ("A3_003", "follow-up keeps inherited filters",
     "migration.test_completion_contracts.CompletionTests.test_followup_cannot_silently_drop_inherited_month"),
    ("A3_004", "wrong remote scope creates no approval before corrected query",
     "migration.test_completion_contracts.CompletionTests.test_wrong_remote_scope_never_creates_approval_before_corrected_query"),
    ("A3_005", "wrong chart population is rejected before tool execution",
     "migration.test_completion_contracts.CompletionTests.test_wrong_filtered_histogram_is_rejected_before_tool_execution"),
    ("A3_006", "grounded filtered count uses deterministic local fallback",
     "migration.test_completion_contracts.CompletionTests.test_grounded_filtered_count_has_deterministic_local_fallback"),
    ("A3_007", "grounded percentage separates population and numerator",
     "migration.test_completion_contracts.CompletionTests.test_grounded_ratio_has_deterministic_percent_fallback"),
    ("A3_008", "prose-only model exhausts a bounded recovery loop",
     "migration.test_recovery_journey.RecoveryJourneyTests.test_no_tool_model_exhausts_bounded_recovery"),
    ("A3_009", "repeated bad tool calls stop before graph recursion",
     "migration.test_completion_contracts.CompletionTests.test_identical_bad_tool_calls_stop_before_graph_recursion"),
    ("A3_010", "Databricks refusal never executes the remote query",
     "migration.test_recovery_journey.RecoveryJourneyTests.test_refusal_does_not_execute"),
    ("A3_011", "request scope survives restart while approval is pending",
     "migration.test_completion_contracts.CompletionTests.test_request_scope_survives_restart_while_waiting_for_approval"),
    ("A3_012", "crash after remote claim never resubmits",
     "migration.test_approval_rollout.LedgerTests.test_process_crash_after_claim_never_resubmits"),
    ("A3_013", "failed tool cannot be hidden by a final success claim",
     "migration.test_diagnostics.ToolOutcomeTests.test_final_claim_after_failed_tool_is_replaced"),
    ("A3_014", "wrong table ID becomes an observation and recovers locally",
     "migration.test_diagnostics.DiagnosticsTests.test_wrong_table_id_is_observation_then_recovers_without_database"),
    ("A3_015", "controller completes approval-to-render even when model stops",
     "migration.test_recovery_journey.PlannedJourneyTests.test_controller_connects_plan_approval_and_render_despite_model_stopping"),
    ("A3_016", "cached chart remains displayable without duplicate UI keys",
     "migration.test_cached_chart_ui.CachedChartUITests.test_repeated_prepared_chart_displays_without_duplicate_widget_keys"),
]


def run_case(case_id: str, capability: str, test_id: str) -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    output = StringIO()
    started = time.monotonic()
    with redirect_stdout(output), redirect_stderr(output):
        result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
    status = "PASS" if result.wasSuccessful() and result.testsRun == 1 else "FAIL"
    record = {"id": case_id, "capability": capability, "test": test_id,
              "status": status, "tests_run": result.testsRun,
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    if status != "PASS":
        record["diagnostic"] = output.getvalue()[-2000:]
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    results = []
    for case in CASES:
        record = run_case(*case)
        results.append(record)
        print(json.dumps({key: record[key] for key in ("id", "status", "elapsed_seconds")}))
    statuses = {status: sum(row["status"] == status for row in results)
                for status in ("PASS", "FAIL")}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "level": 3,
        "name": "production agentic recovery contracts",
        "evidence_type": "deterministic fault injection against GraphAnalysisRuntime and its approval/storage/UI boundaries",
        "coverage": {"selected": len(results), "statuses": statuses,
                     "all_selected_passed": statuses["FAIL"] == 0},
        "results": results,
        "limitations": [
            "Scripted fault injection proves controller behavior, not broad natural-language understanding.",
            "Remote Databricks calls use fakes and never contact a real workspace.",
            "Natural-model accuracy and browser rendering require separate reports.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.output}")
    return 0 if statuses["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
