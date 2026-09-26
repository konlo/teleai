"""Run DeepEval against recorded real TeleAI fixture runs, never mock traces.

Requires the separate DeepEval environment. The local judge is supplementary;
independent fixture oracles remain the authority for numeric/chart correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

EXPECTED_SINGLE_TOOL = {
    "L1_001": "inspect_table_context",
    "L1_016": "aggregate_dataset",
    "L1_017": "local_analysis_sql",
    "L1_036": "local_analysis_sql",
    "L1_062": "pivot_dataset",
    "L1_076": "prepare_histogram",
    "L2_005": "local_analysis_sql",
    "L2_036": "detect_outliers",
    "L2_051": "statistical_test",
    "L2_061": "render_count_rate_chart",
}
JUDGE_CASES = {"L1_001", "L1_016", "L1_017", "L1_036", "L2_005", "L2_036", "L2_051"}


def real_records(source):
    """Accept measured graph reports, retaining synthetic-fault qualifications."""
    if source.get('mode') in {'live-local-model', 'live-databricks-model'}:
        return source['results']
    if source.get('mode') != 'real-model synthetic journeys; deterministic rescue disabled':
        raise ValueError('Only real-model graph reports are accepted')
    records=[]
    for case in source['results']:
        if not case.get('live_calls'):
            continue
        for number, turn in enumerate(case.get('turns', []), 1):
            records.append({'id':case['id']+(':'+str(number) if number>1 else ''),
                'prompt':turn['prompt'], 'status':turn['status'],
                'final_output':turn.get('final_output',''),
                'runtime_metadata':{'recovery_model_calls':turn['model_calls']},
                'tool_calls':[{'tool':name} for name in turn['tools']],
                'reference_facts':('The requested calculation equals '+str(turn['expected'])+'. '
                    + ('A chart was independently verified as generated. ' if turn['chart_count'] else '')
                    + 'Answer in ordinary prose; no JSON or particular output format is required.')})
    return records


def expected_answer(record: dict) -> str:
    if record.get('reference_facts'):
        return record['reference_facts']
    evidence = record.get("evidence") or {}
    if "expected" in evidence:
        return json.dumps(evidence["expected"], ensure_ascii=False, default=str)
    if "metadata" in evidence:
        metadata = evidence["metadata"]
        return json.dumps({"column_count": metadata.get("column_count"),
                           "columns": metadata.get("columns")}, ensure_ascii=False)
    return "The answer must complete the user's request with a verified result."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-model", default="gemma4:e4b")
    parser.add_argument("--judge-id", action="append",
                        help="Judge only these case IDs; tool checks still cover the full report")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
    os.environ["LANGSMITH_TRACING"] = "false"
    from deepeval.test_case import LLMTestCase, ToolCall
    from deepeval.metrics import ToolCorrectnessMetric, GEval
    from deepeval.models import OpenAIModel
    try:
        from deepeval.test_case import SingleTurnParams as Params
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams as Params

    source = json.loads(args.report.read_text())
    try:
        records = real_records(source)
    except ValueError as exc:
        parser.error(str(exc))
    judge_model = OpenAIModel(
        model=args.judge_model,
        base_url="http://localhost:11434/v1", api_key="ollama")
    results = []
    for record in records:
        case_id = record["id"]
        tool_calls = record.get("tool_calls", [])
        entry = {"id": case_id, "product_oracle_status": record["status"],
                 "model_calls": (record.get("runtime_metadata") or {}).get("recovery_model_calls"),
                 "tool_call_names": [call["tool"] for call in tool_calls]}
        expected_tool = EXPECTED_SINGLE_TOOL.get(case_id)
        if expected_tool is not None:
            case = LLMTestCase(input=record["prompt"],
                actual_output=record.get("final_output") or "",
                tools_called=[ToolCall(name=call["tool"]) for call in tool_calls],
                expected_tools=[ToolCall(name=expected_tool)])
            metric = ToolCorrectnessMetric(threshold=1.0,
                should_exact_match=True, include_reason=True, model=judge_model)
            try:
                metric.measure(case)
                entry["tool_correctness"] = metric.score
                entry["tool_reason"] = metric.reason
            except Exception as exc:
                entry["tool_error"] = type(exc).__name__
        judge_cases = set(args.judge_id or JUDGE_CASES)
        if not args.skip_judge and case_id in judge_cases and record.get("final_output"):
            case = LLMTestCase(input=record["prompt"],
                actual_output=record["final_output"],
                expected_output=expected_answer(record))
            metric = GEval(name="Fixture answer grounding",
                evaluation_steps=[
                    "Identify the user's requested calculation or schema fact and its scope.",
                    "Compare the final answer with the reference facts, allowing only harmless rounding or equivalent notation.",
                    "Reference facts may be serialized as JSON, but this is not an output-format requirement. Do not invent formatting requirements or require internal test fields in the user's answer.",
                    "Penalize missing results, technical-error messages, unsupported claims, and false statements of completion.",
                ],
                evaluation_params=[Params.INPUT, Params.ACTUAL_OUTPUT, Params.EXPECTED_OUTPUT],
                threshold=0.8, model=judge_model, async_mode=False)
            try:
                metric.measure(case)
                entry["judge_score"] = metric.score
                entry["judge_reason"] = metric.reason
            except Exception as exc:
                entry["judge_error"] = type(exc).__name__
        results.append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
            "source_report": str(args.report.resolve()), "judge_model": args.judge_model,
            "source_mode":source['mode'],
            "limitations":["LLM judge scores supplement independent numeric/scope/artifact contracts.",
                           "Tool correctness covers only the explicitly configured legacy cases, not general autonomy."],
            "results": results}, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"id": case_id, "tool": entry.get("tool_correctness"),
                          "judge": entry.get("judge_score"),
                          "judge_error": entry.get("judge_error")}), flush=True)
    counts = Counter(record["product_oracle_status"] for record in results)
    print(json.dumps({"cases": len(results), "product_oracle_statuses": counts,
        "tool_scores": [r.get("tool_correctness") for r in results],
        "judged": sum("judge_score" in r for r in results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
