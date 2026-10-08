"""List/export saved journeys, or replay exact prompts through the real agent.

Replay records evidence; it deliberately does not award an unverified PASS.
No UI is automated by this CLI. Use the saved JSON for browser replay separately.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_set.prompt_scenarios import list_scenarios, load_scenario


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n',
                         encoding='utf-8')
    temporary.chmod(0o600)
    temporary.replace(path)


def replay(scenario, runtime, output, *, backend, provider):
    """One runtime, one submission per turn, including turns after failed requests.

    Only prompt strings reach submit. Acceptance descriptions are report-only.
    State repair/resume is the production agent's responsibility.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output path; previous results are preserved')
    report = {'scenario': scenario['name'], 'prompts_sha256': scenario['prompts_sha256'],
              'requested_turns': scenario['prompt_count'],
              'started_at': datetime.now(timezone.utc).isoformat(),
              'backend': backend, 'provider': provider, 'scope': 'production_agent_cli_not_browser',
              'status': 'RUNNING', 'grading': 'NOT_GRADED',
              'limitations': ['answered is not PASS; independently verify source, scope, counts and PNG/UI',
                              'reference expectations describe an earlier database snapshot'],
              'turns': []}
    write_report(output, report)
    try:
        for turn in scenario['turns']:
            start = time.monotonic()
            entry = {'id': turn['id'], 'prompt': turn['prompt'], 'grading': 'NOT_GRADED'}
            try:
                result = runtime.submit(turn['prompt'])
                state = runtime.inspect()
                recovery = state.get('recovery') or {}
                entry.update(agent_status=result.get('status'), text=result.get('text', ''),
                             error_type=result.get('error_type'),
                             run_id=runtime.diagnostics.run_id,
                             error_id=runtime.diagnostics.last_error_id,
                             execution_state=state.get('state'),
                             evidence={key: recovery.get(key) for key in (
                                 'request_id', 'request_text', 'status', 'stop_reason', 'goal',
                                 'required_sources', 'scope', 'model_calls', 'model_seconds',
                                 'artifact_ids', 'evidence_ids', 'metadata_evidence',
                                 'table_preview_evidence', 'value_list_evidence')})
            except Exception as error:
                # No raw exception/credentials/rows in a shareable report.
                entry.update(agent_status='exception', error_type=type(error).__name__,
                             run_id=runtime.diagnostics.run_id,
                             error_id=runtime.diagnostics.last_error_id)
            entry['elapsed_seconds'] = round(time.monotonic() - start, 3)
            report['turns'].append(entry)
            write_report(output, report)
            print(json.dumps({key: entry.get(key) for key in (
                'id', 'agent_status', 'error_type', 'elapsed_seconds', 'grading')},
                             ensure_ascii=False), flush=True)
        report['status'] = ('RECORDED_UNGRADED' if all(
            t['agent_status'] == 'answered' for t in report['turns']) else 'INCOMPLETE_UNGRADED')
        report['recorded_turns'] = len(report['turns'])
        report['answered_turns'] = sum(t['agent_status'] == 'answered' for t in report['turns'])
        return report
    except BaseException:
        report['status'] = 'INTERRUPTED_UNGRADED'
        raise
    finally:
        write_report(output, report)


def open_runtime(backend_name, provider, run_id):
    from dotenv import load_dotenv
    from core.analysis_agent.backends import load_data_backend
    from core.analysis_agent.model_provider import build_analysis_chat_model
    from core.analysis_agent.policy import RuntimePolicy
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    load_dotenv(ROOT / '.env')
    backend = load_data_backend(ROOT, name=backend_name)
    backend.preflight()
    policy = RuntimePolicy()
    model = build_analysis_chat_model(policy, provider=provider)
    # Isolated from app conversations and from other backend runs.
    root = backend.storage_root(ROOT / '.telly_runtime/prompt_scenario_runs' / run_id)
    return GraphAnalysisRuntime(root, 'prompt-scenario-evaluation', run_id, model,
        policy=policy, connection_identity=backend.config.identity(),
        remote_factory=lambda datasets: backend.executor_factory(
            backend.config, datasets, max_rows=policy.max_remote_rows,
            max_coordinate_rows=policy.max_scatter_coordinates),
        reference_context_loader=backend.context_loader,
        sql_dialect=backend.dialect, source_namespace=backend.namespace, intent_mode='llm')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', help='Exact saved name; quote names containing #')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true')
    action.add_argument('--show', action='store_true', help='Show exact prompts; no model/DB calls')
    action.add_argument('--export', type=Path, help='Export complete reusable JSON; no model/DB calls')
    action.add_argument('--run', action='store_true', help='Replay with the live configured agent/DB')
    parser.add_argument('--backend', choices=('mysql', 'databricks'))
    parser.add_argument('--provider', choices=('ollama', 'databricks'))
    parser.add_argument('--output', type=Path, help='New report file; defaults to ignored local runtime dir')
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(list_scenarios(), ensure_ascii=False, indent=2))
        return 0
    if not args.scenario:
        parser.error('--scenario is required')
    scenario = load_scenario(args.scenario)
    if args.show:
        print(json.dumps({'name': scenario['name'], 'turns': [
            {'id': t['id'], 'prompt': t['prompt']} for t in scenario['turns']]},
                        ensure_ascii=False, indent=2))
        return 0
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open('x', encoding='utf-8') as file:
            file.write(json.dumps(scenario, ensure_ascii=False, indent=2) + '\n')
        return 0
    if not args.backend or not args.provider:
        parser.error('--run requires explicit --backend and --provider')
    run_id = uuid4().hex
    output = args.output or ROOT / '.telly_runtime/prompt_scenario_reports' / run_id / 'report.json'
    if output.exists():
        parser.error('Output already exists; use a new path')
    try:
        runtime = open_runtime(args.backend, args.provider, run_id)
    except Exception as error:
        write_report(output, {'scenario': scenario['name'], 'status': 'SETUP_FAILED',
            'grading': 'NOT_GRADED', 'backend': args.backend, 'provider': args.provider,
            'error_type': type(error).__name__, 'recorded_turns': 0})
        print(json.dumps({'status': 'SETUP_FAILED', 'error_type': type(error).__name__,
                          'report': str(output.resolve())}, ensure_ascii=False))
        return 1
    try:
        report = replay(scenario, runtime, output, backend=args.backend, provider=args.provider)
    finally:
        runtime.close()
    print(json.dumps({'report': str(output.resolve()), 'status': report['status'],
                      'grading': 'NOT_GRADED'}, ensure_ascii=False))
    # An ungraded replay must never green-light a CI/release gate.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
