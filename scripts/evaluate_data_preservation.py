"""Local multi-turn preservation baseline using the real graph and optional Ollama.

No Databricks connection, approvals, or operational data. Oracle data is never
passed to the model. PASS covers these fixture turns only, not entire J21/J24.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from scripts.evaluate_analysis_agent import HistogramCapture, fixture_reference_context, preserve_runtime_metadata

FIXTURE = ROOT / 'tests/fixtures/data_preservation_v1.json'


class ModelCallCounter(BaseCallbackHandler):
    """Count attempted calls, including calls that time out; never retain inputs."""
    def __init__(self):
        self.starts = 0
        self.errors = 0

    def on_chat_model_start(self, *args, **kwargs):
        self.starts += 1

    def on_llm_error(self, *args, **kwargs):
        self.errors += 1


def fingerprint(frame):
    header = json.dumps({'columns': list(frame.columns), 'dtypes': frame.dtypes.astype(str).tolist()},
                        ensure_ascii=False).encode()
    return sha256(header + pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def expected_distribution(spec, turn):
    rows = spec['rows']
    if turn.get('where'):
        condition = turn['where']
        rows = [row for row in rows if row[condition['column']] == condition['value']]
    return dict(Counter(float(row[spec['roles']['measure']]) for row in rows
                        if row[spec['roles']['measure']] is not None))


def histogram_matches(expected, observed, rendered_total):
    return expected == observed and rendered_total == sum(expected.values())


def turn_event_count(diagnostics, event):
    """Count actual attempts in the newest run, not recovery-loop counters."""
    runs = [entry.get('run_id') for entry in diagnostics
            if entry.get('event') == 'run_started']
    if not runs:
        return 0
    return sum(entry.get('event') == event and entry.get('run_id') == runs[-1]
               for entry in diagnostics)


def run_case(spec, case, model, counter=None, artifact_dir=None):
    frame = pd.DataFrame(spec['rows']).astype(spec['dtypes'])
    expected_digest = fingerprint(frame)
    calls = []
    chart_evidence = {}
    records = []
    context = fixture_reference_context(spec['source'], frame)
    with tempfile.TemporaryDirectory(prefix='telly-preservation-') as directory:
        def forbidden_factory(_store):
            def execute(_envelope):
                calls.append('forbidden_remote_execution')
                raise AssertionError('This fixture evaluator has no DB access')
            return execute

        def create():
            return GraphAnalysisRuntime(directory, 'fixture-evaluation', case['id'], model,
                connection_identity='fixture-only:no-database', remote_factory=forbidden_factory,
                reference_context_loader=lambda: [context],
                policy=RuntimePolicy())

        runtime = create()
        info = runtime.datasets.register(frame, source=spec['source'], coverage='complete',
                                        predicate_known=True, snapshot=spec['snapshot'])
        # Mirror the UI's active raw selection; unselected roots remain ambiguous.
        runtime.select_dataset(info.id)
        original_metadata = asdict(info)
        try:
            for index, turn in enumerate(case['turns']):
                if turn.get('restart'):
                    runtime.close()
                    runtime = create()
                start = time.monotonic()
                model_starts = counter.starts if counter else 0
                model_errors = counter.errors if counter else 0
                before_ids = set(runtime.datasets.metadata)
                try:
                    with HistogramCapture() as capture:
                        outcome = runtime.submit(turn['prompt'])
                    state = runtime.inspect()
                    recovery = state['recovery']
                    diagnostics = [json.loads(line) for line in runtime.diagnostics.path.read_text().splitlines()]
                    root_unchanged = (fingerprint(runtime.datasets.frames[info.id]) == expected_digest
                                      and asdict(runtime.datasets.metadata[info.id]) == original_metadata)
                    new_histograms = {item['figure']: item for item in capture.histograms}
                    for digest, figure in capture.saved.items():
                        if figure in new_histograms:
                            item = new_histograms[figure]
                            chart_evidence[digest] = (dict(item['distribution']), item['rendered_total'])
                    oracle_pass = False
                    if turn['oracle'] == 'histogram':
                        expected = expected_distribution(spec, turn)
                        for chart_id in recovery.get('artifact_ids', []):
                            card = runtime.artifacts[chart_id]
                            evidence = chart_evidence.get(sha256(card.image).hexdigest())
                            if evidence and card.kind == 'histogram':
                                oracle_pass |= histogram_matches(expected, *evidence)
                    elif turn['oracle'] == 'scalar_mean':
                        values = [row[spec['roles']['measure']] for row in spec['rows']
                                  if row[spec['roles']['measure']] is not None]
                        expected = sum(values) / len(values)
                        # Only current-turn lineage-bound results, never prose or previous values.
                        for dataset_id in recovery.get('evidence_ids', []):
                            if dataset_id not in runtime.datasets.metadata or dataset_id in before_ids:
                                continue
                            result = runtime.datasets.frames[dataset_id]
                            meta = runtime.datasets.metadata[dataset_id]
                            if result.shape == (1, 1) and (meta.parent_id == info.id or info.id in meta.parent_ids):
                                oracle_pass |= abs(float(result.iloc[0, 0]) - expected) < 1e-9
                    complete = outcome.get('status') in {'answered', 'complete'}
                    pending = len(state['requests'])
                    passed = complete and oracle_pass and root_unchanged and not pending and not calls
                    record = {'turn': index + 1, 'status': 'PASS' if passed else 'FAIL',
                              'agent_status': outcome.get('status'), 'root_unchanged': root_unchanged,
                              'oracle_pass': bool(oracle_pass), 'approval_requests': pending,
                              'remote_executions': len(calls),
                              'runtime_model_calls': recovery.get('model_calls', 0),
                              'runtime_tool_calls': turn_event_count(diagnostics, 'tool_started'),
                              'recovery_tool_calls': recovery.get('tool_calls', 0),
                              'error_type': outcome.get('error_type'), 'error_id': outcome.get('error_id')}
                except Exception as exc:
                    record = {'turn': index + 1, 'status': 'FAIL', 'error_type': type(exc).__name__,
                              'remote_executions': len(calls)}
                record['elapsed_seconds'] = round(time.monotonic() - start, 3)
                record['model_attempts'] = counter.starts - model_starts if counter else None
                record['model_errors'] = counter.errors - model_errors if counter else None
                try:
                    record['runtime_metadata'] = preserve_runtime_metadata(
                        runtime, {'id': f"{case['id']}-turn{index + 1}"}, artifact_dir)
                except Exception as exc:
                    record.update(status='FAIL', metadata_error_type=type(exc).__name__)
                records.append(record)
                print(json.dumps({'case': case['id'], **{key: value for key, value in record.items()
                                                       if key != 'runtime_metadata'}}, ensure_ascii=False), flush=True)
                if record.get('approval_requests') or record.get('error_type'):
                    # Do not approve an unnecessary remote call or silently repair the experiment.
                    for following in range(index + 1, len(case['turns'])):
                        records.append({'turn': following + 1, 'status': 'NOT_RUN',
                                        'reason': 'Prior turn left an unresolved execution/approval'})
                    break
        finally:
            runtime.close()
            for handler in list(runtime.diagnostics.logger.handlers):
                runtime.diagnostics.logger.removeHandler(handler)
                handler.close()
    return {'id': case['id'], 'journeys': case['journeys'], 'turns': records,
            'status': 'PASS' if all(row['status'] == 'PASS' for row in records) else 'FAIL'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-local-model', action='store_true')
    parser.add_argument('--model', default='gemma4:e4b')
    parser.add_argument('--id', action='append', help='Select fixture case IDs; omitted cases are not scored')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(FIXTURE.read_text())
    selected = [case for case in spec['local_model_cases'] if not args.id or case['id'] in args.id]
    if set(args.id or []) - {case['id'] for case in spec['local_model_cases']}:
        parser.error('Unknown fixture case ID')
    model = None
    counter = ModelCallCounter()
    if args.live_local_model:
        # No .env loading and no external tracing: this runner only uses a fixed loopback endpoint.
        os.environ['LANGSMITH_TRACING'] = 'false'
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
        from langchain_ollama import ChatOllama
        model = ChatOllama(model=args.model, base_url='http://127.0.0.1:11434',
            reasoning=True, temperature=0, num_ctx=16384, num_predict=2048,
            client_kwargs={'timeout': RuntimePolicy().model_timeout_seconds}, callbacks=[counter])
    report = {'generated_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'live-local-model' if model else 'contract-only',
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        'fixture_sha256': sha256(FIXTURE.read_bytes()).hexdigest(),
        'selected_case_ids': [case['id'] for case in selected],
        'total_defined_cases': len(spec['local_model_cases']),
        'evaluator_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                          for folder in ('core', 'utils') for p in sorted((ROOT/folder).rglob('*.py'))},
        'versions': {name: importlib.metadata.version(name) for name in ('langchain', 'langgraph', 'pandas', 'duckdb', 'langchain-ollama')},
        'model': args.model if model else None,
        'model_settings': {'endpoint': 'loopback-only', 'temperature': 0, 'num_ctx': 16384,
                           'num_predict': 2048, 'timeout_seconds': RuntimePolicy().model_timeout_seconds,
                           'reasoning': True},
        'limitations': ['Synthetic preloaded raw data; no initial Databricks load or browser rendering.',
            'Only model_attempts counts failed calls too; runtime_model_calls may exclude timed-out calls.',
            'Zero model_attempts means deterministic path, not autonomous model reasoning evidence.',
            'PRES_02 scalar is a smoke oracle, not counterfactual proof of arbitrary generated SQL.',
            'Subset coverage of J21/J22/J24; full fault/load/hybrid contracts remain NOT_RUN.'],
        'cases': []}
    for case in selected:
        try:
            record = run_case(spec, case, model, counter, args.output.parent/'preservation_artifacts') if model else {'id': case['id'], 'status': 'NOT_RUN'}
        except Exception as exc:
            record = {'id': case['id'], 'status': 'FAIL', 'error_type': type(exc).__name__,
                      'reason': 'Evaluator setup failed; no success evidence'}
        report['cases'].append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return int(model is not None and any(c['status'] != 'PASS' for c in report['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
