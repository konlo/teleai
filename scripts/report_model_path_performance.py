#!/usr/bin/env python3
"""Measure a grounded analysis turn that must use the configured local model.

The benchmark uses the checked-in bank fixture in fresh temporary conversations.
No Databricks executor or query tool is registered, and each scalar result is
checked against a pandas oracle before it is counted as successful.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(percent / 100 * len(ordered)) - 1)]


def summary(values):
    return {
        'count': len(values),
        'p50_seconds': percentile(values, 50),
        'p95_seconds': percentile(values, 95),
        'max_seconds': max(values, default=None),
        'percentile_method': 'nearest-rank',
    }


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def reference_context(frame):
    payload = json.loads((ROOT / 'test_set/data_context/bank_loan.json').read_text())
    aliases = {item['name']: item.get('aliases', []) for item in payload['columns']}
    stamp = datetime.now(timezone.utc).isoformat()
    return [{
        'table': 'workspace.default.bank_loan',
        'training_status': 'benchmark_fixture',
        'trained_at': stamp,
        'observed_at': stamp,
        'columns': [
            {'name': str(name), 'dtype': str(dtype), 'aliases': aliases.get(str(name), [])}
            for name, dtype in frame.dtypes.items()
        ],
    }]


def numeric_evidence(runtime, evidence_ids):
    values = []
    for dataset_id in evidence_ids:
        try:
            frame = runtime.datasets.frames[dataset_id]
        except (KeyError, OSError, ValueError):
            continue
        for value in frame.select_dtypes(include='number').to_numpy().ravel():
            if pd.notna(value):
                values.append(float(value))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--num-predict', type=int, default=1024)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'docs/model_path_performance_2026-09-14.json')
    args = parser.parse_args(argv)
    if args.iterations < 5:
        parser.error('--iterations must be at least 5 for a p95 sample')
    if args.num_predict < 128:
        parser.error('--num-predict must be at least 128')

    from dotenv import load_dotenv
    from langchain_ollama import ChatOllama
    from core.analysis_agent.runtime import GraphAnalysisRuntime

    load_dotenv(ROOT / '.env')
    endpoint = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    if urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
        parser.error('Only a localhost Ollama endpoint is allowed')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    model_name = os.getenv('OLLAMA_MODEL', 'gemma4:e4b')
    model = ChatOllama(model=model_name, base_url=endpoint, reasoning=True,
        temperature=0, num_ctx=16384, num_predict=args.num_predict,
        client_kwargs={'timeout': 60})

    frame = pd.read_csv(ROOT / 'test_set/data/bank_loan.csv')
    expected = float(frame['age'].corr(frame['balance']))
    prompt = '현재 보유한 bank_loan 데이터에서 age와 balance의 피어슨 상관계수를 계산해줘.'
    rows = []
    with tempfile.TemporaryDirectory(prefix='telly-model-performance-') as scratch:
        for index in range(args.iterations):
            runtime = GraphAnalysisRuntime(scratch, 'benchmark', f'model-{index}', model,
                reference_context_loader=lambda: reference_context(frame))
            started = time.monotonic()
            try:
                runtime.datasets.register(frame, source='workspace.default.bank_loan',
                    query='SELECT * FROM workspace.default.bank_loan', coverage='complete',
                    predicate_known=True)
                result = runtime.submit(prompt)
                recovery = runtime.inspect()['recovery']
                values = numeric_evidence(runtime, recovery.get('evidence_ids', []))
                matches = [value for value in values if math.isclose(value, expected, rel_tol=1e-7, abs_tol=1e-9)]
                success = result.get('status') == 'answered' and bool(matches)
                row = {
                    'iteration': index + 1,
                    'status': 'PASS' if success else 'FAIL',
                    'agent_status': result.get('status'),
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                    'model_calls': recovery.get('model_calls', 0),
                    'model_seconds': recovery.get('model_seconds', 0.0),
                    'tool_calls': len(recovery.get('sent_calls', [])),
                    'evidence_values': values[:10],
                }
            except Exception as exc:
                row = {'iteration': index + 1, 'status': 'FAIL',
                       'elapsed_seconds': round(time.monotonic() - started, 3),
                       'error_type': type(exc).__name__}
            finally:
                runtime.close()
            rows.append(row)
            report = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'model': model_name,
                'num_predict': args.num_predict,
                'endpoint_host': urlparse(endpoint).hostname,
                'prompt': prompt,
                'oracle': {'kind': 'pearson_correlation', 'expected': expected},
                'summary': {
                    **summary([row['elapsed_seconds'] for row in rows if row['status'] == 'PASS']),
                    'requested_iterations': args.iterations,
                    'successes': sum(row['status'] == 'PASS' for row in rows),
                    'failures': sum(row['status'] == 'FAIL' for row in rows),
                    'max_process_peak_rss_bytes': peak_rss_bytes(),
                },
                'results': rows,
                'safety': {'databricks_calls': 0, 'remote_tool_registered': False,
                           'fixture_rows': len(frame)},
                'limitations': [
                    'This measures one correlation intent against one local model and one fixture.',
                    'The five-run p95 is the nearest-rank maximum and needs production traffic calibration.',
                    'It does not represent concurrent model serving or remote Databricks latency.',
                ],
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({key: row.get(key) for key in
                ('iteration', 'status', 'agent_status', 'elapsed_seconds', 'model_calls', 'model_seconds')}), flush=True)
    print(json.dumps(report['summary'], ensure_ascii=False))
    print(f'Report: {args.output}')
    return 0 if report['summary']['failures'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
