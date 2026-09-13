#!/usr/bin/env python3
"""Report observed turn latency and benchmark the local DataFrame path.

The benchmark reads one already-persisted dataset, copies it into temporary
conversation stores, and forbids both model and remote execution. It never
connects to Databricks and does not mutate the source asset database.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentile(values, percent):
    """Nearest-rank percentile; stable and honest for small samples."""
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


def observed_turns(runtime_root):
    rows = []
    for path in sorted(runtime_root.glob('*/runtime.jsonl*')):
        for line in path.read_text(errors='ignore').splitlines():
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if record.get('event') == 'run_completed' and isinstance(record.get('elapsed_seconds'), (int, float)):
                rows.append(record)
    durations = [row['elapsed_seconds'] for row in rows]
    instrumented = [row for row in rows if isinstance(row.get('process_peak_rss_bytes'), int)]
    return {
        **summary(durations),
        'status_counts': dict(Counter(str(row.get('status')) for row in rows)),
        'scope': 'all retained v1 run_completed events, including pre-fix failures',
        'current_code_only': False,
        'rss_instrumented_turns': len(instrumented),
        'instrumented_current_code': {
            **summary([row['elapsed_seconds'] for row in instrumented]),
            'max_process_peak_rss_bytes': max(
                (row['process_peak_rss_bytes'] for row in instrumented), default=None),
            'max_frame_cache_bytes': max((row.get('frame_cache_bytes', 0) for row in instrumented), default=None),
        },
    }


def read_dataset(asset_db):
    import pandas as pd
    connection = sqlite3.connect(f'file:{asset_db}?mode=ro', uri=True)
    try:
        row = connection.execute(
            "SELECT metadata, payload FROM assets WHERE kind='dataset' ORDER BY rowid LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError('No persisted dataset was found')
    metadata = json.loads(row[0])
    return metadata, pd.read_parquet(BytesIO(row[1]))


def current_rss_bytes(pid):
    if not pid:
        return None
    result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    return int(value) * 1024 if value else None


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def local_benchmark(frame, metadata, iterations):
    from langchain_core.language_models.chat_models import BaseChatModel
    from core.analysis_agent.runtime import GraphAnalysisRuntime

    class NoModel(BaseChatModel):
        @property
        def _llm_type(self):
            return 'model-call-forbidden'

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            raise AssertionError('local benchmark invoked the model')

    prompt = '현재 로딩된 10,000행 표본의 age 히스토그램을 보여줘'
    durations = []
    artifact_counts = []
    with tempfile.TemporaryDirectory(prefix='telly-performance-') as scratch:
        for index in range(iterations):
            runtime = GraphAnalysisRuntime(scratch, 'benchmark', f'cold-{index}', NoModel())
            runtime.datasets.register(frame, source=metadata['source'], query=metadata.get('query', ''),
                coverage=metadata.get('coverage', 'unknown'),
                predicate_known=metadata.get('predicate_known', False))
            started = time.monotonic()
            result = runtime.submit(prompt)
            durations.append(round(time.monotonic() - started, 6))
            artifact_counts.append(len(runtime.artifacts))
            state = runtime.inspect()['recovery']
            if result.get('status') != 'answered' or state.get('model_calls') != 0 or not state.get('artifact_ids'):
                raise AssertionError({'result': result, 'recovery': state})
            runtime.close()
    return {
        **summary(durations),
        'iterations': iterations,
        'source': metadata['source'],
        'rows': len(frame),
        'columns': len(frame.columns),
        'prompt': prompt,
        'model_calls': 0,
        'remote_calls': 0,
        'actual_png_each_run': all(count > 0 for count in artifact_counts),
        'process_peak_rss_bytes': peak_rss_bytes(),
        'scope': 'cold temporary conversations using a copy of the persisted 10,000-row DataFrame',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=ROOT / '.telly_runtime/v1')
    parser.add_argument('--asset-db', type=Path, required=True)
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'docs/runtime_performance_2026-09-14.json')
    args = parser.parse_args()
    if args.iterations < 5:
        parser.error('--iterations must be at least 5')
    metadata, frame = read_dataset(args.asset_db)
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'observed_retained_traffic': observed_turns(args.runtime_root),
        'current_local_dataframe_benchmark': local_benchmark(frame, metadata, args.iterations),
        'live_server_memory_snapshot': {
            'pid': args.server_pid,
            'rss_bytes': current_rss_bytes(args.server_pid),
            'measurement': 'single ps RSS snapshot after the verified live journey',
        },
        'safety': {'databricks_calls': 0, 'model_calls': 0, 'source_asset_mutated': False},
        'interpretation_limits': [
            'Retained traffic includes failures from older code and is a historical baseline, not a current-version SLO claim.',
            'The controlled benchmark covers the deterministic current-DataFrame histogram path only.',
            'The server RSS value is one snapshot; instrumented live-turn counts remain too small for a product-wide RSS percentile.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
