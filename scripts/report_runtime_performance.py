#!/usr/bin/env python3
"""Report observed turn latency and benchmark the local DataFrame path.

The benchmark reads one already-persisted dataset, copies it into temporary
conversation stores, and forbids both model and remote execution. It never
connects to Databricks and does not mutate the source asset database.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from hashlib import sha256
import json
import math
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time

import pandas as pd
import pyarrow.parquet as pq

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


def read_dataset(asset_db, *, dataset_id=None, max_rows=10_000, max_blob_bytes=128 * 1024 * 1024):
    """Read a bounded benchmark sample from either persisted asset format."""
    if max_rows < 1:
        raise ValueError('max_rows must be positive')
    connection = sqlite3.connect(f'file:{asset_db}?mode=ro', uri=True)
    try:
        if dataset_id:
            row = connection.execute(
                "SELECT id, metadata, payload IS NULL, length(payload) FROM assets "
                "WHERE kind='dataset' AND id=?", (dataset_id,)).fetchone()
        else:
            row = connection.execute(
                "SELECT id, metadata, payload IS NULL, length(payload) FROM assets "
                "WHERE kind='dataset' ORDER BY (payload IS NULL) DESC, rowid LIMIT 1"
            ).fetchone()
        if row is None:
            raise ValueError('No persisted dataset was found')
        asset_id, metadata_json, file_backed, blob_size = row
        if not file_backed and (blob_size or 0) > max_blob_bytes:
            raise ValueError('Legacy dataset BLOB exceeds the benchmark read limit')
        payload = None if file_backed else connection.execute(
            "SELECT payload FROM assets WHERE id=?", (asset_id,)).fetchone()[0]
    finally:
        connection.close()
    source = asset_db.parent / f'{asset_id}.parquet' if file_backed else BytesIO(payload)
    if file_backed and not source.is_file():
        raise FileNotFoundError('Persisted dataset file is missing')
    parquet = pq.ParquetFile(source)
    batches, count = [], 0
    for batch in parquet.iter_batches(batch_size=min(1024, max_rows)):
        piece = batch.to_pandas().head(max_rows - count)
        batches.append(piece)
        count += len(piece)
        if count >= max_rows:
            break
    if not batches:
        raise ValueError('Persisted dataset is empty; no benchmark sample')
    metadata = json.loads(metadata_json)
    metadata['benchmark_asset_id'] = asset_id
    metadata['persisted_rows'] = parquet.metadata.num_rows
    metadata['benchmark_sample_rows'] = count
    metadata['benchmark_source'] = 'benchmark.sample'
    return metadata, pd.concat(batches, ignore_index=True)


def benchmark_columns(frame):
    numeric = [name for name in frame.select_dtypes(include='number').columns
               if isinstance(name, str) and name and len(name) <= 128 and '\n' not in name]
    if not numeric:
        raise ValueError('Benchmark sample needs one numeric column')
    return numeric


def asset_fingerprint(asset_db, asset_id):
    """Hash the exact persisted dataset row and payload without materializing it."""
    connection = sqlite3.connect(f'file:{asset_db}?mode=ro', uri=True)
    try:
        row = connection.execute(
            "SELECT rowid, metadata, payload IS NULL, length(payload) FROM assets "
            "WHERE id=? AND kind='dataset'", (asset_id,)).fetchone()
        if row is None:
            raise ValueError('Benchmark source asset disappeared')
        rowid, metadata_json, file_backed, blob_size = row
        digest = sha256(metadata_json.encode())
        if file_backed:
            source = asset_db.parent / f'{asset_id}.parquet'
            if not source.is_file():
                raise FileNotFoundError('Persisted dataset file is missing')
            with source.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
        else:
            with connection.blobopen('assets', 'payload', rowid, readonly=True) as blob:
                remaining = blob_size
                while remaining:
                    chunk = blob.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError('Persisted dataset BLOB ended unexpectedly')
                    digest.update(chunk)
                    remaining -= len(chunk)
        return digest.hexdigest()
    finally:
        connection.close()


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

    column = benchmark_columns(frame)[0]
    prompt = f'현재 보유한 데이터의 `{column.replace("`", "``")}` 컬럼 히스토그램을 보여줘'
    durations = []
    artifact_counts = []
    with tempfile.TemporaryDirectory(prefix='telly-performance-') as scratch:
        for index in range(iterations):
            runtime = GraphAnalysisRuntime(scratch, 'benchmark', f'cold-{index}', NoModel())
            info = runtime.datasets.register(frame, source=metadata['benchmark_source'],
                coverage='complete', predicate_known=True)
            runtime.select_dataset(info.id)
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
        'source': metadata['benchmark_source'],
        'persisted_source': metadata['source'],
        'persisted_rows': metadata['persisted_rows'],
        'rows': len(frame),
        'columns': len(frame.columns),
        'prompt': prompt,
        'model_calls': 0,
        'remote_calls': 0,
        'actual_png_each_run': all(count > 0 for count in artifact_counts),
        'process_peak_rss_bytes': peak_rss_bytes(),
        'scope': 'cold temporary conversations using a bounded persisted-data sample',
    }


def concurrent_local_benchmark(frame, metadata, requests, workers):
    """Measure isolated conversations under simultaneous local chart work."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from core.analysis_agent.runtime import GraphAnalysisRuntime

    class NoModel(BaseChatModel):
        @property
        def _llm_type(self):
            return 'model-call-forbidden'

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            raise AssertionError('concurrent benchmark invoked the model')

    column = benchmark_columns(frame)[0]
    prompt = f'현재 보유한 데이터의 `{column.replace("`", "``")}` 컬럼 히스토그램을 보여줘'
    started = time.monotonic()
    durations, errors = [], []
    with tempfile.TemporaryDirectory(prefix='telly-concurrent-performance-') as scratch:
        def run_one(index):
            runtime = GraphAnalysisRuntime(scratch, 'benchmark', f'concurrent-{index}', NoModel())
            try:
                info = runtime.datasets.register(frame.copy(deep=False),
                    source=metadata['benchmark_source'], coverage='complete', predicate_known=True)
                runtime.select_dataset(info.id)
                turn_started = time.monotonic()
                result = runtime.submit(prompt)
                elapsed = round(time.monotonic() - turn_started, 6)
                state = runtime.inspect()['recovery']
                if (result.get('status') != 'answered' or state.get('model_calls') != 0
                        or not state.get('artifact_ids')):
                    raise AssertionError({'result': result, 'recovery': state})
                return elapsed
            finally:
                runtime.close()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one, index): index for index in range(requests)}
            for future in as_completed(futures):
                try:
                    durations.append(future.result())
                except Exception as exc:
                    errors.append({'request': futures[future], 'error_type': type(exc).__name__,
                                   'message': str(exc)[:500]})
    wall_seconds = round(time.monotonic() - started, 6)
    return {
        **summary(durations),
        'requests': requests,
        'workers': workers,
        'successes': len(durations),
        'failures': len(errors),
        'wall_seconds': wall_seconds,
        'throughput_requests_per_second': round(len(durations) / wall_seconds, 3) if wall_seconds else None,
        'errors': errors,
        'model_calls': 0,
        'remote_calls': 0,
        'process_peak_rss_bytes': peak_rss_bytes(),
        'scope': 'parallel cold conversations using independent copies of the persisted local DataFrame',
    }


def local_correlation_benchmark(frame, metadata, iterations):
    """Verify the common two-column correlation intent avoids the slow model path."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from core.analysis_agent.runtime import GraphAnalysisRuntime

    class NoModel(BaseChatModel):
        @property
        def _llm_type(self):
            return 'model-call-forbidden'

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            raise AssertionError('correlation benchmark invoked the model')

    numeric = benchmark_columns(frame)
    if len(numeric) < 2:
        return {'status': 'SKIPPED', 'reason': 'Two numeric columns are required for correlation'}
    left, right = numeric[:2]
    prompt = (f'현재 보유한 데이터의 `{left.replace("`", "``")}`와 '
              f'`{right.replace("`", "``")}`의 피어슨 상관계수를 계산해줘.')
    expected = float(frame[left].corr(frame[right]))
    durations = []
    with tempfile.TemporaryDirectory(prefix='telly-correlation-performance-') as scratch:
        for index in range(iterations):
            runtime = GraphAnalysisRuntime(scratch, 'benchmark', f'correlation-{index}', NoModel())
            try:
                info = runtime.datasets.register(frame, source=metadata['benchmark_source'],
                    coverage='complete', predicate_known=True)
                runtime.select_dataset(info.id)
                started = time.monotonic()
                result = runtime.submit(prompt)
                durations.append(round(time.monotonic() - started, 6))
                recovery = runtime.inspect()['recovery']
                evidence = runtime.datasets.frames[recovery['evidence_ids'][-1]]
                actual = float(evidence.iloc[0, 0])
                if (result.get('status') != 'answered' or recovery.get('model_calls') != 0
                        or not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)):
                    raise AssertionError({'result': result, 'recovery': recovery,
                                          'expected': expected, 'actual': actual})
            finally:
                runtime.close()
    return {
        **summary(durations),
        'iterations': iterations,
        'prompt': prompt,
        'expected': expected,
        'model_calls': 0,
        'remote_calls': 0,
        'scope': 'cold temporary conversations using the persisted local DataFrame and a pandas oracle',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=ROOT / '.telly_runtime/v1')
    parser.add_argument('--asset-db', type=Path, required=True)
    parser.add_argument('--dataset-id', help='select one persisted dataset instead of the first file-backed asset')
    parser.add_argument('--sample-rows', type=int, default=10_000)
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--concurrent-requests', type=int, default=20)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'docs/runtime_performance_2026-09-14.json')
    args = parser.parse_args()
    if args.iterations < 5:
        parser.error('--iterations must be at least 5')
    if args.sample_rows < 1:
        parser.error('--sample-rows must be positive')
    if args.concurrent_requests < 1 or args.workers < 1:
        parser.error('--concurrent-requests and --workers must be positive')
    metadata, frame = read_dataset(args.asset_db, dataset_id=args.dataset_id,
                                   max_rows=args.sample_rows)
    before_digest = asset_fingerprint(args.asset_db, metadata['benchmark_asset_id'])
    local = local_benchmark(frame, metadata, args.iterations)
    concurrent = concurrent_local_benchmark(
        frame, metadata, args.concurrent_requests, min(args.workers, args.concurrent_requests))
    correlation = local_correlation_benchmark(frame, metadata, args.iterations)
    after_digest = asset_fingerprint(args.asset_db, metadata['benchmark_asset_id'])
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'benchmark_input': {'asset_id': metadata['benchmark_asset_id'],
                            'persisted_source': metadata['source'],
                            'persisted_rows': metadata['persisted_rows'],
                            'sample_rows': metadata['benchmark_sample_rows'],
                            'sample_is_full_source': metadata['benchmark_sample_rows'] == metadata['persisted_rows']},
        'observed_retained_traffic': observed_turns(args.runtime_root),
        'current_local_dataframe_benchmark': local,
        'current_concurrent_local_benchmark': concurrent,
        'current_local_correlation_benchmark': correlation,
        'live_server_memory_snapshot': {
            'pid': args.server_pid,
            'rss_bytes': current_rss_bytes(args.server_pid),
            'measurement': 'single ps RSS snapshot after the verified live journey',
        },
        'safety': {'databricks_calls': 0, 'model_calls': 0,
                   'source_asset_mutated': before_digest != after_digest,
                   'source_asset_sha256_before': before_digest,
                   'source_asset_sha256_after': after_digest},
        'interpretation_limits': [
            'The sampled asset is re-scoped as benchmark.sample; a bounded prefix is not the complete source population.',
            'Retained traffic includes failures from older code and is a historical baseline, not a current-version SLO claim.',
            'The controlled benchmark covers the deterministic current-DataFrame histogram path only.',
            'The concurrent benchmark uses isolated conversations in one process; it is not a distributed deployment load test.',
            'The server RSS value is one snapshot; instrumented live-turn counts remain too small for a product-wide RSS percentile.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
