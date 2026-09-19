#!/usr/bin/env python3
"""Local synthetic storage benchmark; no model, credentials, or remote access.

Each measurement runs in a fresh interpreter so ru_maxrss is a process peak,
not a claimed allocation delta. Temporary stores are removed after verification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def peak_rss_bytes():
    measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(measured if sys.platform == 'darwin' else measured * 1024)


def frame_fingerprint(frame):
    import pandas as pd
    return sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def synthetic_frame(spec, rows):
    import numpy as np
    import pandas as pd
    indexes = np.arange(rows, dtype='int64')
    columns = {}
    for name, definition in spec['columns'].items():
        kind = definition['kind']
        if kind == 'sequence':
            columns[name] = indexes
        elif kind == 'numeric_cycle':
            columns[name] = (indexes % definition['modulus']) / definition['divisor']
        elif kind == 'category':
            values = definition['values']
            columns[name] = np.asarray(values, dtype=object)[indexes % len(values)]
        else:
            raise ValueError(f'Unsupported synthetic generator: {kind}')
    return pd.DataFrame(columns)


def worker(mode, manifest_file):
    import pandas as pd
    from core.analysis_agent.assets import AssetDB, PersistentDatasets, PersistentCharts
    from utils.analysis_charts import ChartPreview

    manifest = json.loads(Path(manifest_file).read_text())
    db = AssetDB(manifest['root'], 'storage-validation', 'synthetic')
    store = PersistentDatasets(db, budget=manifest['budget_bytes'])
    started = time.monotonic()
    baseline_peak = peak_rss_bytes()
    result = {}
    if mode == 'register':
        frame = synthetic_frame(manifest['fixture'], manifest['rows'])
        info = store.register(frame, source=manifest['fixture']['source'],
                              coverage='complete', predicate_known=True)
        result = {'dataset_id': info.id, 'rows': len(frame),
                  'frame_bytes_deep': int(frame.memory_usage(index=True, deep=True).sum()),
                  'fingerprint': frame_fingerprint(frame), 'columns': list(frame.columns)}
    elif mode == 'read':
        frame = store.frames[manifest['dataset_id']]
        fingerprint = frame_fingerprint(frame)
        # Mutating a returned object must not change the immutable persisted input.
        first = frame.iloc[0, 0]
        frame.iloc[0, 0] = -999
        del frame
        restored = store.frames[manifest['dataset_id']]
        result = {'fingerprint_matches': fingerprint == manifest['fingerprint'],
                  'mutation_isolated': bool(restored.iloc[0, 0] == first),
                  'rows': len(restored), 'cache_bytes': store.frames.bytes,
                  'cached_ids': list(store.frames.cache),
                  'uncached_when_over_budget': not store.frames.cache}
    elif mode == 'lru':
        small = synthetic_frame(manifest['fixture'], min(10000, manifest['rows']))
        store = PersistentDatasets(db, budget=0)
        first = store.register(small, source=manifest['fixture']['source'])
        second = store.register(small.copy(), source=manifest['fixture']['source'])
        # Size the cache from the representation actually restored from Parquet.
        # RangeIndex can become a materialized Index and use more memory than the
        # input, so the pre-serialization size is not a valid admission boundary.
        _, first_payload = db.get(first.id, 'dataset')
        restored_size = int(pd.read_parquet(BytesIO(first_payload)).memory_usage(index=True, deep=True).sum())
        store = PersistentDatasets(db, budget=restored_size)
        original = frame_fingerprint(small)
        fetched = store.frames[first.id]
        admitted_first = first.id in store.frames.cache
        fetched.iloc[0, 0] = -999
        same_hit = frame_fingerprint(store.frames[first.id]) == original
        store.frames[second.id]
        admitted_second = second.id in store.frames.cache
        first_evicted = first.id not in store.frames.cache
        reloaded = store.frames[first.id]
        result = {'admitted_first': admitted_first, 'admitted_second': admitted_second,
                  'evicted_first': first_evicted,
                  'reload_matches': frame_fingerprint(reloaded) == original,
                  'hit_mutation_isolated': same_hit,
                  'cache_within_budget': store.frames.bytes <= store.frames.budget,
                  'cache_bytes': store.frames.bytes, 'budget_bytes': store.frames.budget}
    elif mode == 'chart':
        from matplotlib.figure import Figure
        image = BytesIO()
        fig = Figure(figsize=(1, 1))
        fig.subplots().plot([0, 1], [0, 1])
        fig.savefig(image, format='png')
        payload_reads = []
        original_get = db.get

        def tracked_get(key, kind):
            if kind == 'dataset':
                payload_reads.append(key)
            return original_get(key, kind)

        db.get = tracked_get
        card = ChartPreview(str(uuid4()), manifest['dataset_id'], 'Storage validation',
                            'Synthetic PNG persistence', 'line', (), 'synthetic', image.getvalue())
        PersistentCharts(db)[card.id] = card
        result = {'chart_id': card.id, 'png_sha256': sha256(card.image).hexdigest(),
                  'dataset_payload_reads_for_chart_insert': len(payload_reads)}
    elif mode == 'reopen':
        frame = store.frames[manifest['dataset_id']]
        card = PersistentCharts(db)[manifest['chart_id']]
        other = AssetDB(manifest['root'], 'different-owner', 'synthetic')
        result = {'rows': len(frame),
                  'fingerprint_matches': frame_fingerprint(frame) == manifest['fingerprint'],
                  'png_matches': sha256(card.image).hexdigest() == manifest['png_sha256'],
                  'owner_isolated': not other.metadata('dataset'),
                  'integrity_check': db.conn.execute('PRAGMA integrity_check').fetchone()[0]}
        other.close()
    elif mode == 'interrupt':
        # Execute the real AssetDB.put transaction, then kill before its context
        # manager commits. This models abrupt process loss, not power/disk loss.
        connection = db.conn

        class BeforeCommitCrash:
            def __enter__(self):
                connection.__enter__()
                return self

            def execute(self, *args):
                return connection.execute(*args)

            def __exit__(self, *args):
                Path(manifest['crash_marker']).write_text('inserted-before-commit')
                os.kill(os.getpid(), signal.SIGKILL)

        db.conn = BeforeCommitCrash()
        db.put('interrupted-asset', 'dataset', {'synthetic': True}, b'x' * 4_000_000)
        raise AssertionError('Crash injection did not execute')
    elif mode == 'recover':
        frame = store.frames[manifest['dataset_id']]
        result = {'uncommitted_asset_absent': 'interrupted-asset' not in db.metadata('dataset'),
                  'original_matches': frame_fingerprint(frame) == manifest['fingerprint'],
                  'original_chart_present': manifest['chart_id'] in db.metadata('chart'),
                  'integrity_check': db.conn.execute('PRAGMA integrity_check').fetchone()[0]}
    else:
        raise ValueError(mode)
    result.update(elapsed_seconds=round(time.monotonic() - started, 4),
                  peak_rss_bytes=peak_rss_bytes(), import_peak_rss_bytes=baseline_peak,
                  database_bytes=(db.directory / 'assets.sqlite').stat().st_size)
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/large_data_workload.json')
    parser.add_argument('--rows', type=int)
    parser.add_argument('--budget-mib', type=int, default=64)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/large_data_validation_2026-09-13.json')
    parser.add_argument('--worker', choices=['register', 'read', 'lru', 'chart', 'reopen', 'interrupt', 'recover'])
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.manifest)))
        return
    fixture = json.loads(args.fixture.read_text())
    rows = args.rows if args.rows is not None else fixture['rows']
    if rows < 1 or args.budget_mib < 0:
        parser.error('Rows must be positive; cache budget must be nonnegative')
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python': platform.python_version(),
              'fixture': str(args.fixture.relative_to(ROOT) if args.fixture.is_relative_to(ROOT) else args.fixture),
              'fixture_sha256': sha256(args.fixture.read_bytes()).hexdigest(),
              'rows': rows, 'cache_budget_bytes': args.budget_mib * 1024**2,
              'remote_calls': 0, 'model_calls': 0,
              'measurement': 'Fresh subprocess ru_maxrss includes interpreter, imports, serialization, retained cache, and caller copies. It is not a hard RSS limit or allocation delta.',
              'scope': 'Synthetic five-column data; 750000 default rows matches an observed row count only. It does not reproduce operational width, dtype, distribution or total data size.',
              'limitations': ['No retention/TTL or disk quota policy is implemented.',
                              'Cache budget bounds retained DataFrame deep size, not process RSS.',
                              'SIGKILL tests process interruption, not power loss, disk-full, filesystem corruption or concurrent writes.',
                              'No product memory or latency SLO has been supplied; no general capacity pass is asserted.'],
              'stages': {}}
    with tempfile.TemporaryDirectory(prefix='telly-storage-benchmark-') as scratch:
        manifest = {'root': str(Path(scratch) / 'assets'), 'fixture': fixture,
                    'rows': rows, 'budget_bytes': report['cache_budget_bytes'],
                    'crash_marker': str(Path(scratch) / 'crash-marker')}
        manifest_path = Path(scratch) / 'manifest.json'
        for stage in ('register', 'read', 'lru', 'chart', 'reopen', 'interrupt', 'recover'):
            manifest_path.write_text(json.dumps(manifest))
            completed = subprocess.run([sys.executable, '-W', 'ignore', __file__, '--worker', stage,
                                        '--manifest', str(manifest_path)], capture_output=True, text=True, timeout=180)
            if stage == 'interrupt':
                result = {'killed_before_commit': completed.returncode == -signal.SIGKILL,
                          'insert_reached': Path(manifest['crash_marker']).exists()}
            elif completed.returncode:
                result = {'error': completed.stderr[-2000:], 'exit_code': completed.returncode}
            else:
                result = json.loads(completed.stdout)
            report['stages'][stage] = result
            if stage in {'register', 'chart'} and 'error' not in result:
                manifest.update(result)
            print(f'{stage}: {result.get("elapsed_seconds", "-")} s', flush=True)
            if 'error' in result:
                break
    stages = report['stages']
    expectations = {'read': ['fingerprint_matches', 'mutation_isolated'],
                    'lru': ['admitted_first', 'admitted_second', 'evicted_first', 'reload_matches', 'hit_mutation_isolated', 'cache_within_budget'],
                    'reopen': ['fingerprint_matches', 'png_matches', 'owner_isolated'],
                    'interrupt': ['killed_before_commit', 'insert_reached'],
                    'recover': ['uncommitted_asset_absent', 'original_matches', 'original_chart_present']}
    report['contract_checks_passed'] = (all(stages.get(stage, {}).get(key) is True
                                         for stage, keys in expectations.items() for key in keys)
        and all(stages.get(stage, {}).get('integrity_check') == 'ok' for stage in ('reopen', 'recover'))
        and stages.get('chart', {}).get('dataset_payload_reads_for_chart_insert') == 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(str(args.output))
    raise SystemExit(0 if report['contract_checks_passed'] else 1)


if __name__ == '__main__':
    main()
