"""Isolated-process million-row agent check; no remote SQL or LLM required."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from migration.test_persistent_runtime import QuietModel
from utils.analysis_image_validation import validate_chart_image


def runtime(path):
    return GraphAnalysisRuntime(path, 'large-eval', 'latest', QuietModel(),
        policy=RuntimePolicy(max_dataset_bytes=1024*1024*1024, frame_cache_bytes=0),intent_mode='contract_fixture')


def seed(path, fixture):
    r = runtime(path)
    key, value, order, unused = fixture['columns']
    def batches():
        for offset in range(0, fixture['rows'], 10_000):
            index = np.arange(offset, min(offset+10_000, fixture['rows']))
            cycle, entity = index // fixture['keys'], index % fixture['keys']
            yield pd.DataFrame({key: entity, value: np.array(fixture['categories'])[(entity+cycle)%3],
                order: pd.to_datetime(cycle, unit='s'), unused: 'x' * fixture['payload_bytes']})
    try:
        raw = r.datasets.register_batches(batches(), columns=fixture['columns'],
            source=fixture['source'], max_rows=fixture['rows'], coverage='complete', predicate_known=True)
        r.select_dataset(raw.id)
    finally:
        r.close()


def evaluate(path, fixture, output):
    r = runtime(path)
    raw = r.context.selected_dataset_id
    file = r.datasets.db.dataset_file(raw)
    before = sha256(file.read_bytes()).hexdigest()
    original_project = r.datasets.frames.project
    original_get = type(r.datasets.frames).__getitem__
    forbidden = []
    def project(key, columns):
        if key == raw:
            forbidden.append('project'); raise AssertionError('Full raw projection is forbidden')
        return original_project(key, columns)
    def get(cache, key):
        if key == raw:
            forbidden.append('get'); raise AssertionError('Full raw decoding is forbidden')
        return original_get(cache, key)
    started = time.monotonic()
    try:
        with patch.object(r.datasets.frames, 'project', side_effect=project), patch.object(type(r.datasets.frames), '__getitem__', get):
            result = r.submit(fixture['prompt'])
        elapsed = time.monotonic() - started
        state = r.inspect()['recovery']
        proof = state.get('latest_selection_evidence') or {}
        counts = {item[fixture['columns'][1]]: item[proof['count_column']] for item in proof.get('counts', [])}
        preserved = sha256(file.read_bytes()).hexdigest() == before
        images = []
        for identity in state.get('artifact_ids', []):
            card = r.artifacts[identity]
            validate_chart_image(card.image)
            destination = output.parent / 'latest_distribution.png'
            destination.write_bytes(card.image)
            images.append(destination.name)
        selected = r.datasets.frames[proof['dataset']['id']] if proof else pd.DataFrame()
        # The generator cycles exactly 100 times. The final cycle is 99;
        # its category is independently specified by key modulo 3.
        key, value, order, _ = fixture['columns']
        oracle_ok = (len(selected) == fixture['keys'] and selected[key].is_unique
            and all(row[value] == fixture['categories'][int(row[key]) % 3]
                    and row[order] == pd.Timestamp(99, unit='s') for row in selected.to_dict('records')))
        report = {'mode': 'production graph deterministic path; synthetic retained Parquet; no model inference',
            'status': 'PASS' if result['status']=='answered' and oracle_ok and counts==fixture['expected_counts']
                and preserved and images and not forbidden else 'FAIL',
            'agent_status': result['status'], 'input_rows': fixture['rows'], 'selected_keys': len(selected),
            'expected_counts': fixture['expected_counts'], 'actual_counts': counts, 'oracle_rows_match': oracle_ok,
            'raw_preserved': preserved, 'raw_full_read_attempts': forbidden, 'cache_bytes': r.datasets.frames.bytes,
            'elapsed_seconds': round(elapsed, 3),
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform=='darwin' else 1024),
            'warehouse_queries': 0, 'model_calls': state.get('model_calls'),
            'execution_mode': proof.get('execution_mode'), 'images': images, 'answer': result.get('text')}
    finally:
        r.close()
    r = runtime(path)
    try:
        report['restart_preserved'] = (r.datasets.metadata[raw].rows == fixture['rows']
            and r.datasets.metadata[proof['dataset']['id']].row_selection['input_dataset_id'] == raw)
    finally:
        r.close()
    if not report['restart_preserved']:
        report['status'] = 'FAIL'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status']=='PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['seed', 'evaluate'])
    parser.add_argument('--storage', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads((ROOT/'tests/fixtures/large_latest_recipe.json').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'seed':
        seed(args.storage, fixture)
    elif args.stage == 'evaluate':
        return evaluate(args.storage, fixture, args.output)
    else:
        with tempfile.TemporaryDirectory(prefix='telly-large-latest-eval-') as directory:
            for stage in ('seed', 'evaluate'):
                subprocess.run([sys.executable, __file__, '--stage', stage, '--storage', directory,
                    '--output', str(args.output.absolute())], check=True)


if __name__ == '__main__':
    raise SystemExit(main())
