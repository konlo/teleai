"""Synthetic retained-data scale evaluation; no inference or warehouse claims."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_outliers import select_outlier_rows


def file_hash(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT/'tests/fixtures/large_outlier_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = json.loads(args.fixture.read_text())
    value, key, payload = fixture['columns']
    with tempfile.TemporaryDirectory(prefix='telly-outlier-scale-') as root:
        db = AssetDB(root, 'evaluation', 'outliers')
        try:
            store = PersistentDatasets(db, budget=0)
            def seed():
                for start in range(0, fixture['rows'], 10000):
                    index = np.arange(start, min(start+10000, fixture['rows']))
                    values = (index % fixture['normal_cycle']).astype(float)
                    values[(index+1) % fixture['exception_period'] == 0] = fixture['exception_value']
                    yield pd.DataFrame({value: values, key: index, payload: 'x'*fixture['payload_bytes']})
            parent = store.register_batches(seed(), columns=fixture['columns'], source=fixture['source'],
                max_rows=fixture['rows'], coverage='complete', predicate_known=True, snapshot='synthetic:v1')
            db.select_dataset(parent.id)
            before = file_hash(db.dataset_file(parent.id))
            store.max_frame_bytes = 64*1024*1024
            reads, batch_sizes = [], []
            original_project, original_batches = store.frames.project, store.frames.batches
            def project(identity, columns):
                reads.append(list(columns))
                if identity == parent.id and list(columns) != [value]:
                    raise AssertionError('Only the measured raw column may be projected')
                return original_project(identity, columns)
            def batches(*a, **kw):
                for batch in original_batches(*a, **kw):
                    batch_sizes.append(batch.num_rows)
                    yield batch
            started = time.monotonic()
            with (patch.object(FrameCache, '__getitem__', side_effect=AssertionError('whole raw read')),
                    patch.object(store.frames, 'project', side_effect=project),
                    patch.object(store.frames, 'batches', side_effect=batches)):
                result = select_outlier_rows(store, parent.id, column=value, method='iqr', tail='upper')
            elapsed = time.monotonic()-started
            child = result['dataset']['id']
            actual = store.frames.project(child, [key, value])
            expected_keys = list(range(fixture['exception_period']-1, fixture['rows'], fixture['exception_period']))
            correct = (len(actual) == fixture['expected_selected_rows'] and
                       actual[key].tolist() == expected_keys and
                       actual[value].eq(fixture['exception_value']).all())
            digest_ok = stored_dataset_digest(store, child) == result['selection_summary']['data_sha256']
            preserved = file_hash(db.dataset_file(parent.id)) == before and db.selected_dataset_id() == parent.id
            report = {'status': 'PASS' if correct and digest_ok and preserved else 'FAIL',
                'mode': 'synthetic production tool, no LLM or remote SQL', 'input_rows': fixture['rows'],
                'selected_rows': len(actual), 'independent_row_oracle': bool(correct),
                'digest_matches': digest_ok, 'original_preserved': preserved,
                'raw_projected_columns': reads, 'max_batch_rows': max(batch_sizes),
                'batches': len(batch_sizes), 'cache_bytes': store.frames.bytes,
                'elapsed_seconds': round(elapsed, 3),
                'whole_process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        finally:
            db.close()
        reopened = AssetDB(root, 'evaluation', 'outliers')
        try:
            restored = PersistentDatasets(reopened, budget=0)
            report['restart_preserved'] = (restored.metadata[child].parent_id == parent.id and
                stored_dataset_digest(restored, child) == result['selection_summary']['data_sha256'] and
                file_hash(reopened.dataset_file(parent.id)) == before)
            if not report['restart_preserved']:
                report['status'] = 'FAIL'
        finally:
            reopened.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
