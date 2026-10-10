"""Read-only capture of the existing browser conversation; never submits work."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(__file__).resolve().parents[3]
SCOPE = ROOT / '.telly_runtime/v1/mysql_eval/44bc5ed80f4a4583ada9f905/ccf9c5f6d58ecf99399cb150810c6051b0223abfa39541b857a049eeb9a3b18f'

def connect(name):
    return sqlite3.connect(f'file:{SCOPE / name}?mode=ro', uri=True, check_same_thread=False)

def capture():
    with connect('assets.sqlite') as db:
        assets = {}
        for ident, kind, metadata, payload in db.execute('SELECT id,kind,metadata,payload FROM assets'):
            file = SCOPE / f'{ident}.parquet'
            content = payload if payload is not None else file.read_bytes()
            assets[ident] = {'kind': kind, 'metadata': json.loads(metadata),
                'metadata_sha256': hashlib.sha256(metadata.encode()).hexdigest(),
                'payload_sha256': hashlib.sha256(content).hexdigest()}
        selection = db.execute("SELECT dataset_id FROM selection WHERE slot='active'").fetchone()
    conn = connect('graph.sqlite')
    try:
        item = SqliteSaver(conn).get_tuple({'configurable': {'thread_id': 'conversation'}})
        state = item.checkpoint['channel_values'] if item else {}
        current = state.get('recovery') or {}
    finally:
        conn.close()
    logs = [json.loads(line) for line in (SCOPE / 'runtime.jsonl').read_text().splitlines()]
    last = next((r['run_id'] for r in reversed(logs) if r.get('run_id')), None)
    run = [r for r in logs if r.get('run_id') == last]
    wanted = ('request_id','request_text','status','stop_reason','intent_origin','goal','model_calls',
        'model_seconds','scope','required_sources','required_columns','artifact_ids','evidence_ids',
        'table_preview_evidence','metadata_evidence','value_list_evidence','chart_axes','kind',
        'failed','scope_error','goal_contract_error','source_scatter_error')
    return {'assets': assets, 'selection': selection[0] if selection else '',
        'current': {k: current[k] for k in wanted if k in current}, 'run_id': last, 'run': run}

if __name__ == '__main__':
    result = capture()
    if len(sys.argv) > 2:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if (result['current'].get('request_text') == sys.argv[2]
                    and result['current'].get('request_id') in {r.get('request_id') for r in result['run'] if r.get('request_id')}
                    and any(r['event'] == 'run_completed' for r in result['run'])):
                break
            time.sleep(1)
            result = capture()
        result['requested_prompt']=sys.argv[2]
        result['current_matches_run']=result['current'].get('request_id') in {r.get('request_id') for r in result['run'] if r.get('request_id')}
        result['request_checkpoint_verified']=(result['current_matches_run'] and result['current'].get('request_text')==sys.argv[2]
            and any(r['event']=='run_completed' for r in result['run']))
    output = Path(__file__).parent / (sys.argv[1] + '.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')
    summary = {k:v for k,v in result.items() if k not in {'assets','run','current'}}
    summary['current'] = {k:v for k,v in result['current'].items() if k != 'metadata_evidence'}
    summary['asset_count'] = len(result['assets'])
    summary['events'] = [r for r in result['run'] if r['event'] in {'run_completed','error','goal_contract_error','model_inference_failed','remote_query_finished'}]
    print(json.dumps(summary, ensure_ascii=False, default=str))
