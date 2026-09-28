"""Verify a reported request against a read-only copy of existing local assets.

The source conversation and approval ledger are never modified. The only remote
executor raises an error; approvals are never granted by this script.
"""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def copy_runtime_snapshot(source, destination):
    """Snapshot under the app's conversation lock; never replay live approvals."""
    copied = []
    with (source / 'runtime.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            for name in ('assets.sqlite', 'graph.sqlite', 'approvals.sqlite'):
                path = source / name
                if not path.exists(): raise FileNotFoundError(name)
                with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as original, \
                        sqlite3.connect(destination / name) as target:
                    original.backup(target)
                copied.append(name)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conversation', required=True)
    parser.add_argument('--owner', default='local-owner')
    parser.add_argument('--storage', type=Path, default=ROOT / '.telly_runtime/v1')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/histogram_request.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/cached_histogram_validation.json')
    parser.add_argument('--live-local-model', action='store_true')
    parser.add_argument('--include-history', action='store_true', help='Replay with a locked SQLite snapshot of graph, transcript and approval state')
    args = parser.parse_args()
    if args.include_history and not args.live_local_model:
        parser.error('--include-history requires --live-local-model')
    fixture = json.loads(args.fixture.read_text())
    key = sha256(json.dumps([args.owner, args.conversation]).encode()).hexdigest()
    path = args.storage / key / 'assets.sqlite'
    assets = []
    if not args.include_history:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            assets = db.execute('SELECT id,kind,metadata,payload FROM assets').fetchall()
    from core.analysis_agent.assets import AssetDB, PersistentDatasets, PersistentCharts
    from core.analysis_tool_contract import AnalysisToolContext
    from core.analysis_runtime_tools import build_analysis_tools
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    remote_attempts = []
    def forbidden(envelope):
        remote_attempts.append(envelope)
        raise AssertionError('This validation never permits remote execution')
    with tempfile.TemporaryDirectory(prefix='telly-cached-check-') as temporary:
        db = AssetDB(temporary, 'validation', 'cache')
        for asset_id, kind, metadata, payload in assets:
            db.put(asset_id, kind, json.loads(metadata), payload)
        destination = db.directory
        db.close()
        copied = copy_runtime_snapshot(path.parent, destination) if args.include_history else []
        model = None
        if args.live_local_model:
            from dotenv import load_dotenv
            from langchain_ollama import ChatOllama
            load_dotenv(ROOT / '.env')
            endpoint = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            if urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
                raise ValueError('Local model endpoint required')
            os.environ['LANGSMITH_TRACING'] = 'false'
            os.environ['LANGCHAIN_TRACING_V2'] = 'false'
            model = ChatOllama(model=os.getenv('OLLAMA_MODEL', 'gemma4:e4b'), base_url=endpoint,
                reasoning=True, temperature=0, num_ctx=16384, num_predict=4096, client_kwargs={'timeout':60})
            runtime = GraphAnalysisRuntime(temporary, 'validation', 'cache', model,
                connection_identity='validation-no-remote', remote_factory=lambda _: forbidden)
            context = runtime.context
        else:
            db = AssetDB(temporary, 'validation', 'cache')
            context = AnalysisToolContext(PersistentDatasets(db), PersistentCharts(db), [], forbidden)
        from core.analysis_catalog import load_saved_reference_context
        context.reference_context.extend(load_saved_reference_context(ROOT / '.telly_table_context'))
        if not context.reference_context:
            context.reference_context.append({'table':fixture['source'], 'columns':[{'name':fixture['column']} ]})
        before = set(context.artifacts)
        if model:
            initial = runtime.inspect()
            if initial['state'] == 'incomplete':
                outcome = {'status':'not_run', 'reason':'Source conversation has unfinished graph work; no old execution was resumed'}
            else:
                outcome = runtime.submit(fixture['request'])
            state = runtime.inspect()['recovery']
            card_ids = state.get('artifact_ids', [])
        else:
            prepare = next(t.run for t in build_analysis_tools(context) if t.name == 'prepare_histogram')
            outcome = prepare(fixture['source'], fixture['column'])
            card_ids = [c['id'] for c in outcome.get('cards', [])]
            state = {}
        success = outcome.get('status') in {'ready', 'answered'} and bool(card_ids) and set(card_ids) <= before and not remote_attempts
        evidence = []
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for card_id in card_ids:
            card = context.artifacts[card_id]
            image_path = args.output.with_suffix('.png')
            image_path.write_bytes(card.image)
            evidence.append({'chart_id':card_id, 'dataset_id':card.dataset_id,
                'png_sha256':sha256(card.image).hexdigest(), 'scope':card.scope,
                'image_path':str(image_path.resolve())})
        report = {'status':'PASS' if success else 'FAIL', 'mode':'actual-local-model' if model else 'local-tools',
            'request':fixture['request'], 'source_conversation':args.conversation,
            'source_assets_read_only':True, 'original_chart_reused':bool(card_ids) and set(card_ids) <= before,
            'remote_executions':len(remote_attempts), 'result':outcome, 'recovery':state, 'evidence':evidence,
            'include_history':args.include_history, 'copied_databases':copied,
            'initial_state':initial['state'] if model else None,
            'initial_message_count':initial['message_count'] if model else 0,
            'events':[json.loads(line) for line in runtime.diagnostics.path.read_text().splitlines()] if model else []}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
        print(json.dumps({k:report[k] for k in ('status','mode','original_chart_reused','remote_executions')}, ensure_ascii=False))
        if model: runtime.close()
        else: db.close()
        return int(not success)


if __name__ == '__main__': raise SystemExit(main())
