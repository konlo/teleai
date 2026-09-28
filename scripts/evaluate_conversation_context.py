"""Live multi-turn context evaluation with independent per-turn fixture oracles."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from PIL import Image
import sqlglot
from sqlglot import exp
from langchain_core.messages import AIMessage, ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from utils.analysis_datasets import stored_dataset_digest


def run_case(spec, fixture, model, repeat):
    frames = {d['source']: pd.DataFrame(d['rows'], columns=d['columns']) for d in fixture['datasets']}
    refs = [{'table': name, 'observed_at': datetime.now(timezone.utc).isoformat(),
             'columns': [{'name': c, 'dtype': str(frame[c].dtype)} for c in frame]}
            for name, frame in frames.items()]
    with tempfile.TemporaryDirectory(prefix='teleai-context-eval-') as root:
        def open_runtime(conversation):
            return GraphAnalysisRuntime(root, 'context-eval', conversation, model,
                summary_trigger_tokens=spec.get('summary_trigger_tokens', 6000),
                summary_keep_messages=2 if spec.get('require_summary') else 8,
                policy=RuntimePolicy(model_timeout_seconds=45), reference_context_loader=lambda: refs)
        def seed(runtime, selected):
            ids = {}
            for source, frame in frames.items():
                info = runtime.datasets.register(frame, source=source, coverage='complete',
                    predicate_known=True, snapshot='fixture-v1')
                ids[source] = (info.id, stored_dataset_digest(runtime.datasets, info.id))
            runtime.select_dataset(ids[selected][0])
            return ids
        runtime = open_runtime('primary')
        identities = seed(runtime, fixture['datasets'][0]['source'])
        original_identities = identities
        turns = []
        blocked = False
        for number, turn in enumerate(spec['turns'], 1):
            if blocked:
                turns.append({'turn': number, 'prompt': turn['prompt'], 'status': 'NOT_RUN',
                              'reason': 'Previous turn left unfinished checkpoint; no manual resume'})
                continue
            if turn.get('reopen_before') or turn.get('new_session_before') or turn.get('return_session_before'):
                runtime.close()
                runtime = open_runtime('isolated' if turn.get('new_session_before') else 'primary')
                identities = seed(runtime, turn['select_source']) if turn.get('new_session_before') else original_identities
            prior_ids = {m.id for m in runtime.events()}
            started = time.monotonic()
            try:
                outcome = runtime.submit(turn['prompt'])
            except Exception as exc:
                outcome = {'status': 'incomplete', 'error_type': type(exc).__name__, 'text': ''}
            state = runtime.inspect()['recovery']
            events = [m for m in runtime.events() if m.id not in prior_ids]
            calls = [c for m in events if isinstance(m, AIMessage) for c in m.tool_calls]
            observations = []
            for message in events:
                if isinstance(message, ToolMessage):
                    try:
                        value = json.loads(message.content)
                    except (TypeError, ValueError):
                        value = {}
                    observations.append({'tool': message.name, 'status': value.get('status'),
                        'error_code': value.get('error_code'), 'chart_spec': value.get('chart_spec'),
                        'cards': value.get('cards', [])})
            evidence, match = [], False
            for key in state.get('evidence_ids', []):
                info = runtime.datasets.metadata[key]
                frame = runtime.datasets.frames[key]
                evidence.append({'source': info.source, 'query': info.query,
                                 'rows': frame.to_dict(orient='records')})
                if turn['kind'] == 'scalar' and frame.shape == (1, 1):
                    try:
                        value_ok = abs(float(frame.iloc[0, 0])-turn['expected']) < 1e-8
                        tree = sqlglot.parse_one(info.query, read='duckdb')
                        operation_ok = any(n.sql_name() == turn['operation'] and
                            any(c.name == turn['column'] for c in n.find_all(exp.Column))
                            for n in tree.find_all(exp.AggFunc))
                        match = match or (value_ok and operation_ok and info.source == turn['source'])
                    except (ValueError, TypeError):
                        pass
            cards = state.get('artifact_ids', [])
            if turn['kind'] == 'chart':
                for key in cards:
                    card = runtime.artifacts[key]
                    info = runtime.datasets.metadata[card.dataset_id]
                    values = runtime.datasets.frames[card.dataset_id]
                    png_ok = False
                    try:
                        with Image.open(BytesIO(card.image)) as image:
                            image.verify()
                        png_ok = True
                    except (ValueError, OSError):
                        pass
                    spec_ok = any((o['chart_spec'] or {}).get('bins') == turn['bins'] and
                                  any(c.get('id') == key for c in o['cards']) for o in observations)
                    values_ok = (turn['column'] in values and
                        sorted(values[turn['column']].dropna().tolist()) == sorted(turn['expected_values']))
                    match = match or (png_ok and spec_ok and values_ok and card.kind == 'histogram'
                                      and info.source == turn['source'])
            if turn['kind'] == 'explanation':
                match = all(word in outcome.get('text', '') for word in turn['must_contain']) and not calls and not cards
            unchanged = all(stored_dataset_digest(runtime.datasets, key) == digest for key, digest in identities.values())
            passed = outcome['status'] == 'answered' and match and unchanged and not runtime.inspect()['requests']
            turns.append({'turn': number, 'prompt': turn['prompt'], 'oracle': turn,
                'status': 'PASS' if passed else 'FAIL', 'agent_status': outcome['status'],
                'false_completion': outcome['status'] == 'answered' and not match,
                'error_type': outcome.get('error_type'), 'answer': outcome.get('text', ''),
                'evidence': evidence, 'observations': observations, 'tools': [c['name'] for c in calls],
                'state': {k: state.get(k) for k in ('scope', 'required_columns', 'operations', 'kind', 'profile_kind', 'stop_reason')},
                'model_calls': state.get('model_calls'), 'raw_preserved': unchanged,
                'elapsed_seconds': round(time.monotonic()-started, 3)})
            print(json.dumps({'case': spec['id'], 'repeat': repeat, 'turn': number,
                              'status': turns[-1]['status'], 'agent': outcome['status']}, ensure_ascii=False), flush=True)
            blocked = bool(runtime.agent.get_state(runtime.config).next)
        logs = [json.loads(line) for line in runtime.diagnostics.path.read_text().splitlines()]
        summary_messages = [m for m in runtime.agent.get_state(runtime.config).values.get('messages', [])
                            if m.additional_kwargs.get('lc_source') == 'summarization']
        summary_count = sum(e.get('event') == 'summarization_finished' and
                            e.get('status') == 'ok' and e.get('summarized') is True for e in logs)
        summary_seen = bool(summary_messages) or summary_count > 0
        runtime.close()
        return {'id': spec['id'], 'repeat': repeat,
            'status': ('FAIL' if not all(t['status'] == 'PASS' for t in turns) else
                       'NOT_EXERCISED' if spec.get('require_summary') and not summary_seen else 'PASS'),
            'summary_required': bool(spec.get('require_summary')), 'summary_seen': summary_seen,
            'summary_events': [e for e in logs if 'summar' in str(e.get('event', '')).lower()],
            'turns': turns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--id', action='append')
    a = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    fixture = json.loads((ROOT / 'tests/fixtures/conversation_context_journeys.json').read_text())
    results = []
    for repeat in range(1, a.repeats+1):
        for case in fixture['cases']:
            if a.id and case['id'] not in a.id:
                continue
            model = build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=45), provider='databricks')
            results.append(run_case(case, fixture, model, repeat))
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({'mode': 'actual Databricks model; production recovery; synthetic local data',
                'generated_at': datetime.now(timezone.utc).isoformat(), 'warehouse_sql_executions': 0,
                'limitations': ['Known fixture, not a population success estimate; deterministic paths may call no model.',
                    'Summary journey lowers trigger to 1200 tokens to exercise compaction.',
                    'Explanation turn checks topic terms and absence of actions; prose quality requires human review.',
                    'No manual resume or state correction between turns. Unfinished checkpoints block later turns.'],
                'results': results}, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
