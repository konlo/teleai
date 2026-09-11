"""Repeatable baseline evaluation. Never connects to Databricks.

Run: .venv/bin/python scripts/validate_analysis_agent.py [--live-local-model]
Live mode permits only an explicitly local Ollama endpoint and synthetic data.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from core.analysis_loop import AnalysisSession
from core.analysis_approval import ApprovalQueue
from core.analysis_instructions import ANALYSIS_INSTRUCTIONS
from core.analysis_runtime_tools import build_runtime_tools
from utils.analysis_datasets import DatasetStore, AnalysisNeed, assess_reuse


def evaluate(live=False):
    fixture = json.loads((ROOT/'tests/fixtures/analysis_acceptance.json').read_text())
    store = DatasetStore()
    info = store.register(pd.DataFrame(fixture['rows']), source=fixture['source'],
                          coverage='complete', predicate_known=True)
    session = AnalysisSession('acceptance-baseline', ANALYSIS_INSTRUCTIONS, [])
    session.tools = build_runtime_tools(session, store)
    checks = []
    def check(name, ok, evidence):
        checks.append(dict(name=name, status='PASS' if ok else 'FAIL', evidence=evidence))
    def answer(messages, tools):
        return {'role': 'assistant', 'content': '상태 확인', 'tool_calls': []}
    queue = ApprovalQueue('contract')
    req = queue.propose(source=fixture['source'], query='SELECT 1', reason='fixture', goal='test')
    calls = []
    def execute(r):
        calls.append(r.query)
        return {'status': 'ready'}
    try:
        queue.execute(req.id, execute)
    except PermissionError:
        pass
    check('unapproved_zero_execution', not calls, {'executions': len(calls)})
    queue.approve(req.id)
    queue.execute(req.id, execute)
    try:
        queue.execute(req.id, execute)
    except PermissionError:
        pass
    check('duplicate_execution_blocked', len(calls)==1, {'executions': len(calls)})
    rejected = queue.propose(source=fixture['source'], query='SELECT 2', reason='fixture', goal='test')
    queue.decline(rejected.id)
    try:
        queue.execute(rejected.id, execute)
    except PermissionError:
        pass
    check('rejected_zero_additional_execution', len(calls)==1, {'executions': len(calls)})
    pending = session.approvals.propose(source=fixture['source'], query='SELECT 3', reason='fixture', goal='test')
    session.state = 'awaiting_approval'
    session.submit('지금 어떤 승인을 기다리고 있어?', answer)
    status = session.approvals.get(pending.id).status
    check('status_question_preserves_pending', status=='proposed', {'actual': status})
    recreated = AnalysisSession(session.id, session.instructions, [])
    check('session_recreation_preserves_history', recreated.history==session.history,
          {'before_messages': len(session.history), 'after_messages': len(recreated.history),
           'scope': '새 세션 객체 복원 경로 검사; 실제 프로세스 강제종료 검증은 후속'})
    partial = store.register(pd.DataFrame(fixture['rows'][:2]), source=fixture['source'],
                             coverage='truncated', predicate_known=True)
    decision = assess_reuse(partial, AnalysisNeed(fixture['source'], info.columns))
    check('truncated_population_requires_proposal', decision.action=='query_source', {'decision': decision.action})
    session = AnalysisSession('live-acceptance', ANALYSIS_INSTRUCTIONS, [])
    # Keep only original source for the conversation; diagnostic partial data is unrelated.
    store.frames.pop(partial.id); store.metadata.pop(partial.id)
    session.tools = build_runtime_tools(session, store)
    tool = next(t for t in session.tools if t.name=='recommend_chart_images')
    cards = tool.run(dataset_id=info.id)
    check('chart_artifacts_are_real_png', bool(cards['cards']) and all(
        session.artifacts[c['id']].image.startswith(b'\x89PNG') for c in cards['cards']),
        {'cards': len(cards['cards'])})
    turns = []
    if live:
        import os
        from dotenv import load_dotenv
        from core.analysis_model import OllamaAnalysisModel
        load_dotenv(ROOT/'.env')
        endpoint = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        if urlparse(endpoint).hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Live evaluation only permits localhost Ollama')
        model = OllamaAnalysisModel(os.getenv('OLLAMA_MODEL', 'gemma4:e4b'), endpoint)
        for turn in fixture['turns']:
            start = len(session.history)
            try:
                result = session.submit(turn['prompt'], model)
                observations = [json.loads(m['content']) for m in session.history[start:]
                                if m['role']=='tool' and m.get('name')=='local_analysis_sql']
                # Require an actual final scalar calculation, not a number in raw preview rows.
                scalar = None
                for obs in observations:
                    dataset = obs.get('dataset', {})
                    frame = store.frames.get(dataset.get('id'))
                    if frame is not None and frame.shape==(1,1):
                        scalar = float(frame.iloc[0,0])
                ok = result['status']=='answered' and scalar is not None and abs(scalar-turn['expected'])<1e-8
                record = dict(prompt=turn['prompt'], expected=turn['expected'], actual=scalar,
                              status='PASS' if ok else 'FAIL', answer=result,
                              observations=observations)
            except Exception as exc:
                record = dict(prompt=turn['prompt'], status='FAIL', error_type=type(exc).__name__)
            turns.append(record)
            print(json.dumps({k:v for k,v in record.items() if k not in {'observations','answer'}}, ensure_ascii=False), flush=True)
    return dict(generated_at=datetime.now(timezone.utc).isoformat(), runtime='current AnalysisSession',
                remote_database_executions=0, checks=checks, live_turns=turns,
                limitations=['No live Databricks', 'No LangGraph runtime yet',
                             'Final answer semantic correctness requires review',
                             'No real process crash or browser journey in this harness'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live-local-model', action='store_true')
    args=parser.parse_args()
    report=evaluate(args.live_local_model)
    dest=ROOT/'docs/analysis_acceptance_results.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
    print(f'Report: {dest}')
    print(json.dumps(report['checks'],ensure_ascii=False,indent=2))
    return int(any(c['status']=='FAIL' for c in report['checks']+report['live_turns']))

if __name__=='__main__':
    sys.exit(main())
