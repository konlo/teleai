"""Live inference checks for pending-message interpretation and compaction.

The executor rejects every query. Fixtures are synthetic; these checks establish
state preservation and live integration, not a general natural-language score.
Fault injection and bounded retry evidence live in test_auxiliary_model_recovery.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from langchain_core.messages import HumanMessage, AIMessage
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime


def evaluate(provider):
    model=build_analysis_chat_model(RuntimePolicy(),provider=provider)
    results=[];executions=[]
    def execute(envelope):
        executions.append(True)
        raise AssertionError('No query may execute in this evaluation')
    with tempfile.TemporaryDirectory(prefix='teleai-aux-live-') as directory:
        r=GraphAnalysisRuntime(directory,'evaluation','pending',model,
            remote_factory=lambda _:execute,connection_identity='synthetic-only')
        try:
            grant=r.propose_query('synthetic.events','SELECT 1','synthetic state check')['requests'][0]
            checkpoint=r.agent.get_state(r.config).config
            for prompt in ('지금 어떤 승인을 기다리고 있어?', '왜 승인을 받아야 해?', '음...'):
                start=time.monotonic();outcome=r.submit(prompt)
                preserved=(r.ledger.get(grant['id'])['status']=='proposed'
                    and r.agent.get_state(r.config).config==checkpoint and not executions)
                events=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                action=next((e.get('action') for e in reversed(events)
                             if e.get('event')=='approval_intent_checked'),None)
                expected='uncertain' if prompt=='음...' else 'status'
                results.append({'kind':'pending_message','prompt':prompt,
                    'status':'PASS' if preserved and action==expected else 'FAIL',
                    'expected':expected,'actual':action,'approval_preserved':preserved,
                    'error_type':outcome.get('error_type'),'elapsed_seconds':round(time.monotonic()-start,3),
                    'invalid_response_events':sum(e.get('event')=='approval_intent_invalid_response' for e in events),
                    'model_recovery':r.inspect()['model_recovery']})
            before=r.inspect()['model_recovery']
            r.submit('지금 어떤 승인을 기다리고 있어?')
            after=r.inspect()['model_recovery']
            results.append({'kind':'cached_status','status':'PASS' if before==after else 'FAIL'})
        finally:r.close()
        r=GraphAnalysisRuntime(directory,'evaluation','summary',model,
            summary_trigger_tokens=100,summary_keep_messages=2)
        try:
            history=[]
            for _ in range(5):
                history.extend([HumanMessage(content='아직 데이터를 로딩하지 않았고 분석 목적을 논의하고 있습니다. '*30,id=str(uuid4())),
                                AIMessage(content='데이터를 조회하지 않았습니다.',id=str(uuid4()))])
            r.agent.update_state(r.config,{'messages':history},as_node='model')
            r.resume();r.events();before=[m.id for m in r.events()]
            start=time.monotonic();outcome=r.submit('앞선 대화에서 확인한 상태를 간단히 설명해줘')
            state=r.inspect()['recovery'];after=[m.id for m in r.events()]
            preserved=after[:len(before)]==before
            results.append({'kind':'summary_graph','status':'PASS' if preserved
                and outcome['status']=='answered' and state.get('summary_model_calls',0)>0 else 'FAIL',
                'transcript_preserved':preserved,'agent_status':outcome['status'],
                'summary_model_calls':state.get('summary_model_calls',0),
                'model_calls':state.get('model_calls'), 'output':outcome.get('text',''),
                'error_type':outcome.get('error_type'),'elapsed_seconds':round(time.monotonic()-start,3)})
        finally:r.close()
    return {'provider':provider,'data':'synthetic only','query_executions':len(executions),
            'results':results,'status':'PASS' if not executions and all(r['status']=='PASS' for r in results) else 'FAIL'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=['databricks','ollama'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    result=evaluate(args.provider)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(result['results']),'query_executions':result['query_executions']}))
    raise SystemExit(0 if result['status']=='PASS' else 1)
