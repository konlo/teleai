"""Replay reported request with the configured model; never executes remote data."""
import json, os, tempfile, time
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,AIMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime

load_dotenv()
fixture=json.loads(Path('tests/fixtures/histogram_request.json').read_text())
contexts=[json.loads(p.read_text()) for p in Path('.telly_table_context/contexts').glob('*.json')]
context=next(c for c in contexts if c['table_fqn']==fixture['source'])
model=ChatOllama(model=os.getenv('OLLAMA_MODEL','gemma4:e4b'),base_url=os.getenv('OLLAMA_BASE_URL','http://localhost:11434'),reasoning=True,temperature=0,num_ctx=16384,num_predict=4096,client_kwargs={'timeout':60})
with tempfile.TemporaryDirectory() as root:
    def forbidden(_):raise AssertionError('Remote execution is not authorized by this check')
    r=GraphAnalysisRuntime(root,'test','live',model,connection_identity='validation',remote_factory=lambda _:forbidden)
    r.context.reference_context.append({'table':context['table_fqn'],'training_status':context.get('training_status'),'columns':context['columns']})
    r.agent.update_state(r.config,{'messages':[HumanMessage(content=fixture['previous_request']),AIMessage(content='현재 선택한 테이블: '+fixture['source'])]},as_node='model')
    if r.agent.get_state(r.config).next:r.resume()
    started=time.monotonic()
    result=r.submit(fixture['request'])
    events=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
    report={'result':result,'elapsed_seconds':round(time.monotonic()-started,3),'database_calls':0,'events':events}
    Path('docs/live_recovery_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    r.close()
    if result['status']!='awaiting_approval':raise SystemExit(1)
