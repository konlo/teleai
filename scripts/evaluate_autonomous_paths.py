"""Live synthetic graph journeys with deterministic planning disabled; no remote SQL."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_agent import fixture_reference_context
from utils.analysis_datasets import stored_dataset_digest


class FaultThenLiveModel(BaseChatModel):
    delegate: Any
    first_call: dict | None = None
    tracker: dict = Field(default_factory=dict)
    @property
    def _llm_type(self): return 'fault-injection-then-live-model'
    def bind_tools(self,tools,**kwargs):
        return self.model_copy(update={'delegate':self.delegate.bind_tools(tools,**kwargs)})
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        if self.first_call and not self.tracker.get('injected'):
            self.tracker['injected']=True
            args={k:self.tracker['dataset_id'] if v=='$raw' else v for k,v in self.first_call['args'].items()}
            msg=AIMessage(content='',tool_calls=[{'name':self.first_call['name'],'args':args,'id':'injected-fault'}])
        else:
            self.tracker['live_calls']=self.tracker.get('live_calls',0)+1
            msg=self.delegate.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])


CASES=[
 {'id':'discover_mean','prompt':'로컬 평균 집계 도구의 입력 규칙을 검색해서 확인한 뒤 reading 평균을 계산해줘.', 'expected':9.,'search_required':True},
 {'id':'repair_sql','prompt':'reading 평균을 알려줘.', 'expected':9.,
  'first_call':{'name':'local_analysis_sql','args':{'dataset_id':'$raw','query':'SELECT AVG(missing_column) AS mean FROM data'}}},
 {'id':'local_outage','prompt':'reading 평균을 알려줘.', 'expected':9.,'outage':True,
  'first_call':{'name':'aggregate_dataset','args':{'dataset_id':'$raw','aggregation':'mean','value_column':'reading'}}},
 {'id':'compound','prompt':'reading 평균을 계산하고 reading 히스토그램도 보여줘.', 'expected':9.,'chart':True},
 {'id':'followup','prompt':'reading >= 10인 행의 건수를 알려줘.', 'expected':2.,
  'followup':{'prompt':'그중 reading 평균을 알려줘.', 'expected':15.}},
]


def evaluate_case(spec,delegate):
    remote=[]
    def remote_factory(_):
        def forbidden(envelope):
            remote.append(True)
            raise AssertionError('Unapproved remote execution')
        return forbidden
    with tempfile.TemporaryDirectory(prefix='teleai-autonomy-') as root:
        model=FaultThenLiveModel(delegate=delegate,first_call=spec.get('first_call'))
        r=GraphAnalysisRuntime(root,'evaluation',spec['id'],model)
        frame=pd.DataFrame({'reading':[2.,4.,10.,20.], 'cohort':['red','red','blue','blue']})
        raw=r.datasets.register(frame,source='unfamiliar.observations',coverage='complete',predicate_known=True,snapshot='synthetic-v1')
        r.context.reference_context[:]=[fixture_reference_context(raw.source,frame)]
        r.select_dataset(raw.id)
        digest=stored_dataset_digest(r.datasets,raw.id)
        model.tracker['dataset_id']=raw.id
        turns=[]
        try:
            for turn in [spec]+([spec['followup']] if spec.get('followup') else []):
                start=time.monotonic()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(r.recovery,'_next_local',return_value=None))
                    stack.enter_context(patch.object(r.recovery,'_budget_local_rescue',return_value=None))
                    stack.enter_context(patch.object(r.recovery,'_cached_chart_call',return_value=None))
                    if spec.get('outage'):
                        stack.enter_context(patch('core.analysis_runtime_tools.build_aggregate_dataset',return_value={
                            'status':'unavailable','error_code':'local_worker_unavailable','retryable':False,
                            'message':'This local aggregate tool is unavailable for this turn. Use a different local tool with the same raw dataset and request scope; no remote reload is necessary.'}))
                    outcome=r.submit(turn['prompt'])
                state=r.inspect()['recovery']
                evidence=state.get('evidence_ids',[])
                actual=None
                if evidence:
                    result=r.datasets.frames[evidence[-1]]
                    if result.shape==(1,1):actual=float(result.iloc[0,0])
                charts=state.get('artifact_ids',[])
                chart_valid=all(r.artifacts[c].image.startswith(b'\x89PNG') for c in charts)
                requests=r.inspect()['requests']
                valid=(outcome['status']=='answered' and actual==turn['expected'] and
                       digest==stored_dataset_digest(r.datasets,raw.id) and not requests and not remote and
                       (not turn.get('chart') or (charts and chart_valid)))
                messages=r.events()
                called=[call['name'] for m in messages if isinstance(m,AIMessage) for call in m.tool_calls]
                search_used='search_analysis_tools' in called
                if turn.get('search_required'):valid=valid and search_used
                turns.append({'prompt':turn['prompt'],'status':'PASS' if valid else 'FAIL',
                    'agent_status':outcome['status'],'final_output':outcome.get('text',''),
                    'expected':turn['expected'],'actual':actual,'model_calls':state['model_calls'],
                    'search_used':search_used,'tools':called,'chart_count':len(charts),
                    'raw_unchanged':digest==stored_dataset_digest(r.datasets,raw.id),
                    'approval_requests':len(requests),'remote_executions':len(remote),
                    'stop_reason':state.get('stop_reason'),'elapsed_seconds':round(time.monotonic()-start,3)})
                if spec.get('followup') and len(turns)==1:
                    # Restart actual runtime; all scope must come from checkpoint.
                    r.close()
                    r=GraphAnalysisRuntime(root,'evaluation',spec['id'],model)
                    r.context.reference_context[:]=[fixture_reference_context(raw.source,frame)]
            return {'id':spec['id'],'status':'PASS' if all(x['status']=='PASS' for x in turns) else 'FAIL',
                    'injected_first_call':bool(spec.get('first_call')),'live_calls':model.tracker.get('live_calls',0),'turns':turns}
        finally:r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=['databricks','ollama'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--id',action='append')
    a=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    model=build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=45),provider=a.provider)
    results=[]
    for spec in CASES:
        if a.id and spec['id'] not in a.id:continue
        try:record=evaluate_case(spec,model)
        except Exception as exc:record={'id':spec['id'],'status':'FAIL','error_type':type(exc).__name__}
        results.append(record)
        report={'generated_at':datetime.now(timezone.utc).isoformat(),'provider':a.provider,
                'mode':'real-model synthetic journeys; deterministic rescue disabled',
                'limitations':['Some cases inject a first faulty call; subsequent decisions use the live model.',
                               'No real remote SQL executor is connected; this does not validate loading.'],
                'results':results}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'id':record['id'],'status':record['status'],'live_calls':record.get('live_calls')},ensure_ascii=False),flush=True)
    return 0 if results and all(r['status']=='PASS' for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
