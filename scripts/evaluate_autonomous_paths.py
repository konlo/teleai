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
import numpy as np
from io import BytesIO
from PIL import Image, ImageStat
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
    inject_count: int = 1
    tracker: dict = Field(default_factory=dict)
    @property
    def _llm_type(self): return 'fault-injection-then-live-model'
    def bind_tools(self,tools,**kwargs):
        return self.model_copy(update={'delegate':self.delegate.bind_tools(tools,**kwargs)})
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        if self.first_call and self.tracker.get('injected',0)<self.inject_count:
            self.tracker['injected']=self.tracker.get('injected',0)+1
            args={k:self.tracker['dataset_id'] if v=='$raw' else v for k,v in self.first_call['args'].items()}
            msg=AIMessage(content='',tool_calls=[{'name':self.first_call['name'],'args':args,'id':f'injected-fault-{self.tracker["injected"]}'}])
        else:
            self.tracker['live_calls']=self.tracker.get('live_calls',0)+1
            msg=self.delegate.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])


CASES=[
 {'id':'repeated_discovery','prompt':'reading 평균을 알려줘.', 'expected':9.,'inject_count':3,
  'first_call':{'name':'list_analysis_context','args':{}}},
 {'id':'discover_mean','prompt':'로컬 평균 집계 도구의 입력 규칙을 검색해서 확인한 뒤 reading 평균을 계산해줘.', 'expected':9.,'search_required':True},
 {'id':'repair_sql','prompt':'reading 평균을 알려줘.', 'expected':9.,
  'first_call':{'name':'local_analysis_sql','args':{'dataset_id':'$raw','query':'SELECT AVG(missing_column) AS mean FROM data'}}},
 {'id':'local_outage','prompt':'reading 평균을 알려줘.', 'expected':9.,'outage':True,
  'first_call':{'name':'aggregate_dataset','args':{'dataset_id':'$raw','aggregation':'mean','value_column':'reading'}}},
 {'id':'repeated_outage','prompt':'reading 평균을 알려줘.', 'expected':9.,'outage':True,'inject_count':2,
  'first_call':{'name':'aggregate_dataset','args':{'dataset_id':'$raw','aggregation':'mean','value_column':'reading'}}},
 {'id':'local_timeout','prompt':'reading 평균을 알려줘.', 'expected':9.,'timeout':True,
  'first_call':{'name':'aggregate_dataset','args':{'dataset_id':'$raw','aggregation':'mean','value_column':'reading'}}},
 {'id':'compound','prompt':'reading 평균을 계산하고 reading 히스토그램도 보여줘.', 'expected':9.,'chart':True},
 {'id':'followup','prompt':'reading >= 10인 행의 건수를 알려줘.', 'expected':2.,
  'followup':{'prompt':'그중 reading 평균을 알려줘.', 'expected':15.}},
]


def output_proof(runtime,state,expected,chart_required=False):
    # A compound turn may finish with a chart dataset after its scalar dataset.
    # Grade both independent obligations, never just the last evidence ID.
    scalars=[]
    for ident in state.get('evidence_ids',[]):
        result=runtime.datasets.frames[ident]
        if result.shape==(1,1):scalars.append(float(result.iloc[0,0]))
    actual=expected if expected in scalars else (scalars[-1] if scalars else None)
    cards=[runtime.artifacts[c] for c in state.get('artifact_ids',[])]
    valid=True
    for card in cards:
        image=Image.open(BytesIO(card.image));image.load()
        valid=valid and image.format=='PNG' and min(image.size)>100 and max(ImageStat.Stat(image.convert('RGB')).stddev)>5
        if chart_required:
            spec=card.render_spec
            wanted=np.histogram([2.,4.,10.,20.],bins=spec.get('bin_edges',[]))[0]
            valid=valid and card.kind=='histogram' and np.array_equal(wanted,spec.get('bin_counts'))
    return actual,scalars,bool(valid and (cards or not chart_required))


def evaluate_case(spec,delegate,policy=None):
    policy=policy or RuntimePolicy()
    remote=[]
    def remote_factory(_):
        def forbidden(envelope):
            remote.append(True)
            raise AssertionError('Unapproved remote execution')
        return forbidden
    with tempfile.TemporaryDirectory(prefix='teleai-autonomy-') as root:
        model=FaultThenLiveModel(delegate=delegate,first_call=spec.get('first_call'),inject_count=spec.get('inject_count',1))
        def make_runtime():
            runtime=GraphAnalysisRuntime(root,'evaluation',spec['id'],model,policy=policy)
            # Faults belong to execution planning, never the JSON intent role.
            # Production provider roles still interpret the original prompt.
            from core.analysis_agent.goal_interpreter import GoalInterpreter
            runtime.recovery.goal_interpreter=GoalInterpreter(delegate,runtime.context,runtime.diagnostics,runtime.max_context_chars)
            runtime.recovery.goal_interpreter.model_recovery=runtime.model_recovery
            runtime.diagnostic_identity['goal_model']=getattr(runtime.recovery.goal_interpreter.model,'model',type(delegate).__name__)
            return runtime
        r=make_runtime()
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
                    if spec.get('timeout'):
                        stack.enter_context(patch('core.analysis_runtime_tools.build_aggregate_dataset',
                                                  side_effect=TimeoutError('synthetic local worker timeout')))
                    outcome=r.submit(turn['prompt'])
                state=r.inspect()['recovery']
                actual,scalar_outputs,chart_valid=output_proof(r,state,turn['expected'],bool(turn.get('chart')))
                charts=state.get('artifact_ids',[])
                requests=r.inspect()['requests']
                valid=(outcome['status']=='answered' and actual==turn['expected'] and
                       digest==stored_dataset_digest(r.datasets,raw.id) and not requests and not remote and
                       (not turn.get('chart') or (charts and chart_valid)))
                messages=r.events()
                called=[call['name'] for m in messages if isinstance(m,AIMessage) for call in m.tool_calls]
                search_used='search_analysis_tools' in called
                log=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                repair_events=[e for e in log if e['event'].startswith(('tool_repair_','tool_progress_'))]
                if turn.get('search_required'):valid=valid and search_used
                turns.append({'prompt':turn['prompt'],'status':'PASS' if valid else 'FAIL',
                    'agent_status':outcome['status'],'final_output':outcome.get('text',''),
                    'expected':turn['expected'],'actual':actual,'model_calls':state['model_calls'],
                    'scalar_outputs':scalar_outputs,'chart_proof_valid':chart_valid,
                    'search_used':search_used,'tools':called,'chart_count':len(charts),
                    'raw_unchanged':digest==stored_dataset_digest(r.datasets,raw.id),
                    'approval_requests':len(requests),'remote_executions':len(remote),
                    'stop_reason':state.get('stop_reason'),'elapsed_seconds':round(time.monotonic()-start,3)})
                turns[-1]['repair_events']=repair_events
                turns[-1]['goal']=state.get('goal')
                turns[-1]['intent_events']=[e for e in log if e['event'].startswith('goal_')]
                turns[-1]['injected_calls']=model.tracker.get('injected',0)
                turns[-1]['phase_order']=[e['event'] for e in log if e['event'] in
                    {'goal_interpretation_completed','tool_started'}]
                turns[-1]['executed_tools']=[e['tool'] for e in log if e['event']=='tool_started']
                turns[-1]['proposed_calls']=[call for m in messages if isinstance(m,AIMessage) for call in m.tool_calls]
                turns[-1]['failure_observations']=[{'tool':m.name,'result':json.loads(m.content)}
                    for m in messages if isinstance(m,ToolMessage) and m.content.startswith('{')
                    and json.loads(m.content).get('status') in {'error','needs_context','unavailable','needs_data'}]
                turns[-1]['scope_and_preflight_events']=[e for e in log if e['event'].startswith(('request_scope_','tool_proposal_','proposal_preflight_'))]
                turns[-1]['model_recovery']=r.inspect()['model_recovery']
                turns[-1]['model_failure_events']=[e for e in log if e['event']=='model_inference_failed']
                turns[-1]['error_type']=outcome.get('error_type')
                turns[-1]['error_category']=outcome.get('error_category')
                if spec.get('followup') and len(turns)==1:
                    # Restart actual runtime; all scope must come from checkpoint.
                    r.close()
                    r=make_runtime()
                    r.context.reference_context[:]=[fixture_reference_context(raw.source,frame)]
            return {'id':spec['id'],'status':'PASS' if all(x['status']=='PASS' for x in turns) else 'FAIL',
                    'intent_mode':'llm','fault_phase':'execution_planner_only','injected_first_call':bool(spec.get('first_call')),'live_calls':model.tracker.get('live_calls',0),'turns':turns}
        finally:r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=['databricks','ollama'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--id',action='append')
    a=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    policy=RuntimePolicy.from_env()
    model=build_analysis_chat_model(policy,provider=a.provider)
    results=[]
    for spec in CASES:
        if a.id and spec['id'] not in a.id:continue
        try:record=evaluate_case(spec,model,policy)
        except Exception as exc:record={'id':spec['id'],'status':'FAIL','error_type':type(exc).__name__}
        results.append(record)
        report={'generated_at':datetime.now(timezone.utc).isoformat(),'provider':a.provider,'policy':policy.public(),
                'mode':'real-model synthetic journeys; deterministic rescue disabled',
                'limitations':['Some cases inject a first faulty call; subsequent decisions use the live model.',
                               'No real remote SQL executor is connected; this does not validate loading.'],
                'results':results}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'id':record['id'],'status':record['status'],'live_calls':record.get('live_calls')},ensure_ascii=False),flush=True)
    return 0 if results and all(r['status']=='PASS' for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
