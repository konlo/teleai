"""Actual-model goals, independent synthetic values, no regex-derived expected answers.
This is a focused acceptance suite, not a general agent/DeepEval/Spider score.
"""
import json,tempfile,time
from pathlib import Path
from dotenv import load_dotenv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
import argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=pd.DataFrame({'reading':range(20),'cohort':['A','B']*10,'event_time':range(20,40)})
def schema(source):
 return [{'table':source,'columns':[{'name':c,'dtype':str(DATA[c].dtype)} for c in DATA]}]
parser=argparse.ArgumentParser(description='Actual model goal interpretation and independent local result oracles; no remote DB')
parser.add_argument('--output',type=Path,default=ROOT/'docs/evaluation/2026-10-05_llm_goal/live_model.json')
parser.add_argument('--id',action='append',help='Run named cases only; preserve their independent oracles')
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
load_dotenv('.env')
cases=[
 ('preview','observations에서 열 개 행을 보여줘','row_preview'),
 ('negated_chart','observations의 reading 히스토그램은 그리지 말고 컬럼 목록만 보여줘','metadata'),
 ('negated_avg','observations의 reading 평균은 계산하지 말고 합계만 알려줘','calculation'),
 ('filtered_sum','observations에서 reading이 3~8인 것만 합계를 알려줘','calculation')]
if args.id:
 unknown=set(args.id)-{c[0] for c in cases}
 if unknown:parser.error('Unknown case IDs: '+repr(sorted(unknown)))
 cases=[c for c in cases if c[0] in args.id]
results=[]
for name,prompt,cap in cases:
 with tempfile.TemporaryDirectory() as root:
  model=build_analysis_chat_model(RuntimePolicy(),provider='ollama')
  r=GraphAnalysisRuntime(root,'actual-goal',name,model,sql_dialect='mysql',reference_context_loader=lambda:schema('lab.observations'),source_namespace='lab')
  try:
   raw=r.datasets.register(DATA,source='lab.observations',coverage='complete',predicate_known=True);r.select_dataset(raw.id)
   start=time.monotonic();out=r.submit(prompt);state=r.inspect()['recovery']
   final=None
   if state.get('evidence_ids'):
    f=r.datasets.frames[state['evidence_ids'][-1]];final=f.to_dict(orient='records')
   cases_ok=bool(state.get('goal') and [x['capability'] for x in state['goal']['tasks']]==[cap])
   if name=='preview':cases_ok=cases_ok and (state.get('table_preview_evidence') or {}).get('rows')==10
   if name=='negated_chart':cases_ok=cases_ok and state.get('metadata_kind')=='columns' and not state.get('chart')
   if name=='negated_avg':cases_ok=cases_ok and state.get('operations')==['SUM'] and final and next(iter(final[0].values()))==190
   if name=='filtered_sum':cases_ok=cases_ok and final and next(iter(final[0].values()))==33
   result={'id':name,'prompt':prompt,'status':out['status'],'text':out['text'],'seconds':round(time.monotonic()-start,3),'goal':state.get('goal'),'model_calls':state.get('model_calls'),'goal_error':state.get('goal_contract_error'),'result':final,'pass':bool(cases_ok and out['status']=='answered'),
       'diagnostics':[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()],
       'stop_reason':state.get('stop_reason')}
   results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
   args.output.write_text(json.dumps({'model':model.model,'goal_model':getattr(r.recovery.goal_interpreter.model,'model',type(model).__name__),'mode':'actual Ollama, isolated synthetic raw data, no remote database','passed':sum(c['pass'] for c in results),'total':len(cases),'cases':results},ensure_ascii=False,indent=2)+'\n')
  except Exception as exc:
   print(json.dumps({'id':name,'error':type(exc).__name__,'detail':str(exc)[:100]},ensure_ascii=False),flush=True)
   results.append({'id':name,'error':type(exc).__name__,'pass':False})
  finally:
   args.output.write_text(json.dumps({'model':model.model,'goal_model':getattr(r.recovery.goal_interpreter.model,'model',type(model).__name__),'mode':'actual Ollama, isolated synthetic raw data, no remote database','passed':sum(c['pass'] for c in results),'total':len(cases),'cases':results},ensure_ascii=False,indent=2)+'\n')
   r.close()

sys.exit(0 if len(results)==len(cases) and all(c['pass'] for c in results) else 1)
