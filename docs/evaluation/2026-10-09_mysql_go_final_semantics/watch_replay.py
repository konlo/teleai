import importlib.util,json,time
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('capture',HERE/'capture.py');cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)
spec=importlib.util.spec_from_file_location('inspect_case',HERE/'inspect_case.py');check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
plan=json.loads((HERE/'plan.json').read_text());seen={int(f.name[:2]) for f in HERE.glob('[0-9][0-9]_check.json')}
while len(seen)<len(plan):
 d=cap.capture()
 logs=[json.loads(l) for l in (cap.SCOPE/'runtime.jsonl').read_text().splitlines()]
 with cap.connect('graph.sqlite') as db:
  checkpoints=list(SqliteSaver(db).list({'configurable':{'thread_id':'conversation'}}))
 for item in checkpoints:
  c=item.checkpoint['channel_values'].get('recovery',{})
  if not c.get('request_id'):continue
  turn=next((t for t in plan if t['prompt'].strip()==str(c.get('request_text')).strip()),None)
  if not turn or turn['id'] in seen:continue
  rid=next((e['run_id'] for e in logs if e.get('request_id')==c.get('request_id')),None)
  run=[e for e in logs if rid and e.get('run_id')==rid]
  final=next((e for e in run if e['event']=='run_completed'),None)
  if not final:continue
  n=turn['id'];snapshot={**d,'current':c,'run_id':rid,'run':run,'requested_prompt':turn['prompt'],'current_matches_run':True,'request_checkpoint_verified':True}
  (HERE/f'{n:02}.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,default=str))
  try:out=check.inspect(n)
  except Exception as e:out={'id':n,'verdict':'CHECKER_ERROR','error_type':type(e).__name__}
  if final.get('error_id'):
   err=next((e for e in reversed(run) if e['event']=='error'),{})
   out={'id':n,'verdict':'FAIL','reason':err.get('error_type'),'run_id':rid,'error_id':err.get('error_id')}
  (HERE/f'{n:02}_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));seen.add(n)
  (HERE/'progress.json').write_text(json.dumps({'completed':sorted(seen),'last':{k:v for k,v in out.items() if k not in ['chart_metadata','goal','prompt']}}))
 time.sleep(1)
