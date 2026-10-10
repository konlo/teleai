import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('capture',HERE/'capture.py');cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)
spec=importlib.util.spec_from_file_location('inspect_case',HERE/'inspect_case.py');check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
plan=json.loads((HERE/'plan.json').read_text());seen={int(f.name[:2]) for f in HERE.glob('[0-9][0-9]_check.json')}
while len(seen)<len(plan):
 d=cap.capture();c=d['current'];run=d['run'];final=next((e for e in reversed(run) if e['event']=='run_completed'),None)
 if final:
  req=c.get('request_text');turn=next((t for t in plan if t['prompt'].strip()==str(req).strip()),None)
  if turn and turn['id'] not in seen and c.get('request_id') in {e.get('request_id') for e in run if e.get('request_id')}:
   n=turn['id'];d.update(requested_prompt=turn['prompt'],current_matches_run=True,request_checkpoint_verified=True)
   (HERE/f'{n:02}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str))
   try:out=check.inspect(n)
   except Exception as e:out={'id':n,'verdict':'CHECKER_ERROR','error_type':type(e).__name__}
   if final.get('error_id'):
    err=next((e for e in reversed(run) if e['event']=='error'),{})
    out={'id':n,'verdict':'FAIL','reason':err.get('error_type'),'run_id':d['run_id'],'error_id':err.get('error_id')}
   (HERE/f'{n:02}_check.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
   seen.add(n);(HERE/'progress.json').write_text(json.dumps({'completed':sorted(seen),'last':{k:v for k,v in out.items() if k not in ['chart_metadata','goal','prompt']}}))
 time.sleep(1)
