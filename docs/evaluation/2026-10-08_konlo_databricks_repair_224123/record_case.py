"""Read-only receipt capture and independent checks after actual UI submission."""
from pathlib import Path
import importlib.util,json,sys,time

HERE=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

n=int(sys.argv[1]);prompt=json.loads((HERE/'plan.json').read_text())[n-1]['prompt']
capture=module('capture');deadline=time.monotonic()+45
while True:
    d=capture.capture();current=d['current']
    match=(current.get('request_id') in {r.get('request_id') for r in d['run'] if r.get('request_id')})
    verified=(match and current.get('request_text')==prompt and any(r['event']=='run_completed' for r in d['run']))
    verified = verified and current.get('status') != 'working'
    terminal_error=any(e['event']=='run_completed' and e.get('error_id') for e in d['run']) and any(m.get('data',{}).get('content')==prompt for m in d['recent_transcript'])
    if verified or terminal_error or time.monotonic()>=deadline:break
    time.sleep(1)
d.update(requested_prompt=prompt,current_matches_run=match,request_checkpoint_verified=verified)
(HERE/f'{n:02}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n')
if verified:module('inspect_case').inspect(n)
else:
    final=next((e for e in reversed(d['run']) if e['event']=='run_completed'),{})
    error=next((e for e in reversed(d['run']) if e['event']=='error'),{})
    result={'id':n,'verdict':'FAIL' if terminal_error else 'PENDING','status':final.get('status'),'reason':error.get('error_type'),'run_id':d['run_id'],'error_id':error.get('error_id'),'current_request':current.get('request_text'),'requested_prompt':prompt,'note':'failed before goal checkpoint; prior checkpoint retained' if terminal_error else ''}
    (HERE/f'{n:02}_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
