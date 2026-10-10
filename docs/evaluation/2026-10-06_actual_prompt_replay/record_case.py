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
    if verified or time.monotonic()>=deadline:break
    time.sleep(1)
d.update(requested_prompt=prompt,current_matches_run=match,request_checkpoint_verified=verified)
(HERE/f'{n:02}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str)+'\n')
if verified:module('inspect_case').inspect(n)
else:print(json.dumps({'id':n,'verdict':'PENDING','current_request':current.get('request_text')}))
