"""Validate observed chat DOM separately from DB/checkpoint oracle checks."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);plan=json.loads((p/'plan.json').read_text());messages=json.loads((p/'browser_messages.json').read_text())
users=[i for i,m in enumerate(messages) if m['role']=='Chat message from user']
assert len(users)==len(plan),'missing/duplicate actual prompt submissions'
results=[]
for n,(i,turn) in enumerate(zip(users,plan),1):
 end=users[n] if n<len(users) else len(messages)
 replies=messages[i+1:end];c=json.loads((p/f'{n:02}.json').read_text())['current']
 r={'id':n,'verdict':'PASS','visible_assistant_messages':len(replies)}
 try:
  assert ' '.join(messages[i]['text'].split())==' '.join(turn['prompt'].split()),'actual prompt differs from frozen scenario'
  assert replies and all(m['role']=='Chat message from assistant' for m in replies),'missing actual answer'
  text='\n'.join(m['text'] for m in replies);images=sum(m['images'] for m in replies);tables=sum(m['tables'] for m in replies);grids=sum(m['grids'] for m in replies)
  assert images==len(c.get('artifact_ids',[])),'chart not attached to current actual reply or stale chart leaked'
  proof=c.get('metadata_evidence')
  if proof:
   assert proof['table'] in text,'actual reply has wrong/missing schema subject'
   for col in proof['columns']:assert col in text,'missing displayed column '+col
   if proof['kind']=='dtypes':
    assert tables>=1,'types proof exists but table not shown'
    for col in proof['schema']:assert col['name']+'\t'+col['dtype'] in text,'type not displayed '+col['name']
  if c.get('table_preview_evidence'):assert grids+tables>=1,'records claimed without visible native data table'
  if c.get('value_list_evidence'):
   for v in c['value_list_evidence'].get('values',[]):assert str(v) in text,'value not shown'
  if any(t['capability']=='table_list' for t in (c.get('goal') or {}).get('tasks',[])):assert tables>=1,'inventory claimed without actual table'
  if c.get('status')=='needs_context':assert not images and not tables and not grids,'clarification leaked old output'
  r.update(images=images,tables=tables,grids=grids)
 except AssertionError as e:r.update(verdict='FAIL',reason=str(e))
 results.append(r)
summary={'cases':results,'passed':sum(r['verdict']=='PASS' for r in results),'total':len(results),
 'method':'CUA observed rendered chat text and actual img/table/native dataframe DOM, joined by frozen original prompt order; DB values and PNG contents graded independently'}
(p/'browser_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
