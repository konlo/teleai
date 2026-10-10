"""Summarize saved browser evidence without executing the product or SQL."""
from datetime import datetime, timezone
import hashlib,json,statistics
from pathlib import Path
from inspect_case import inspect
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
checks=[inspect(n) for n in range(1,11)]
docs=[json.loads((HERE/f'{n:02}.json').read_text()) for n in range(1,11)]
plan=json.loads((HERE/'plan.json').read_text()); oracle=json.loads((HERE/'oracles.json').read_text())
messages=json.loads((HERE/'browser_messages.json').read_text())
users=[m for m in messages if m['role']=='Chat message from user']
answers=[]
for m in messages:
 if m['role']=='Chat message from user':answers.append({'text':'','images':0,'grids':0,'table_rows':[]})
 else:
  assert answers
  answers[-1]['text']+=m['text']+'\n'
  answers[-1]['images']+=m['images'];answers[-1]['grids']+=m['grids'];answers[-1]['table_rows']+=m['table_rows']
assert len(users)==len(answers)==10
assert [u['text'] for u in users]==[p['prompt'] for p in plan]
for n in (3,9):
 name='stormtrooper' if n==3 else 'alibaba_ssd'
 expected=[[c['name'],c['dtype']] for c in oracle['schemas'][name]]
 assert answers[n-1]['table_rows']==expected, (n,'rendered schema mismatch')
for n in (2,10):
 assert answers[n-1]['grids']==1 and 'teleai_default.stormtrooper' in answers[n-1]['text']
for n in (7,8):assert answers[n-1]['images']==1
for n in (1,2,3,4,5,6,9,10):assert answers[n-1]['images']==0
for n in (4,5,6,7,8):
 expected=[{'column':'age','op':'ge','value':30},{'column':'age','op':'le','value':40},
 {'column':'education','op':'eq','value':'primary'} if n<6 else {'column':'education','op':'in','value':['primary','secondary']}]
 scope=docs[n-1]['current']['scope']
 assert scope['conditions']==expected and not scope['any_conditions'] and not scope['unresolved'],n
prior={};changes=[];missing=[]
for n,d in enumerate(docs,1):
 assets=d['assets']
 for ident,old in prior.items():
  if ident not in assets:missing.append({'case':n,'id':ident})
  elif any(assets[ident][k]!=old[k] for k in ('kind','metadata_sha256','payload_sha256')):
   changes.append({'case':n,'id':ident})
 prior.update(assets)
assert not changes and not missing
build=json.loads((HERE.parent/'build_release_candidate.json').read_text())
mutations=[name for name,sha in build['source_hashes'].items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
assert not mutations
elapsed=[next(e['elapsed_seconds'] for e in d['run'] if e['event']=='run_completed') for d in docs]
result={'captured_at':datetime.now(timezone.utc).isoformat(),'scope':'local single-user mysql/ollama/gemma4:e4b',
 'conversation':json.loads((HERE/'session.json').read_text()),'cases':checks,
 'pass':sum(c['verdict']=='PASS' for c in checks),'fail':sum(c['verdict']!='PASS' for c in checks),
 'first_submissions':10,'resubmissions':0,'rendered_user_messages':len(users),'rendered_answer_groups':len(answers),'rendered_assistant_blocks':sum(m['role']=='Chat message from assistant' for m in messages),
 'database_oracles_and_scope_checks':'PASS','rendered_full_schema_pairs':[13,107],
 'prior_asset_mutations':changes,'prior_asset_losses':missing,'retained_assets':len(prior),
 'source_mutations_during_run':mutations,'elapsed_seconds':elapsed,'median_seconds':statistics.median(elapsed),
 'maximum_seconds':max(elapsed),'model_calls':[d['current'].get('model_calls') for d in docs],
 'rebin_remote_queries':sum(e['event']=='remote_query_finished' for e in docs[7]['run']),
 'official_deepeval_score':None,'official_spider2_score':None}
(HERE/'frozen_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','conversation')},ensure_ascii=False))
