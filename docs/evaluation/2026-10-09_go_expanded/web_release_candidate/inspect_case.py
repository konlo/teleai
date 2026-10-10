"""Read-only independent DB, data, image and output checks for the new journey."""
from io import BytesIO
from pathlib import Path
import json,sqlite3
import numpy as np
import pandas as pd
from PIL import Image,ImageStat
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
ORACLE=json.loads((HERE/'oracles.json').read_text());PLAN=json.loads((HERE/'plan.json').read_text())
SCOPE=ROOT/'.telly_runtime/v1'/json.loads((HERE/'session.json').read_text())['scope']
def inspect(n):
 d=json.loads((HERE/f'{n:02}.json').read_text());c=d['current']
 out={'id':n,'prompt':PLAN[n-1]['prompt'],'status':c.get('status'),'run_id':d['run_id']}
 db=sqlite3.connect(f'file:{SCOPE/"assets.sqlite"}?mode=ro',uri=True)
 def frame(ident):
  payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
  return pd.read_parquet(BytesIO(payload) if payload is not None else SCOPE/f'{ident}.parquet')
 try:
  assert d['request_checkpoint_verified'] and c['status']=='complete'
  cards=[d['assets'][i]['metadata'] for i in c.get('artifact_ids',[])]
  for ident in c.get('artifact_ids',[]):
   payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
   im=Image.open(BytesIO(payload));im.load()
   assert im.format=='PNG' and min(im.size)>100 and max(ImageStat.Stat(im.convert('RGB')).stddev)>5
  if n==1:
   receipt=next(iter(c['remote_query_evidence'].values()))
   assert receipt['source']=='information_schema.tables'
   data=frame(receipt['dataset_id'])
   assert set(data.TABLE_NAME)==set(ORACLE['tables']) and len(data)==len(ORACLE['tables'])
  elif n in (2,10):
   proof=c['table_preview_evidence'];data=frame(proof['dataset_id'])
   assert proof['source']=='teleai_default.stormtrooper' and data.shape==(10,13) and not cards
   assert list(data.columns)==[v['name'] for v in ORACLE['schemas']['stormtrooper']]
  elif n in (3,9):
   table='stormtrooper' if n==3 else 'alibaba_ssd';proof=c['metadata_evidence']
   wanted=ORACLE['schemas'][table]
   assert proof['table']=='teleai_default.'+table and proof['kind']=='dtypes'
   assert proof['schema']==[{'name':v['name'],'dtype':v['dtype']} for v in wanted]
   if proof.get('type_authority')!='current_database_metadata':
    # Fresh DB-observed metadata is valid when no forced refresh was requested.
    # Do not accept pandas dtypes, static fixtures, stale or partial snapshots.
    observations=[json.loads(t['data']['content']) for t in d['recent_transcript'] if t['data'].get('name')=='inspect_table_context']
    valid=[v for v in observations if v.get('authority')=='saved_snapshot' and v.get('table_context',{}).get('table')=='teleai_default.'+table]
    assert valid and not c['goal']['fresh_source_required']
    ctx=valid[0]['table_context']
    assert ctx['freshness']=='fresh' and ctx['training_status']=='observed_schema' and ctx['observed_at']
    assert [{'name':v['name'],'dtype':v['dtype']} for v in ctx['columns']]==proof['schema']
    out['metadata_basis']={'authority':'fresh_database_observed_snapshot','observed_at':ctx['observed_at']}
   assert not cards
  elif n in (4,5,6):
   op='SUM' if n==4 else 'AVG'
   assert c['operations']==[op] and c['required_sources']==['teleai_default.bank_loan'] and c['required_columns']==['balance'] and not cards
   target=ORACLE['primary' if n in (4,5) else 'primary_secondary'][0]['total' if n==4 else 'mean']
   actual=[float(frame(i).iloc[0,0]) for i in c['evidence_ids'] if frame(i).shape==(1,1)]
   assert any(abs(v-float(target))<0.00011 for v in actual)
   out['actual']=actual;out['expected']=target
  elif n in (7,8):
   assert len(cards)==1;card=cards[0];s=card['render_spec']
   assert card['kind']=='histogram' and card['columns'][0]=='age' and s['total_count']==208699
   groups=ORACLE['age_frequency'];wanted=np.histogram([g['age'] for g in groups],bins=s['bin_edges'],weights=[g['n'] for g in groups])[0]
   assert np.array_equal(wanted,s['bin_counts'])
   if n==8:
    assert s['bins']==5
    assert not any(e['event']=='remote_query_finished' for e in d['run'])
  out['verdict']='PASS'
 except (AssertionError,KeyError,TypeError,ValueError) as e:
  out.update(verdict='FAIL',reason=str(e) or 'independent output mismatch',error_type=type(e).__name__)
 finally:db.close()
 return out
