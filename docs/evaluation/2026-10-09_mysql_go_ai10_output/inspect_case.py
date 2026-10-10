"""Independent checks for saved AI ten-turn journey, never model inputs."""
from io import BytesIO
from pathlib import Path
import hashlib,json,sqlite3
import numpy as np
import pandas as pd
from PIL import Image,ImageStat
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
ORACLE=json.loads((HERE/'oracles.json').read_text())
SESSION=json.loads((HERE/'session.json').read_text());SCOPE=ROOT/'.telly_runtime/v1'/SESSION['scope']
PLAN=json.loads((HERE/'plan.json').read_text())

def inspect(n):
 d=json.loads((HERE/f'{n:02}.json').read_text());c=d['current']
 out={'id':n,'prompt':PLAN[n-1]['prompt'],'status':c.get('status'),'run_id':d['run_id']}
 db=sqlite3.connect(f'file:{SCOPE/"assets.sqlite"}?mode=ro',uri=True)
 def frame(ident):
  payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
  return pd.read_parquet(BytesIO(payload) if payload is not None else SCOPE/f'{ident}.parquet')
 try:
  assert d['request_checkpoint_verified'] and c['status']=='complete'
  cards=[d['assets'][i]['metadata'] for i in c['artifact_ids']]
  for ident in c['artifact_ids']:
   content=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
   image=Image.open(BytesIO(content));image.load()
   assert image.format=='PNG' and min(image.size)>100 and max(ImageStat.Stat(image.convert('RGB')).stddev)>5
  if n==1:
   assert any(t['capability']=='table_list' for t in c['goal']['tasks'])
   evidence=next(iter(c['remote_query_evidence'].values()));f=frame(evidence['dataset_id'])
   assert evidence['source']=='information_schema.tables'
   assert set(f.TABLE_NAME)==set(ORACLE['tables']) and len(f)==len(ORACLE['tables'])
  elif n in (2,10):
   table='bank_loan' if n==2 else 'stormtrooper';proof=c['metadata_evidence'];schema=ORACLE['schemas'][table]
   assert proof['table']==ORACLE['namespace']+'.'+table and proof['columns']==[v['name'] for v in schema]
   assert not cards
   if n==10:
    assert c['metadata_kind']=='dtypes' and proof['kind']=='dtypes'
    browser=json.loads((HERE/'browser_messages.json').read_text())
    reply=browser[-1]
    assert reply['role']=='Chat message from assistant' and reply['tables']==1 and reply['images']==0
    for col in schema:assert col['name']+'\t'+col['dtype'] in reply['text']
    assert proof['type_authority']=='current_database_metadata'
    assert proof['schema']==[{'name':v['name'],'dtype':v['dtype']} for v in schema]
  elif n==3:
   proof=c['value_list_evidence'];f=frame(proof['dataset_id'])
   assert set(f.education)==set(ORACLE['education_counts']) and not cards
  elif n==4:
   assert len(cards)==1;card=cards[0];s=card['render_spec']
   assert card['kind']=='bar' and card['columns']==['education']
   assert dict(zip(s['labels'],s['counts']))==ORACLE['education_counts']
   assert s['total_count']==ORACLE['bank_rows']
  elif n in (5,6,7):
   assert len(cards)==1;card=cards[0];s=card['render_spec']
   assert card['kind']=='histogram' and card['columns'][0]=='age'
   assert s['total_count']==ORACLE['primary_30_40'] if n==5 else s['total_count']==ORACLE['primary_secondary_30_40']
   groups=[g for g in ORACLE['bank_age_education_groups'] if 30<=g['age']<=40 and g['education'] in (['primary'] if n==5 else ['primary','secondary'])]
   if s.get('category'):
    assert s['category']=='education'
    for label, counts in zip(s['legend_labels'],s['series_counts']):
     rows=[g for g in groups if g['education']==label]
     wanted=np.histogram([g['age'] for g in rows],bins=s['bin_edges'],weights=[g['count'] for g in rows])[0]
     assert np.array_equal(wanted,counts)
   else:
    wanted=np.histogram([g['age'] for g in groups],bins=s['bin_edges'],weights=[g['count'] for g in groups])[0]
    assert np.array_equal(wanted,s['bin_counts'])
   if n==7:assert s['bins']==5
  elif n==8:
   assert len(cards)==1;card=cards[0];s=card['render_spec']
   assert card['kind']=='scatter' and s['x']=='age' and s['y']=='balance'
   assert s['drawable_rows']==ORACLE['bank_rows'] and not c['scope']['conditions']
   f=frame(card['dataset_id']);w=s['weight_column']
   data=f[['age','balance',w]].rename(columns={w:'__frequency'}).astype('int64').sort_values(['age','balance'],ignore_index=True)
   assert hashlib.sha256(data.to_json(orient='values').encode()).hexdigest()==ORACLE['bank_scatter_oracle']['sorted_coordinates_and_frequencies_sha256']
  elif n==9:
   proof=c['table_preview_evidence'];f=frame(proof['dataset_id'])
   assert proof['source']==ORACLE['namespace']+'.stormtrooper'
   assert f.shape==(10,len(ORACLE['schemas']['stormtrooper'])) and not cards
  out['verdict']='PASS'
 except (AssertionError,KeyError,TypeError,ValueError) as error:
  out.update(verdict='FAIL',reason=str(error) or 'independent output contract mismatch',error_type=type(error).__name__)
 finally:db.close()
 return out
