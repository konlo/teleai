"""Read-only independent arithmetic, PNG, DOM, source and preservation checks."""
from io import BytesIO
import hashlib,json,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageStat
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
d=json.loads((H/'12.json').read_text());before=json.loads((H/'10.json').read_text())
o=json.loads((H/'oracles.json').read_text());c=d['current']
scope=ROOT/'.telly_runtime/v1'/json.loads((H/'session.json').read_text())['scope']
result={'prompt':d['requested_prompt'],'run_id':d['run_id'],'attempt':'explicit retest after initial case11 failure and product repair','build':'build_go_final.json'}
db=sqlite3.connect(f'file:{scope/"assets.sqlite"}?mode=ro',uri=True)
def frame(ident):
    payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
    return pd.read_parquet(BytesIO(payload) if payload is not None else scope/f'{ident}.parquet')
try:
    assert d['request_checkpoint_verified'] and c['status']=='complete'
    assert c['required_sources']==['teleai_default.bank_loan'] and c['required_columns']==['age']
    assert c['operations']==['AVG']
    assert {t['capability'] for t in c['goal']['tasks']}=={'calculation','chart'}
    assert c['scope']['conditions']==[{'column':'age','op':'ge','value':30},{'column':'age','op':'le','value':40},{'column':'education','op':'in','value':['primary','secondary']}]
    assert not c['scope']['any_conditions'] and not c['scope']['measure_conditions'] and not c['scope']['ratio']
    receipts=list(c['remote_query_evidence'].values());assert len(receipts)==1
    receipt=receipts[0];assert receipt['source']=='teleai_default.bank_loan' and receipt['coverage']=='complete'
    scalar=frame(receipt['dataset_id']);assert scalar.shape==(1,1)
    actual=float(scalar.iloc[0,0]);count=sum(v['n'] for v in o['age_frequency'])
    expected=sum(v['age']*v['n'] for v in o['age_frequency'])/count
    assert abs(actual-expected)<0.00011
    assert len(c['artifact_ids'])==1;ident=c['artifact_ids'][0];card=d['assets'][ident]['metadata']
    s=card['render_spec'];assert card['kind']=='histogram' and card['columns']==['age']
    assert s['total_count']==count==208699
    bins=np.histogram([v['age'] for v in o['age_frequency']],bins=s['bin_edges'],weights=[v['n'] for v in o['age_frequency']])[0]
    assert np.array_equal(bins,s['bin_counts'])
    parent=frame(card['dataset_id'])
    assert sorted(zip(parent.age.astype(int),parent.__frequency.astype(int)))==sorted((v['age'],v['n']) for v in o['age_frequency'])
    payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
    im=Image.open(BytesIO(payload));im.load()
    assert im.format=='PNG' and min(im.size)>100 and max(ImageStat.Stat(im.convert('RGB')).stddev)>5
    changed=[i for i,v in before['assets'].items() if d['assets'].get(i)!=v]
    assert not changed and len(before['assets'])==8
    messages=json.loads((H/'browser_messages_latest.json').read_text())
    latest=max(i for i,v in enumerate(messages) if v['role']=='Chat message from user')
    answer=messages[latest+1:];text='\n'.join(v['text'] for v in answer)
    assert sum(v['images'] for v in answer)==1 and '평균: 34.8200' in text and '208,699' in text and 'bank_loan' in text
    build=json.loads((H.parent/'build_go_final.json').read_text())
    source_changes=[f for f,h in build['source_hashes'].items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
    assert not source_changes
    result.update(verdict='PASS',expected_mean=expected,actual_mean=actual,frequency_count=count,
        bins=s['bins'],bin_counts=s['bin_counts'],png_dimensions=im.size,ui_images=1,
        previous_assets_preserved=8,changed_previous_assets=changed,source_hash_changes=source_changes,
        scalar_rows_loaded=1,histogram_frequency_rows_reused=len(parent),chart_asset_reused=ident in before['assets'],
        elapsed_seconds=next(e['elapsed_seconds'] for e in d['run'] if e['event']=='run_completed'),
        model_calls=c['model_calls'])
except Exception as e:
    result.update(verdict='FAIL',reason=str(e),error_type=type(e).__name__)
finally:db.close()
(H/'12_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
raise SystemExit(0 if result['verdict']=='PASS' else 1)
