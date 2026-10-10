"""Read-only independent checks of the final ten browser submissions."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import sqlite3

import pandas as pd
from PIL import Image, ImageStat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SESSION=json.loads((HERE/'session.json').read_text())
SCOPE=ROOT/'.telly_runtime/v1/mysql_eval/44bc5ed80f4a4583ada9f905'/SESSION['scope']
ORACLE=json.loads((HERE/'oracles.json').read_text())
PLAN=json.loads((HERE/'plan.json').read_text())
UI=json.loads((HERE/'acceptance_ui.json').read_text())
CAPTURES=[json.loads((HERE/f'acceptance_{i:02}.json').read_text()) for i in range(1,11)]
DB=sqlite3.connect(f'file:{SCOPE/"assets.sqlite"}?mode=ro',uri=True)


def frame(ident):
    payload=DB.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
    return pd.read_parquet(BytesIO(payload) if payload is not None else SCOPE/f'{ident}.parquet')


def card(capture):
    ident=capture['current']['artifact_ids'][0]
    metadata=capture['assets'][ident]['metadata']
    payload=DB.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
    assert payload.startswith(b'\x89PNG'), 'chart is not an actual PNG'
    image=Image.open(BytesIO(payload)).convert('RGB')
    assert min(image.size)>100 and max(ImageStat.Stat(image).stddev)>10, 'blank/invalid chart'
    return metadata


def check(i,d):
    current=d['current'];goal=current['goal']
    assert d['request_checkpoint_verified'] and current['status']=='complete'
    assert current['request_text']==PLAN[i-1]['prompt']
    assert UI[i-1]['id']==i and UI[i-1]['prompt']==PLAN[i-1]['prompt']
    assert PLAN[i-1]['prompt'] in UI[i-1]['snapshot'], 'request not present in actual browser'
    if i==1:
        ident=next(k for k,a in d['assets'].items() if a['metadata'].get('source')=='information_schema.tables')
        assert frame(ident).TABLE_NAME.tolist()==ORACLE['tables']
    elif i==2:
        proof=current['metadata_evidence']
        assert proof['table']=='teleai_default.bank_loan'
        assert proof['columns']==[c['name'] for c in ORACLE['schemas']['bank_loan']]
    elif i==3:
        proof=current['value_list_evidence']
        assert proof['source']=='teleai_default.bank_loan' and proof['column']=='education'
        assert set(proof['values'])==set(ORACLE['education_counts']) and not proof['has_more']
    elif i==4:
        chart=card(d);spec=chart['render_spec']
        assert chart['kind']=='bar' and chart['columns']==['education']
        assert dict(zip(spec['labels'],spec['counts']))==ORACLE['education_counts']
        assert spec['total_count']==ORACLE['bank_rows']
    elif i in (5,6,7):
        chart=card(d);spec=chart['render_spec']
        assert chart['kind']=='histogram' and chart['columns']==['age']
        expected=ORACLE['primary_30_40'] if i==5 else ORACLE['primary_secondary_30_40']
        assert spec['total_count']==expected and sum(spec['bin_counts'])==expected
        predicates=current['scope']['conditions']
        by_column={c['column'] for c in predicates}
        assert by_column=={'age','education'} and not current['scope']['any_conditions']
        assert {'column':'age','op':'ge','value':30} in predicates
        assert {'column':'age','op':'le','value':40} in predicates
        assert (any(c['column']=='education' and ((c['op']=='eq' and c['value']=='primary')
                     or (c['op']=='in' and c['value']==['primary'])) for c in predicates) if i==5
                else any(c['column']=='education' and c['op']=='in'
                         and set(c['value'])=={'primary','secondary'} for c in predicates))
        if i==7:
            assert spec['bins']==5 and spec['bin_counts']==ORACLE['histogram_5']['counts']
            assert spec['bin_edges']==ORACLE['histogram_5']['edges']
            assert not any(r['event']=='remote_query_finished' for r in d['run'])
    elif i==8:
        chart=card(d);spec=chart['render_spec'];points=frame(chart['dataset_id'])
        assert chart['kind']=='scatter' and spec['x']=='age' and spec['y']=='balance'
        assert not current['scope']['conditions'] and not current['scope']['any_conditions']
        assert spec['drawable_rows']==ORACLE['bank_rows']
        assert len(points)==spec['coordinate_count']==ORACLE['scatter']['coordinate_count']
        assert not points[['age','balance']].duplicated().any()
        assert int(points[spec['weight_column']].sum())==ORACLE['bank_rows']
        normalized=points[['age','balance',spec['weight_column']]].rename(
            columns={spec['weight_column']:'__frequency'}).astype('int64').sort_values(
                ['age','balance'],ignore_index=True)
        assert hashlib.sha256(normalized.to_json(orient='values').encode()).hexdigest()==(
            ORACLE['scatter']['sorted_coordinates_and_frequencies_sha256'])
    elif i==9:
        proof=current['table_preview_evidence'];rows=frame(proof['dataset_id'])
        assert proof['source']=='teleai_default.stormtrooper' and proof['rows']==10
        assert rows.shape==(10,13) and rows.columns.tolist()==[c['name'] for c in ORACLE['schemas']['stormtrooper']]
        assert not current['scope']['conditions'] and not current['scope']['any_conditions']
    elif i==10:
        proof=current['metadata_evidence']
        assert proof['table']=='teleai_default.stormtrooper' and proof['schema']==ORACLE['schemas']['stormtrooper']
        assert proof['type_authority']=='current_database_metadata'
        assert not current['artifact_ids']
        assert d['assets']==CAPTURES[8]['assets'], 'metadata request changed assets'


results=[]
for i,d in enumerate(CAPTURES,1):
    error=None
    try:check(i,d)
    except (AssertionError,KeyError,ValueError) as exc:error=str(exc) or 'assertion mismatch'
    run=d['run'];completed=next((r for r in run if r['event']=='run_completed'),{})
    results.append({'id':i,'prompt':PLAN[i-1]['prompt'],'passed':error is None,'error':error,
        'model_calls':d['current'].get('model_calls'),'elapsed_seconds':completed.get('elapsed_seconds'),
        'remote_queries':sum(r['event']=='remote_query_finished' for r in run),
        'automatic_repairs':[r for r in run if r['event'] in
            {'goal_contract_error','goal_source_audit_conflict','goal_population_audited','goal_obligations_coalesced'}]})

changed=[]
for before,after in zip(CAPTURES,CAPTURES[1:]):
    for ident,asset in before['assets'].items():
        if asset!=after['assets'].get(ident):changed.append(ident)
    if before['selection']!=after['selection']:changed.append('selection')

report={'evaluation':'final build / actual browser / one fresh continuous conversation / no manual resubmissions',
    'session':SESSION,'baseline_journey_passed':3,'baseline_journey_total':10,
    'passed':sum(r['passed'] for r in results),'total':10,'results':results,
    'preservation':{'changed_or_missing_assets':sorted(set(changed)),
                    'final_asset_count':len(CAPTURES[-1]['assets'])},
    'tests':{'total':640,'passed':636,'skipped':4,'log':'acceptance_unit_tests.log'},
    'limitations':['MySQL local evaluation only; company Databricks not validated in this run',
                   'Same frozen ten prompts after repair; not unseen-language generalization',
                   'Local model latency remains material; no general production GO',
                   'Not an official DeepEval or Spider 2.0 score'],
    'intermediate_failures':['intermediate_inventory_failure.json','r2_03_complete.json',
                             'final_06.json','corrected_09.json','repaired_10.json','recovered_10.json']}
older=json.loads((HERE/'older_data_preservation.json').read_text())
images=json.loads((HERE/'acceptance_rendered_images.json').read_text())
report['preservation']['older_conversations']=older
report['browser_images']={'count':len(images),'all_loaded':all(
    i['complete'] and i['naturalWidth']>100 and i['naturalHeight']>100 for i in images),
    'evidence':'acceptance_rendered_images.json'}
report['manual_submissions']=len(UI)
report['acceptance_passed']=(report['passed']==10 and not changed and len(UI)==10
    and len(images)==5 and report['browser_images']['all_loaded']
    and all(not p['changed_or_missing'] and p['selection_unchanged'] for p in older))
report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/'core/analysis_agent'/name for name in
        ('goal_contract.py','goal_interpreter.py','source_references.py','population_audit.py',
         'goal_normalization.py','recovery.py','value_list.py')]}
report['latency_seconds']={'min':min(r['elapsed_seconds'] for r in results),
                         'max':max(r['elapsed_seconds'] for r in results),
                         'total':round(sum(r['elapsed_seconds'] for r in results),3)}
(HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'total':10,'acceptance_passed':report['acceptance_passed'],
    'changed':changed,'results':[
    {k:r[k] for k in ('id','passed','error','elapsed_seconds')} for r in results]},ensure_ascii=False))
DB.close()
