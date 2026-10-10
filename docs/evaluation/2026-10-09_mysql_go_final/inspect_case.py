"""Independent read-only checks. Does not invoke the agent or submit UI work."""
from io import BytesIO
from pathlib import Path
import hashlib,json,sqlite3,sys
import numpy as np
import pandas as pd
from PIL import Image,ImageStat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SESSION=json.loads((HERE/'session.json').read_text())
SCOPE=ROOT/'.telly_runtime/v1'/SESSION['scope']
PLAN=json.loads((HERE/'plan.json').read_text())
ORACLE=json.loads((HERE/'oracles.json').read_text())


def inspect(n):
    path=HERE/f'{n:02}.json'
    if not path.exists():return {'id':n,'verdict':'PENDING'}
    d=json.loads(path.read_text());c=d['current']
    output={'id':n,'prompt':PLAN[n-1]['prompt'],'status':c.get('status'),
            'goal':c.get('goal'),'reason':c.get('stop_reason'),'run_id':d['run_id'],
            'model_calls':c.get('model_calls'),'chart_metadata':[]}
    if not d.get('request_checkpoint_verified'):
        return {**output,'verdict':'PENDING'}
    db=sqlite3.connect(f'file:{SCOPE/"assets.sqlite"}?mode=ro',uri=True)
    def frame(ident):
        payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
        return pd.read_parquet(BytesIO(payload) if payload is not None else SCOPE/f'{ident}.parquet')
    def metadata(table,types=False):
        p=c['metadata_evidence'];expected=ORACLE['schemas'][table]
        assert p['table']==ORACLE['namespace']+'.'+table, 'wrong schema subject'
        assert p['columns']==[x['name'] for x in expected], 'incomplete/wrong source columns'
        if types:
            assert p['schema']==[{'name':x['name'],'dtype':x['dtype']} for x in expected]
            assert p['type_authority']=='current_database_metadata'
        output['verified_metadata_columns']=len(expected)
        assert not c['artifact_ids'], 'unrequested chart'
    def preview(table):
        p=c['table_preview_evidence'];f=frame(p['dataset_id'])
        assert p['source']==ORACLE['namespace']+'.'+table
        assert f.shape==(10,len(ORACLE['schemas'][table])) and p['rows']==10
        assert f.columns.tolist()==[x['name'] for x in ORACLE['schemas'][table]]
        output['verified_preview_shape']=list(f.shape)
        if n==16:
            assert f.age.between(30,40).all() and f.education.isin(['primary','secondary']).all(), 'preview lost preceding population conditions'
    try:
        assert c['request_text'].strip()==PLAN[n-1]['prompt'].strip()
        if n in (29,30,31):
            assert c.get('required_sources')==['teleai_default.bank_loan'], 'requested qualified source was silently replaced'
        if n in (27,28):
            assert c['status']=='needs_context' and c.get('goal_question'), 'missing source needs actionable clarification'
            assert 'tianchi_ssd_shared300k_raw' in c['goal_question'], 'wrong missing subject'
            assert not c['artifact_ids'] and not c['evidence_ids'], 'missing source silently produced a different result'
            output.update(verdict='PASS',verified_missing_source=True)
            (HERE/f'{n:02}_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
            return output
        assert c['status']=='complete', str(c.get('stop_reason') or c['status'])
        cards=[]
        for ident in c['artifact_ids']:
            a=d['assets'][ident];payload=db.execute('SELECT payload FROM assets WHERE id=?',(ident,)).fetchone()[0]
            assert payload.startswith(b'\x89PNG'), 'not an actual PNG'
            im=Image.open(BytesIO(payload)).convert('RGB')
            assert min(im.size)>100 and max(ImageStat.Stat(im).stddev)>10, 'blank image'
            cards.append(a['metadata']);output['chart_metadata'].append(a['metadata'])
        if n in (1,24,26,41):
            ident=next(k for k,a in d['assets'].items() if a['metadata'].get('source','').endswith('information_schema.tables'))
            actual=frame(ident); names=next(k for k in actual.columns if k.lower()=='table_name'); schemas=next(k for k in actual.columns if k.lower()=='table_schema'); assert sorted(actual.loc[actual[schemas]=='teleai_default',names].tolist())==ORACLE['tables']
        elif n in (2,15,30):metadata('bank_loan')
        elif n==4:
            p=c['value_list_evidence'];assert p['source']==ORACLE['namespace']+'.bank_loan' and p['column']=='education'
            assert set(p['values'])==set(ORACLE['education_counts']) and not p['has_more']
        elif n==5:
            assert len(cards)==1;card=cards[0];s=card['render_spec']
            assert card['kind']=='bar' and card['columns']==['education']
            assert dict(zip(s['labels'],s['counts']))==ORACLE['education_counts']
        elif n in (3,6,7,8,9,10,11,12,13,14):
            assert len(cards)==1;card=cards[0];s=card['render_spec']
            assert card['kind']=='histogram' and card['columns'][0]=='age'
            expected=ORACLE['bank_rows'] if n==3 else ORACLE['primary_30_40'] if n==6 else ORACLE['primary_secondary_30_40']
            assert s['total_count']==expected, 'wrong population total'
            groups=[g for g in ORACLE['bank_age_education_groups'] if n==3 or
                    (30<=g['age']<=40 and g['education'] in (['primary'] if n==6 else ['primary','secondary']))]
            if 'series_counts' in s:
                assert s['category']=='education' and s['legend']
                actual=s['series_counts']
                for label,counts in zip(s['legend_labels'],actual):
                    subset=[g for g in groups if g['education']==label]
                    wanted=np.histogram([g['age'] for g in subset],bins=s['bin_edges'],weights=[g['count'] for g in subset])[0]
                    assert np.array_equal(wanted,counts), 'wrong grouped bin counts'
            else:
                wanted=np.histogram([g['age'] for g in groups],bins=s['bin_edges'],weights=[g['count'] for g in groups])[0]
                assert np.array_equal(wanted,s['bin_counts']), 'wrong histogram bins'
            if n>=9:
                assert s['category']=='education' and s['legend']
                assert set(s['legend_labels'])=={'primary','secondary'}
                assert len(set(s['colors']))==2
                if n in (12,13):assert s.get('stacked') is True, 'requested stacking absent from chart contract/output'
        elif n in (16,21):preview('bank_loan')
        elif n in (17,18,19,20,22,23,29,31):
            assert len(cards)==1;card=cards[0];s=card['render_spec']
            assert card['kind']=='scatter' and s['x']=='age' and s['y']=='balance'
            if n in (17,18):
                # The original request does not remove its earlier population.
                # A failed metadata/preview request is not a scope-reset order.
                assert s['drawable_rows']==ORACLE['primary_secondary_30_40'], 'unrequested prior-population removal'
            if n in (19,20,23,29,31):
                assert not c['scope']['conditions'] and not c['scope']['any_conditions']
                assert s['drawable_rows']==ORACLE['bank_rows']
                f=frame(card['dataset_id']);w=s['weight_column']
                normalized=f[['age','balance',w]].rename(columns={w:'__frequency'}).astype('int64').sort_values(['age','balance'],ignore_index=True)
                assert hashlib.sha256(normalized.to_json(orient='values').encode()).hexdigest()==ORACLE['bank_scatter_oracle']['sorted_coordinates_and_frequencies_sha256']
            elif n==22:
                actual=frame(card['dataset_id'])[['age','balance']].reset_index(drop=True)
                previous=json.loads((HERE/'21.json').read_text())
                expected=frame(previous['current']['table_preview_evidence']['dataset_id'])[['age','balance']].reset_index(drop=True)
                assert len(actual)==10, 'explicit preview scope lost'
                pd.testing.assert_frame_equal(actual,expected)
                assert not any(e['event']=='remote_query_finished' for e in d['run']), 'unnecessary remote query'
        elif n==25:
            counts=[]
            for ident in c['evidence_ids']:
                if ident in d['assets'] and d['assets'][ident]['kind']=='dataset':
                    a=d['assets'][ident]['metadata'];f=frame(ident)
                    if a.get('source')=='teleai_default._tianchi_ssd_shared300k_import':
                        counts.extend(int(v) for v in f.to_numpy().ravel() if str(v).isdigit())
            assert ORACLE['import_rows'] in counts, 'missing/wrong source actual row count'
            output['verified_row_count']=ORACLE['import_rows']
        elif n in (27,28):
            raise AssertionError('absent requested table must not silently produce completion for another source')
        elif n==32:metadata('alibaba_ssd')
        elif n==33:preview('alibaba_ssd')
        elif n in (34,35):
            assert len(cards)==1;card=cards[0];s=card['render_spec'];o=ORACLE['historical_large_frequency']
            assert card['kind']=='bar' and card['columns']==['n_1']
            assert dict(zip(s['labels'],s['counts']))==dict(zip(o['labels'],o['counts']))
            assert s['total_count']==o['total_count']
            if n==35:
                assert s['y_limits']==[0,10000]
                before=json.loads((HERE/'34.json').read_text());old=before['assets'][before['current']['artifact_ids'][0]]['metadata']
                assert old['dataset_id']==card['dataset_id'], 'axis change reloaded population'
                assert not any(e['event']=='remote_query_finished' for e in d['run'])
        elif n in (36,37,38):
            output['manual_required']='Verify inspected alibaba schema/date candidates with samples/types; inference must be marked.'
        elif n in (39,40):metadata('alibaba_ssd',True)
        elif n==42:metadata('ncr_ride')
        elif n==43:preview('ncr_ride')
        elif n==44:preview('stormtrooper')
        elif n==45:metadata('stormtrooper')
        elif n==46:metadata('stormtrooper',True)
        output['verdict']='REVIEW' if 'manual_required' in output else 'PASS'
    except (AssertionError,KeyError,TypeError,ValueError) as exc:
        output.update(verdict='FAIL',error=str(exc) or 'contract mismatch')
    finally:db.close()
    (HERE/f'{n:02}_check.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    summary={k:v for k,v in output.items() if k not in {'goal','chart_metadata','prompt'}}
    summary['charts']=[{'kind':a['kind'],'columns':a['columns'],
        **{k:v for k,v in a['render_spec'].items() if k in
           {'total_count','legend_labels','group_totals','stacked','drawable_rows','coordinate_count','y_limits'}}}
        for a in output['chart_metadata']]
    print(json.dumps(summary,ensure_ascii=False))
    return output

if __name__=='__main__':inspect(int(sys.argv[1]))
