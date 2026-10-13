"""Strict recorded-prompt grading against separate actual DB oracles.

No model calls. Oracle expectations never enter the agent. Unsupported chart
forms remain REVIEW_REQUIRED; a receipt or answered string alone is not PASS.
"""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
from PIL import Image


def asset_frame(db, root, key):
    metadata, payload = db.execute('SELECT metadata,payload FROM assets WHERE id=? AND kind=?',
                                  (key,'dataset')).fetchone()
    return json.loads(metadata),pd.read_parquet(BytesIO(payload) if payload else root/(key+'.parquet'))


def grade(report, root, oracle, schemas):
    matrix=pd.DataFrame(oracle['results']['age_education_matrix']['rows'],columns=['age','education','frequency'])
    expected_inventory=sorted(map(tuple,oracle['results']['inventory']['rows']))
    ai=report['scenario']=='prompt_ai_test_scenario_#1'
    inventory={1} if ai else {1,24,26}
    metadata={2,10} if ai else {2,15}
    values={3} if ai else {4}
    preview={9} if ai else {16,21}
    absent=set() if ai else {25,27,28,29,30,31}
    results=[];known={}
    runtime_events=[json.loads(line) for line in (root/'runtime.jsonl').read_text().splitlines()]
    with sqlite3.connect('file:'+str(root/'assets.sqlite')+'?mode=ro',uri=True) as db:
        for turn in report['turns']:
            n=turn['id'];e=turn.get('evidence',{});checks={};status='FAIL'
            datasets=turn.get('datasets',{})
            preserved=all(k in datasets and datasets[k]['digest']==digest
                          for k,digest in known.items())
            known.update({k:v['digest'] for k,v in turn.get('datasets',{}).items()})
            checks['prior_assets_immutable']=preserved
            if turn['agent_status']!='answered':
                results.append({'id':n,'status':'FAIL','agent_status':turn['agent_status'],
                    'reason':e.get('stop_reason') or turn.get('error_type') or 'unverified_response',
                    'checks':checks}); continue
            if n in absent:
                status='REVIEW_REQUIRED'
                checks['unsupported_namespace']='Explicit MySQL-only names are not remapped to Databricks; inspect precise refusal manually'
                # A successful answer about a different physical source is an
                # established failure, not merely an unsupported-case review.
                if n == 25:
                    requested='_tianchi_ssd_shared300k_import'
                    checks['no_silent_source_substitution']=all(
                        s.rsplit('.',1)[-1].casefold()==requested.casefold()
                        for s in e.get('required_sources',[])) and bool(e.get('required_sources'))
            elif n in inventory:
                ids=[r['dataset_id'] for task in turn.get('verified_tasks',[]) for r in task.get('receipts',{}).get('remote_query',{}).values()]
                if ids:
                    meta,frame=asset_frame(db,root,ids[-1])
                    checks['actual_inventory']=meta['source']=='workspace.information_schema.tables' and sorted(map(tuple,frame.values.tolist()))==expected_inventory
            elif n in metadata:
                proof=e.get('metadata_evidence') or {};table='stormtrooper' if ai and n==10 else 'bank_loan'
                native=schemas['results'][table+'_schema']['rows'];names=[r[0] for r in native]
                checks['physical_subject']=proof.get('table')=='workspace.default.'+table
                checks['actual_columns']=proof.get('columns')==names
                if ai and n==10:
                    checks['native_db_types']=(proof.get('type_authority')=='current_database_metadata'
                        and {c.get('name'):c.get('dtype') for c in proof.get('schema',[])}
                        ==dict((r[0],r[1]) for r in native))
                    checks['no_chart']=not turn.get('charts')
            elif n in values:
                proof=e.get('value_list_evidence') or {}
                checks['physical_subject']=proof.get('source')=='workspace.default.bank_loan'
                checks['all_distinct_values']=set(proof.get('values',[]))==set(matrix.education) and proof.get('has_more') is False
            elif n in preview:
                proof=e.get('table_preview_evidence') or {};table='stormtrooper' if ai else 'bank_loan'
                checks['physical_subject']=proof.get('source')=='workspace.default.'+table
                if proof.get('dataset_id'):
                    meta,frame=asset_frame(db,root,proof['dataset_id'])
                    checks['ten_real_rows']=proof.get('rows')==10 and len(frame)>=10 and list(frame.columns)==[r[0] for r in schemas['results'][table+'_schema']['rows']]
                    if not ai and n==16:
                        checks['filters_retained']=frame.head(10).age.between(30,40).all() and frame.head(10).education.isin(['primary','secondary']).all()
            else:
                cards=turn.get('charts',{})
                checks['current_chart']=bool(cards)
                if cards:
                    key=next(reversed(cards));raw=db.execute('SELECT metadata,payload FROM assets WHERE id=? AND kind=?',(key,'chart')).fetchone()
                    card=json.loads(raw[0]);spec=card.get('render_spec',{});info=turn['datasets'][card['dataset_id']]
                    try:
                        with Image.open(BytesIO(raw[1])) as im:im.verify()
                        checks['valid_png']=True
                        with Image.open(BytesIO(raw[1])) as im:
                            checks['visible_plot_content']=(im.format=='PNG' and
                                im.width*im.height<=16*1024*1024 and
                                bool(np.var(np.asarray(im.convert('RGB')),axis=(0,1)).max()>1))
                    except Exception:checks['valid_png']=False
                    checks['physical_subject']=info['source']=='workspace.default.bank_loan'
                    scatter=ai and n==8 or not ai and n in {17,18,19,20,22,23}
                    if scatter:
                        checks['scatter_axes']=card['kind']=='scatter' and card['columns']==['age','balance']
                        if not ai and n==22:
                            actual_meta,actual_frame=asset_frame(db,root,card['dataset_id'])
                            # A ten-row raw preview may retain additional fields;
                            # the plotted axes are checked separately above.
                            checks['last_ten_rows']=len(actual_frame)==10 and all(
                                name in actual_frame for name in ['age','balance'])
                            earlier=next((t for t in report['turns'] if t['id']==21),{})
                            preview_id=(earlier.get('evidence',{}).get('table_preview_evidence') or {}).get('dataset_id')
                            if preview_id and all(c in actual_frame for c in ['age','balance']):
                                _,preview_frame=asset_frame(db,root,preview_id)
                                checks['prior_preview_values']=(actual_frame[['age','balance']].reset_index(drop=True)
                                    .equals(preview_frame.head(10)[['age','balance']].reset_index(drop=True)))
                                checks['preview_lineage']=(card['dataset_id']==preview_id
                                    or actual_meta.get('parent_id')==preview_id)
                            else:checks['prior_preview_values']=False
                            checks['no_remote_submission']=not any(x.get('event')=='remote_query_started'
                                and x.get('run_id')==turn['run_id'] and not x.get('cached_receipt')
                                for x in runtime_events)
                        else:
                            expected_rows=208699 if not ai and n in {17,18} else 750000
                            checks['population_count']=spec.get('drawable_rows')==expected_rows
                            if expected_rows==750000:
                                checks['whole_coordinates']=spec.get('coordinate_count')==oracle['results']['scatter_coordinate_count']['rows'][0][0]
                    else:
                        filtered=matrix
                        primary=ai and n==5 or not ai and n==6
                        combined=ai and n in {6,7} or not ai and 7<=n<=14
                        if primary or combined:
                            filtered=matrix[matrix.age.between(30,40)&matrix.education.isin(['primary','secondary'] if combined else ['primary'])]
                        wanted='education' if ai and n==4 or not ai and n==5 else 'age'
                        if not ai and 6<=n<=14:
                            # These human prompts leave the distribution axis open.
                            # Accept either meaningful observed axis, keeping exact
                            # population/bin/legend checks; never invent a gold axis.
                            wanted=card['columns'][0] if card['columns'] and card['columns'][0] in {'age','education'} else wanted
                        checks['distribution_axis']=card['kind']==('bar' if wanted=='education' else 'histogram') and wanted in card['columns']
                        checks['population_count']=spec.get('total_count',sum(spec.get('counts',[])))==int(filtered.frequency.sum())
                        if wanted=='education' and spec.get('labels'):
                            checks['every_category_frequency']=dict(zip(spec['labels'],spec['counts']))==filtered.groupby('education').frequency.sum().to_dict()
                        elif 'bin_edges' in spec:
                            edges=np.asarray(spec['bin_edges']);actual=spec['bin_counts']
                            expected=np.histogram(filtered.age,bins=edges,weights=filtered.frequency)[0]
                            checks['every_bin_frequency']=expected.tolist()==actual
                        else:status='REVIEW_REQUIRED'
                        if ai and n==7:checks['five_bins']=spec.get('bins')==5
                        if not ai and 9<=n<=14:
                            checks['education_legend']=set(spec.get('legend_labels',[]))=={'primary','secondary'}
                            colors=spec.get('colors',[]);checks['distinct_colors']=len(set(colors))>=2
                        if not ai and n in {12,13,14}:checks['stacked']=spec.get('stacked') is True
            checks={k:v.item() if isinstance(v,np.generic) else v for k,v in checks.items()}
            if status!='REVIEW_REQUIRED':status='PASS' if len(checks)>1 and all(v is True for v in checks.values()) else 'FAIL'
            elif any(v is False for v in checks.values()):status='FAIL'
            results.append({'id':n,'status':status,'agent_status':turn['agent_status'],'checks':checks})
    return {'scenario':report['scenario'],'build':report['build'],'scope':'Independent DB matrix/schema/inventory, persisted PNG, asset digests; no UI automation here',
        'requested_turns':report['requested_turns'],'recorded_turns':len(results),'counts':dict(Counter(r['status'] for r in results)),
        'release_decision':'NO_GO' if len(results)!=report['requested_turns'] or any(r['status']!='PASS' for r in results) else 'JOURNEY_PASS_NOT_GLOBAL_GO','results':results}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['report','asset-root','oracle','schemas','output']:p.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args();result=grade(json.loads(args.report.read_text()),args.asset_root,json.loads(args.oracle.read_text()),json.loads(args.schemas.read_text()))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['counts']))
    sys.exit(0 if result['release_decision']=='JOURNEY_PASS_NOT_GLOBAL_GO' else 1)
