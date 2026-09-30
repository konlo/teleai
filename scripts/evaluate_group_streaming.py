"""Million-row retained-data group statistics against an independent integer oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from utils.analysis_group_summary import summarize_groups
from utils.analysis_charts import render_chart_spec
from utils.analysis_image_validation import validate_chart_image
from tests.test_group_streaming import FIXTURE, GROUP, VALUE, PAYLOAD, frame_slice, metrics


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def expected(rows):
    totals={};filtered=0
    # Independent row generator arithmetic, no production reductions or SQL.
    for index in range(rows):
        value=index%FIXTURE['value_cycle']
        if index%FIXTURE['null_value_period']==0 or value<3:continue
        filtered+=1
        if index%FIXTURE['null_group_period']==0:continue
        key=index%FIXTURE['groups']
        group=totals.setdefault(key,{'count':0,'sum':0,'hist':{},'positive':0,'chosen':[]})
        group['count']+=1;group['sum']+=value
        group['hist'][value]=group['hist'].get(value,0)+1
        if index%FIXTURE['null_status_period'] and index%3>0:group['positive']+=1
        if index%FIXTURE['null_status_period'] and index%3==2:group['chosen'].append(value)
    records=[]
    for key,g in sorted(totals.items()):
        ranks=[(g['count']-1)//2,g['count']//2];seen=0;middle=[]
        for value,n in sorted(g['hist'].items()):
            middle.extend(value for rank in ranks if seen<=rank<seen+n);seen+=n
        records.append({GROUP:key,'rows':g['count'],'sum':g['sum'],'mean':g['sum']/g['count'],
            'median':sum(middle)/2,'min':min(g['hist']),'max':max(g['hist']),
            'positive_percent':100*g['positive']/g['count'],
            'chosen_mean':sum(g['chosen'])/len(g['chosen']) if g['chosen'] else -1})
    return pd.DataFrame(records),filtered


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    n=FIXTURE['large_rows'];oracle,filtered=expected(n)
    with tempfile.TemporaryDirectory() as folder:
        db=AssetDB(folder,'evaluation','large-groups');store=PersistentDatasets(db,budget=0,max_full_read_bytes=1024)
        try:
            raw=store.register_batches((frame_slice(i,min(i+10000,n)) for i in range(0,n,10000)),
                columns=FIXTURE['columns'],source=FIXTURE['source'],max_rows=n,
                coverage='complete',predicate_known=True,snapshot='synthetic-v1')
            db.select_dataset(raw.id);original=file_hash(db.dataset_file(raw.id))
            observed=[];sizes=[];batches=store.frames.batches
            def track(*values,**kwargs):
                observed.append(list(values[1]))
                for batch in batches(*values,**kwargs):sizes.append(batch.num_rows);yield batch
            started=time.monotonic()
            with patch.object(FrameCache,'project',side_effect=AssertionError('full-column read')), patch.object(
                    FrameCache,'__getitem__',side_effect=AssertionError('whole-frame read')), patch.object(
                    store.frames,'batches',side_effect=track):
                result=summarize_groups(store,raw.id,group_columns=[GROUP],metrics=metrics(),
                    conditions=[{'column':VALUE,'op':'ge','value':3}])
            elapsed=time.monotonic()-started
            child=result['dataset']['id'];actual=store.frames.project(child,list(oracle))
            pd.testing.assert_frame_equal(actual,oracle,check_dtype=False,rtol=1e-10)
            counters=result['group_summary_result']
            card,chart_summary,_=render_chart_spec(store,child,kind='bar',x=GROUP,y='mean',aggregation='none')
            validate_chart_image(card.image)
            preserved=file_hash(db.dataset_file(raw.id))==original and db.selected_dataset_id()==raw.id
            assert counters['source_rows']==n and counters['filtered_rows']==filtered and preserved
            assert all(PAYLOAD not in columns for columns in observed) and max(sizes)<=1024
            report={'status':'PASS','mode':'synthetic production tool; no model or remote SQL',
                'source_rows':n,'groups':len(actual),'metrics':len(metrics()),'independent_oracle':True,
                'input_rows_verified':True,'original_preserved':preserved,'columns_read':observed,
                'max_batch_rows':max(sizes),'batch_count':len(sizes),'cache_bytes':store.frames.bytes,
                'elapsed_seconds':round(elapsed,3),'execution_mode':counters['execution_mode'],
                'chart_png_valid':True,'chart_dataset_is_aggregate':card.dataset_id==child,
                'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        finally:db.close()
        reopened=AssetDB(folder,'evaluation','large-groups')
        try:
            restored=PersistentDatasets(reopened,budget=0)
            report['restart_preserved']=restored.metadata[child].parent_id==raw.id and file_hash(reopened.dataset_file(raw.id))==original
            pd.testing.assert_frame_equal(restored.frames.project(child,list(oracle)),oracle,check_dtype=False,rtol=1e-10)
            assert report['restart_preserved']
        finally:reopened.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.png').write_bytes(card.image)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
