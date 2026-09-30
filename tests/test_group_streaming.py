import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.analysis_agent.assets import AssetDB, FrameCache, PersistentDatasets
from utils.analysis_group_summary import summarize_groups

FIXTURE=json.loads((Path(__file__).parent/'fixtures/group_streaming.json').read_text())
GROUP,VALUE,FLAG,PAYLOAD=FIXTURE['columns']


def frame_slice(start,stop):
    indices=np.arange(start,stop)
    frame=pd.DataFrame({GROUP:indices%FIXTURE['groups'],VALUE:(indices%FIXTURE['value_cycle']).astype(float),
                        FLAG:indices%3,PAYLOAD:['unused']*len(indices)})
    for column,period in [(GROUP,'null_group_period'),(VALUE,'null_value_period'),(FLAG,'null_status_period')]:
        frame.loc[indices%FIXTURE[period]==0,column]=np.nan
    return frame


def metrics():
    return [{'name':'rows','aggregation':'count'},
            *[{'name':kind,'aggregation':kind,'value_column':VALUE} for kind in ('sum','mean','median','min','max')],
            {'name':'positive_percent','aggregation':'conditional_percent','condition':{'column':FLAG,'op':'gt','value':0}},
            {'name':'chosen_mean','aggregation':'conditional_mean','value_column':VALUE,
             'condition':{'column':FLAG,'op':'eq','value':2},'empty_value':-1}]


class GroupStreamingTests(unittest.TestCase):
    def test_agent_completes_grouped_eda_from_retained_data_without_reload(self):
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        from migration.test_persistent_runtime import QuietModel
        with tempfile.TemporaryDirectory() as folder:
            runtime=GraphAnalysisRuntime(folder,'owner','agent-groups',QuietModel())
            frame=frame_slice(0,FIXTURE['rows'])
            raw=runtime.datasets.register_batches([frame],columns=list(frame),source=FIXTURE['source'],
                max_rows=len(frame),coverage='complete',predicate_known=True)
            runtime.select_dataset(raw.id)
            try:
                with patch.object(FrameCache,'project',side_effect=AssertionError('whole-column read')), patch.object(
                        FrameCache,'__getitem__',side_effect=AssertionError('whole-frame read')):
                    result=runtime.submit(FIXTURE['agent_prompt'])
                self.assertEqual(result['status'],'answered',result)
                proof=runtime.inspect()['recovery']['group_summary_evidence']
                self.assertEqual(proof['group_summary_result']['execution_mode'],'streamed_local_sql')
                self.assertEqual(proof['dataset']['parent_id'],raw.id)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                self.assertFalse(runtime.ledger.uncertain())
            finally:runtime.close()

    def test_large_integer_sum_and_zero_numerators_keep_exact_values(self):
        with tempfile.TemporaryDirectory() as folder:
            db=AssetDB(folder,'owner','exact-numbers');store=PersistentDatasets(db,budget=0)
            frame=frame_slice(0,FIXTURE['rows'])
            frame[GROUP]=0
            frame[VALUE]=pd.Series([400000000001]*len(frame),dtype='int64')
            frame.loc[0,VALUE]+=1  # Odd total above float64's exact integer range.
            frame[FLAG]=0
            raw=store.register_batches([frame],columns=list(frame),source=FIXTURE['source'],
                max_rows=len(frame),coverage='complete',predicate_known=True)
            try:
                output=summarize_groups(store,raw.id,group_columns=[GROUP],metrics=metrics())
                actual=store.frames.project(output['dataset']['id'],['sum','positive_percent','chosen_mean'])
                expected=400000000001*len(frame)+1
                self.assertGreater(expected,2**53)
                self.assertEqual(actual['sum'].iloc[0],expected)
                self.assertTrue(pd.api.types.is_integer_dtype(actual['sum']))
                self.assertEqual(actual['positive_percent'].iloc[0],0)
                self.assertEqual(actual['chosen_mean'].iloc[0],-1)
            finally:db.close()

    def test_grouped_metrics_stream_only_needed_columns_and_preserve_root(self):
        with tempfile.TemporaryDirectory() as folder:
            db=AssetDB(folder,'owner','groups')
            store=PersistentDatasets(db,budget=0,max_full_read_bytes=1024)
            frame=frame_slice(0,FIXTURE['rows'])
            raw=store.register_batches([frame],columns=list(frame),source=FIXTURE['source'],
                max_rows=len(frame),coverage='complete',predicate_known=True,snapshot='fixture-v1')
            db.select_dataset(raw.id)
            original=db.dataset_file(raw.id).read_bytes()
            observed=[];batches=store.frames.batches
            def track(*args,**kwargs):
                observed.append(list(args[1]));yield from batches(*args,**kwargs)
            try:
                with patch.object(FrameCache,'__getitem__',side_effect=AssertionError('whole-frame decode')), patch.object(
                        FrameCache,'project',side_effect=AssertionError('whole-column decode')), patch.object(
                        store.frames,'batches',side_effect=track):
                    output=summarize_groups(store,raw.id,group_columns=[GROUP],metrics=metrics(),
                        conditions=[{'column':VALUE,'op':'ge','value':3}])
                actual=store.frames.project(output['dataset']['id'],[GROUP,*[m['name'] for m in metrics()]])
                expected=[]
                filtered=frame.loc[frame[VALUE]>=3].dropna(subset=[GROUP])
                for key,group in filtered.groupby(GROUP):
                    values=group[VALUE].dropna().tolist()
                    chosen=group.loc[group[FLAG]==2,VALUE].dropna()
                    expected.append({GROUP:key,'rows':len(group),'sum':sum(values),
                        'mean':sum(values)/len(values),'median':float(np.median(values)),
                        'min':min(values),'max':max(values),
                        'positive_percent':100*(group[FLAG]>0).sum()/len(group),
                        'chosen_mean':float(chosen.mean()) if len(chosen) else -1})
                pd.testing.assert_frame_equal(actual,pd.DataFrame(expected),check_dtype=False,rtol=1e-10)
                self.assertEqual(output['group_summary_result']['execution_mode'],'streamed_local_sql')
                self.assertEqual(output['group_summary_result']['source_rows'],len(frame))
                self.assertEqual(output['group_summary_result']['group_input_rows'],len(filtered))
                self.assertTrue(observed)
                self.assertTrue(all(PAYLOAD not in columns for columns in observed))
                self.assertEqual(db.dataset_file(raw.id).read_bytes(),original)
                self.assertEqual(db.selected_dataset_id(),raw.id)
                self.assertEqual(store.frames.bytes,0)
                child=output['dataset']['id']
            finally:db.close()
            reopened=AssetDB(folder,'owner','groups')
            try:
                restored=PersistentDatasets(reopened,budget=0)
                self.assertEqual(restored.metadata[child].parent_id,raw.id)
                self.assertEqual(restored.metadata[child].snapshot,'fixture-v1')
            finally:reopened.close()

    def test_interrupted_or_incomplete_batches_never_publish(self):
        for mode in ('interrupted','truncated','too_many_groups','output_limit'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                db=AssetDB(folder,'owner',mode);store=PersistentDatasets(db,budget=0)
                frame=frame_slice(0,FIXTURE['rows'])
                raw=store.register_batches([frame],columns=list(frame),source=FIXTURE['source'],
                    max_rows=len(frame),coverage='complete',predicate_known=True)
                db.select_dataset(raw.id);original=db.dataset_file(raw.id).read_bytes()
                batches=store.frames.batches
                def fail(*args,**kwargs):
                    stream=batches(*args,**kwargs)
                    try:
                        yield next(stream)
                        if mode=='interrupted':raise OSError('injected disk read failure')
                        if mode in {'too_many_groups','output_limit'}:yield from stream
                    finally:stream.close()
                try:
                    with patch.object(store.frames,'batches',side_effect=fail):
                        with self.assertRaises(Exception):
                            summarize_groups(store,raw.id,group_columns=[GROUP],metrics=metrics(),
                                             max_groups=1 if mode=='too_many_groups' else 1000,
                                             max_output_rows=1 if mode=='output_limit' else 1000)
                    self.assertEqual(set(store.metadata),{raw.id})
                    self.assertEqual(db.dataset_file(raw.id).read_bytes(),original)
                    self.assertEqual(db.selected_dataset_id(),raw.id)
                finally:db.close()


if __name__=='__main__':unittest.main()
