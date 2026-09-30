"""Missing rows must be excluded only at the explicitly requested stage."""
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from tests import test_latest_distribution as local
from tests import test_remote_latest as remote
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_latest import latest_distribution
from ui.analysis_chart_delivery import chart_references

POLICY='결측행을 최신행 선택 전에 제외해줘'


def missing_frame():
    frame=local.source_frame()
    newest=frame.sort_values(local.CLOCK).drop_duplicates(local.KEY,keep='last').iloc[[0]].copy()
    newest[local.CLOCK]+=pd.Timedelta(days=100)
    newest[local.VALUE]=None
    missing_key=frame.iloc[[0]].copy();missing_key[local.KEY]=None
    return pd.concat([frame,newest,missing_key],ignore_index=True)


class LatestNullContinuationTests(unittest.TestCase):
    def test_local_and_remote_followup_preserves_fallback_older_valid_record(self):
        for mode in ('local','remote'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                helper=local.LatestDistributionTests() if mode=='local' else remote.RemoteLatestTests()
                def open_runtime():
                    result=helper.runtime(root,missing_frame())
                    return (result,[]) if mode=='local' else result
                r,calls=open_runtime()
                try:
                    prompt=local.FIXTURE['cases'][1]['prompt'] if mode=='local' else helper.prompt()
                    r.submit(prompt)
                    self.assertFalse(chart_references(r.events()[-1]))
                    self.assertEqual(r.inspect()['recovery']['latest_pending']['error_code'],'latest_null_policy')
                    raw=r.context.selected_dataset_id
                    digest=stored_dataset_digest(r.datasets,raw) if raw else None
                finally:r.close()
                r,calls=open_runtime()
                try:
                    outcome=r.submit(POLICY)
                    proof=r.inspect()['recovery'].get('latest_selection_evidence')
                    self.assertTrue(proof,outcome)
                    self.assertEqual({row[local.VALUE]:row[proof['count_column']] for row in proof['counts']},local.FIXTURE['expected_counts'])
                    self.assertEqual(proof['excluded_rows'],2)
                    self.assertEqual(proof['selected_keys'],5)
                    self.assertIn('선택 전에',outcome['text'])
                    self.assertEqual(proof['distribution']['row_selection']['null_policy'],'drop_before_selection')
                    state=r.inspect()['recovery'];args={k:state['latest_per_key_spec'][k] for k in
                        (('source','key_columns','order_column','value_column') if mode=='remote' else ('dataset_id','key_columns','order_column','value_column'))}
                    self.assertFalse(r.recovery._proposed_scope_valid({'name':'prepare_remote_latest_distribution' if mode=='remote' else 'analyze_latest_distribution','args':args},state))
                    self.assertTrue(chart_references(r.events()[-1]))
                    if mode=='remote':self.assertEqual(len(calls),1)
                    else:self.assertEqual(stored_dataset_digest(r.datasets,raw),digest)
                    if mode=='remote':
                        r.submit(prompt+' '+POLICY)
                        self.assertEqual(len(calls),1)
                finally:r.close()

    def test_large_retained_null_policy_scans_batches_not_full_projection(self):
        helper=local.LatestDistributionTests()
        frame=missing_frame()
        bad=frame.iloc[[-1]]
        frame=pd.concat([frame,pd.concat([bad]*20_001)],ignore_index=True)
        with tempfile.TemporaryDirectory() as root:
            r=helper.runtime(root,frame)
            try:
                raw=r.context.selected_dataset_id;digest=stored_dataset_digest(r.datasets,raw)
                original_project=r.datasets.frames.project
                def project(key,columns):
                    if key==raw:raise AssertionError('Full original projection')
                    return original_project(key,columns)
                with patch.object(r.datasets.frames,'project',side_effect=project):
                    result=r.submit(local.FIXTURE['cases'][1]['prompt']+' '+POLICY)
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertTrue(proof,result)
                self.assertEqual(proof['execution_mode'],'bounded_local_sql')
                self.assertEqual(proof['excluded_rows'],20_003)
                self.assertEqual(proof['selected_keys'],5)
                self.assertEqual(stored_dataset_digest(r.datasets,raw),digest)
                self.assertNotIn(raw,r.datasets.frames.cache)
            finally:r.close()

    def test_unspecified_stage_after_selection_do_not_publish(self):
        helper=local.LatestDistributionTests()
        for policy in ('결측값 제외해줘','결측행을 최신행 선택 후에 제외해줘',
                '결측행을 최신행 선택 전에 제외하지 말아줘',
                '결측행을 최신행 선택 전에 제외해줘. 아니 선택 후에 제외해줘'):
            with tempfile.TemporaryDirectory() as root:
                r=helper.runtime(root,missing_frame())
                try:
                    r.submit(local.FIXTURE['cases'][1]['prompt']+' '+policy)
                    self.assertFalse(chart_references(r.events()[-1]))
                    self.assertFalse(r.events()[-1].additional_kwargs['analysis_complete'])
                finally:r.close()

    def test_infinity_is_not_silently_treated_as_missing(self):
        from utils.analysis_latest_sql import select_latest
        frame=local.source_frame()
        frame[local.VALUE]=[float('inf')]+[2.]*(len(frame)-1)
        refs=remote.references()
        next(c for c in refs[0]['columns'] if c['name']==local.VALUE)['dtype']='double'
        with tempfile.TemporaryDirectory() as root,patch.object(remote,'references',return_value=refs):
            helper=remote.RemoteLatestTests();r,_=helper.runtime(root,frame)
            try:
                r.submit(helper.prompt()+' '+POLICY)
                self.assertEqual(r.inspect()['recovery']['latest_error_code'],'latest_nonfinite_policy')
                self.assertFalse(chart_references(r.events()[-1]))
                raw=r.datasets.register(frame,source='fixture.local',coverage='complete',predicate_known=True)
                output=latest_distribution(r.context,raw.id,[local.KEY],local.CLOCK,local.VALUE,null_policy='drop_before_selection')
                self.assertEqual(output['error_code'],'latest_nonfinite_policy')
                _,issue,_=select_latest(r.datasets,raw,list(frame.columns),[local.KEY],[local.CLOCK],null_policy='drop_before_selection')
                self.assertEqual(issue['error_code'],'latest_nonfinite_policy')
                self.assertFalse(r.artifacts)
            finally:r.close()
    def test_all_missing_do_not_publish(self):
        helper=local.LatestDistributionTests()
        for mode in ('local','remote'):
            frame=local.source_frame();frame[local.VALUE]=None
            with tempfile.TemporaryDirectory() as root:
                if mode=='local':r=helper.runtime(root,frame);prompt=local.FIXTURE['cases'][1]['prompt']
                else:
                    remote_helper=remote.RemoteLatestTests();r,_=remote_helper.runtime(root,frame);prompt=remote_helper.prompt()
                try:
                    r.submit(prompt+' '+POLICY)
                    self.assertFalse(chart_references(r.events()[-1]))
                    self.assertFalse(r.events()[-1].additional_kwargs['analysis_complete'])
                finally:r.close()
