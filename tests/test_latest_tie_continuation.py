"""Whole-graph recovery: explicit tie policy survives restart and stays source-bound."""
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
from tests import test_latest_distribution as local
from tests import test_remote_latest as remote
from core.analysis_agent.latest_selection import accepted_remote, continue_order
from utils.analysis_datasets import stored_dataset_digest
from ui.analysis_chart_delivery import chart_references


def tied_frame():
    frame=local.source_frame()
    winner=frame.sort_values(local.CLOCK).drop_duplicates(local.KEY,keep='last').iloc[[0]].copy()
    winner[local.INGEST]+=pd.Timedelta(days=100)
    winner[local.VALUE]='replacement'
    return pd.concat([frame,winner],ignore_index=True)


class LatestTieContinuationTests(unittest.TestCase):
    def test_local_followup_restart_and_raw_preservation(self):
        helper=local.LatestDistributionTests()
        with tempfile.TemporaryDirectory() as root:
            r=helper.runtime(root,tied_frame())
            raw=r.context.selected_dataset_id
            digest=stored_dataset_digest(r.datasets,raw)
            try:
                r.submit(local.FIXTURE['cases'][1]['prompt'])
                self.assertFalse(chart_references(r.events()[-1]))
                self.assertEqual(r.inspect()['recovery']['latest_pending']['error_code'],'latest_order_tie')
            finally:r.close()
            r=helper.runtime(root)
            try:
                outcome=r.submit(local.INGEST+' 기준으로 해줘')
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertTrue(proof,outcome)
                actual=r.datasets.frames[proof['dataset']['id']]
                expected=tied_frame().sort_values([local.CLOCK,local.INGEST]).drop_duplicates(local.KEY,keep='last')
                self.assertEqual(dict(zip(actual[local.KEY],actual[local.VALUE])),dict(zip(expected[local.KEY],expected[local.VALUE])))
                self.assertEqual(stored_dataset_digest(r.datasets,raw),digest)
                self.assertTrue(chart_references(r.events()[-1]))
            finally:r.close()

    def test_remote_followup_recomputes_once_then_reuses_after_restart(self):
        helper=remote.RemoteLatestTests()
        resolved=helper.prompt()+' 동률이면 '+local.INGEST+'이 가장 큰 행을 사용해줘.'
        with tempfile.TemporaryDirectory() as root:
            r,calls=helper.runtime(root,tied_frame())
            try:
                # An unrelated selected original must not prevent remote continuation.
                raw=r.datasets.register(local.source_frame(),source='fixture.other.input',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                r.submit(helper.prompt())
                self.assertFalse(chart_references(r.events()[-1]))
                self.assertEqual(len(calls),1)
            finally:r.close()
            r,calls=helper.runtime(root,tied_frame())
            try:
                outcome=r.submit(local.INGEST+' 기준으로 해줘')
                state=r.inspect()['recovery'];proof=state.get('latest_selection_evidence')
                self.assertTrue(proof,outcome)
                expected=tied_frame().sort_values([local.CLOCK,local.INGEST]).drop_duplicates(local.KEY,keep='last')[local.VALUE].value_counts().to_dict()
                self.assertEqual({row[local.VALUE]:row[proof['count_column']] for row in proof['counts']},expected)
                self.assertEqual(len(calls),1)
                self.assertIn('_order DESC, _tie0 DESC',calls[0])
                args={k:state['latest_per_key_spec'][k] for k in ('source','key_columns','order_column','value_column')}
                args['result_dataset_id']=proof['input_result_id']
                self.assertFalse(accepted_remote(r.context,r.artifacts,state['latest_per_key_spec'],args,proof,state['remote_query_evidence']))
            finally:r.close()
            r,calls=helper.runtime(root,tied_frame())
            try:
                outcome=r.submit(resolved)
                self.assertIn('재사용',outcome['text'])
                self.assertFalse(calls)
            finally:r.close()

    def test_unresolved_second_tie_and_changed_source_never_choose_arbitrarily(self):
        helper=local.LatestDistributionTests()
        frame=tied_frame()
        frame=pd.concat([frame,frame.iloc[[-1]]],ignore_index=True)
        with tempfile.TemporaryDirectory() as root:
            r=helper.runtime(root,frame)
            try:
                r.submit(local.FIXTURE['cases'][1]['prompt'])
                old=deepcopy(r.inspect()['recovery'])
                self.assertIsNone(continue_order(local.INGEST+' 오름차순',old,r.context))
                r.submit(local.INGEST+' 기준으로 해줘')
                self.assertFalse(chart_references(r.events()[-1]))
                self.assertFalse(r.events()[-1].additional_kwargs['analysis_complete'])
                other=r.datasets.register(local.source_frame(),source='fixture.other.source',coverage='complete',predicate_known=True)
                r.select_dataset(other.id)
                self.assertIsNone(continue_order(local.INGEST+' 기준으로 해줘',old,r.context))
            finally:r.close()

    def test_remote_numeric_bins_with_explicit_tie_order(self):
        helper=remote.RemoteLatestTests()
        frame=tied_frame()
        frame[local.VALUE]=frame[local.VALUE].map({'P100':100,'P200':200,'P300':300,'replacement':400})
        refs=remote.references()
        next(c for c in refs[0]['columns'] if c['name']==local.VALUE)['dtype']='int64'
        prompt=(local.FIXTURE['source']+' 테이블에서 '+local.KEY+'별 '+local.CLOCK
            +'이 가장 늦은 행 하나만 사용해서 '+local.VALUE+'에 대한 histogram을 보여줘. 4개 구간. 동률이면 '
            +local.INGEST+'이 가장 큰 행을 사용해줘.')
        with tempfile.TemporaryDirectory() as root,patch.object(remote,'references',return_value=refs):
            r,calls=helper.runtime(root,frame)
            try:
                outcome=r.submit(prompt)
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertTrue(proof,outcome)
                import numpy as np
                winners=frame.sort_values([local.CLOCK,local.INGEST]).drop_duplicates(local.KEY,keep='last')
                counts,edges=np.histogram(winners[local.VALUE],bins=4)
                self.assertEqual(proof['chart_spec']['counts'],counts.tolist())
                self.assertEqual(proof['chart_spec']['edges'],edges.tolist())
                self.assertEqual(len(calls),1)
                previous=deepcopy(r.inspect()['recovery'])
                other=r.datasets.register(frame,source='fixture.changed.selection',coverage='complete',predicate_known=True)
                r.select_dataset(other.id)
                self.assertIsNone(continue_order('6개 구간으로 바꿔줘',previous,r.context))
            finally:r.close()
