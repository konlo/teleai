"""Compare real generated remote bins and rendered bars to independent NumPy."""
from dataclasses import asdict
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from matplotlib.axes import Axes
from tests import test_remote_latest as remote_fixture
from tests.test_remote_latest import references, execute_fixture
from pathlib import Path
import json
NUMERIC_PROMPT=json.loads((Path(__file__).parent/"fixtures/remote_latest_per_key.json").read_text())["numeric_prompt"]
from tests.test_latest_distribution import source_frame, KEY, CLOCK, VALUE, FIXTURE
from utils.analysis_remote_latest import prepare, plan


class RemoteLatestHistogramTests(unittest.TestCase):
    def run_histogram(self, values, bins):
        helper = remote_fixture.RemoteLatestTests()
        frame = source_frame()
        frame[VALUE] = values
        refs = references()
        next(c for c in refs[0]['columns'] if c['name']==VALUE)['dtype']='double'
        with tempfile.TemporaryDirectory() as root, patch('tests.test_remote_latest.references',return_value=refs):
            runtime,calls = helper.runtime(root,frame)
            try:
                bars=[]
                original=Axes.bar
                def observe(ax,x,height,*a,**kw):
                    bars.append((np.array(x),np.array(height),np.array(kw['width'])))
                    return original(ax,x,height,*a,**kw)
                prompt=NUMERIC_PROMPT+f' {bins}개 구간으로 그려줘.'
                with patch.object(Axes,'bar',observe):
                    outcome=runtime.submit(prompt)
                proof=runtime.inspect()['recovery'].get('latest_selection_evidence')
                self.assertIsNotNone(proof, runtime.inspect()['recovery'])
                expected_values=frame.sort_values(CLOCK).drop_duplicates(KEY,keep='last')[VALUE]
                counts,edges=np.histogram(expected_values,bins=bins)
                card=runtime.artifacts[proof['cards'][0]['id']]
                self.assertEqual(card.kind,'histogram')
                self.assertEqual(card.render_spec['bins'],bins)
                np.testing.assert_array_equal(card.render_spec['counts'],counts)
                np.testing.assert_allclose(card.render_spec['edges'],edges)
                self.assertEqual(len(bars),1)
                np.testing.assert_array_equal(bars[0][1],counts)
                np.testing.assert_allclose(bars[0][0],edges[:-1])
                np.testing.assert_allclose(bars[0][2],np.diff(edges))
                self.assertEqual(len(calls),1)
                self.assertEqual(runtime.datasets.metadata[proof['input_result_id']].rows,bins+1)
                self.assertEqual(outcome['status'],'answered')
                runtime.submit(prompt)
                self.assertEqual(len(calls),1,'same histogram must reuse verified result')
                runtime.submit(NUMERIC_PROMPT+f' {bins+1}개 구간으로 그려줘.')
                self.assertEqual(len(calls),2,'different bins must not reuse old bins')
                runtime.submit(f'구간 수만 {bins+2}개로 바꿔줘')
                next_proof=runtime.inspect()['recovery'].get('latest_selection_evidence')
                self.assertIsNotNone(next_proof)
                self.assertEqual(next_proof['chart_spec']['bins'],bins+2)
                self.assertEqual(next_proof['selected_keys'],proof['selected_keys'])
                self.assertEqual(len(calls),3)
            finally:runtime.close()

    def test_negative_boundary_empty_and_constant_bins(self):
        n=len(source_frame())
        for values,bins in ((np.arange(n,dtype=float)-n/2,5), (np.ones(n)*7,4), (np.arange(n)*0.1,25)):
            with self.subTest(bins=bins): self.run_histogram(values,bins)

    def test_nonfinite_and_corrupted_edges_are_not_completed(self):
        helper=remote_fixture.RemoteLatestTests()
        frame=source_frame(); frame[VALUE]=np.arange(len(frame),dtype=float)
        refs=references();next(c for c in refs[0]['columns'] if c['name']==VALUE)['dtype']='double'
        for value in (float('inf'),float('-inf'),float('nan')):
            invalid=frame.copy();invalid.loc[:,VALUE]=value
            with self.subTest(value=value),tempfile.TemporaryDirectory() as root, patch('tests.test_remote_latest.references',return_value=refs):
                r,calls=helper.runtime(root,invalid)
                try:
                    r.submit(NUMERIC_PROMPT+' 5개 구간으로 그려줘.')
                    self.assertFalse(r.artifacts)
                    self.assertIsNone(r.inspect()['recovery'].get('latest_selection_evidence'))
                finally:r.close()
        with tempfile.TemporaryDirectory() as root,patch('tests.test_remote_latest.references',return_value=refs):
            r,_=helper.runtime(root,frame)
            try:
                spec=plan(r.context,FIXTURE['source'],[KEY],CLOCK,VALUE,categorical=False,bins=5)['remote_latest_plan']
                result=execute_fixture(spec['query'],frame);result.loc[1,'__upper']+=0.1
                info=r.datasets.register(result,source=spec['source'],query=spec['query'],coverage='complete')
                with self.assertRaises(ValueError):prepare(r.context,spec['source'],[KEY],CLOCK,VALUE,info.id,categorical=False,bins=5)
                self.assertFalse(r.artifacts)
            finally:r.close()

    def test_explicit_composite_keys_use_all_key_columns(self):
        helper=remote_fixture.RemoteLatestTests()
        frame=source_frame();frame['partition_tag']=['p' if i%2 else 'q' for i in range(len(frame))]
        refs=references();refs[0]['columns'].append({'name':'partition_tag','dtype':'object'})
        prompt=helper.prompt().replace(KEY+'별',KEY+', partition_tag 조합 기준으로 unique한 값마다')
        with tempfile.TemporaryDirectory() as root,patch('tests.test_remote_latest.references',return_value=refs):
            r,calls=helper.runtime(root,frame)
            try:
                outcome=r.submit(prompt)
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertIsNotNone(proof,outcome)
                expected=frame.sort_values(CLOCK).drop_duplicates([KEY,'partition_tag'],keep='last')[VALUE].value_counts().to_dict()
                self.assertEqual({v[VALUE]:v[proof['count_column']] for v in proof['counts']},expected)
                self.assertEqual(len(calls),1)
            finally:r.close()
