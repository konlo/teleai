"""Planning known numeric roles must not decode large persisted columns."""
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from langchain_core.messages import HumanMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.assets import FrameCache
from scripts.evaluate_analysis_statistics import ForbiddenModel


class MetadataPlanningTests(unittest.TestCase):
    def test_group_and_pivot_bind_without_any_data_scan(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'test','metadata-planning',ForbiddenModel())
            frame=pd.DataFrame({'segment':['a','a','b'],'measure':[2.,4.,7.], 'flag':['x','y','y'],'unused':['wide'*500]*3})
            frame=pd.concat([frame]*100,ignore_index=True)
            info=r.datasets.register_batches([frame],columns=list(frame.columns),source='arbitrary.dynamic',
                max_rows=500,coverage='complete',predicate_known=True)
            r.select_dataset(info.id)
            try:
                with patch.object(FrameCache,'__getitem__',side_effect=AssertionError('full root read')), \
                     patch.object(FrameCache,'project',side_effect=AssertionError('column decode during planning')):
                    state,_=r.recovery._state({'messages':[HumanMessage(content='각 segment별 measure 평균과 건수를 계산해줘',id='groups')]})
                    self.assertTrue(state['group_summary_requested'])
                    self.assertEqual(state['group_summary_columns'],['segment'])
                    self.assertEqual({m['aggregation'] for m in state['group_summary_metrics']},{'mean','count'})
                    self.assertTrue(r.recovery._numeric_column_known('measure',state))
                    self.assertFalse(r.recovery._numeric_column_known('flag',state))
                self.assertEqual(r.datasets.frames.bytes,0)
            finally:r.close()
