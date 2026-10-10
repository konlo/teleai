"""Unknown handles are repairable input errors, never silent rebindings."""
import tempfile, unittest
from contextlib import ExitStack
from unittest.mock import patch
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel, goal
from utils.analysis_datasets import stored_dataset_digest

class DatasetHandleRecoveryTests(unittest.TestCase):
    def test_unknown_handle_is_rejected_then_model_proposes_registered_handle(self):
        with tempfile.TemporaryDirectory() as root:
            model=GoalModel(goals=[goal('calculation',{'operations':['AVG']},columns=['reading'])],calls=[
                {'name':'aggregate_dataset','args':{'dataset_id':'invented-handle','aggregation':'mean','value_column':'reading'}},
                {'name':'aggregate_dataset','args':{'dataset_id':'$fixture','aggregation':'mean','value_column':'reading'}}])
            r=GraphAnalysisRuntime(root,'owner','handles',model,sql_dialect='mysql')
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            before=stored_dataset_digest(r.datasets,raw.id)
            try:
                with ExitStack() as stack:
                    for method in ['_next_local','_budget_local_rescue','_cached_chart_call']:
                        stack.enter_context(patch.object(r.recovery,method,return_value=None))
                    result=r.submit('reading 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(state['attempts'],1)
                self.assertEqual(model.position,2)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,-1]),9.)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                messages=r.agent.get_state(r.config).values['messages']
                feedback=[m for m in messages if m.additional_kwargs.get('lc_source')=='proposal_preflight']
                self.assertTrue(feedback)
                self.assertIn('unknown_dataset_id',str(feedback[-1].content))
                self.assertIn(raw.id,str(feedback[-1].content))
            finally:r.close()
