"""A model call limit must not discard its last valid tool response."""
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel, goal
from utils.analysis_datasets import stored_dataset_digest

class LastModelToolAdmissionTests(unittest.TestCase):
    def test_last_permitted_response_runs_valid_sql_without_another_model_call(self):
        for column, expected in [('reading','answered'),('other','exhausted')]:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as root:
                model=GoalModel(goals=[goal('calculation',{'operations':['AVG']},columns=['reading'])],
                    calls=[{'name':'local_analysis_sql','args':{'dataset_id':'$fixture',
                        'query':f'SELECT AVG({column}) AS average FROM data'}}])
                r=GraphAnalysisRuntime(root,'owner','last-model-tool',model,sql_dialect='mysql')
                raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.],'other':[100.,200.,300.,400.]}),
                    source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
                digest=stored_dataset_digest(r.datasets,raw.id)
                r.recovery.max_model_calls=3;r.model_recovery.max_calls=3
                try:
                    with ExitStack() as stack:
                        for method in ['_next_local','_budget_local_rescue','_cached_chart_call']:
                            stack.enter_context(patch.object(r.recovery,method,return_value=None))
                        result=r.submit('reading 평균을 알려줘')
                    self.assertEqual(result['status'],expected,result)
                    self.assertEqual(model.position,1)
                    state=r.inspect()['recovery'];self.assertEqual(state['model_calls'],3)
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                    if column=='reading':
                        frame=r.datasets.frames[state['evidence_ids'][-1]]
                        self.assertEqual(float(frame.iloc[0,0]),9.)
                    else:
                        self.assertEqual(state['stop_reason'],'model_call_budget')
                        # A read-only intermediate result is allowed, but the
                        # wrong requested measure must never count as evidence.
                        self.assertFalse(state['evidence_ids'])
                finally:r.close()
