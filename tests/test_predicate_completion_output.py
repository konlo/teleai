"""Supported exclusion predicates must survive execution through final text."""
import tempfile,unittest
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import goal,GoalModel
from utils.analysis_datasets import stored_dataset_digest

class PredicateCompletionOutputTests(unittest.TestCase):
    def test_exclusion_mean_is_published_without_losing_the_raw_data(self):
        plan=goal('calculation',{'operations':['AVG']},columns=['reading'],conditions=[
            {'column':'label','op':'not_in','value':['red']}])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','exclusion-output',GoalModel(goals=[plan]),sql_dialect='mysql')
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.],'label':['red','red','blue','blue']}),
                source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id)
            try:
                result=r.submit('Exclude red and compute the reading mean')
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('평균: 15.0',result['text'])
                self.assertIn('label 제외',result['text'])
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
            finally:r.close()
