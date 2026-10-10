"""Repair presentation input separately from semantic interpretation retries."""
from contextlib import ExitStack
import tempfile, unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel, goal
from utils.analysis_datasets import stored_dataset_digest

class ChartInputRecoveryTests(unittest.TestCase):
    def test_compound_mean_and_chart_repairs_missing_legend_after_goal_retries(self):
        with tempfile.TemporaryDirectory() as root:
            g=goal('calculation',{'operations':['AVG']},columns=['reading'])
            g['tasks'].append({'capability':'chart','options':{'kind':'histogram','axes':{'x':'reading'},'legend':True}})
            model=GoalModel(goals=[g],calls=[
                {'name':'aggregate_dataset','args':{'dataset_id':'$fixture','aggregation':'mean','value_column':'reading'}},
                {'name':'prepare_histogram','args':{'source':'lab.observations','column':'reading','dataset_id':'$fixture'}},
                {'name':'prepare_histogram','args':{'source':'lab.observations','column':'reading','dataset_id':'$fixture','legend':True}}])
            r=GraphAnalysisRuntime(root,'owner','presentation',model)
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            before=stored_dataset_digest(r.datasets,raw.id)
            original=r.recovery.goal_interpreter.interpret
            def semantic_retries(*args,**kwargs):
                result=original(*args,**kwargs);result['attempts']=2;return result
            try:
                with ExitStack() as stack:
                    for method in ['_next_local','_budget_local_rescue','_cached_chart_call']:
                        stack.enter_context(patch.object(r.recovery,method,return_value=None))
                    stack.enter_context(patch.object(r.recovery.goal_interpreter,'interpret',side_effect=semantic_retries))
                    result=r.submit('reading 평균과 reading 히스토그램을 범례와 함께 보여줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                values=[float(r.datasets.frames[i].iloc[0,0]) for i in state['evidence_ids'] if r.datasets.frames[i].shape==(1,1)]
                self.assertIn(9.,values)
                card=r.artifacts[state['artifact_ids'][0]]
                self.assertTrue(card.render_spec['legend'])
                self.assertEqual(card.render_spec['legend_labels'],['reading'])
                self.assertEqual(sum(card.render_spec['bin_counts']),4)
                np.testing.assert_array_equal(card.render_spec['bin_counts'],np.histogram([2.,4.,10.,20.],bins=card.render_spec['bin_edges'])[0])
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                messages=r.agent.get_state(r.config).values['messages']
                feedback=[m for m in messages if m.additional_kwargs.get('lc_source')=='proposal_preflight']
                self.assertTrue(feedback)
                self.assertIn('chart_presentation_mismatch',str(feedback[-1].content))
                self.assertIn('legend',str(feedback[-1].content))
                self.assertNotIn('Rewrite the SELECT',str(feedback[-1].content))
            finally:r.close()

    def test_repeated_wrong_legend_stops_without_false_chart_completion(self):
        with tempfile.TemporaryDirectory() as root:
            g=goal('chart',{'kind':'histogram','axes':{'x':'reading'},'legend':True},columns=['reading'])
            wrong={'name':'prepare_histogram','args':{'source':'lab.observations','column':'reading','dataset_id':'$fixture'}}
            model=GoalModel(goals=[g],calls=[wrong,wrong])
            r=GraphAnalysisRuntime(root,'owner','repeat',model)
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            before=stored_dataset_digest(r.datasets,raw.id)
            try:
                with ExitStack() as stack:
                    for method in ['_next_local','_budget_local_rescue','_cached_chart_call']:
                        stack.enter_context(patch.object(r.recovery,method,return_value=None))
                    result=r.submit('reading 히스토그램에 범례를 보여줘')
                self.assertIn(result['status'],{'blocked','exhausted'})
                state=r.inspect()['recovery']
                self.assertEqual(state['stop_reason'],'proposal_validation_failed')
                self.assertEqual(state['proposal_error'],'chart_presentation_mismatch')
                self.assertEqual(model.position,2)
                self.assertEqual(len(r.artifacts),0)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
            finally:r.close()
