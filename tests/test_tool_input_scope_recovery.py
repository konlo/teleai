"""Malformed tool arguments must be repaired before SQL population validation."""
from contextlib import ExitStack
import tempfile,unittest
from unittest.mock import patch
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel,goal
from utils.analysis_datasets import stored_dataset_digest

class ToolInputScopeRecoveryTests(unittest.TestCase):
    def test_count_query_keys_repaired_then_restart_keeps_filter_for_mean(self):
        with tempfile.TemporaryDirectory() as root:
            conditions=[{'column':'reading','op':'ge','value':10}]
            count=goal('row_count',conditions=conditions)
            mean=goal('calculation',{'operations':['AVG']},columns=['reading'],conditions=conditions)
            mean['source_reference']='previous_analysis'
            model=GoalModel(goals=[count,count,mean,mean],calls=[
                {'name':'local_analysis_sql','args':{'sql':'SELECT COUNT(*) AS n FROM data WHERE reading >= 10'}},
                {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT COUNT(*) AS n FROM data WHERE reading >= 10'}},
                {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT AVG(reading) AS mean FROM data WHERE reading >= 10'}}])
            r=GraphAnalysisRuntime(root,'owner','keys',model)
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            before=stored_dataset_digest(r.datasets,raw.id)
            try:
                for prompt,expected in [('reading >= 10인 행의 건수를 알려줘',2.),('그중 reading 평균을 알려줘',15.)]:
                    with ExitStack() as stack:
                        for method in ['_next_local','_budget_local_rescue','_cached_chart_call']:
                            stack.enter_context(patch.object(r.recovery,method,return_value=None))
                        result=r.submit(prompt)
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery']
                    self.assertEqual(state['scope']['conditions'],conditions)
                    self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),expected)
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                    if expected==2.:
                        feedback=[m for m in r.agent.get_state(r.config).values['messages'] if m.additional_kwargs.get('lc_source')=='proposal_preflight']
                        self.assertTrue(feedback)
                        self.assertIn('invalid_tool_arguments',str(feedback[-1].content))
                        self.assertIn('query',str(feedback[-1].content))
                        r.close();r=GraphAnalysisRuntime(root,'owner','keys',model)
            finally:r.close()

    def test_offline_filtered_scalar_menu_retains_local_sql(self):
        from types import SimpleNamespace
        from core.analysis_agent.model_context import ProgressiveToolsMiddleware
        from tests.test_model_context_budget import Request
        from langchain_core.messages import HumanMessage,SystemMessage
        names=['query_databricks','inspect_table_context','list_analysis_context','search_analysis_tools',
               'read_analysis_skill','aggregate_dataset','local_analysis_sql','use_dataset','prepare_numeric_dataset',
               'render_histogram','pivot_dataset','summarize_groups','profile_dataset']
        tools=[SimpleNamespace(name=n) for n in names]
        request=Request([HumanMessage(content='calculate')],tools,{'recovery':{'calculation':True,'required_sources':['lab.observations']}},SystemMessage(content='policy'))
        local=ProgressiveToolsMiddleware(tools,remote_available=False).wrap_model_call(request,lambda r:r)
        remote=ProgressiveToolsMiddleware(tools,remote_available=True).wrap_model_call(request,lambda r:r)
        self.assertIn('local_analysis_sql',{t.name for t in local.tools})
        self.assertNotIn('local_analysis_sql',{t.name for t in remote.tools})
