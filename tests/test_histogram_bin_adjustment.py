"""A chart adjustment rebins verified data without replacing its population."""
import tempfile,unittest
import pandas as pd
import json
from uuid import uuid4
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.goal_contract import validate_goal
from core.analysis_agent.task_selection import verify_goal
from tests.test_llm_goal import GoalModel,goal
from utils.analysis_datasets import stored_dataset_digest

class FrequencyChoiceModel(GoalModel):
    """Choose a renderer from an observed frequency receipt, not prose routing."""
    frequency_selected: bool=False

    def _generate(self,messages,**kwargs):
        if (not self.frequency_selected and self.position>=len(self.calls)
                and any(m.additional_kwargs.get('lc_source')=='action_cycle' for m in messages)):
            for message in reversed(messages):
                if not isinstance(message,ToolMessage) or message.name!='local_analysis_sql':continue
                observed=json.loads(message.content)
                if observed.get('status')!='ready':continue
                self.frequency_selected=True
                call={'name':'render_histogram','id':str(uuid4()),'args':{
                    'dataset_id':observed['dataset']['id'],'value_column':'reading',
                    'weight_column':'__frequency','bins':8}}
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[call]))])
        return super()._generate(messages,**kwargs)

class HistogramBinAdjustmentTests(unittest.TestCase):
    def test_rebin_is_a_supported_edit_and_preserves_original(self):
        histogram=goal('chart',{'kind':'histogram','axes':{'x':'reading'},'bins':8},columns=['reading'])
        adjustment=goal('chart_adjust',{'bins':5})
        model=FrequencyChoiceModel(goals=[histogram,histogram,adjustment,adjustment],advisory_after_calls=True,
            calls=[{'name':'local_analysis_sql','args':{'dataset_id':'$fixture',
                'query':'SELECT reading, COUNT(*) AS __frequency FROM data GROUP BY reading'}}])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','bins',model,sql_dialect='mysql')
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);before=stored_dataset_digest(r.datasets,raw.id)
            model.evaluation_dataset_id=raw.id
            try:
                self.assertEqual(r.submit('reading 히스토그램을 8개 구간으로 그려줘')['status'],'answered')
                previous=r.inspect()['recovery']['artifact_ids'][-1]
                result=r.submit('방금 히스토그램의 구간을 5개로 바꿔줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery'];card=r.context.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(card.render_spec['bins'],5)
                self.assertEqual(card.render_spec['total_count'],4)
                self.assertIn(previous,r.context.artifacts)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                self.assertFalse(state['remote_query_ids'])
            finally:r.close()
        for bins in [0,1,101,5.5,True]:
            with self.subTest(bins=bins),self.assertRaises(ValueError):validate_goal(goal('chart_adjust',{'bins':bins}))

    def test_selected_edit_keys_cannot_be_replaced_with_axis_limit(self):
        selection={'mode':'execute','capabilities':['chart_adjust'],'source_reference':'explicit',
            'current_result_only':False,'source_mentions':[],'chart_edit_fields':['bins']}
        with self.assertRaises(ValueError):verify_goal(goal('chart_adjust',{'y_max':5}),selection,None)
