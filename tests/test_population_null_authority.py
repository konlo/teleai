"""A population reviewer cannot invent a missing-value filter for a mean."""
from contextlib import ExitStack
import json,tempfile,unittest
from unittest.mock import patch
import pandas as pd
from jsonschema import Draft202012Validator,ValidationError
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration,ChatResult
from core.analysis_agent.population_audit import model_for
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel,goal
from utils.analysis_datasets import stored_dataset_digest

class BadNullAudit(GoalModel):
    def _generate(self,messages,**kwargs):
        current=json.loads(next(m.content for m in messages if isinstance(m.content,str) and m.content.startswith('{')))
        value={'change':'modify','evidence_quote':current['CURRENT_USER_REQUEST'],
            'included_values':[],'removed_columns':[],
            'conditions':[{'column':'reading','op':'is_null','value':None}],'any_conditions':[]}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(value)))])

class PopulationNullAuthorityTests(unittest.TestCase):
    def test_schema_blocks_new_null_role_and_preserves_declared_null_role(self):
        scope={'conditions':[{'column':'reading','op':'ge','value':10}],'any_conditions':[]}
        proposed={**scope}
        candidate={'change':'modify','evidence_quote':'그중 reading 평균을 알려줘',
            'included_values':[],'removed_columns':[],
            'conditions':[{'column':'reading','op':'is_null','value':None}],'any_conditions':[]}
        with patch('core.analysis_agent.model_roles.json_role',side_effect=lambda model,schema,*args:schema):
            ordinary=model_for(None,scope,candidate['evidence_quote'],proposed)
            declared=model_for(None,scope,candidate['evidence_quote'],{'conditions':candidate['conditions'],'any_conditions':[]})
        with self.assertRaises(ValidationError):Draft202012Validator(ordinary).validate(candidate)
        Draft202012Validator(declared).validate(candidate)

    def test_hallucinated_null_audit_cannot_override_correct_goal_after_count(self):
        with tempfile.TemporaryDirectory() as root:
            conditions=[{'column':'reading','op':'ge','value':10}]
            count=goal('row_count',conditions=conditions)
            mean=goal('calculation',{'operations':['AVG']},columns=['reading'],conditions=conditions)
            mean['source_reference']='previous_analysis'
            model=GoalModel(goals=[count,count,mean,mean],calls=[
                {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT COUNT(*) FROM data','requested_conditions':conditions}},
                {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT AVG(reading) FROM data','requested_conditions':conditions}}])
            r=GraphAnalysisRuntime(root,'owner','null-audit',model)
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            before=stored_dataset_digest(r.datasets,raw.id)
            try:
                for prompt,expected in [('reading >= 10인 행의 건수를 알려줘',2.),('그중 reading 평균을 알려줘',15.)]:
                    if expected==15.:r.recovery.goal_interpreter.population_model=BadNullAudit()
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
                        r.close();r=GraphAnalysisRuntime(root,'owner','null-audit',model)
                events=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                self.assertTrue(any(e['event']=='goal_population_audit_conflict' and e.get('resolution')=='retain_unchanged_reviewed_population' for e in events))
            finally:r.close()
