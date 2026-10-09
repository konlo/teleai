"""Catalog growth cannot prevent intent inference or lose a pending request."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage
from core.analysis_agent.goal_interpreter import GoalInterpreter, INSTRUCTIONS, protocol_examples
from core.analysis_agent.model_context import ModelContextBudgetMiddleware, ModelContextBudgetExceeded
from tests.test_model_context_budget import Request
from tests import test_llm_goal as llm_goals
from tests.test_llm_goal import GoalModel, goal


class GoalInputBudgetTests(unittest.TestCase):
    def test_inventory_then_explicit_schema_fits_real_16k_budget(self):
        refs=[{'table':f'workspace.catalog.table_{i}', 'columns':[],
               'training_status':'discovered_name'} for i in range(200)]
        refs[-1]['columns']=[{'name':f'field_{i}','dtype':'varchar'} for i in range(100)]
        refs[-1]['training_status']='runtime_schema'
        context=SimpleNamespace(reference_context=refs,selected_dataset_id='',
            datasets=SimpleNamespace(metadata={}),source_namespace='workspace')
        interpreter=GoalInterpreter.__new__(GoalInterpreter);interpreter.context=context
        current={'request_id':'now','request_text':'table_199 컬럼들을 보여줘',
            'confirmed_analysis':{'status':'complete','required_sources':['workspace.information_schema.tables'],
                'scope':{'conditions':[]},'request_text':'지금 볼 수있는 table를 보여줘'}}
        payload=interpreter.payload(current,[])
        self.assertEqual(payload['tables'][0]['table'],refs[-1]['table'])
        self.assertEqual(payload['table_count'],200)
        self.assertEqual(len(payload['available_table_names']),32)
        self.assertEqual(len(refs[-1]['columns']),100)
        req=Request([HumanMessage(content=json.dumps(payload,ensure_ascii=False,separators=(',',':')))],[],
            {'recovery':current},SystemMessage(content=INSTRUCTIONS+'\nProtocol examples:\n'+protocol_examples()))
        projected=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=2048)).wrap_model_call(req,lambda r:r)
        self.assertEqual(json.loads(projected.messages[0].content)['request'],current['request_text'])

    def test_failure_before_inference_keeps_request_confirmed_goal_and_restart_resume(self):
        with tempfile.TemporaryDirectory() as root:
            model=GoalModel(goals=[goal('metadata',{'kind':'columns'})])
            harness=llm_goals.LLMGoalTests();r,raw=harness.runtime(root,model)
            try:
                self.assertEqual(r.submit('observations 컬럼을 보여줘')['status'],'answered')
                confirmed=r.inspect()['recovery']['confirmed_analysis']
                assets=set(r.datasets.metadata)
                with patch.object(r.recovery.goal_interpreter,'interpret',side_effect=ModelContextBudgetExceeded('fixture')):
                    failed=r.submit('같은 테이블 컬럼을 다시 보여줘')
                self.assertEqual(failed['error_type'],'ModelContextBudgetExceeded')
                state=r.inspect()['recovery']
                self.assertEqual(state['request_text'],'같은 테이블 컬럼을 다시 보여줘')
                self.assertTrue(state['goal_pending'])
                self.assertNotEqual(state['request_id'],confirmed['request_id'])
                for key in ('request_id','required_sources','scope'):
                    self.assertEqual(state['confirmed_analysis'][key],confirmed[key])
                self.assertNotIn('confirmed_analysis',state['confirmed_analysis'])
                self.assertEqual(set(r.datasets.metadata),assets)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                from core.analysis_agent.runtime import GraphAnalysisRuntime
                from tests.test_row_preview import schema
                r.close();r=GraphAnalysisRuntime(root,'goal-test','conversation',model,sql_dialect='mysql',
                    reference_context_loader=lambda:schema('lab.observations'))
                self.assertEqual(r.inspect()['recovery']['request_text'],state['request_text'])
                self.assertEqual(r.resume()['status'],'answered')
                self.assertEqual(set(r.datasets.metadata),assets)
            finally:r.close()
