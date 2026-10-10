"""Plan history survives sync/reopen and describes actual observations only."""
from copy import deepcopy
import unittest
from tests.test_analysis_extensions import AnalysisExtensionsTests,rolling_contract
from core.analysis_agent.execution_plan import ExecutionPlans
from core.analysis_agent.python_result_contract import digest


class PlanObservationHistoryTests(unittest.TestCase):
    setUp=AnalysisExtensionsTests.setUp
    tearDown=AnalysisExtensionsTests.tearDown

    def scenario_state(self):
        return {**self.state,'goal':{'tasks':[{'capability':'custom_analysis','options':{
            'description':'rolling','result_contract':rolling_contract()}},
            {'capability':'export','options':{'format':'csv'}}]},'export_spec':{'format':'csv'}}

    def test_dependency_revision_survives_sync_and_noop_is_not_a_revision(self):
        plans=ExecutionPlans(self.db);state=self.scenario_state();plan=plans.sync(state)
        a,b=[t['id'] for t in plan['tasks']];edges=[{'task_id':b,'depends_on':[a]}]
        result=plans.dependencies('request1',edges)
        self.assertGreater(result['plan']['revision'],plan['revision'])
        self.assertEqual(plans.dependencies('request1',edges)['plan']['revision'],result['plan']['revision'])
        refreshed=plans.sync(state)
        self.assertEqual(refreshed['tasks'][1]['depends_on'],[a])
        self.assertEqual(refreshed['revision'],result['plan']['revision'])
        self.assertEqual(sum(e['kind']=='dependencies_revised' for e in plans.history('request1')),1)
        with self.assertRaises(ValueError):plans.dependencies('request1',[{'task_id':a,'depends_on':[b]}])

    def test_failure_changed_code_verified_output_history_is_durable_without_code_text(self):
        plans=ExecutionPlans(self.db);state=self.scenario_state();plans.sync(state)
        first={'id':'failed-call','name':'execute_analysis_python','args':{'code':'result=df.copy()'}}
        state['sent_calls']=['failed-call'];state['failed_signatures']={'signature':1}
        state['tool_failures']={'signature':{'tool':first['name'],'status':'error',
            'error_code':'semantic_mismatch','call_id':'failed-call'}}
        plans.sync(state,{'failed-call':first});count=len(plans.history('request1'))
        plans.sync(state,{'failed-call':first});self.assertEqual(len(plans.history('request1')),count)
        inspection={'id':'inspect-call','name':'inspect_dataset','args':{'dataset_id':self.raw.id}}
        state['sent_calls'].append(inspection['id'])
        plans.sync(state,{'failed-call':first,'inspect-call':inspection})
        second={'id':'corrected-call','name':first['name'],'args':{'code':'corrected private code'}}
        state['sent_calls'].append(second['id'])
        calls={'failed-call':first,'inspect-call':inspection,'corrected-call':second}
        plans.sync(state,calls)
        state['custom_analysis_evidence']={'output_dataset_id':self.raw.id,'semantic_verified':True,
                                           'result_contract_hash':digest(rolling_contract())}
        plans.sync(state,calls)
        events=ExecutionPlans(self.db).inspect('request1')['recent_events']
        kinds=[e['kind'] for e in events]
        self.assertLess(kinds.index('tool_failure_observed'),kinds.index('tool_plan_changed'))
        self.assertLess(kinds.index('tool_plan_changed'),kinds.index('task_verified'))
        self.assertNotIn('private code',str(events));self.assertNotIn('df.copy()',str(events))
        self.assertTrue(next(e for e in events if e['kind']=='tool_plan_changed')['goal_preserved'])
        before=deepcopy(events);plans.sync(state,calls)
        self.assertEqual(plans.history('request1'),before)
        self.assertEqual(plans.history('different-request'),[])
