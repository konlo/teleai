"""Fault injection must exercise planner repair, not corrupt intent JSON."""
import unittest
from scripts.evaluate_autonomous_paths import evaluate_case
from tests.test_llm_goal import GoalModel,goal


class AutonomyFaultPhaseTests(unittest.TestCase):
    def test_fault_is_injected_after_verified_intent(self):
        model=GoalModel(goals=[goal('calculation',{'operations':['AVG']},
            sources=['unfamiliar.observations'],columns=['reading'])])
        result=evaluate_case({'id':'phase_contract','prompt':'Compute the mean',
            'expected':9.,'first_call':{'name':'local_analysis_sql','args':{
                'dataset_id':'$raw','query':'SELECT AVG(nonexistent) FROM data'}}},model)
        turn=result['turns'][0]
        self.assertEqual(result['fault_phase'],'execution_planner_only')
        self.assertEqual(turn['goal']['tasks'][0]['capability'],'calculation')
        self.assertEqual(turn['injected_calls'],1)
        self.assertEqual(turn['phase_order'][0],'goal_interpretation_completed')
        self.assertIn('local_analysis_sql',turn['executed_tools'])
        self.assertTrue(turn['raw_unchanged'])
        self.assertEqual(turn['remote_executions'],0)


if __name__=='__main__':unittest.main()
