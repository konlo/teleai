"""Supported decisions are grounded in text/facts, without regex intent routing."""
import tempfile
import unittest
from copy import deepcopy
from core.analysis_agent.goal_grounding import required_paths, verify, schema_with_evidence
from core.analysis_agent.goal_contract import response_schema
from tests.test_llm_goal import goal, GoalModel, LLMGoalTests


def evidence(plan, request):
    return [{'path':p,'origin':'request','quote':request,'reference':''} for p in required_paths(plan)]


class GoalDecisionEvidenceTests(unittest.TestCase):
    def test_quote_must_be_real_current_and_cover_every_decision(self):
        request='Show four records from lab.observations where reading >= 3'
        plan=goal('row_preview',{'limit':4},columns=['reading'],conditions=[{'column':'reading','op':'ge','value':3}])
        entries=evidence(plan,request)
        proof=verify(plan,entries,{'request':request})
        self.assertEqual(proof['status'],'supported')
        self.assertFalse(proof['linguistic_entailment_proven'])
        for invalid in (entries[:-1],entries+[entries[0]],
                        [{**e,'quote':'An invented request'} for e in entries]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):verify(plan,invalid,{'request':request})
        grammar=schema_with_evidence(response_schema())
        self.assertIn('decision_evidence',grammar['required'])
        self.assertNotIn('decision_evidence',response_schema()['properties'])
        bound=schema_with_evidence(response_schema(),plan)['properties']['decision_evidence']
        self.assertEqual(bound['type'],'object')
        self.assertEqual(bound['required'],required_paths(plan))
        self.assertFalse(bound['additionalProperties'])

    def test_inherited_condition_requires_exact_observed_value_and_type(self):
        request='Keep that condition and show four rows'
        predicate={'column':'reading','op':'ge','value':3}
        plan=goal('row_preview',{'limit':4},conditions=[predicate])
        entries=evidence(plan,request)
        entries[-1]={'path':'/conditions/0','origin':'verified_previous','quote':'',
                     'reference':'/scope/conditions/0'}
        data={'request':request,'verified_previous':{'scope':{'conditions':[predicate]}}}
        self.assertEqual(verify(plan,entries,data)['status'],'supported')
        for wrong in (4,True):
            changed=deepcopy(plan);changed['conditions'][0]['value']=wrong
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):verify(changed,entries,data)
        data['verified_previous']['scope']['conditions']=[]
        with self.assertRaises(ValueError):verify(plan,entries,data)

    def test_provider_schema_uses_required_object_keys_and_real_support_choices(self):
        import jsonschema
        request='Keep that condition and show four rows'
        predicate={'column':'reading','op':'ge','value':3}
        plan=goal('row_preview',{'limit':4},conditions=[predicate])
        data={'request':request,'verified_previous':{'scope':{'conditions':[predicate]}}}
        grammar=schema_with_evidence(response_schema(),plan,data)['properties']['decision_evidence']
        entries={e['path']:{k:v for k,v in e.items() if k!='path'} for e in evidence(plan,request)}
        entries['/conditions/0']={'origin':'verified_previous','quote':'','reference':'/scope/conditions/0'}
        jsonschema.validate(entries,grammar)
        self.assertEqual(verify(plan,entries,data)['status'],'supported')
        for bad in ('/scope/conditions/1','/invented'):
            invalid=deepcopy(entries);invalid['/conditions/0']['reference']=bad
            with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(invalid,grammar)
        invalid=deepcopy(entries);invalid['/tasks/0']['quote']='not present'
        with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(invalid,grammar)

    def test_tasks_cannot_be_requested_by_old_context_and_flags_need_support(self):
        request='Draw only these displayed rows'
        plan=goal('chart',{'kind':'histogram'},columns=['reading']);plan['current_result_only']=True
        entries=evidence(plan,request)
        self.assertIn('/current_result_only',required_paths(plan))
        entries[0].update(origin='verified_previous',quote='',reference='/task')
        with self.assertRaises(ValueError):verify(plan,entries,{'request':request,'verified_previous':{'task':plan['tasks'][0]}})

    def test_real_graph_accepts_supported_goal_and_rejects_fabricated_support_without_execution(self):
        request='observations table row 4개 보여줘'
        for valid in (True,False):
            with self.subTest(valid=valid),tempfile.TemporaryDirectory() as root:
                plan=goal('row_preview',{'limit':4})
                plan['decision_evidence']=evidence(plan,request if valid else 'invented quote')
                runtime,raw=LLMGoalTests().runtime(root,GoalModel(goals=[plan]))
                try:
                    out=runtime.submit(request);state=runtime.inspect()['recovery']
                    if valid:
                        self.assertEqual(out['status'],'answered',out)
                        proof=state['goal_decision_evidence']
                        stored=runtime.plans.inspect(state['request_id'])
                        self.assertEqual(stored['plan']['decision_evidence'],proof)
                        self.assertEqual(stored['plan']['status'],'complete')
                        self.assertTrue(any(e['kind']=='task_verified' for e in stored['recent_events']))
                        altered=deepcopy(state);altered['goal']['tasks'][0]['options']['limit']=20
                        self.assertEqual(runtime.plans.admission(altered,'inspect_dataset')['error_code'],
                                         'goal_decision_support_stale')
                        altered=deepcopy(state);altered['request_text']='another request'
                        self.assertEqual(runtime.plans.admission(altered,'inspect_dataset')['error_code'],
                                         'goal_decision_support_stale')
                    else:
                        self.assertNotEqual(out['status'],'answered',out)
                        self.assertFalse(state['sent_calls'])
                    self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                    self.assertEqual(len(runtime.datasets.metadata),1)
                finally:runtime.close()
