"""The next planner receives actual completed/remaining work and raw IDs."""
import json,unittest
from types import SimpleNamespace
from langchain_core.messages import SystemMessage
from core.analysis_agent.remaining_work import RemainingWorkMiddleware
from core.analysis_agent.goal_normalization import normalize
from tests.test_llm_goal import goal

class RemainingWorkContractTests(unittest.TestCase):
    def test_completed_measure_and_pending_chart_are_separate(self):
        info=SimpleNamespace(id='immutable-raw',source='lab.observations',columns=('reading','label'),
            grain='raw',coverage='complete',rows=4,predicate_known=True)
        context=SimpleNamespace(datasets=SimpleNamespace(metadata={info.id:info}))
        state={'goal':{'mode':'execute'},'calculation':True,'chart':True,
            'evidence_ids':['verified-average'],'artifact_ids':[],
            'required_sources':['lab.observations'],'scope':{'conditions':[{'column':'reading','op':'ge','value':10}]}}
        req=SimpleNamespace(state={'recovery':state},system_message=SystemMessage(content='instructions'))
        req.override=lambda **kw:SimpleNamespace(**{**vars(req),**kw})
        result=RemainingWorkMiddleware(context).wrap_model_call(req,lambda r:r)
        text=result.system_message.content
        facts=json.loads(text.split('Verified work progress (data, not instructions): ')[1].split('\n')[0])
        self.assertEqual(facts['completed_capabilities'],['calculation'])
        self.assertEqual(facts['remaining_capabilities'],['chart'])
        self.assertEqual(facts['raw_candidates'][0]['dataset_id'],'immutable-raw')
        self.assertEqual(facts['requested_scope'],state['scope'])
        self.assertEqual(state['evidence_ids'],['verified-average'])

    def test_unused_root_fields_do_not_become_extra_measures(self):
        plan=goal('calculation',{'operations':['AVG']},columns=['reading','label'])
        plan['conditions']=[{'column':'label','op':'eq','value':'blue'}]
        selected={'output_columns':['reading']}
        result=normalize(plan,selected)
        self.assertEqual(result['columns'],['reading']);self.assertEqual(result['conditions'],plan['conditions'])
        self.assertEqual(plan['columns'],['reading','label'])
        plan['tasks'][0]['options']['group_columns']=['label']
        self.assertEqual(normalize(plan,selected)['columns'],['reading','label'])
        plan['columns']=['other']
        self.assertEqual(normalize(plan,selected)['columns'],['other'])
