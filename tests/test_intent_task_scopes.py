"""Scoped plan/failure/budget contracts; scripted decisions are not LLM scores."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.outputs import ChatGeneration,ChatResult

from tests.test_llm_goal import GoalModel,goal
from tests.test_row_preview import DATA,schema
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.goal_contract import pending_state
from core.analysis_agent.intent_plan import validate
from core.analysis_agent.task_execution import start

FIXTURE=json.loads((Path(__file__).parent/'fixtures/intent_task_scopes.json').read_text())

def step(key,cap,opts,column=None,quote=None,terms=None,basis='source_population',deps=None,ref=''):
    g=goal(cap,opts,sources=[FIXTURE['source']],columns=[column] if column else [],conditions=terms)
    g['source_reference']='explicit' if 'observations' in (quote or FIXTURE['request']) else 'previous_analysis'
    for field in ('objective','mode','question','conditions','any_conditions','current_result_only','ratio','measure_conditions'):g.pop(field,None)
    q=quote or FIXTURE['request']
    review=[]
    if basis=='unfiltered_source':
        review=[{'quote':q,'kind':'source_scope','columns':[],'status':'resolved','scope_basis':basis}]
    elif terms:review=[{'quote':q,'kind':'predicate','columns':[t['column'] for t in terms],'status':'resolved'}]
    return {'id':key,'depends_on':deps or [],'request_quote':q,'goal':g,
        'population':{'basis':basis,'output_reference_id':ref},
        'population_delta':{'conditions':terms or [],'any_conditions':[],
            'change':'clear' if basis=='unfiltered_source' else 'modify' if terms else 'keep',
            'removed_columns':[],'included_values':[]},'restriction_review':{'clauses':review}}


def plan(steps):return {'objective':FIXTURE['request'],'mode':'execute','question':'','steps':steps}

class PlanModel(GoalModel):
    plan_responses: list=[]
    plan_calls: int=0
    def _generate(self,messages,**kwargs):
        if any('intent_plan_outline_v2:' in str(m.content) for m in messages):
            self.plan_calls+=1
            value=deepcopy(self.plan_responses[0])
            value['steps']=[{k:s[k] for k in ('id','depends_on','request_quote')}|{'capability':s['goal']['tasks'][0]['capability'],'grouped':s['goal']['tasks'][0]['options'].get('grouped',False),'scope_change':('clear' if s['population']['basis']=='unfiltered_source' else 'filter' if s['population_delta']['conditions'] else 'keep')} for s in value['steps']]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(value)))])
        if any('intent_plan_v2:' in str(m.content) for m in messages):
            self.plan_calls+=1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(
                self.plan_responses[min(self.plan_calls-1,len(self.plan_responses)-1)])))])
        return super()._generate(messages,**kwargs)

class InterruptedPlanModel(PlanModel):
    def _generate(self,messages,**kwargs):
        for message in reversed(messages):
            if message.additional_kwargs.get('lc_source')=='action_cycle':
                view=json.loads('{'+str(message.content).split('\n{',1)[1])
                if view.get('task_plan',{}).get('active_step')=='s2':
                    raise RuntimeError('injected model interruption after verified preview')
                break
        return super()._generate(messages,**kwargs)

class TaskScopeTests(unittest.TestCase):
    def runtime(self,root,model):
        r=GraphAnalysisRuntime(root,'scoped-test','conversation',model,sql_dialect='mysql',
            reference_context_loader=lambda:schema(FIXTURE['source']),source_namespace='lab')
        raw=r.datasets.register(DATA,source=FIXTURE['source'],coverage='complete',predicate_known=True)
        r.select_dataset(raw.id)
        return r,raw

    def test_two_readings_then_model_chosen_execution_with_independent_scope(self):
        terms=[{'column':FIXTURE['column'],'op':'between','value':[3,8]}]
        spec=plan([step('s1','chart',{'kind':'histogram','axes':{'x':FIXTURE['column']},
            'legend':False,'stacked':False,'palette':'default'},FIXTURE['column'],
            FIXTURE['whole_quote'],basis='unfiltered_source'),
            step('s2','calculation',{'operations':['SUM']},FIXTURE['column'],FIXTURE['filtered_quote'],terms)])
        model=PlanModel(plan_responses=[spec])
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,model)
            try:
                r.recovery.goal_interpreter.requires_grounding=True
                with patch('core.analysis_agent.intent_plan.json_role',side_effect=lambda m,s,t:m):
                    out=r.submit(FIXTURE['request'])
                self.assertEqual(out['status'],'answered',out)
                state=r.inspect()['recovery'];done=state['intent_plan']['completed']
                self.assertEqual(model.plan_calls,2)
                self.assertEqual(len(done),2)
                self.assertEqual(done[0]['scope']['conditions'],[])
                self.assertEqual(len(done[1]['scope']['conditions']),2)
                self.assertEqual(r.datasets.frames[done[1]['evidence_ids'][-1]].iloc[0,0],FIXTURE['sum'])
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertTrue(r.datasets.frames[raw.id].equals(DATA))
                saved=r.plans.inspect(state['request_id'])['plan']
                self.assertEqual(saved['scope_mode'],'per_task')
                self.assertEqual([t['status'] for t in saved['tasks']],['verified','verified'])
                self.assertNotEqual(saved['tasks'][0]['scope'],saved['tasks'][1]['scope'])
                self.assertIn('작업 1',out['text']);self.assertIn('작업 2',out['text'])
            finally:r.close()

    def test_repeated_chart_capability_keeps_two_receipts(self):
        prompt=FIXTURE['repeat_request'];cols=['reading','event_time']
        spec=plan([step('s'+str(i+1),'chart',{'kind':'histogram','axes':{'x':c},
            'legend':False,'stacked':False,'palette':'default'},c,prompt) for i,c in enumerate(cols)])
        model=PlanModel(plan_responses=[spec])
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,model)
            try:
                r.recovery.goal_interpreter.requires_grounding=True
                with patch('core.analysis_agent.intent_plan.json_role',side_effect=lambda m,s,t:m):out=r.submit(prompt)
                self.assertEqual(out['status'],'answered',out)
                current=r.inspect()['recovery']
                self.assertEqual(len(current['artifact_ids']),2)
                self.assertEqual([d['goal']['columns'] for d in current['intent_plan']['completed']],[[c] for c in cols])
                self.assertEqual(len({d['artifact_ids'][0] for d in current['intent_plan']['completed']}),2)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_preview_dependency_uses_only_verified_ten_rows(self):
        prompt=FIXTURE['dependency_request']
        spec=plan([step('s1','row_preview',{'limit':10},quote=prompt),step('s2','chart',
            {'kind':'histogram','axes':{'x':'reading'},'legend':False,'stacked':False,'palette':'default'},
            'reading',prompt,basis='displayed_result',deps=['s1'],ref='s1')])
        model=PlanModel(plan_responses=[spec])
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,model)
            try:
                r.recovery.goal_interpreter.requires_grounding=True
                with patch('core.analysis_agent.intent_plan.json_role',side_effect=lambda m,s,t:m):out=r.submit(prompt)
                self.assertEqual(out['status'],'answered',out)
                current=r.inspect()['recovery'];done=current['intent_plan']['completed']
                self.assertEqual(done[0]['receipts']['row_preview']['rows'],10)
                self.assertEqual(done[1]['output_reference']['display_evidence']['rows'],10)
                self.assertEqual(r.datasets.metadata[raw.id].rows,20)
            finally:r.close()

    def test_interruption_restart_preserves_verified_preview_and_pending_scope(self):
        prompt=FIXTURE['dependency_request']
        spec=plan([step('s1','row_preview',{'limit':10},quote=prompt),step('s2','chart',
            {'kind':'histogram','axes':{'x':'reading'},'legend':False,'stacked':False,'palette':'default'},
            'reading',prompt,basis='displayed_result',deps=['s1'],ref='s1')])
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,InterruptedPlanModel(plan_responses=[spec]))
            try:
                r.recovery.goal_interpreter.requires_grounding=True
                with patch('core.analysis_agent.intent_plan.json_role',side_effect=lambda m,s,t:m):
                    failed=r.submit(prompt)
                self.assertEqual(failed['status'],'incomplete',failed)
                saved=r.inspect()['recovery']
                self.assertEqual([d['id'] for d in saved['intent_plan']['completed']],['s1'])
                self.assertEqual(saved['intent_plan']['active_step_id'],'s2')
                original=raw.id
                self.assertEqual(len(r.artifacts),0)
            finally:r.close()
            r=GraphAnalysisRuntime(root,'scoped-test','conversation',GoalModel(),sql_dialect='mysql',
                reference_context_loader=lambda:schema(FIXTURE['source']),source_namespace='lab')
            try:
                recovered=r.resume()
                self.assertEqual(recovered['status'],'answered',recovered)
                state=r.inspect()['recovery']
                done=state['intent_plan']['completed']
                self.assertEqual([d['id'] for d in done],['s1','s2'])
                self.assertEqual(done[0]['receipts'],saved['intent_plan']['completed'][0]['receipts'])
                self.assertEqual(done[1]['output_reference']['display_evidence']['rows'],10)
                self.assertEqual(r.context.selected_dataset_id,original)
                self.assertTrue(r.datasets.frames[original].equals(DATA))
                self.assertGreaterEqual(state['model_calls'],saved['model_calls'])
                reply=r.events()[-1]
                self.assertEqual(len(reply.additional_kwargs['analysis_table_previews']),1)
                self.assertEqual(len(reply.additional_kwargs['analysis_artifact_ids']),1)
            finally:r.close()

    def test_cyclic_or_forward_dependency_and_unquoted_predicate_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,GoalModel())
            try:
                cur=pending_state(HumanMessage(content=FIXTURE['request'],id='one'),{},r.context)
                data=r.recovery.goal_interpreter.payload(cur,[])
                valid=plan([step('s1','row_preview',{'limit':10})])
                for edit in ('forward','quote','predicate','scope','source'):
                    bad=deepcopy(valid)
                    if edit=='forward':bad['steps'][0]['depends_on']=['s2']
                    if edit=='quote':bad['steps'][0]['request_quote']='fabricated request'
                    if edit=='predicate':bad['steps'][0]['population_delta'].update(change='modify',
                        conditions=[{'column':'reading','op':'ge','value':3}])
                    if edit=='scope':bad['steps'][0]['population'].update(basis='displayed_result',output_reference_id='missing')
                    if edit=='source':bad['steps'][0]['goal']['sources']=['other_namespace.unrequested']
                    with self.subTest(edit=edit),self.assertRaises(ValueError):validate(bad,cur,data,r.context)
            finally:r.close()

    def test_provider_without_prefix_items_still_enforces_selected_task_fields(self):
        from core.analysis_agent.intent_plan import schema as plan_schema
        from jsonschema import Draft202012Validator,ValidationError
        outline={'mode':'execute','steps':[{'id':'s1','capability':'metadata','scope_change':'keep'}]}
        provider=plan_schema(outline)
        # Some grammar providers ignore prefixItems; items must independently
        # protect every step instead of becoming unconstrained JSON.
        provider['properties']['steps'].pop('prefixItems')
        valid=plan([step('s1','metadata',{'kind':'columns'})])
        Draft202012Validator(provider).validate(valid)
        for field,value in [('ratio',{'column':'reading'}),('unexpected','value')]:
            bad=deepcopy(valid);bad['steps'][0]['goal'][field]=value
            with self.subTest(field=field),self.assertRaises(ValidationError):
                Draft202012Validator(provider).validate(bad)
        bad=deepcopy(valid);bad['steps'][0]['goal']['tasks']=[{'capability':'row_preview','options':{'limit':10}}]
        with self.assertRaises(ValidationError):Draft202012Validator(provider).validate(bad)
        bad=deepcopy(valid);bad['steps'][0]['restriction_review']['clauses']=[{
            'kind':'predicate','status':'resolved','columns':['reading'],'quote':'reading',
            'relative_period':{'anchor':'source_max','unit':'year','amount':1,'column':'event_time'}}]
        with self.assertRaises(ValidationError):Draft202012Validator(provider).validate(bad)

    def test_observed_previous_identity_relabels_origin_without_replacing_source(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,GoalModel())
            try:
                prompt='같은 자료에서 다시 열 행 보여줘'
                cur=pending_state(HumanMessage(content=prompt,id='followup'),{},r.context)
                prior={'status':'complete','required_sources':[FIXTURE['source']],'scope':{}}
                data=r.recovery.goal_interpreter.payload(cur,[])
                data['verified_previous']=prior;data['requested_previous_analysis']=prior
                spec=plan([step('s1','row_preview',{'limit':10},quote=prompt)])
                spec['steps'][0]['goal']['source_reference']='explicit'
                accepted=validate(spec,cur,data,r.context)
                self.assertEqual(accepted['steps'][0]['goal']['sources'],[FIXTURE['source']])
                self.assertEqual(accepted['steps'][0]['goal']['source_reference'],'previous_analysis')
                bad_prompt='different_ns.observations 행 열 개 보여줘'
                bad=deepcopy(spec);bad['steps'][0]['request_quote']=bad_prompt
                bad_cur={**cur,'request_text':bad_prompt}
                bad_data=r.recovery.goal_interpreter.payload(bad_cur,[])
                for reference in ('explicit','previous_analysis'):
                    bad['steps'][0]['goal']['source_reference']=reference
                    with self.subTest(reference=reference),self.assertRaises(ValueError):
                        validate(bad,bad_cur,bad_data,r.context)
                bad['steps'][0]['goal'].update(source_reference='explicit',sources=['different_ns.observations'])
                admitted=validate(bad,bad_cur,bad_data,r.context)
                self.assertEqual(admitted['steps'][0]['goal']['sources'],['different_ns.observations'])
            finally:r.close()

    def test_ungrouped_chart_grammar_excludes_invented_category(self):
        from core.analysis_agent.intent_schema import schema
        from jsonschema import Draft202012Validator,ValidationError
        prompt='관측값 전체의 분포를 보여줘'
        spec=plan([step('s1','chart',{'kind':'histogram','axes':{'x':FIXTURE['column']},
            'legend':False,'stacked':False,'palette':'default'},FIXTURE['column'],
            prompt,basis='unfiltered_source')])
        outline={'mode':'execute','steps':[{'id':'s1','capability':'chart',
            'request_quote':prompt,'grouped':False,'scope_change':'clear'}]}
        validator=Draft202012Validator(schema(outline))
        validator.validate(spec)
        bad=deepcopy(spec);bad['steps'][0]['goal']['tasks'][0]['options']['category']=FIXTURE['column']
        with self.assertRaises(ValidationError):validator.validate(bad)
        bad=deepcopy(spec);bad['steps'][0]['restriction_review']['clauses'][0]['quote']='paraphrased'
        with self.assertRaises(ValidationError):validator.validate(bad)

    def test_transition_keeps_budget_without_recounting_auxiliary_calls(self):
        from core.analysis_agent.task_execution import activate
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,GoalModel())
            try:
                cur=pending_state(HumanMessage(content=FIXTURE['request'],id='budget'),{},r.context)
                cur.update(model_calls=5,model_seconds=35.,accounted_aux_calls=3,accounted_aux_seconds=25.)
                spec=plan([step('s1','row_preview',{'limit':10}),step('s2','metadata',{'kind':'columns'})])
                data=r.recovery.goal_interpreter.payload(cur,[])
                accepted=validate(spec,cur,data,r.context)
                active=start(cur,accepted,r.context,False)
                active['intent_plan']['completed']=[{'id':'s1','output_reference':None}]
                nxt=activate(active,accepted['steps'][1],r.context,False)
                for key in ('model_calls','model_seconds','accounted_aux_calls','accounted_aux_seconds','request_started_at'):
                    self.assertEqual(nxt[key],cur[key])
                self.assertIsNone(nxt.get('table_preview_evidence'))
                changed=deepcopy(nxt);changed['goal']['columns']=['forged']
                self.assertEqual(r.plans.admission(changed,'inspect_table_context')['error_code'],'task_goal_changed')
            finally:r.close()

if __name__=='__main__':unittest.main()
