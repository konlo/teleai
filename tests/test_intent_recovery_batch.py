"""Failed-intent lineage, semantic replanning and deadline dispatch boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.outputs import ChatGeneration,ChatResult

from tests.test_intent_task_scopes import FIXTURE,PlanModel,plan,step
from tests.test_row_preview import DATA,schema
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.goal_contract import pending_state
from core.analysis_agent.intent_plan import validate
from core.analysis_catalog import resolve_table_context
from utils.analysis_datasets import stored_dataset_digest


class ReplanningModel(PlanModel):
    def _generate(self,messages,**kwargs):
        outline=any('intent_plan_outline_v2:' in str(m.content) for m in messages)
        review=any('intent_plan_v2:' in str(m.content) for m in messages)
        if outline or review:
            self.plan_calls+=1
            cap='profile' if self.plan_calls<=2 else 'value_list'
            p=plan([step('s1',cap,{'kind':'distinct'} if cap=='profile' else {},
                'reading',quote='reading 값들을 보여줘')])
            if outline:
                p['steps']=[{'id':'s1','depends_on':[],'request_quote':'reading 값들을 보여줘',
                    'capability':cap,'grouped':False,'scope_change':'keep'}]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(p)))])
        return super()._generate(messages,**kwargs)


class ReviewOutlineModel(ReplanningModel):
    def _generate(self,messages,**kwargs):
        if any('intent_plan_outline_v2:' in str(m.content) for m in messages):
            return super()._generate(messages,**kwargs)
        if any('intent_plan_v2:' in str(m.content) for m in messages):
            self.plan_calls+=1
            p=plan([step('s1','value_list',{},'reading',quote='reading 값들을 보여줘')])
            if self.plan_calls==2:
                p={'mode':'revise_outline','reason':'The question asks observed values, not cardinality',
                    'outline':{'mode':'execute','objective':'reading 값들을 보여줘','question':'',
                        'steps':[{'id':'s1','depends_on':[],'request_quote':'reading 값들을 보여줘',
                            'capability':'value_list','grouped':False,'scope_change':'keep'}]}}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(p)))])
        return super()._generate(messages,**kwargs)


class IntentRecoveryBatchTests(unittest.TestCase):
    def runtime(self,root,model=None):
        r=GraphAnalysisRuntime(root,'batch','case',model or PlanModel(),
            reference_context_loader=lambda:schema(FIXTURE['source']),sql_dialect='mysql')
        raw=r.datasets.register(DATA,source=FIXTURE['source'],coverage='complete',predicate_known=True)
        r.select_dataset(raw.id);return r,raw

    def test_failed_uninterpreted_request_is_human_context_even_with_confirmed_table(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root)
            try:
                old={'request_id':'old','request_text':'reading 3~8만 보여줘','goal_pending':True,
                    'status':'incomplete','confirmed_analysis':{'required_sources':[raw.source],'scope':{}}}
                current=pending_state(HumanMessage(content='같은 조건으로 다시 보여줘',id='new'),old,r.context)
                data=r.recovery.goal_interpreter.payload(current,[HumanMessage(content=old['request_text'],id='old')])
                self.assertEqual(data['unparsed_user_intent'][0]['request_text'],old['request_text'])
                self.assertEqual(data['conversation_text_not_evidence'][0]['text'],old['request_text'])
                completed=pending_state(HumanMessage(content='다음',id='later'),{**old,'status':'complete'},r.context)
                self.assertEqual(completed['unparsed_user_intent'],[])
            finally:r.close()

    def test_same_condition_cannot_fall_back_to_older_unfiltered_success(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root)
            try:
                q='같은 조건의 reading histogram 다시 보여줘'
                current=pending_state(HumanMessage(content=q,id='new'),{},r.context)
                current['confirmed_analysis']={'required_sources':[raw.source],'scope':{'conditions':[],'any_conditions':[]}}
                data=r.recovery.goal_interpreter.payload(current,[])
                data['unparsed_user_intent']=[{'request_id':'failed','request_text':'reading 3~8만 보여줘','truncated':False}]
                p=plan([step('s1','chart',{'kind':'histogram','axes':{'x':'reading'},
                    'legend':False,'stacked':False,'palette':'default'},'reading',quote=q)])
                with self.assertRaisesRegex(ValueError,'Latest human intent'):validate(p,current,data,r.context)
                terms=[{'column':'reading','op':'between','value':[3,8]}]
                p['steps'][0]['pending_population']={'change':'modify','conditions':terms,'any_conditions':[],
                    'restriction_review':{'clauses':[{'kind':'predicate','columns':['reading'],
                        'quote':'reading 3~8만 보여줘','status':'resolved'}]}}
                accepted=validate(p,current,data,r.context)
                self.assertEqual(accepted['steps'][0]['goal']['conditions'],terms)
                bad=deepcopy(p);bad['steps'][0]['pending_population']['restriction_review']['clauses'][0]['quote']='assistant inferred filter'
                with self.assertRaises(ValueError):validate(bad,current,data,r.context)
                self.assertEqual(len(r.datasets.metadata),1)
            finally:r.close()

    def test_qualified_namespaces_never_match_only_table_basename(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root)
            try:
                self.assertEqual(resolve_table_context(r.context.reference_context,r.datasets,'other.observations')['status'],'needs_context')
                self.assertEqual(resolve_table_context(r.context.reference_context,r.datasets,'observations')['table_context']['table'],raw.source)
            finally:r.close()

    def test_model_selected_replan_preserves_original_and_shared_request_budget(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'TELLY_INTENT_PROTOCOL':'v2'}),patch(
                'core.analysis_agent.intent_plan.intent_role',side_effect=lambda model,*args:model):
            model=ReplanningModel(calls=[{'name':'review_analysis_intent','args':{'reason':'The original user asked values, not a distinct count profile'}},
                {'name':'inspect_value_list','args':{'source':FIXTURE['source'],'column':'reading'}}])
            r,raw=self.runtime(root,model)
            try:
                r.recovery.goal_interpreter.requires_grounding=True
                model.evaluation_dataset_id=raw.id;before=stored_dataset_digest(r.datasets,raw.id)
                outcome=r.submit('reading 값들을 보여줘')
                self.assertEqual(outcome['status'],'answered',outcome)
                state=r.inspect()['recovery'];self.assertEqual(state['intent_revisions'],1)
                self.assertGreaterEqual(state['model_calls'],5)
                self.assertEqual(set(state['value_list_evidence']['values']),set(DATA.reading))
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_completed_model_response_after_deadline_is_not_dispatched_or_refunded(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root)
            try:
                from core.analysis_agent.model_recovery import ModelAttemptBudgetExceeded
                m=r.model_recovery;m.policy=replace(m.policy,turn_slo_seconds=.01)
                def response():
                    time.sleep(.02)
                    return AIMessage(content='',tool_calls=[{'id':'never-dispatch','name':'local_analysis_sql',
                        'args':{'dataset_id':raw.id,'query':'SELECT COUNT(*) FROM data'}}])
                with self.assertRaises(ModelAttemptBudgetExceeded):m.invoke({'request_id':'late'},response)
                measured={'request_id':'late'};r.model_attempts.sync(measured)
                self.assertEqual(measured['model_calls'],1)
                self.assertEqual(r.model_attempts.get('late')['failures'],0)
                self.assertEqual(len(r.datasets.metadata),1)
            finally:r.close()

    def test_review_can_correct_wrong_outline_before_any_execution(self):
        with tempfile.TemporaryDirectory() as root,patch.dict('os.environ',{'TELLY_INTENT_PROTOCOL':'v2'}),patch(
                'core.analysis_agent.intent_plan.intent_role',side_effect=lambda model,*args:model):
            model=ReviewOutlineModel(calls=[{'name':'inspect_value_list','args':{'source':FIXTURE['source'],'column':'reading'}}])
            r,raw=self.runtime(root,model)
            try:
                r.recovery.goal_interpreter.requires_grounding=True
                outcome=r.submit('reading 값들을 보여줘')
                self.assertEqual(outcome['status'],'answered',outcome)
                state=r.inspect()['recovery'];self.assertTrue(state['value_list_evidence'])
                self.assertFalse(state['profile_kind'])
                self.assertEqual(state['model_calls'],4)
                self.assertEqual(len(r.datasets.metadata),1)
            finally:r.close()
