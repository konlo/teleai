"""Failure does not erase interpreted intent or turn a preview into a population."""
import unittest
from types import SimpleNamespace
from langchain_core.messages import HumanMessage
from core.analysis_agent.goal_contract import pending_state, compile_goal
from core.analysis_agent.intent_memory import unfinished, model_view, prior_intent
from core.analysis_agent.task_selection import validate, verify_goal
from tests.test_llm_goal import goal


class PendingIntentPopulationTests(unittest.TestCase):
    def context(self):
        return SimpleNamespace(selected_dataset_id='',sql_dialect='mysql',
            source_namespace='lab',datasets=SimpleNamespace(metadata={}),
            reference_context=[{'table':'lab.observations','columns':[
                {'name':'reading','dtype':'double'},{'name':'cohort','dtype':'varchar'}]}])

    def test_failed_execution_keeps_requested_scope_but_never_copies_result_evidence(self):
        ctx=self.context()
        old={'status':'complete','request_id':'old','required_sources':['lab.observations'],
            'scope':{'conditions':[{'column':'cohort','op':'eq','value':'A'}]},
            'evidence_ids':['old-proof']}
        expanded=[{'column':'cohort','op':'in','value':['A','B']},
                  {'column':'reading','op':'between','value':[12,19]}]
        failed=compile_goal({'confirmed_analysis':old,'request_id':'failed','intent_origin':'llm',
            'request_text':'B도 포함해줘','status':'working'},
            goal('chart',{'kind':'histogram','axes':{'x':'reading'},'category':'cohort'},
                 sources=['lab.observations'],columns=['reading'],conditions=expanded),ctx)
        failed.update(status='exhausted',evidence_ids=['unverified'],artifact_ids=['unverified'])
        current=pending_state(HumanMessage(content='같은 조건으로 다시 그려줘',id='next'),failed,ctx)
        self.assertEqual(current['confirmed_analysis']['evidence_ids'],['old-proof'])
        pending=prior_intent(current)
        self.assertEqual(pending['scope']['conditions'],failed['scope']['conditions'])
        self.assertNotIn('evidence_ids',pending)
        self.assertNotIn('artifact_ids',model_view(current))
        self.assertEqual(prior_intent(current,True),old)
        current.update(status='blocked',goal_interpretation_error=True)
        self.assertEqual(unfinished(current),pending)
        other=goal('metadata',{'kind':'columns'},sources=['lab.other'])
        self.assertFalse(compile_goal(current,other,ctx)['scope']['conditions'])
        metadata=goal('metadata',{'kind':'columns'},sources=['lab.observations'])
        self.assertEqual(compile_goal(current,metadata,ctx)['scope']['conditions'],pending['scope']['conditions'])

    def test_source_population_is_not_a_reuse_or_refresh_setting(self):
        from core.analysis_agent.population_basis import validate as population
        value=population({'basis':'source_population','quote':''},'전체 원본으로 관계를 그려줘')
        self.assertFalse(value['current_result_only'])
        for quote in ('','방금 표시한 10행'):
            with self.assertRaises(ValueError):population({'basis':'displayed_result','quote':quote},'전체 원본으로 관계를 그려줘')
        value=population({'basis':'displayed_result','quote':'방금 표시한 10행'},'방금 표시한 10행으로 관계를 그려줘')
        self.assertTrue(value['current_result_only'])

    def test_semantic_review_can_correct_a_wrong_preliminary_obligation(self):
        selected=validate({'mode':'execute','capabilities':['table_list'],
            'source_reference':'explicit','source_mentions':[],
            'chart_kind':''},'이 테이블의 컬럼')
        selected['current_result_only']=False
        corrected=goal('metadata',{'kind':'columns'},sources=['lab.observations'])
        with self.assertRaises(ValueError):verify_goal(corrected,selected,self.context())
        verify_goal(corrected,selected,self.context(),enforce_obligations=False)
        selected['source_mentions']=[{'name':'missing','quote':'missing'}]
        with self.assertRaises(ValueError):verify_goal(corrected,selected,self.context(),enforce_obligations=False)

    def test_provider_grammar_binds_a_model_read_literal_table_without_guessing(self):
        from langchain_ollama import ChatOllama
        from core.analysis_agent.goal_interpreter import GoalInterpreter
        interpreter=GoalInterpreter(ChatOllama(model='fixture-only-no-inference',num_ctx=16384),
            self.context(),SimpleNamespace(emit=lambda *args,**kwargs:None),16000)
        selected={'mode':'execute','capabilities':['metadata'],'source_reference':'explicit',
            'source_mentions':[{'name':'observations','quote':'observations'}],
            'current_result_only':False,'chart_kind':''}
        narrowed=interpreter.selected_goal_model(selected).format
        self.assertEqual(narrowed['properties']['sources'],{'const':['lab.observations']})
        self.assertEqual(selected['bound_sources'],['lab.observations'])
        from core.analysis_agent.goal_contract import response_schema
        reviewed=response_schema();interpreter.bind_literal_schema(reviewed,selected)
        self.assertEqual(reviewed['properties']['sources'],narrowed['properties']['sources'])
        selected['source_mentions']=[{'name':'other.lab.observations','quote':'other.lab.observations'}]
        reviewed=response_schema();interpreter.bind_literal_schema(reviewed,selected)
        self.assertEqual(reviewed['properties']['sources'],{'const':['other.lab.observations']})

    def test_empty_redundant_columns_are_derived_only_from_declared_axes(self):
        from core.analysis_agent.goal_normalization import normalize
        from core.analysis_agent.goal_contract import validate_goal
        histogram=goal('chart',{'kind':'histogram','axes':{'x':'reading'}})
        fixed=normalize(histogram)
        self.assertEqual(fixed['columns'],['reading'])
        validate_goal(fixed)
        scatter=goal('chart',{'kind':'scatter','axes':{'x':'reading','y':'cohort'}})
        self.assertEqual(normalize(scatter)['columns'],['reading','cohort'])
        conflict=goal('chart',{'kind':'histogram','axes':{'x':'reading'}},columns=['cohort'])
        self.assertEqual(normalize(conflict),conflict)
        with self.assertRaises(ValueError):validate_goal(normalize(conflict))
        unspecified=goal('chart',{'kind':'recommend'})
        self.assertEqual(normalize(unspecified),unspecified)

    def test_categorical_frequency_and_grouped_numeric_chart_have_distinct_grammars(self):
        from core.analysis_agent.goal_schema import task_schema
        variants=task_schema(['chart'],'histogram',numeric_columns=['reading'])['anyOf'][0]['properties']['options']['anyOf']
        self.assertEqual(len(variants),2)
        self.assertNotIn('category',variants[0]['properties'])
        self.assertIn('legend',variants[0]['required'])
        self.assertEqual(variants[1]['properties']['axes']['properties']['x']['enum'],['reading'])
        self.assertIn('category',variants[1]['required'])
        categorical_only=task_schema(['chart'],'histogram',numeric_columns=[])['anyOf'][0]['properties']['options']['anyOf']
        self.assertEqual(len(categorical_only),1)
        self.assertNotIn('category',categorical_only[0]['properties'])

    def test_missing_required_task_targets_never_escape_into_execution_planning(self):
        from core.analysis_agent.goal_contract import validate_goal,response_schema
        from core.analysis_agent.goal_interpreter import GoalInterpreter
        for columns in ([],['reading','cohort']):
            with self.assertRaises(ValueError):validate_goal(goal('value_list',columns=columns))
        validate_goal(goal('value_list',columns=['cohort']))
        for cap,minimum in (('value_list',1),('calculation',1),('statistics',1),('time_series',2)):
            schema=response_schema()
            GoalInterpreter.bind_required_columns(schema,{'capabilities':[cap],'chart_kind':''})
            self.assertEqual(schema['properties']['columns']['minItems'],minimum)

    def test_duplicate_qualified_literals_do_not_expose_their_suffix_tokens(self):
        from core.analysis_agent.subject_identity import candidates
        result=candidates('other.events와 other.events 컬럼')
        self.assertEqual([x['source'] for x in result],['other.events'])

    def test_new_named_source_excludes_foreign_displayed_population_before_inference(self):
        from unittest.mock import patch
        from core.analysis_agent.population_basis import read
        model=SimpleNamespace(invoke=lambda messages: (_ for _ in ()).throw(AssertionError('no eligible result')))
        interpreter=SimpleNamespace(context=self.context(),selection_model=model)
        current={'request_id':'next','request_text':'other에서 열 행을 보여줘'}
        data={'literal_table_subjects':[{'name':'other','quote':'other'}],
              'verified_previous':{'required_sources':['lab.observations'],'chart_available':True,
                  'table_preview_evidence':{'source':'lab.observations','rows':10}},
              'selected_dataset':{'source':'lab.observations'}}
        self.assertFalse(read(interpreter,current,data)['current_result_only'])

    def test_included_membership_values_keep_other_dimensions_and_need_current_quote(self):
        from core.analysis_agent.population_audit import reconcile,model_for
        from langchain_ollama import ChatOllama
        prior={'conditions':[{'column':'reading','op':'between','value':[12,19]},
                            {'column':'cohort','op':'eq','value':'A'}],'any_conditions':[]}
        delta={'change':'modify','evidence_quote':'include B too','conditions':[],
               'any_conditions':[],'removed_columns':[],
               'included_values':[{'column':'cohort','values':['B'],'quote':'include B too'}]}
        result=reconcile(delta,prior,'Keep the range; include B too',True)
        self.assertEqual(result['conditions'],[prior['conditions'][0],
            {'column':'cohort','op':'in','value':['A','B']}])
        delta['included_values'].append({'column':'cohort','values':['C'],'quote':'include B too'})
        self.assertEqual(reconcile(delta,prior,'Keep the range; include B too',True)['conditions'][-1],
                         {'column':'cohort','op':'in','value':['A','B','C']})
        delta['included_values'][0]['quote']='older request'
        with self.assertRaisesRegex(ValueError,'CURRENT quote'):
            reconcile(delta,prior,'Keep the range; include B too',True)
        model=model_for(ChatOllama(model='fixture-only-no-inference'),prior)
        self.assertEqual(model.format['properties']['included_values']['items']['properties']['column'],
                         {'type':'string','enum':['cohort']})
        self.assertEqual(model_for(model,{}).format['properties']['included_values'],{'const':[]})
        narrowed=model_for(model,prior,'Include B too')
        self.assertEqual(narrowed.format['properties']['evidence_quote']['enum'],['','Include B too'])

    def test_numeric_or_that_erases_range_is_rejected_before_a_query(self):
        from core.analysis_agent.population_audit import reconcile
        delta={'change':'modify','evidence_quote':'reading from 12 to 19','conditions':[],
               'any_conditions':[{'column':'reading','op':'ge','value':12},
                                 {'column':'reading','op':'le','value':19}]}
        with self.assertRaisesRegex(ValueError,'Overlapping numeric bounds'):
            reconcile(delta,{},delta['evidence_quote'],True)

    def test_literal_subject_reader_receives_observed_fields_of_current_subject(self):
        from unittest.mock import patch
        from core.analysis_agent.subject_identity import read
        from langchain_core.messages import AIMessage
        import json
        captured=[]
        def invoke(messages):
            captured.append(json.loads(messages[1].content))
            return AIMessage(content=json.dumps({'candidate_roles':{'cohort':'other','DB':'other'}}))
        model=SimpleNamespace(invoke=invoke)
        interpreter=SimpleNamespace(selection_model=model,model_recovery=None,
            budget=SimpleNamespace(wrap_model_call=lambda r,h:h(r)),
            diagnostics=SimpleNamespace(emit=lambda *a,**k:None))
        data={'verified_previous':{'required_sources':['lab.observations']},
              'tables':self.context().reference_context}
        with patch('core.analysis_agent.model_roles.json_role',return_value=model):
            self.assertEqual(read(interpreter,{'request_id':'next','request_text':'cohort DB 값'},data),[])
        self.assertEqual(captured[0]['ACTIVE_SOURCE'],['lab.observations'])
        self.assertEqual(captured[0]['OBSERVED_COLUMNS'],[
            {'table':'lab.observations','columns':['reading','cohort']}])

    def test_restart_followup_uses_failed_intent_and_keeps_original_bytes(self):
        import tempfile
        import pandas as pd
        from tests.test_llm_goal import GoalModel
        from tests import test_llm_goal as harnesses
        from tests.test_row_preview import schema
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        with tempfile.TemporaryDirectory() as root:
            harness=harnesses.LLMGoalTests()
            model=GoalModel(goals=[goal('metadata',{'kind':'columns'})])
            runtime,raw=harness.runtime(root,model)
            before=runtime.datasets.frames[raw.id].copy(deep=True)
            self.assertEqual(runtime.submit('컬럼 이름')['status'],'answered')
            confirmed=runtime.inspect()['recovery']
            plan=goal('chart',{'kind':'histogram'},columns=['reading'],conditions=[
                {'column':'reading','op':'between','value':[3,8]}])
            failed=compile_goal({'request_id':'injected-failure','request_text':'3에서8까지',
                'status':'working','intent_origin':'llm','confirmed_analysis':confirmed},plan,runtime.context)
            failed.update(status='exhausted',stop_reason='injected_render_failure')
            runtime.agent.update_state(runtime.config,{'recovery':failed})
            runtime.close()
            runtime=GraphAnalysisRuntime(root,'goal-test','conversation',
                GoalModel(goals=[goal('metadata',{'kind':'columns'})]),sql_dialect='mysql',
                reference_context_loader=lambda:schema('lab.observations'))
            try:
                self.assertEqual(runtime.submit('같은 조건의 테이블 컬럼')['status'],'answered')
                current=runtime.inspect()['recovery']
                self.assertEqual(current['scope']['conditions'],failed['scope']['conditions'])
                pd.testing.assert_frame_equal(before,runtime.datasets.frames[raw.id])
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
            finally:runtime.close()
