"""LLM authority and execution boundaries; scripted goals are not language scores."""
import json
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from tests.test_actual_agent_evaluation import EvaluationModel
from tests.test_row_preview import DATA, schema
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.goal_contract import validate_goal


def goal(capability, options=None, *, sources=None, columns=None, conditions=None):
    return {'objective':'requested work','mode':'execute',
            'sources':sources if sources is not None else ['lab.observations'],
            'columns':columns or [],'conditions':conditions or [],'any_conditions':[],
            'measure_conditions':[],'ratio':None,'current_result_only':False,
            'fresh_source_required':False,'question':'',
            'tasks':[{'capability':capability,'options':options or {}}]}


class GoalModel(EvaluationModel):
    goals: list = []
    goal_calls: int = 0

    def _generate(self, messages, **kwargs):
        if any('goal_schema_v1' in str(m.content) for m in messages):
            self.goal_calls += 1
            value = self.goals[min(self.goal_calls-1,len(self.goals)-1)]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(value)))])
        return super()._generate(messages, **kwargs)


class LLMGoalTests(unittest.TestCase):
    def runtime(self, root, model, **kwargs):
        r=GraphAnalysisRuntime(root,'goal-test','conversation',model,sql_dialect='mysql',
            reference_context_loader=lambda:schema('lab.observations'),**kwargs)
        raw=r.datasets.register(DATA,source='lab.observations',coverage='complete',predicate_known=True)
        r.select_dataset(raw.id)
        return r,raw

    def test_natural_language_requires_model_and_does_not_call_legacy_parser(self):
        for prompt in ['observations table row 10개 보여줘','observations에서 열 개 행을 보여줘',
                       'Give me ten records from observations']:
            with self.subTest(prompt=prompt),tempfile.TemporaryDirectory() as root:
                model=GoalModel(goals=[goal('row_preview',{'limit':10})])
                r,raw=self.runtime(root,model)
                try:
                    with patch.object(r.recovery,'_legacy_state',side_effect=AssertionError('raw text reached regex router')):
                        result=r.submit(prompt)
                    self.assertEqual(result['status'],'answered',result)
                    self.assertEqual(model.goal_calls,2)
                    state=r.inspect()['recovery']
                    self.assertEqual(state['intent_origin'],'llm')
                    self.assertEqual(state['table_preview_evidence']['rows'],10)
                    self.assertEqual(r.context.selected_dataset_id,raw.id)
                finally:r.close()

    def test_negated_chart_and_average_follow_model_goal(self):
        cases=[('reading 히스토그램은 그리지 말고 컬럼 목록만 보여줘',
                goal('metadata',{'kind':'columns'}),False,[]),
               ('reading 평균은 계산하지 말고 합계만 알려줘',
                goal('calculation',{'operations':['SUM']},columns=['reading']),False,['SUM'])]
        for prompt,plan,chart,operations in cases:
            with self.subTest(prompt=prompt),tempfile.TemporaryDirectory() as root:
                model=GoalModel(goals=[plan]);r,raw=self.runtime(root,model)
                try:
                    result=r.submit(prompt);self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery']
                    self.assertEqual(state['chart'],chart)
                    self.assertEqual(state['operations'],operations)
                    if operations:
                        frame=r.datasets.frames[state['evidence_ids'][0]]
                        self.assertEqual(float(frame.iloc[0,0]),sum(DATA.reading))
                    self.assertEqual(len(r.artifacts),0)
                finally:r.close()

    def test_unsupported_goal_cannot_execute_and_original_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            model=GoalModel(goals=[{'execute_sql':'DROP TABLE observations'}]);r,raw=self.runtime(root,model)
            try:
                result=r.submit('observations table row 10개 보여줘')
                self.assertNotEqual(result['status'],'answered',result)
                self.assertEqual(len(r.datasets.metadata),1)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertFalse(r.inspect()['recovery']['sent_calls'])
            finally:r.close()

    def test_schema_and_structural_validation_does_not_guess_meaning(self):
        with self.assertRaises(ValueError):validate_goal(goal('row_preview',{'limit':1000}))
        with self.assertRaises(ValueError):validate_goal(goal('calculation',{'operations':['DROP']}))
        with self.assertRaises(ValueError):validate_goal(goal('row_preview',{'status':'complete'}))
        with self.assertRaises(ValueError):validate_goal(goal('row_preview',{'limit':True}))
        for cap in ('statistics','time_series','winsorization','outliers','pivot','group_summary','latest_per_key'):
            with self.subTest(cap=cap),self.assertRaises(ValueError):validate_goal(goal(cap))

    def test_filtered_measure_comes_from_goal_not_word_matching(self):
        with tempfile.TemporaryDirectory() as root:
            plan=goal('calculation',{'operations':['SUM']},columns=['reading'],conditions=[
                {'column':'reading','op':'ge','value':3},{'column':'reading','op':'le','value':8}])
            r,raw=self.runtime(root,GoalModel(goals=[plan]))
            try:
                out=r.submit('그 범위만 모두 더해줘')
                self.assertEqual(out['status'],'answered',out)
                proof=r.inspect()['recovery']
                self.assertEqual(float(r.datasets.frames[proof['evidence_ids'][0]].iloc[0,0]),33)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_unrequested_tool_cannot_add_chart_or_statistic(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,GoalModel(goals=[goal('metadata',{'kind':'columns'})]))
            try:
                r.submit('차트 없이 컬럼 목록')
                state=r.inspect()['recovery']
                for tool in ('render_histogram','statistical_test','winsorize_numeric'):
                    self.assertFalse(r.recovery._proposed_scope_valid({'name':tool,'args':{}},state))
                self.assertFalse(state['chart'])
            finally:r.close()

    def test_wide_catalog_is_a_bounded_model_view_not_removed_schema(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,GoalModel(goals=[goal('metadata',{'kind':'columns'})]))
            try:
                refs=[{'table':f'lab.table_{i}','columns':[{'name':f'column_{j}','dtype':'varchar',
                    'description':'설명'*100} for j in range(200)]} for i in range(32)]
                r.context.reference_context[:]=refs
                payload=r.recovery.goal_interpreter.payload({'request_id':'new','request_text':'테이블들을 살펴봐'},[])
                self.assertEqual(sum(len(t['columns']) for t in payload['tables']),len(DATA.columns))
                self.assertEqual(payload['tables'][0]['origin'],'retained_dataset_projection_not_database_schema')
                self.assertEqual(payload['table_count'],32)
                self.assertEqual(len(r.context.reference_context[0]['columns']),200)
                current={'request_id':'new','request_text':'같은 테이블의 타입',
                         'confirmed_analysis':{'status':'complete','required_sources':['lab.table_31']}}
                payload=r.recovery.goal_interpreter.payload(current,[])
                self.assertEqual(sum(len(t['columns']) for t in payload['tables']),8)
                self.assertEqual(payload['tables'][-1]['column_count'],200)
                self.assertEqual(len(r.context.reference_context[-1]['columns']),200)
            finally:r.close()

    def test_retained_schema_fallback_never_replaces_a_named_subject(self):
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,GoalModel(goals=[goal('calculation',{'operations':['AVG']},columns=['reading'])]))
            try:
                r.context.reference_context[:]=[]
                payload=r.recovery.goal_interpreter.payload({'request_id':'local','request_text':'reading 평균'},[])
                self.assertEqual(payload['tables'][0]['table'],raw.source)
                self.assertEqual(payload['tables'][0]['columns'][0]['name'],'reading')
                absent=r.recovery.goal_interpreter.payload({'request_id':'new',
                    'request_text':'otherdb.missing 컬럼',
                    'requested_subject':{'sources':['otherdb.missing'],'request_id':'new'}},[])
                self.assertFalse(any(t['table']==raw.source for t in absent['tables']))
            finally:r.close()

    def test_pending_or_invalid_goal_is_never_complete(self):
        from core.analysis_agent.completion import completion_ready
        for state in ({'intent_origin':'llm','goal_pending':True},
                      {'intent_origin':'llm','goal_interpretation_error':True}):
            self.assertFalse(completion_ready(state))

    def test_unrelated_selected_dataset_is_repaired_by_model_before_preview(self):
        with tempfile.TemporaryDirectory() as root:
            bad=goal('row_preview',{'limit':10},sources=['lab.other'])
            bad['current_result_only']=True
            model=GoalModel(goals=[bad,goal('row_preview',{'limit':10})])
            r,raw=self.runtime(root,model)
            try:
                result=r.submit('observations의 열 개 행을 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(model.goal_calls,3)
                self.assertEqual(r.inspect()['recovery']['table_preview_evidence']['source'],'lab.observations')
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_followup_gets_previous_confirmed_goal_and_restart_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            preview=goal('row_preview',{'limit':10})
            total=goal('calculation',{'operations':['SUM']},columns=['reading'])
            model=GoalModel(goals=[preview,preview,total,total])
            r,raw=self.runtime(root,model)
            try:
                self.assertEqual(r.submit('observations 열 개 행 보여줘')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['table_preview_evidence']['rows'],10)
                self.assertEqual(r.submit('그 테이블 reading 합계만')['status'],'answered')
                self.assertEqual(model.goal_calls,4)
                self.assertEqual(r.inspect()['recovery']['required_sources'],['lab.observations'])
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                saved=set(r.datasets.metadata)
                r.close()
                r=GraphAnalysisRuntime(root,'goal-test','conversation',model,sql_dialect='mysql',
                    reference_context_loader=lambda:schema('lab.observations'))
                self.assertEqual(set(r.datasets.metadata),saved)
                self.assertEqual(r.inspect()['recovery']['intent_origin'],'llm')
            finally:r.close()
