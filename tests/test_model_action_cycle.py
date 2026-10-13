"""Model action authority and failure loopback; scripted choices are not scores."""
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.support_report import summarize,brief
from tests.test_llm_goal import GoalModel,goal
from tests.test_row_preview import DATA,schema
from utils.analysis_datasets import stored_dataset_digest

class RewordedQueryModel(GoalModel):
    def _generate(self,messages,**kwargs):
        result=super()._generate(messages,**kwargs)
        for call in result.generations[0].message.tool_calls:
            if call['name']=='query_databricks':
                call['args']['reason']='Return the requested records.'
                call['args']['query']=call['args']['query'].replace('SELECT','select').replace('LIMIT','limit')
        return result


class ModelActionCycleTests(unittest.TestCase):
    def test_query_reason_and_formatting_do_not_replace_execution_constraints(self):
        from dataclasses import asdict
        from core.analysis_agent.row_preview import matches_plan
        context=SimpleNamespace(sql_dialect='mysql')
        plan={'source':'lab.observations','query':'SELECT * FROM `lab`.`observations` LIMIT 10','reason':'original'}
        self.assertTrue(matches_plan(context,{**plan,'query':'select * from lab.observations limit 10','reason':'different'},plan))
        for query in ['SELECT * FROM lab.observations LIMIT 11','SELECT * FROM lab.other LIMIT 10',
                      'SELECT * FROM lab.observations WHERE reading > 5 LIMIT 10']:
            self.assertFalse(matches_plan(context,{**plan,'query':query},plan))
        model=RewordedQueryModel(goals=[goal('row_preview',{'limit':10})]);executed=[]
        def factory(datasets):
            def execute(envelope):
                executed.append(envelope)
                info=datasets.register(DATA.head(10),source=envelope['source'],query=envelope['query'],
                    coverage='complete',predicate_known=True)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'action-owner','reworded',model,sql_dialect='mysql',
                reference_context_loader=lambda:schema('lab.observations'),remote_factory=factory,
                connection_identity='fixture-mysql')
            try:
                result=r.submit('Show ten records from observations')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(executed),1)
                self.assertEqual(r.inspect()['recovery']['table_preview_evidence']['rows'],10)
                self.assertEqual(executed[0]['reason'],'Return the requested records.')
            finally:r.close()

    def runtime(self,root,model,**kwargs):
        r=GraphAnalysisRuntime(root,'action-owner','action-conversation',model,
            sql_dialect='mysql',reference_context_loader=lambda:schema('lab.observations'),**kwargs)
        raw=r.datasets.register(DATA,source='lab.observations',coverage='complete',predicate_known=True)
        r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
        return r,raw

    def test_model_can_inspect_contract_instead_of_executing_candidate(self):
        model=GoalModel(goals=[goal('metadata',{'kind':'columns'})],calls=[
            {'name':'get_analysis_tool_contract','args':{'name':'inspect_table_context'}},
            {'name':'inspect_table_context','args':{'table':'lab.observations'}}])
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,model)
            try:
                out=r.submit('List the observed columns')
                self.assertEqual(out['status'],'answered',out)
                state=r.inspect()['recovery']
                self.assertEqual([item['choices'][0]['tool'] for item in state['action_cycle_history']],
                    ['get_analysis_tool_contract','inspect_table_context'])
                self.assertEqual(model.position,2)
                self.assertEqual(len(state['action_cycle_history']),2)
                events=r.plans.history(state['request_id'])
                self.assertEqual(len([e for e in events if e['kind']=='model_action_selected']),2)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_dependency_refresh_is_retryable_and_advice_does_not_mutate_failure_state(self):
        from core.analysis_agent.action_cycle import guidance
        call={'name':'inspect_table_context','args':{'table':'lab.observations'}}
        diagnostic=SimpleNamespace(emit=lambda *a,**kw:None)
        def validate(candidate,state):
            state['scope_error']='unselected_candidate_error'
            return True
        recovery=SimpleNamespace(context=SimpleNamespace(allowed_tool_names={'inspect_table_context'}),
            _signature=lambda c:'signature',_proposed_scope_valid=validate,
            diagnostics=diagnostic,max_tool_calls=12)
        state={'request_id':'fixture','tool_failures':{'signature':{'tool':'inspect_table_context',
            'status':'needs_refresh','call_id':'stale-schema'}},'failed_signatures':{'signature':1},
            'chart_presentation_spec':{'legend':True}}
        answer=guidance(recovery,state,{},call)
        view=json.loads('{'+answer['messages'][0].content.split('\n{',1)[1])
        self.assertEqual(view['advisory_call'],call)
        self.assertEqual(view['execution_constraints']['chart_presentation_spec'],{'legend':True})
        self.assertNotIn('scope_error',state)
        state['tool_failures']['signature']['status']='error'
        answer=guidance(recovery,state,{},call)
        view=json.loads('{'+answer['messages'][0].content.split('\n{',1)[1])
        self.assertIsNone(view['advisory_call'])

    def test_failed_sql_returns_to_model_and_alternative_preserves_original(self):
        model=GoalModel(goals=[goal('calculation',{'operations':['SUM']},columns=['reading'])],calls=[
            {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT SUM(reading) FROM not_a_registered_relation'}},
            {'name':'aggregate_dataset','args':{'dataset_id':'$fixture','aggregation':'sum','value_column':'reading'}}])
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,model)
            before=stored_dataset_digest(r.datasets,raw.id)
            try:
                out=r.submit('Add up the reading values')
                self.assertEqual(out['status'],'answered',out)
                state=r.inspect()['recovery']
                self.assertEqual(state['action_cycle_history'][0]['choices'][0]['tool'],'local_analysis_sql')
                self.assertEqual(state['action_cycle_history'][1]['choices'][0]['tool'],'aggregate_dataset')
                self.assertTrue(state['action_cycle_history'][1]['observed_call_ids'])
                result=r.datasets.frames[state['evidence_ids'][0]]
                self.assertEqual(float(result.iloc[0,0]),float(DATA.reading.sum()))
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertFalse(state['remote_query_ids'])
                report=summarize(r.diagnostics.path)
                self.assertEqual(report['action_cycle']['choices'],2)
                self.assertEqual(report['action_cycle']['after_failure'],1)
                self.assertIn('관찰·행동 루프',brief(report))
            finally:r.close()

    def test_unexecuted_prose_cannot_be_completed_by_controller_candidate(self):
        model=GoalModel(goals=[goal('metadata',{'kind':'columns'})],answer='Done without tools')
        with tempfile.TemporaryDirectory() as root:
            r,raw=self.runtime(root,model)
            try:
                # Preserve the prepared candidate, but force an independent
                # model that declines to call it. The controller cannot act.
                from tests.test_actual_agent_evaluation import EvaluationModel
                with patch.object(GoalModel,'_generate',side_effect=lambda messages,**kw:
                    EvaluationModel._generate(model,messages,**kw) if any(
                    m.additional_kwargs.get('lc_source')=='action_cycle' for m in messages)
                    else original(model,messages,**kw)):
                    out=r.submit('List the observed columns')
                self.assertNotEqual(out['status'],'answered')
                self.assertFalse(r.inspect()['recovery']['sent_calls'])
                self.assertFalse(r.inspect()['recovery'].get('metadata_evidence'))
            finally:r.close()


original=GoalModel._generate
