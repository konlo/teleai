"""Clarification continuity and source-MAX observations, not language scores."""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from core.analysis_agent.clarification_memory import carry, remember, resume
from core.analysis_agent.goal_contract import pending_state
from core.analysis_agent.relative_period import boundaries, probe_query
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.source_period_probe import observe
from core.analysis_agent.population_audit import audit
from tests.test_population_requirement_guard import reader,current
from tests.test_population_audit_local_repair import delta
from tests.test_llm_goal import GoalModel,goal
from utils.analysis_datasets import stored_dataset_digest

FIXTURE=json.loads((Path(__file__).parent/'fixtures/temporal_population_requirements.json').read_text())
SPEC={'anchor':'source_max','amount':1,'unit':'year','column':FIXTURE['time_column']}

class RelativePeriodTests(unittest.TestCase):
    def test_calendar_boundaries_include_leap_day_and_preserve_iso_format(self):
        self.assertEqual(boundaries('2024-02-29',1,'year'),('2023-02-28','2024-02-29'))
        self.assertEqual(boundaries('2026-03-31 12:30:00+00:00',1,'month'),
            ('2026-02-28 12:30:00+00:00','2026-03-31 12:30:00+00:00'))
        self.assertEqual(boundaries('2026-01-01T00:00:00Z',1,'day')[0],'2025-12-31T00:00:00Z')

    def test_probe_is_one_read_only_source_scan_and_backend_specific(self):
        from core.analysis_sql import validate_query
        from core.analysis_load_plan import source_plan
        for dialect in ('mysql','databricks'):
            query=probe_query(FIXTURE['source'],FIXTURE['time_column'],'string',dialect)
            tree=validate_query(query,dialect=dialect)
            self.assertEqual(source_plan(FIXTURE['source'],query,dialect=dialect).grain,'aggregate')
            self.assertIn('MAX(',query)
            self.assertEqual(query.count(' FROM '),1)
            self.assertEqual('TRY_CAST' in query,dialect=='databricks')
        with self.assertRaises(ValueError):probe_query(FIXTURE['source'],'reading','double','mysql')

    def test_answer_retains_original_task_duration_and_source_across_pending_state(self):
        ctx=SimpleNamespace(selected_dataset_id='',datasets=SimpleNamespace(metadata={}),reference_context=[])
        old=current('Plot reading for the last year')
        plan=goal('chart',{'kind':'histogram'},columns=['reading'])
        remember(old,plan,'오늘 기준인가요, 데이터의 마지막 날짜 기준인가요?')
        old.update(status='needs_context')
        new=pending_state(HumanMessage(content='데이터 마지막 날짜 기준이지',id='next'),old,ctx)
        model=reader([{'action':'answer','request_id':'new','answer_quote':new['request_text']}])
        resume(model,new,[])
        self.assertIn('last year',new['request_text'])
        self.assertIn('데이터 마지막 날짜 기준이지',new['request_text'])
        self.assertEqual(new['requested_subject']['sources'],plan['sources'])
        self.assertEqual(new['model_calls'],0)
        self.assertNotIn('artifact_ids',carry(old))

    def test_new_request_does_not_inherit_pending_time(self):
        state=current('Show a different table')
        state['pending_clarification']={'request_text':'Plot reading for last year','request_id':'old','question':'anchor?'}
        model=reader([{'action':'new_request','request_id':'','answer_quote':''}])
        resume(model,state,[])
        self.assertEqual(state['request_text'],'Show a different table')
        self.assertNotIn('pending_clarification',state)

    def test_legacy_checkpoint_recovers_from_actual_original_user_turn(self):
        state=current('데이터의 마지막 날짜 기준이에요')
        state['pending_clarification']={'question':'anchor?','legacy':True}
        messages=[HumanMessage(content='Plot reading for the last year',id='original'),
            HumanMessage(content='데이터 마지막 날짜 기준이지',id='reply')]
        model=reader([{'action':'answer','request_id':'original','answer_quote':state['request_text']}])
        resume(model,state,messages)
        self.assertTrue(state['request_text'].startswith('Plot reading for the last year'))

    def runtime(self,root,invalid=0,formats=1):
        calls=[]
        def factory(datasets):
            def execute(envelope):
                calls.append(envelope)
                info=datasets.register(pd.DataFrame({'__period_max':['2024-02-29'],
                    '__period_count':[4],'__period_invalid':[invalid],'__period_formats':[formats]}),
                    source=FIXTURE['source'],query=envelope['query'],coverage='complete',
                    grain='aggregate',aggregation='MAX/COUNT')
                return {'status':'ready','dataset':asdict(info)}
            return execute
        runtime=GraphAnalysisRuntime(root,'owner','relative',GoalModel(goals=[]),
            connection_identity='fixture',remote_factory=factory,
            reference_context_loader=lambda:[{'table':FIXTURE['source'],'columns':[
                {'name':FIXTURE['time_column'],'dtype':'string'},{'name':'reading','dtype':'double'}]}])
        return runtime,calls

    def test_max_probe_replay_reuses_receipt_and_never_replaces_original(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,calls=self.runtime(root)
            try:
                raw=runtime.datasets.register(pd.DataFrame({'reading':[1,2]}),source=FIXTURE['source'],coverage='complete')
                runtime.select_dataset(raw.id);digest=stored_dataset_digest(runtime.datasets,raw.id)
                state=current('last year from source maximum')
                first=observe(runtime,state,FIXTURE['source'],SPEC)
                second=observe(runtime,state,FIXTURE['source'],SPEC)
                self.assertEqual(first,second);self.assertEqual(len(calls),1)
                self.assertEqual(first['conditions'][0]['value'],'2023-02-28')
                self.assertEqual(len(state['observation_query_ids']),1)
                self.assertNotIn(state['observation_query_ids'][0],state.get('remote_query_ids',[]))
                self.assertEqual(len(runtime.execution_status()['queries']),1)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),digest)
            finally:runtime.close()

    def test_bad_or_mixed_dates_do_not_produce_a_verified_period(self):
        for invalid,formats in [(1,1),(0,2)]:
            with self.subTest(invalid=invalid,formats=formats),tempfile.TemporaryDirectory() as root:
                runtime,_=self.runtime(root,invalid,formats)
                try:
                    result=observe(runtime,current('last year'),FIXTURE['source'],SPEC)
                    self.assertNotEqual(result['status'],'ready')
                    self.assertNotIn('conditions',result)
                finally:runtime.close()

    def test_probe_rejects_receipt_from_another_source(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,calls=self.runtime(root)
            try:
                execute=runtime.remote_execute
                def mismatched(envelope):
                    execute(envelope)
                    info=runtime.datasets.register(pd.DataFrame({'__period_max':['2024-02-29'],
                        '__period_count':[4],'__period_invalid':[0],'__period_formats':[1]}),
                        source='other.observations',query=envelope['query'],coverage='complete',
                        grain='aggregate',aggregation='MAX/COUNT')
                    return {'status':'ready','dataset':asdict(info)}
                runtime.remote_execute=mismatched
                result=observe(runtime,current('last year'),FIXTURE['source'],SPEC)
                self.assertEqual(result['status'],'invalid_receipt')
                self.assertNotIn('conditions',result)
                self.assertEqual(len(calls),1)
            finally:runtime.close()

    def test_result_only_goal_does_not_probe_or_widen_to_source_max(self):
        request='Plot reading for the last year from the displayed result'
        state=current(request);response=delta()
        response['restriction_review']={'clauses':[{'quote':'last year','kind':'relative_time',
            'columns':[FIXTURE['time_column']],'status':'unresolved','relative_period':SPEC}],
            'question':''}
        model=reader([response]);probe=Mock()
        model.context.runtime_services={'resolve_source_period':probe}
        plan=goal('chart',{'kind':'histogram'},columns=['reading'])
        plan['current_result_only']=True
        result=audit(model,state,{'tables':[{'columns':[{'name':FIXTURE['time_column']}]}]},plan)
        self.assertEqual(result['mode'],'clarify')
        self.assertEqual(result['tasks'],[])
        probe.assert_not_called()

    def test_audit_resolves_answer_without_asking_literal_dates_again(self):
        request='Plot reading for the last year\n추가 확인 답변: 데이터 마지막 날짜 기준이지'
        state=current(request)
        response=delta();response['restriction_review']={'clauses':[{
            'quote':'last year','kind':'relative_time','columns':[FIXTURE['time_column']],
            'status':'unresolved','relative_period':SPEC}],'question':''}
        model=reader([response])
        terms=[{'column':FIXTURE['time_column'],'op':'ge','value':'2023-02-28'},
               {'column':FIXTURE['time_column'],'op':'le','value':'2024-02-29'}]
        model.context.runtime_services={'resolve_source_period':Mock(return_value={'status':'ready','conditions':terms})}
        data={'tables':[{'table':FIXTURE['source'],'columns':[{'name':FIXTURE['time_column']}]}]}
        result=audit(model,state,data,goal('chart',{'kind':'histogram'},columns=['reading']))
        self.assertEqual(result['mode'],'execute');self.assertEqual(result['conditions'],terms)
        self.assertFalse(result['question'])
        self.assertEqual(data['resolved_temporal_population']['conditions'],terms)

    def test_unknown_submission_is_not_replayed_by_goal_repair(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,_=self.runtime(root)
            try:
                execute=Mock(side_effect=OSError('lost reply'))
                runtime.remote_execute=execute
                state=current('last year')
                self.assertEqual(observe(runtime,state,FIXTURE['source'],SPEC)['status'],'unavailable')
                self.assertEqual(observe(runtime,state,FIXTURE['source'],SPEC)['status'],'unknown_submission')
                state['request_id']='another-user-turn'
                self.assertEqual(observe(runtime,state,FIXTURE['source'],SPEC)['status'],'unknown_submission')
                self.assertEqual(execute.call_count,1)
                self.assertEqual(runtime.ledger.uncertain()[0]['status'],'unknown')
            finally:runtime.close()

    def test_failed_clarification_retains_original_beyond_history_window(self):
        state=current('데이터의 마지막 날짜 기준이에요')
        state['pending_clarification']={'question':'anchor?','legacy':True}
        model=reader([{'action':'answer','request_id':'original','answer_quote':state['request_text']}])
        resume(model,state,[HumanMessage(content='Plot reading for the last year',id='original')])
        state['status']='incomplete'
        saved=carry(state)
        self.assertIn('last year',saved['request_text'])
        self.assertIn('마지막 날짜 기준',saved['request_text'])
        self.assertFalse(saved['legacy'])

    def test_relative_placeholders_are_replaced_only_by_observed_bounds(self):
        request='Plot reading for the last year; source max date'
        state=current(request);response=delta()
        response.update(change='modify',evidence_quote=request,conditions=[{
            'column':FIXTURE['time_column'],'op':'between','value':['latest date','last year']}])
        response['restriction_review']={'clauses':[{'quote':'last year','kind':'relative_time',
            'columns':[FIXTURE['time_column']],'status':'unresolved','relative_period':SPEC}],
            'question':'anchor?'}
        model=reader([response]);terms=[{'column':FIXTURE['time_column'],'op':'ge','value':'2023-02-28'},
            {'column':FIXTURE['time_column'],'op':'le','value':'2024-02-29'}]
        model.context.runtime_services={'resolve_source_period':Mock(return_value={'status':'ready','conditions':terms})}
        result=audit(model,state,{'tables':[{'columns':[{'name':FIXTURE['time_column']}]}]},
            goal('chart',{'kind':'histogram'},columns=['reading']))
        self.assertEqual(result['conditions'],terms)
        self.assertEqual(model.population_model.invoke.call_count,1)

    def test_runtime_source_max_period_filters_rows_and_produces_png(self):
        request='Plot reading for the last year. 데이터의 마지막 날짜 기준이에요'
        response=delta();response['restriction_review']={'clauses':[{'quote':'last year',
            'kind':'relative_time','columns':[FIXTURE['time_column']],'status':'unresolved',
            'relative_period':SPEC}],'question':'anchor?'}
        plan=goal('chart',{'kind':'histogram'},columns=['reading'])
        model=GoalModel(goals=[plan,plan])
        calls=[]
        frame=pd.DataFrame({'reading':[100,1,2,3],FIXTURE['time_column']:[
            '2024-12-31','2025-01-01','2025-06-01','2026-01-01']})
        def factory(datasets):
            def execute(envelope):
                calls.append(envelope)
                from core.analysis_sql import validate_query
                validate_query(envelope['query'])
                receipt=datasets.register(pd.DataFrame({'__period_max':[frame[FIXTURE['time_column']].max()],
                    '__period_count':[len(frame)],'__period_invalid':[0],'__period_formats':[1]}),
                    source=FIXTURE['source'],query=envelope['query'],coverage='complete',grain='aggregate',aggregation='MAX/COUNT')
                return {'status':'ready','dataset':asdict(receipt)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','relative-png',model,
                connection_identity='fixture',remote_factory=factory,
                reference_context_loader=lambda:[{'table':FIXTURE['source'],'columns':[
                    {'name':'reading','dtype':'double'},{'name':FIXTURE['time_column'],'dtype':'string'}]}])
            try:
                raw=runtime.datasets.register(frame,source=FIXTURE['source'],coverage='complete',predicate_known=True)
                runtime.select_dataset(raw.id);digest=stored_dataset_digest(runtime.datasets,raw.id)
                runtime.recovery.goal_interpreter.population_model=reader([response]).population_model
                result=runtime.submit(request)
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual(state['temporal_resolution']['anchor'],'source_max')
                card=runtime.artifacts[state['artifact_ids'][-1]]
                counts=runtime.datasets.frames[card.dataset_id]
                self.assertEqual(int(counts['__frequency'].sum()),3)
                self.assertEqual(set(counts['reading']),{1,2,3})
                from utils.analysis_image_validation import validate_chart_image
                validate_chart_image(card.image)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),digest)
                self.assertEqual(len(calls),1)
                query=f'SELECT `reading`, COUNT(*) AS `__frequency` FROM `{FIXTURE["source"].split(".")[0]}`.`{FIXTURE["source"].split(".")[1]}` WHERE `reading` IS NOT NULL GROUP BY `reading`'
                self.assertFalse(runtime.recovery._scope_valid(query,state,'reading'))
            finally:runtime.close()

    def test_observed_date_bindings_do_not_approve_other_new_filters(self):
        from core.analysis_agent.goal_grounding import bind_observed_period,verify
        from tests.test_goal_decision_evidence import evidence
        original=goal('chart',{'kind':'histogram'},columns=['reading'])
        period={'column':FIXTURE['time_column'],'op':'ge','value':'2025-01-01'}
        revised=deepcopy(original);revised['conditions']=[period]
        data={'request':'last year reading histogram','resolved_temporal_population':{'conditions':[period]}}
        bindings=bind_observed_period(original,revised,evidence(original,data['request']),data)
        self.assertEqual(verify(revised,bindings,data)['status'],'supported')
        revised['conditions'].append({'column':'reading','op':'gt','value':100})
        bindings=bind_observed_period(original,revised,evidence(original,data['request']),data)
        with self.assertRaisesRegex(ValueError,'Missing decision evidence'):verify(revised,bindings,data)
