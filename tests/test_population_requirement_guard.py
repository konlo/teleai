"""Original restrictions cannot vanish behind an unchanged-filter decision."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

import pandas as pd
from langchain_core.messages import AIMessage
from core.analysis_agent.population_audit import audit
from core.analysis_agent.population_basis import read
from core.analysis_agent.population_requirements import validate
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import goal,GoalModel
from tests.test_population_audit_local_repair import delta
from utils.analysis_datasets import stored_dataset_digest

FIXTURE=json.loads((Path(__file__).parent/'fixtures/temporal_population_requirements.json').read_text())

def reader(replies):
    model=Mock(spec=['invoke'])
    model.invoke.side_effect=[AIMessage(content=json.dumps(v)) for v in replies]
    return SimpleNamespace(population_model=model,selection_model=model,model_recovery=None,
        diagnostics=Mock(),context=SimpleNamespace(reference_context=[],datasets=SimpleNamespace(metadata={})),
        budget=SimpleNamespace(wrap_model_call=lambda req,fn:fn(req)))

def current(request):
    return {'request_id':'new','request_text':request,'confirmed_analysis':{'status':'complete',
        'required_sources':[FIXTURE['source']], 'scope':{'conditions':[],'any_conditions':[]}}}

def unresolved(quote):
    value=delta()
    value['restriction_review']={'clauses':[{'quote':quote,'kind':'relative_time',
        'columns':[],'status':'unresolved'}],'question':FIXTURE['question']}
    return value

class PopulationRequirementGuardTests(unittest.TestCase):
    def test_stray_reference_is_repaired_without_regenerating_filter_meaning(self):
        prompt=FIXTURE['ambiguous_requests'][0]['request']
        model=reader([{'basis':'source_population','quote':'','changes_filters':True,
            'filter_change_quote':prompt,'output_reference_id':'wrong-display'}])
        result=read(model,current(prompt),{'verified_previous':{'chart_available':True}})
        self.assertTrue(result['changes_filters'])
        self.assertEqual(result['filter_change_quote'],prompt)
        self.assertNotIn('output_reference',result)
        self.assertEqual(model.selection_model.invoke.call_count,1)

    def test_unchanged_selector_cannot_skip_independent_temporal_review(self):
        for case in FIXTURE['ambiguous_requests']:
            with self.subTest(prompt=case['request']):
                model=reader([unresolved(case['quote'])]);state=current(case['request'])
                plan=goal('chart',{'kind':'histogram'},columns=[FIXTURE['measure']])
                result=audit(model,state,{},plan,{'population_basis':'source_population','changes_filters':False})
                self.assertEqual(result['mode'],'clarify')
                self.assertEqual(result['tasks'],[])
                self.assertEqual(result['question'],FIXTURE['question'])
                self.assertEqual(state['confirmed_analysis']['scope']['conditions'],[])
                self.assertEqual(model.population_model.invoke.call_count,1)

    def test_missing_review_cannot_fall_back_to_empty_previous_filters(self):
        bad=delta();bad.pop('restriction_review')
        model=reader([bad,bad])
        with self.assertRaisesRegex(ValueError,'restriction review contract'):
            audit(model,current(FIXTURE['ambiguous_requests'][0]['request']),{},
                goal('chart',{'kind':'histogram'}),{'changes_filters':False})

    def test_model_echo_is_not_used_as_the_clarification(self):
        case=FIXTURE['ambiguous_requests'][0]
        response=unresolved(case['quote'])
        response['restriction_review']['question']=case['request']
        model=reader([response])
        result=audit(model,current(case['request']),{},goal('chart',{'kind':'histogram'}))
        self.assertEqual(result['question'],FIXTURE['question'])
        self.assertNotEqual(result['question'],case['request'])

    def test_support_summary_reports_clarification_without_request_quotes(self):
        from core.analysis_agent.support_report import summarize,brief
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'runtime.jsonl'
            events=[{'event':'run_started','run_id':'a'*32},
                {'event':'goal_restrictions_need_clarification','run_id':'a'*32,
                 'question':'PRIVATE REQUEST','clause_count':1,'unresolved_count':1},
                {'event':'run_completed','run_id':'a'*32,'status':'needs_context'}]
            path.write_text('\n'.join(json.dumps(e) for e in events))
            report=summarize(path)
            self.assertEqual(report['status'],'needs_context')
            self.assertEqual(report['restriction_review']['clarifications'],1)
            self.assertNotIn('PRIVATE REQUEST',brief(report))

    def test_absolute_dates_have_two_boundaries_and_sql_scope_must_match(self):
        column=FIXTURE['time_column'];request=FIXTURE['explicit_request']
        conditions=[{'column':column,'op':'between','value':['2025-01-01','2026-01-01']}]
        review={'clauses':[{'quote':'2025-01-01부터 2026-01-01까지','kind':'absolute_time',
            'columns':[column],'status':'resolved'}],'question':''}
        raw={'change':'modify','conditions':conditions,'any_conditions':[],'removed_columns':[]}
        result={k:raw[k] for k in ('conditions','any_conditions')}
        validate(review,request,result,raw,[{'columns':[{'name':column}]}])
        from core.analysis_agent.intent_scope import scope_matches
        compiled={'conditions':[{'column':column,'op':'ge','value':'2025-01-01'},
            {'column':column,'op':'le','value':'2026-01-01'}],'any_conditions':[]}
        self.assertFalse(scope_matches('SELECT reading FROM lab.observations',compiled))
        self.assertTrue(scope_matches("SELECT reading FROM lab.observations WHERE captured_at BETWEEN '2025-01-01' AND '2026-01-01'",compiled))
        schema=[{'columns':[{'name':column}]}]
        with self.assertRaisesRegex(ValueError,'no matching population predicate'):
            validate(review,request,{'conditions':[],'any_conditions':[]},raw,schema)
        raw['conditions']=[{'column':column,'op':'ge','value':'2025-01-01'}]
        with self.assertRaisesRegex(ValueError,'both boundaries'):
            validate(review,request,{k:raw[k] for k in result},raw,schema)

    def test_relative_dates_cannot_be_invented_as_absolute_resolution(self):
        request=FIXTURE['ambiguous_requests'][2]['request']
        column=FIXTURE['time_column']
        condition={'column':column,'op':'between','value':['2025-01-01','2026-01-01']}
        population={'conditions':[condition],'any_conditions':[]}
        for kind in ('absolute_time','relative_time'):
            review={'clauses':[{'quote':'last year','kind':kind,'columns':[column],
                'status':'resolved'}],'question':''}
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                validate(review,request,population,{'change':'modify'},[{'columns':[{'name':column}]}])

    def test_real_runtime_clarifies_before_any_remote_submission_and_keeps_original(self):
        request=FIXTURE['ambiguous_requests'][2]['request']
        plan=goal('chart',{'kind':'histogram'},columns=[FIXTURE['measure']])
        remote=Mock(side_effect=AssertionError('Unresolved time must not query'))
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','period',GoalModel(goals=[plan,plan]),
                connection_identity='fixture',remote_factory=lambda datasets:remote)
            try:
                info=runtime.datasets.register(pd.DataFrame({'reading':[1,2,3],
                    'captured_at':['2025-01-01','2025-06-01','2026-01-01']}),
                    source=FIXTURE['source'],coverage='complete',predicate_known=True)
                runtime.select_dataset(info.id)
                before=stored_dataset_digest(runtime.datasets,info.id)
                runtime.recovery.goal_interpreter.population_model=reader([unresolved('last year')]).population_model
                result=runtime.submit(request)
                self.assertEqual(runtime.inspect()['recovery']['goal']['mode'],'clarify',result)
                self.assertFalse(runtime.inspect()['recovery']['sent_calls'])
                remote.assert_not_called()
                self.assertEqual(stored_dataset_digest(runtime.datasets,info.id),before)
                self.assertEqual(runtime.context.selected_dataset_id,info.id)
                self.assertEqual(len(runtime.artifacts),0)
            finally:runtime.close()

    def test_explicit_date_followup_restores_period_and_renders_only_matching_rows(self):
        request=FIXTURE['explicit_request'];column=FIXTURE['time_column']
        conditions=[{'column':column,'op':'between','value':['2025-01-01','2026-01-01']}]
        response=delta();response.update(change='modify',evidence_quote=request,conditions=conditions)
        response['restriction_review']={'clauses':[{'quote':'2025-01-01부터 2026-01-01까지',
            'kind':'absolute_time','columns':[column],'status':'resolved'}],'question':''}
        # Deliberately omit dates from the initial goal: the independent reader
        # must restore them before any execution, rather than trusting that goal.
        plan=goal('chart',{'kind':'histogram'},columns=[FIXTURE['measure']])
        from copy import deepcopy
        from tests.test_goal_decision_evidence import evidence
        corrected=deepcopy(plan);corrected['conditions']=conditions
        plan['decision_evidence']=evidence(plan,request)
        corrected['decision_evidence']=evidence(corrected,request)
        model=GoalModel(goals=[plan,plan,corrected])
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','absolute',model)
            try:
                info=runtime.datasets.register(pd.DataFrame({'reading':[100,1,2,200],
                    'captured_at':['2024-12-31','2025-01-01','2025-06-01','2026-01-02']}),
                    source=FIXTURE['source'],coverage='complete',predicate_known=True)
                runtime.select_dataset(info.id);before=stored_dataset_digest(runtime.datasets,info.id)
                population_reader=reader([response]).population_model
                runtime.recovery.goal_interpreter.population_model=population_reader
                result=runtime.submit(request)
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual({p['op'] for p in state['scope']['conditions']},{'ge','le'})
                card=runtime.artifacts[state['artifact_ids'][-1]]
                counts=runtime.datasets.frames[card.dataset_id]
                self.assertEqual(int(counts['__frequency'].sum()),2)
                self.assertEqual(set(counts[FIXTURE['measure']]),{1,2})
                from utils.analysis_image_validation import validate_chart_image
                validate_chart_image(card.image)
                self.assertEqual(stored_dataset_digest(runtime.datasets,info.id),before)
                self.assertEqual(model.goal_calls,3)
                self.assertEqual(population_reader.invoke.call_count,1)
            finally:runtime.close()
