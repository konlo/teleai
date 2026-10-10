"""Cross-turn population and chart contracts, independent of natural wording."""
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pandas as pd
from core.analysis_agent.population_audit import reconcile
from core.analysis_agent.source_references import source_mentions, qualified_mentions
from core.analysis_agent.goal_contract import compile_goal
from tests.test_llm_goal import goal, GoalModel
from tests import test_llm_goal as llm_goals
from utils.analysis_grouped_distribution import render
from utils.analysis_datasets import DatasetStore


class PopulationPresentationTests(unittest.TestCase):
    def test_same_source_preview_inherits_population_regardless_of_reference_label(self):
        from core.analysis_agent.population_audit import applies
        prior={'required_sources':['lab.events'],'scope':{'conditions':[
            {'column':'reading','op':'ge','value':10}]}}
        preview=goal('row_preview',{'limit':10},sources=['lab.events'],columns=[])
        preview['source_reference']='explicit'
        self.assertTrue(applies(preview,{'confirmed_analysis':prior}))
        preview['sources']=['lab.other_events']
        self.assertFalse(applies(preview,{'confirmed_analysis':prior}))

    def test_catalog_subject_recheck_preserves_positive_and_excluded_literals(self):
        import json
        from unittest.mock import Mock
        from langchain_core.messages import AIMessage
        from core.analysis_agent.subject_identity import read
        for requested in ([], ['_events']):
            with self.subTest(requested=requested):
                model=Mock()
                model.invoke.side_effect=[AIMessage(content='{"candidate_roles":{"Inspect":"other","_events":"other","rows":"other"}}' if requested else
                    '{"candidate_roles":{"Exclude":"other","_events":"other"}}'),
                    AIMessage(content=json.dumps({'requested_identifiers':requested}))]
                interpreter=SimpleNamespace(selection_model=model, model_recovery=None,
                    diagnostics=SimpleNamespace(emit=Mock()),
                    budget=SimpleNamespace(wrap_model_call=lambda req,fn:fn(req)))
                current={'request_id':'one','request_text':
                    'Inspect _events rows' if requested else 'Exclude _events'}
                result=read(interpreter,current,{'literal_source_mentions':[
                    {'source':'lab._events','quote':'_events'}]})
                self.assertEqual(result,[{'name':n,'quote':n} for n in requested])
                self.assertEqual(model.invoke.call_count,2)

    def test_partial_delta_retains_an_unmentioned_range_and_requires_explicit_removal(self):
        age={'column':'signal','op':'between','value':[12,19]}
        group={'column':'segment','op':'eq','value':'A'}
        scope={'conditions':[age,group],'any_conditions':[]}
        delta={'conditions':[{'column':'segment','op':'in','value':['A','B']}],
            'any_conditions':[],'change':'modify','evidence_quote':'include B','removed_columns':[]}
        merged=reconcile(delta,scope,'include B with different colors',True)
        self.assertIn(age,merged['conditions'])
        self.assertNotIn(group,merged['conditions'])
        delta.update(conditions=[],removed_columns=['signal'],evidence_quote='remove signal filter')
        self.assertEqual(reconcile(delta,scope,'remove signal filter',True)['conditions'],[group])

    def test_scope_delta_keep_cannot_drop_prior_filters_and_clear_needs_current_quote(self):
        scope={'conditions':[{'column':'reading','op':'between','value':[12,19]}],'any_conditions':[]}
        response={'conditions':[],'any_conditions':[],'change':'keep','evidence_quote':''}
        self.assertEqual(reconcile(response,scope,'10행만 보여줘',True),scope)
        response['change']='clear'
        with self.assertRaises(ValueError):reconcile(response,scope,'10행만 보여줘',True)
        response['evidence_quote']='전체 데이터'
        with self.assertRaises(ValueError):reconcile(response,scope,'10행만 보여줘',True)
        self.assertEqual(reconcile(response,scope,'전체 데이터로 해줘',True)['conditions'],[])
        response['change']='new_source'
        with self.assertRaises(ValueError):reconcile(response,scope,'전체 데이터로 해줘',True)
        response.update(change='modify',evidence_quote='')
        self.assertEqual(reconcile(response,{},'분포를 그려줘',True),{'conditions':[],'any_conditions':[]})

    def test_qualified_namespace_cannot_ground_another_suffix(self):
        context=SimpleNamespace(reference_context=[{'table':'warehouse.lab.events'}],
            datasets=SimpleNamespace(metadata={}))
        self.assertEqual(source_mentions('wrong.lab.events 컬럼 보여줘',context),[])
        self.assertEqual(source_mentions('other.events 컬럼 보여줘',context),[])
        self.assertEqual(source_mentions('lab.events 컬럼 보여줘',context)[0]['source'],'warehouse.lab.events')
        self.assertEqual(qualified_mentions('`other`.`events` 컬럼')[0]['source'],'other.events')
        self.assertEqual(source_mentions('`wrong`.`lab`.`events` 컬럼',context),[])

    def test_equivalent_scope_is_kept_without_authorization_but_or_does_not_become_and(self):
        scope={'conditions':[{'column':'signal','op':'between','value':[12,19]},
                             {'column':'segment','op':'in','value':['A','B']}],'any_conditions':[]}
        audit={'conditions':[{'column':'segment','op':'in','value':['B','A']},
                              {'column':'signal','op':'ge','value':12},
                              {'column':'signal','op':'le','value':19}],
               'any_conditions':[],'change':'modify','evidence_quote':''}
        self.assertEqual(reconcile(audit,scope,'같은 조건으로 다시 그려줘',True),scope)
        audit['any_conditions']=audit.pop('conditions');audit['conditions']=[]
        with self.assertRaises(ValueError):reconcile(audit,scope,'같은 조건으로 다시 그려줘',True)

    def test_task_selection_grounds_unknown_table_and_narrows_chart_grammar(self):
        from core.analysis_agent.task_selection import validate,verify_goal,schema_for
        from core.analysis_agent.goal_contract import response_schema
        selected={'mode':'execute','capabilities':['row_count'],'source_reference':'explicit',
            'source_mentions':[{'name':'missing_events','quote':'missing_events'}],
            'chart_kind':''}
        selected={**validate(selected,'missing_events 행 수 알려줘'),'current_result_only':False}
        context=SimpleNamespace(reference_context=[{'table':'lab.events'}],datasets=SimpleNamespace(metadata={}))
        with self.assertRaises(ValueError):verify_goal(goal('row_count',sources=['lab.events']),selected,context)
        verify_goal(goal('row_count',sources=['missing_events']),selected,context)
        schema=response_schema(['chart'],'execute','scatter')
        self.assertEqual(schema['properties']['tasks']['minItems'],1)
        variants=schema['properties']['tasks']['items']['anyOf'][0]['properties']['options']['anyOf']
        self.assertEqual(len(variants),1)
        for key in ('legend','stacked','category','palette','bins'):
            self.assertNotIn(key,variants[0]['properties'])
        initial=schema_for({'selected_dataset':None,'verified_previous':{},'requested_previous_subject':None})['anyOf'][0]
        self.assertEqual(initial['properties']['source_reference']['enum'],['explicit'])
        inventory={**{k:v for k,v in selected.items() if k!='current_result_only'},'capabilities':['table_list'],'source_reference':'selected_dataset',
                   'source_mentions':[]}
        fixed=validate(inventory,'사용할 자료 목록',{'selected_dataset':None})
        self.assertEqual(fixed['source_reference'],'explicit')
        from core.analysis_agent.subject_identity import candidates
        literal=candidates('`unknown`.`events`의 signal 컬럼과 balance를 보여줘')
        self.assertEqual(literal[0]['source'],'unknown.events')
        self.assertEqual([x['source'] for x in literal],['unknown.events','signal','balance'])

    def test_null_predicate_serializer_uses_is_null(self):
        from core.analysis_agent.predicate_sql import where_sql
        for dialect in ('mysql','databricks'):
            self.assertIn('IS NULL',where_sql([{'column':'reading','op':'eq','value':None}],dialect))
            self.assertIn('NOT',where_sql([{'column':'reading','op':'ne','value':None}],dialect))

    def test_metadata_followup_keeps_population_across_completion_and_restart(self):
        with tempfile.TemporaryDirectory() as root:
            harness=llm_goals.LLMGoalTests();r,raw=harness.runtime(root,GoalModel(goals=[goal('metadata',{'kind':'columns'})]))
            try:
                current={'request_id':'previous','status':'complete','required_sources':['lab.observations'],
                    'scope':{'sources':['lab.observations'],'conditions':[{'column':'reading','op':'ge','value':3}],
                             'any_conditions':[],'measure_conditions':[],'ratio':None}}
                compiled=compile_goal({'confirmed_analysis':current},goal('metadata',{'kind':'columns'}),r.context)
                self.assertEqual(compiled['scope']['conditions'],current['scope']['conditions'])
                other=goal('metadata',{'kind':'columns'},sources=['lab.other'])
                self.assertFalse(compile_goal({'confirmed_analysis':current},other,r.context)['scope']['conditions'])
            finally:r.close()

    def test_stacked_png_preserves_independent_series_and_rejects_old_card(self):
        with tempfile.TemporaryDirectory() as root:
            harness=llm_goals.LLMGoalTests();r,raw=harness.runtime(root,GoalModel(goals=[goal('metadata',{'kind':'columns'})]))
            try:
                query='SELECT reading, cohort, COUNT(*) AS __frequency FROM lab.observations GROUP BY reading, cohort'
                data=pd.DataFrame({'reading':[1,2,1,2],'cohort':['A','A','B','B'],'__frequency':[2,3,5,7]})
                info=r.datasets.register(data,source='lab.observations',query=query,grain='aggregate',aggregation=query,coverage='complete')
                old=render(r.datasets,info.id,'reading','cohort','__frequency',4)
                card=render(r.datasets,info.id,'reading','cohort','__frequency',4,stacked=True)
                self.assertTrue(card.image.startswith(b'\x89PNG'))
                self.assertEqual(old.render_spec['series_counts'],card.render_spec['series_counts'])
                self.assertEqual(card.render_spec['total_count'],17)
                self.assertTrue(card.render_spec['stacked'])
                state={'kind':'histogram','required_columns':['reading','cohort'],
                    'chart_group_spec':{'value_column':'reading','category':'cohort','bins':4,'stacked':True,'legend':True}}
                self.assertFalse(r.recovery._valid_card(old,state,info.id))
                self.assertTrue(r.recovery._valid_card(card,state,info.id))
                contrast=render(r.datasets,info.id,'reading','cohort','__frequency',4,stacked=True,palette='high_contrast')
                self.assertNotEqual(contrast.image,card.image)
                self.assertNotEqual(contrast.render_spec['colors'],card.render_spec['colors'])
                self.assertEqual(contrast.render_spec['series_counts'],card.render_spec['series_counts'])
                state['chart_group_spec']['palette']='high_contrast'
                self.assertFalse(r.recovery._valid_card(card,state,info.id))
                self.assertTrue(r.recovery._valid_card(contrast,state,info.id))
            finally:r.close()

    def test_source_count_uses_one_exact_query_without_loading_raw(self):
        from dataclasses import asdict
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        queries=[];source='lab.counter_events'
        observed=[{'table':source,'columns':[{'name':'reading','dtype':'double'}]}]
        plan=goal('row_count',sources=[source],conditions=[{'column':'reading','op':'between','value':[12,19]}])
        with tempfile.TemporaryDirectory() as root:
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    self.assertIn('COUNT(*)',envelope['query'])
                    self.assertIn('`reading` >= 12',envelope['query'])
                    self.assertIn('`reading` <= 19',envelope['query'])
                    info=store.register(pd.DataFrame({'count':[716]}),source=source,query=envelope['query'],coverage='complete',grain='aggregate')
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            r=GraphAnalysisRuntime(root,'owner','count',GoalModel(goals=[plan]),sql_dialect='mysql',
                reference_context_loader=lambda:observed,connection_identity='fixture',remote_factory=factory)
            try:
                result=r.submit('조건에 맞는 행 수를 알려줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),1)
                self.assertEqual(len(r.datasets.metadata),1)
                self.assertEqual(next(iter(r.datasets.metadata.values())).rows,1)
                self.assertIn('716',result['text'])
            finally:r.close()

    def test_filtered_coordinate_scatter_keeps_population_and_reuses_exact_receipt(self):
        from dataclasses import asdict
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        from core.analysis_agent.source_scatter import WEIGHT, valid_card
        data=pd.DataFrame({'signal':[1,2,2,3,4],'response':[9,8,8,7,6]})
        source='lab.measurements';queries=[]
        observed=[{'table':source,'columns':[{'name':c,'dtype':'double'} for c in data.columns]}]
        plan=goal('chart',{'kind':'scatter','axes':{'x':'signal','y':'response'}},
            sources=[source],columns=['signal','response'],conditions=[{'column':'signal','op':'ge','value':2}])
        with tempfile.TemporaryDirectory() as root:
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    self.assertIn('`signal` >= 2',envelope['query'])
                    frame=data[data.signal>=2].groupby(['signal','response']).size().reset_index(name=WEIGHT)
                    info=store.register(frame,source=source,query=envelope['query'],coverage='complete',grain='aggregate')
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            r=GraphAnalysisRuntime(root,'owner','filtered',GoalModel(goals=[plan]),sql_dialect='mysql',
                reference_context_loader=lambda:observed,connection_identity='fixture',remote_factory=factory)
            try:
                result=r.submit('signal이 2 이상인 데이터의 signal과 response 관계를 그려줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery'];card=r.artifacts[state['artifact_ids'][0]]
                self.assertEqual(card.render_spec['drawable_rows'],4)
                self.assertTrue(valid_card(r.context,state,card))
                wrong=deepcopy(state);wrong['scope']['conditions']=[]
                self.assertFalse(valid_card(r.context,wrong,card))
                result=r.submit('동일한 조건의 관계를 다시 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),1)
            finally:r.close()
