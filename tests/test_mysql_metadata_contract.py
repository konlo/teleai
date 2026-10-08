"""Engine-neutral schema requests and MySQL metadata evidence contracts."""
from datetime import datetime, timezone
from dataclasses import asdict
import tempfile
import unittest

import pandas as pd
from langchain_core.messages import HumanMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_metadata_discovery import make_plan, stored_column_definitions
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_charts import histogram_from_counts
from utils.analysis_datasets import stored_dataset_digest


class NoInference(EvaluationModel):
    def _generate(self, *args, **kwargs):
        raise AssertionError('Observed schema requests must not loop through inference')


class MySQLMetadataContractTests(unittest.TestCase):
    def context(self, dialect):
        prefix = 'lab' if dialect == 'mysql' else 'catalog.lab'
        return [{'table':prefix+'.'+table, 'observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':name, 'dtype':dtype} for name, dtype in columns]}
                for table, columns in [('observations', [('reading','double'), ('cohort','string'),
                    ('event_time','timestamp')]), ('other', [('different','int')])]]

    def test_elliptical_column_listing_after_chart_uses_source_schema_for_both_backends(self):
        for dialect in ['mysql', 'databricks']:
            with self.subTest(dialect=dialect), tempfile.TemporaryDirectory() as root:
                context = self.context(dialect)
                r = GraphAnalysisRuntime(root, 'owner', 'followup', NoInference(), sql_dialect=dialect,
                    reference_context_loader=lambda:context, summary_trigger_tokens=1,intent_mode='contract_fixture')
                try:
                    source = context[0]['table']
                    raw = r.datasets.register(pd.DataFrame({'reading':[1,2], 'cohort':['A','B']}),
                        source=source, coverage='complete', predicate_known=True)
                    query = f'SELECT reading, COUNT(*) AS __frequency FROM {source} GROUP BY reading'
                    agg = r.datasets.register(pd.DataFrame({'reading':[1,2], '__frequency':[1,1]}),
                        source=source, query=query, grain='aggregate', aggregation=query,
                        coverage='complete', predicate_known=True)
                    card = histogram_from_counts(r.datasets,agg.id,'reading','__frequency')
                    r.artifacts[card.id] = card
                    r.select_chart(card.id)
                    digest = stored_dataset_digest(r.datasets,raw.id)
                    for prompt in ['테이블의 column 다시 보여줘', 'columns를 다시 보여줘',
                                   '컬럼을 다시 알려줘', 'show the table column', 'fields를 확인해줘',
                                   '이 테이블의 column들을 보여줘','이 테이블의 fields들을 보여줘']:
                        result = r.submit(prompt)
                        self.assertEqual(result['status'], 'answered', result)
                        state = r.inspect()['recovery']
                        self.assertEqual(state['metadata_kind'],'columns')
                        self.assertEqual(state['required_sources'],[source])
                        self.assertEqual(state['metadata_evidence']['columns'],['reading','cohort','event_time'])
                        self.assertNotIn('__frequency',result['text'])
                        self.assertEqual(state['model_calls'],0)
                        self.assertEqual(r.context.selected_dataset_id,agg.id)
                        self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                    self.assertEqual(r.submit('other 컬럼을 보여줘')['status'],'answered')
                    result = r.submit('테이블의 column 다시 보여줘')
                    self.assertIn(context[1]['table'],result['text'])
                    self.assertIn('different',result['text'])
                    r.close()
                    r = GraphAnalysisRuntime(root,'owner','followup',NoInference(),sql_dialect=dialect,
                        reference_context_loader=lambda:context,summary_trigger_tokens=1,intent_mode='contract_fixture')
                    self.assertIn('different',r.submit('column 다시 보여줘')['text'])
                finally:
                    r.close()

    def test_ambiguous_table_requires_context_without_inference(self):
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root,'owner','ambiguous',NoInference(),sql_dialect='mysql',
                reference_context_loader=lambda:self.context('mysql'),intent_mode='contract_fixture')
            try:
                result = r.submit('테이블의 column 다시 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertFalse(r.inspect()['recovery'].get('metadata_evidence'))
                self.assertEqual(r.inspect()['recovery']['status'],'needs_context')
                self.assertIn('어느 테이블',result['text'])
                self.assertEqual(r.submit('observations 컬럼을 보여줘')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['metadata_evidence']['columns'],
                                 ['reading','cohort','event_time'])
            finally:
                r.close()

    def test_old_failed_column_checkpoint_resumes_without_model_or_budget_reset(self):
        with tempfile.TemporaryDirectory() as root:
            context = self.context('mysql')
            r = GraphAnalysisRuntime(root,'owner','paused',NoInference(),sql_dialect='mysql',
                reference_context_loader=lambda:context,summary_trigger_tokens=1,intent_mode='contract_fixture')
            try:
                info = r.datasets.register(pd.DataFrame({'reading':[1]}),source=context[0]['table'])
                r.select_dataset(info.id)
                human = HumanMessage(id='legacy-schema',content='테이블의 column 다시 보여줘')
                r.agent.update_state(r.config,{'messages':[human], 'recovery':{
                    'request_id':human.id,'status':'working','metadata_kind':None,'chart':False,
                    'required_sources':[],'operations':[],'model_calls':3,'model_seconds':200.}},
                    as_node='ObservedSummarizationMiddleware.before_model')
                self.assertEqual(r.agent.get_state(r.config).next,('model',))
                r.close()
                r = GraphAnalysisRuntime(root,'owner','paused',NoInference(),sql_dialect='mysql',
                    reference_context_loader=lambda:context,summary_trigger_tokens=1,intent_mode='contract_fixture')
                result = r.resume()
                self.assertEqual(result['status'],'answered',result)
                self.assertIn('event_time',result['text'])
                self.assertEqual(r.inspect()['recovery']['model_calls'],3)
                self.assertEqual(r.context.selected_dataset_id,info.id)
                self.assertEqual(set(r.datasets.metadata),{info.id})
            finally:
                r.close()

    def test_stale_elliptical_schema_refreshes_zero_rows_and_preserves_selected_aggregate(self):
        for dialect in ['mysql','databricks']:
            with self.subTest(dialect=dialect), tempfile.TemporaryDirectory() as root:
                context = self.context(dialect)
                context[0]['observed_at'] = '2000-01-01T00:00:00Z'
                source = context[0]['table']
                queries = []
                def factory(store):
                    def execute(envelope):
                        queries.append(envelope['query'])
                        self.assertEqual(envelope['source'],source)
                        self.assertEqual(envelope['query'],
                            'SELECT * FROM '+'.'.join('`'+p+'`' for p in source.split('.'))+' LIMIT 0')
                        info = store.register(pd.DataFrame({'new_field':pd.Series([],dtype='float64')}),
                            source=source,query=envelope['query'],snapshot=datetime.now(timezone.utc).isoformat())
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                r = GraphAnalysisRuntime(root,'owner','stale',NoInference(),sql_dialect=dialect,
                    reference_context_loader=lambda:context,connection_identity='fixture',remote_factory=factory,intent_mode='contract_fixture')
                try:
                    query=f'SELECT reading, COUNT(*) AS __frequency FROM {source} GROUP BY reading'
                    agg=r.datasets.register(pd.DataFrame({'reading':[1],'__frequency':[2]}),source=source,
                        query=query,grain='aggregate',aggregation=query,coverage='complete')
                    r.select_dataset(agg.id)
                    result=r.submit('테이블의 column 다시 보여줘')
                    self.assertEqual(result['status'],'answered',result)
                    self.assertEqual(r.inspect()['recovery']['metadata_evidence']['columns'],['new_field'])
                    self.assertEqual(r.context.selected_dataset_id,agg.id)
                    self.assertEqual(r.inspect()['recovery']['model_calls'],0)
                    self.assertEqual(len(queries),1)
                    self.assertIn('new_field',r.submit('column 다시 보여줘')['text'])
                    self.assertEqual(len(queries),1)
                finally:r.close()

    def test_unique_value_request_is_not_misclassified_as_column_listing(self):
        with tempfile.TemporaryDirectory() as root:
            model = NoInference()
            runtime = GraphAnalysisRuntime(root, 'owner', 'distinct', model, sql_dialect='mysql',intent_mode='contract_fixture')
            try:
                original = runtime.datasets.register(pd.DataFrame({'reading':[1, 1, 2]}),
                    source='lab.observations', coverage='complete', predicate_known=True)
                runtime.select_dataset(original.id)
                result = runtime.submit('observations reading 컬럼들의 고유값 개수를 알려줘')
                self.assertEqual(result['status'], 'answered', result)
                state = runtime.inspect()['recovery']
                self.assertIsNone(state['metadata_kind'])
                self.assertEqual(state['profile_evidence']['profile']['columns'][0]['distinct_count'], 2)
                self.assertIn('reading: 고유값 2개', result['text'])
            finally:
                runtime.close()

    def test_plural_schema_request_uses_observed_schema_for_both_engines(self):
        for dialect, target in [('mysql', 'lab.observations'),
                                ('databricks', 'catalog.lab.observations')]:
            for request in ['observations 컬럼들을 보여줘', 'observations 필드들을 알려줘',
                            'show columns for observations']:
                with self.subTest(dialect=dialect, request=request), tempfile.TemporaryDirectory() as root:
                    runtime = GraphAnalysisRuntime(root, 'owner', 'schema', NoInference(),
                        sql_dialect=dialect, reference_context_loader=lambda: [{
                            'table':target, 'observed_at':datetime.now(timezone.utc).isoformat(),
                            'columns':[{'name':'reading', 'dtype':'double'},
                                       {'name':'event_time', 'dtype':'timestamp'}]}],intent_mode='contract_fixture')
                    try:
                        original = runtime.datasets.register(pd.DataFrame({'keep':[2, 3]}),
                            source='local.protected', coverage='complete', predicate_known=True)
                        runtime.select_dataset(original.id)
                        result = runtime.submit(request)
                        self.assertEqual(result['status'], 'answered', result)
                        self.assertIn('reading', result['text'])
                        self.assertIn('event_time', result['text'])
                        self.assertEqual(runtime.inspect()['recovery']['metadata_kind'], 'columns')
                        self.assertEqual(runtime.context.selected_dataset_id, original.id)
                        self.assertEqual(set(runtime.datasets.metadata), {original.id})
                    finally:
                        runtime.close()

    def test_mysql_column_comments_are_bound_to_exact_query_table_and_freshness(self):
        target = 'lab.observations'
        plan = make_plan(target, dialect='mysql')['metadata_plan']
        self.assertEqual(plan['source'], 'information_schema.columns')
        self.assertIn('column_comment AS comment', plan['query'])
        self.assertEqual(make_plan(target)['status'], 'needs_context')
        self.assertEqual(make_plan('catalog.lab.observations', dialect='mysql')['status'], 'needs_context')
        for variant in ['valid', 'wrong_table', 'wrong_query', 'stale', 'full_page']:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as root:
                runtime = GraphAnalysisRuntime(root, 'owner', 'comments', EvaluationModel(), sql_dialect='mysql',intent_mode='contract_fixture')
                try:
                    row = dict(table_catalog='', table_schema='lab', table_name='observations',
                        column_name='reading', data_type='double', comment='sensor reading')
                    if variant == 'wrong_table':
                        row['table_name'] = 'another'
                    runtime.datasets.register(pd.DataFrame([row]*(65 if variant=='full_page' else 1)),
                        source=plan['source'], query=plan['query']+(' ' if variant=='wrong_query' else ''),
                        snapshot='2000-01-01T00:00:00Z' if variant=='stale' else datetime.now(timezone.utc).isoformat())
                    result = stored_column_definitions(runtime.datasets, target, dialect='mysql')
                    if variant == 'valid':
                        self.assertEqual(result['columns'][0]['description'], 'sensor reading')
                    else:
                        self.assertIsNone(result)
                finally:
                    runtime.close()
