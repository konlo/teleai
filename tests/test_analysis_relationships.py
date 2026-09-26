"""Independent FK discovery, scope, approval and metadata/raw separation checks."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from core.analysis_agent.intent_scope import scope_matches
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_relationships import inspect_relationships, metadata_join_scope, relationship_plan, stored_relationships
from scripts.evaluate_spider2_teleai import schema_context
from tests.test_actual_agent_evaluation import EvaluationModel
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore, stored_dataset_digest


class RelationshipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.sqlite'
        with sqlite3.connect(self.path) as db:
            db.executescript('''
                CREATE TABLE labels (key INTEGER PRIMARY KEY, label TEXT);
                CREATE TABLE events (left_key INTEGER REFERENCES labels(key), amount REAL);
                INSERT INTO labels VALUES (1,'A'),(2,'B');
                INSERT INTO events VALUES (1,2),(1,4),(2,8),(NULL,9);
            ''')
        self.references = schema_context(self.path)
        self.context = AnalysisToolContext(DatasetStore(), {}, self.references, lambda **_: None)
        self.query = "SELECT COUNT(*) AS count FROM events e JOIN labels l ON e.left_key=l.key WHERE l.label='A'"
        self.scope = {'conditions':[{'column':'label','op':'eq','value':'A'}],
                      'any_conditions':[], 'unresolved':[], 'join_edges':[]}

    def test_database_metadata_supplies_exact_relationship_and_preserves_scope(self):
        observed = inspect_relationships(self.context, 'events')
        self.assertEqual(observed['status'], 'ready')
        self.assertEqual(observed['foreign_keys'][0]['columns'], ['left_key'])
        grounded = metadata_join_scope(self.query, self.context, self.scope, dialect='sqlite')
        self.assertTrue(scope_matches(self.query, grounded, dialect='sqlite'))
        self.assertFalse(scope_matches(self.query.replace("='A'", "='B'"), grounded, dialect='sqlite'))
        self.assertEqual(self.scope['conditions'][0]['column'], 'label')
        self.assertFalse(self.scope['join_edges'])
        self.assertIsNone(metadata_join_scope(self.query,self.context,self.scope,dialect='sqlite',
                                             required_sources=['events']))
        self.assertIsNone(metadata_join_scope(self.query,self.context,self.scope,dialect='sqlite',
                                             required_sources=['events','labels','unrequested']))

    def test_stale_ambiguous_unobserved_wrong_key_and_wrong_join_are_blocked(self):
        for variant in ('stale','ambiguous','unobserved','malformed','wrong_key','outer','cross','extra_on','unknown_column','extra_where'):
            with self.subTest(variant=variant):
                context = AnalysisToolContext(DatasetStore(), {}, deepcopy(self.references), lambda **_: None)
                events = next(c for c in context.reference_context if c['table']=='events')
                query = self.query
                if variant=='stale':events['observed_at']='2000-01-01T00:00:00Z'
                if variant=='ambiguous':events['foreign_keys'].append({**events['foreign_keys'][0], 'columns':['amount']})
                if variant=='unobserved':events.pop('relationship_authority')
                if variant=='malformed':events['foreign_keys'].append({'target_table':'labels','columns':['amount']})
                if variant=='wrong_key':query=query.replace('e.left_key=l.key','e.amount=l.key')
                if variant=='outer':query=query.replace('JOIN labels','LEFT JOIN labels')
                if variant=='cross':query='SELECT COUNT(*) FROM events e CROSS JOIN labels l'
                if variant=='extra_on':query=query.replace('e.left_key=l.key',"e.left_key=l.key AND l.label='B'")
                if variant=='unknown_column':query=query.replace('e.left_key=l.key','e.missing=l.key')
                if variant=='extra_where':query += ' AND e.amount>2'
                grounded=metadata_join_scope(query,context,self.scope,dialect='sqlite')
                self.assertTrue(grounded is None or not scope_matches(query,grounded,dialect='sqlite'))

    def test_composite_key_requires_every_component_and_implicit_pk_is_resolved(self):
        with sqlite3.connect(self.path) as db:
            db.executescript('''
                CREATE TABLE pair (part INTEGER, seq INTEGER, PRIMARY KEY(part, seq));
                CREATE TABLE detail (p INTEGER, s INTEGER, FOREIGN KEY(p,s) REFERENCES pair);
            ''')
        self.context.reference_context = schema_context(self.path)
        query='SELECT COUNT(*) FROM detail d JOIN pair p ON d.p=p.part AND d.s=p.seq'
        grounded=metadata_join_scope(query,self.context,{'conditions':[]},dialect='sqlite')
        self.assertTrue(scope_matches(query,grounded,dialect='sqlite'))
        self.assertIsNone(metadata_join_scope(query.replace(' AND d.s=p.seq',''),self.context,{},dialect='sqlite'))

    def test_real_graph_discovers_then_stages_only_scope_correct_sql(self):
        for variant in ('valid','wrong_filter','missing_where','no_join_requested','missing_source'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as root:
                query=self.query
                if variant=='wrong_filter':query=query.replace("='A'","='B'")
                if variant=='missing_where':query=query.split(' WHERE ')[0]
                model=EvaluationModel(calls=[{'name':'inspect_table_relationships','args':{'table':'events'}},
                    {'name':'query_databricks','args':{'source':'events | labels','query':query,'reason':'Joined count'}}])
                r=GraphAnalysisRuntime(root,'owner',variant,model,reference_context_loader=lambda:self.references,
                    sql_dialect='sqlite',connection_identity='local-public-test',
                    remote_factory=lambda _: lambda request:self.fail('Unapproved SQL'))
                try:
                    prompt="Count records with label='A' after join events and labels"
                    if variant=='no_join_requested':prompt="Count records with label='A' from events and labels"
                    if variant=='missing_source':prompt="Count records after join events"
                    result=r.submit(prompt)
                    self.assertEqual(result['status']=='awaiting_approval',variant=='valid',result)
                    self.assertEqual(bool(result.get('requests')),variant=='valid')
                    if variant=='valid':
                        self.assertEqual(r.inspect()['recovery']['join_relationship_basis'],'fresh_database_catalog')
                        self.assertNotEqual(r.respond(result['requests'][0]['id'],approved=False)['status'],'answered')
                finally:r.close()

    def test_approved_relationship_metadata_survives_restart_preserves_raw(self):
        target='catalog.schema.events';plan=relationship_plan(target)['metadata_plan'];calls=[]
        stamp=datetime.now(timezone.utc).isoformat()
        references=[{'table':target,'observed_at':stamp,'columns':[{'name':'link','dtype':'int64'}]}]
        def factory(datasets):
            def run(request):
                calls.append(request['query'])
                data=pd.DataFrame([dict(constraint_name='fk',source_column='link',target_catalog='catalog',
                    target_schema='schema',target_table='labels',target_column='key',ordinal_position=1)])
                info=datasets.register(data,source=request['source'],query=request['query'],snapshot=stamp)
                return {'status':'ready','dataset':asdict(info)}
            return run
        model=EvaluationModel(calls=[{'name':'inspect_table_relationships','args':{'table':target}},
            {'name':'query_databricks','args':plan}])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','discovery',model,reference_context_loader=lambda:references,
                remote_factory=factory,connection_identity='synthetic')
            raw=r.datasets.register(pd.DataFrame({'link':[1,2]}),source=target)
            r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id)
            result=r.submit('테이블 관계를 확인해줘')
            self.assertEqual(result['status'],'awaiting_approval',result)
            self.assertFalse(calls);r.close()
            r=GraphAnalysisRuntime(root,'owner','discovery',EvaluationModel(),reference_context_loader=lambda:references,
                remote_factory=factory,connection_identity='synthetic')
            try:
                r.respond(result['requests'][0]['id'],approved=True)
                self.assertEqual(calls,[plan['query']])
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertEqual(stored_relationships(r.datasets,target)['foreign_keys'][0]['columns'],['link'])
                self.assertEqual(inspect_relationships(r.context,target)['status'],'ready')
            finally:r.close()

    def test_model_join_choice_does_not_grant_user_join_intent(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','unrequested-join',EvaluationModel(),
                reference_context_loader=lambda:self.references,sql_dialect='sqlite')
            try:
                current={'join':True,'requested_join':False,'required_sources':['events','labels'],
                         'scope':self.scope}
                call={'name':'query_databricks','args':{'query':self.query,'source':'events | labels'}}
                self.assertFalse(r.recovery._proposed_scope_valid(call,current))
                self.assertFalse(r.recovery._scope_valid(self.query,current))
            finally:r.close()

    def test_stored_relationships_reject_incomplete_stale_or_modified_observations(self):
        target='catalog.schema.events';plan=relationship_plan(target)['metadata_plan']
        for variant in ('page_full','stale','different_query','wrong_ordinal','boolean_ordinal','float_ordinal','split_target'):
            with self.subTest(variant=variant):
                store=DatasetStore()
                row=dict(constraint_name='fk',source_column='link',target_catalog='catalog',
                    target_schema='schema',target_table='labels',target_column='key',ordinal_position=1)
                rows=[row]
                if variant=='page_full':rows=[row]*65
                if variant=='wrong_ordinal':rows=[{**row,'ordinal_position':2}]
                if variant=='boolean_ordinal':rows=[{**row,'ordinal_position':True}]
                if variant=='float_ordinal':rows=[{**row,'ordinal_position':1.0}]
                if variant=='split_target':rows += [{**row,'ordinal_position':2,'source_column':'other','target_table':'different'}]
                store.register(pd.DataFrame(rows),source=plan['source'],
                    query=plan['query']+(' ' if variant=='different_query' else ''),
                    snapshot='2000-01-01T00:00:00Z' if variant=='stale' else datetime.now(timezone.utc).isoformat())
                self.assertIsNone(stored_relationships(store,target))

    def test_approved_joined_count_completes_without_loading_joined_raw_rows(self):
        for materialize in (False, True):
            with self.subTest(materialize=materialize), tempfile.TemporaryDirectory() as root:
                calls=[]
                def factory(datasets):
                    def execute(request):
                        calls.append(request['query'])
                        with sqlite3.connect(f'file:{self.path}?mode=ro',uri=True) as db:
                            frame=pd.read_sql_query(request['query'],db)
                        info=datasets.register(frame,source=request['source'],query=request['query'],
                            grain='aggregate',coverage='complete',predicate_known=True)
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                model=EvaluationModel(calls=[{'name':'query_databricks','args':{
                    'source':'events | labels','query':self.query,'reason':'Joined count'}}])
                r=GraphAnalysisRuntime(root,'owner','join-count',model,reference_context_loader=lambda:self.references,
                    sql_dialect='sqlite',connection_identity='public-local-fixture',remote_factory=factory)
                try:
                    request="Count records with label='A' after join events and labels"
                    if materialize:request += '; also save the joined dataset'
                    staged=r.submit(request)
                    self.assertEqual(staged['status'],'awaiting_approval',staged)
                    self.assertEqual(calls,[])
                    result=r.respond(staged['requests'][0]['id'],approved=True)
                    self.assertEqual(result['status']=='answered',not materialize,result)
                    self.assertEqual(calls,[self.query])
                    self.assertEqual(len(r.datasets.metadata),1)
                    self.assertEqual(next(iter(r.datasets.frames.values())).iloc[0,0],2)
                    if not materialize:
                        self.assertIn('건수: 2',result['text'])
                        self.assertTrue(r.inspect()['recovery']['join_query_evidence'])
                    else:
                        self.assertFalse(r.inspect()['recovery'].get('join_query_evidence'))
                finally:r.close()
