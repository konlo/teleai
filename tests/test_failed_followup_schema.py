"""Typed empty schemas and failed turns preserve executable follow-up context."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
from langchain_core.messages import HumanMessage
from core.analysis_agent.assets import AssetDB, PersistentDatasets
from core.analysis_catalog import resolve_table_context
from core.analysis_agent.goal_contract import compile_goal, pending_state
from core.analysis_agent.conversation_context import prior_analysis
from tests.test_llm_goal import goal, GoalModel


class FailedFollowupSchemaTests(unittest.TestCase):
    def test_structured_category_goal_uses_remote_counts_and_real_png_without_raw_reload(self):
        from core.analysis_agent.runtime import GraphAnalysisRuntime
        with tempfile.TemporaryDirectory() as root:
            source='catalog.dynamic.records';queries=[]
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    self.assertIn('COUNT(*)',envelope['query'])
                    self.assertNotIn('LIMIT 100000',envelope['query'])
                    data=pd.DataFrame({'segment':['A','B'],'__frequency':[123456,7890]})
                    info=store.register(data,source=source,query=envelope['query'],grain='aggregate',
                        coverage='complete',predicate_known=True,aggregation=envelope['query'])
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            plan=goal('chart',{'kind':'histogram','axes':{'x':'segment'}},sources=[source],columns=['segment'])
            runtime=GraphAnalysisRuntime(root,'owner','chart',GoalModel(goals=[plan]),
                connection_identity='fixture',remote_factory=factory,sql_dialect='databricks')
            try:
                schema=pa.table({'segment':pa.array([],type=pa.string())}).to_pandas(types_mapper=pd.ArrowDtype)
                original=runtime.datasets.register(schema,source=source,query='SELECT * FROM '+source+' LIMIT 0',predicate_known=True)
                out=runtime.submit('그 분포를 보여줘')
                self.assertEqual(out['status'],'answered',out)
                state=runtime.inspect()['recovery'];card=runtime.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(dict(zip(card.render_spec['labels'],card.render_spec['counts'])),{'A':123456,'B':7890})
                self.assertTrue(card.image.startswith(b'\x89PNG'))
                self.assertEqual(len(queries),1)
                self.assertEqual(runtime.datasets.metadata[original.id].rows,0)
            finally:runtime.close()

    def test_typed_empty_arrow_string_compiles_frequency_chart_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','schema');store=PersistentDatasets(db)
            source='catalog.dynamic.records'
            typed=pa.table({'segment':pa.array([],type=pa.string()),
                            'reading':pa.array([],type=pa.int64())})
            info=store.register(typed.to_pandas(types_mapper=pd.ArrowDtype),source=source,
                query='SELECT * FROM '+source+' LIMIT 0',predicate_known=True)
            untyped=store.register(pd.DataFrame(columns=['mystery']),source='catalog.dynamic.unknown',
                query='SELECT * FROM catalog.dynamic.unknown LIMIT 0',predicate_known=True)
            db.close();db=AssetDB(root,'owner','schema');store=PersistentDatasets(db)
            try:
                context=SimpleNamespace(datasets=store,reference_context=[],selected_dataset_id='',sql_dialect='databricks')
                observed=resolve_table_context([],store,source)
                types={c['name']:c['dtype'] for c in observed['table_context']['columns']}
                self.assertEqual(types['segment'],'string')
                self.assertTrue(types['reading'].startswith('int'))
                self.assertEqual(resolve_table_context([],store,'catalog.dynamic.unknown')['table_context']['columns'][0]['dtype'],'')
                plan=goal('chart',{'kind':'histogram','axes':{'x':'segment'}},sources=[source],columns=['segment'])
                current=compile_goal(pending_state(HumanMessage(id='next',content='그 분포'),{},context),plan,context,True)
                self.assertTrue(current['categorical_distribution'])
                self.assertEqual(current['kind'],'bar')
                self.assertFalse(current['chart_spec_requested'])
                self.assertEqual(set(store.metadata),{info.id,untyped.id})
            finally:db.close()

    def test_all_incomplete_statuses_keep_verified_subject_but_ui_selection_supersedes(self):
        context=SimpleNamespace(selected_dataset_id='original',datasets=SimpleNamespace(metadata={}),
                                reference_context=[],sql_dialect='databricks')
        verified={'status':'complete','required_sources':['catalog.a.events'],
                  'required_columns':['reading'],'kind':'histogram',
                  'selection_at_confirmation':'original',
                  'scope':{'conditions':[{'column':'reading','op':'ge','value':5}]}}
        for status in ('working','exhausted','blocked','cancelled','failed'):
            with self.subTest(status=status):
                previous={'status':status,'required_sources':['catalog.a.unfinished'],
                          'scope':{'conditions':[]},'confirmed_analysis':deepcopy(verified)}
                current=pending_state(HumanMessage(id='new',content='같은 조건으로 다시'),previous,context)
                self.assertEqual(current['confirmed_analysis'],verified)
                self.assertEqual(current['previous_scope'],verified['scope'])
                self.assertNotIn('confirmed_analysis',current['confirmed_analysis'])
                plan=goal('chart',{'kind':'histogram','axes':{'x':'reading'}},sources=[],columns=['reading'])
                plan['source_reference']='previous_analysis'
                compiled=compile_goal(current,plan,context,True)
                self.assertEqual(compiled['required_sources'],verified['required_sources'])
                self.assertEqual(previous['status'],status)
                context.selected_dataset_id='new-selection'
                self.assertEqual(prior_analysis(previous,context),{})
                context.selected_dataset_id='original'
