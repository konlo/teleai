"""Independent scope oracles: variants, rejection, restart, and chart lineage."""
from copy import deepcopy
from dataclasses import asdict
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.intent_scope import resolve_request_scope, scope_matches
from core.analysis_agent.numeric_scope import intervals
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_mysql_metadata_contract import NoInference
from utils.analysis_datasets import stored_dataset_digest


TABLE = 'catalog.lab.measurements'
DATA = pd.DataFrame({'label':['A','A','A','A','B','A'], 'reading':[29,30,35,40,35,41]})


class NumericScopeTests(unittest.TestCase):
    def runtime(self, root, dialect='databricks'):
        runtime = GraphAnalysisRuntime(root, 'owner', 'numeric', NoInference(), sql_dialect=dialect,intent_mode='contract_fixture')
        if not runtime.datasets.metadata:
            info = runtime.datasets.register(DATA, source=TABLE, coverage='complete', predicate_known=True)
            runtime.select_dataset(info.id)
        return runtime

    def test_equivalent_wordings_and_sql_dialects(self):
        wordings = ('reading이 30~40인', 'reading가 30〜40인', 'reading 30∼40',
            'reading은 30–40', 'reading=30-40', 'reading 30—40 구간',
            'reading 30 to 40', 'reading between 30 and 40',
            'reading 30부터 40까지', 'reading이 30에서 40 사이', 'reading 30 이상 40 이하')
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            try:
                for wording in wordings:
                    with self.subTest(wording=wording):
                        scope = resolve_request_scope('label이 A이고 '+wording+' 사람들을 시각화해줘', runtime.context)
                        self.assertEqual(scope['unresolved'], [])
                        self.assertEqual(scope['conditions'], [
                            {'column':'label','op':'eq','value':'A'},
                            {'column':'reading','op':'ge','value':30},
                            {'column':'reading','op':'le','value':40}])
                        for dialect in ('mysql','databricks'):
                            with self.subTest(dialect=dialect):
                                correct = "SELECT * FROM "+TABLE+" WHERE label='A' AND reading BETWEEN 30 AND 40"
                                self.assertTrue(scope_matches(correct,scope,dialect=dialect))
                                self.assertTrue(scope_matches(correct.replace('reading BETWEEN 30 AND 40',
                                    'reading >= 30 AND reading <= 40'),scope,dialect=dialect))
                                for wrong in ("label='A'", "label='A' AND reading BETWEEN 20 AND 40",
                                              "label='B' AND reading BETWEEN 30 AND 40",
                                              "label='A' AND reading > 30 AND reading < 40"):
                                    self.assertFalse(scope_matches('SELECT * FROM '+TABLE+' WHERE '+wrong,
                                        scope,dialect=dialect))
            finally:runtime.close()

    def test_negative_decimal_and_thousands_bounds(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                for text, low, high in (('reading -2.5~-0.5',-2.5,-0.5),
                    ('reading -10--2',-10,-2),('reading 1,000~2,000',1000,2000)):
                    with self.subTest(text=text):
                        scope=resolve_request_scope(text,runtime.context)
                        self.assertEqual(scope['unresolved'],[])
                        self.assertEqual(scope['conditions'],[
                            {'column':'reading','op':'ge','value':low},
                            {'column':'reading','op':'le','value':high}])
            finally:runtime.close()

    def test_multiple_category_values_stay_bound_to_one_column(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                for text in ('label이 A과 B 이고 reading이 30~40인',
                    'label이 A와 B이고 reading이 30~40인',
                    'label A와 B의 reading이 30~40인',
                    "label이 'A' 및 'B'인 reading이 30~40인",
                    'label이 A, never_observed 이고 reading이 30~40인'):
                    with self.subTest(text=text):
                        scope=resolve_request_scope(text,runtime.context)
                        self.assertEqual(scope['unresolved'],[])
                        category=next(c for c in scope['conditions'] if c['column']=='label')
                        self.assertEqual(category['op'],'in')
                        self.assertEqual(category['value'],['A', 'never_observed' if 'never_observed' in text else 'B'])
                        values=', '.join("'"+v+"'" for v in category['value'])
                        self.assertTrue(scope_matches('SELECT * FROM '+TABLE+
                            ' WHERE label IN ('+values+') AND reading BETWEEN 30 AND 40',scope))
                        self.assertFalse(scope_matches('SELECT * FROM '+TABLE+
                            ' WHERE reading BETWEEN 30 AND 40',scope))
                incomplete=resolve_request_scope('label이 A과 B reading이 30~40인',runtime.context)
                self.assertIn('unbound_categorical_list',incomplete['unresolved'])
                self.assertFalse(scope_matches('SELECT * FROM '+TABLE+
                    " WHERE label='A' AND reading BETWEEN 30 AND 40",incomplete))
            finally:runtime.close()

    def test_unknown_or_unattached_range_is_never_silently_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                for text in ('label이 A이고 missing이 30~40인',
                    'label이 A이고 30~40인 사람', 'missing between 30 and 40'):
                    with self.subTest(text=text):
                        scope=resolve_request_scope(text,runtime.context)
                        self.assertIn('unbound_numeric_range',scope['unresolved'])
                        self.assertFalse(scope_matches("SELECT * FROM "+TABLE+" WHERE label='A'",scope))
                reversed_scope=resolve_request_scope('reading이 40~30인',runtime.context)
                self.assertIn('invalid_range_bounds',reversed_scope['unresolved'])
            finally:runtime.close()

    def test_literal_dates_and_unrelated_categorical_strings_are_not_ranges(self):
        self.assertEqual(list(intervals("label='30~40'")),[])
        self.assertEqual(list(intervals("label='2026-10-03'")),[])
        self.assertEqual(list(intervals('2026-10-03')) ,[])
        self.assertEqual(list(intervals('2026-08')),[])
        self.assertEqual(list(intervals('측정값(reading, 1~31일)별 건수')),[])
        self.assertEqual(list(intervals('dataset_id=12345678-1234-5678-9012-123456789012')),[])

    def test_verified_filtered_raw_sql_continues_to_real_images_without_another_model_call(self):
        class Once(NoInference):
            calls:int=0
            def _generate(self,*args,**kwargs):
                self.calls+=1
                if self.calls>1:raise AssertionError('verified rows must continue locally')
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[{
                    'name':'query_databricks','id':'scope-query', 'args':{'source':TABLE,
                    'query':"SELECT * FROM "+TABLE+" WHERE label='A' AND reading IN (30,35,40)",
                    'reason':'apply both requested conditions'}}]))])
        with tempfile.TemporaryDirectory() as root:
            model=Once();queries=[]
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    # Independently defined oracle; never derive the expected
                    # filtered rows from the agent's extracted conditions.
                    frame=DATA[(DATA.label=='A') & DATA.reading.between(30,40)]
                    info=store.register(frame,source=TABLE,query=envelope['query'],
                        coverage='complete',predicate_known=True)
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            runtime=GraphAnalysisRuntime(root,'owner','once',model,connection_identity='fixture',
                remote_factory=factory,intent_mode='contract_fixture')
            original=runtime.datasets.register(DATA,source=TABLE,coverage='complete',predicate_known=True)
            runtime.select_dataset(original.id)
            digest=stored_dataset_digest(runtime.datasets,original.id)
            try:
                result=runtime.submit('label이 A이고 reading (30,35,40)인 사람들을 시각화 해줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual(model.calls,1)
                self.assertEqual(len(queries),1)
                self.assertEqual(len(state['scope']['conditions']),2)
                self.assertTrue(state['artifact_ids'])
                for key in state['artifact_ids']:
                    card=runtime.artifacts[key]
                    self.assertTrue(card.image.startswith(b'\x89PNG'))
                    self.assertEqual(runtime.datasets.metadata[card.dataset_id].rows,3)
                self.assertEqual(stored_dataset_digest(runtime.datasets,original.id),digest)
                self.assertEqual(runtime.datasets.metadata[runtime.context.selected_dataset_id].rows,3)
                self.assertIn(original.id,runtime.datasets.metadata)
            finally:runtime.close()

    def test_exhausted_paused_request_keeps_repaired_scope_and_ends_without_inference(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                with patch.object(runtime.recovery,'_next_local',return_value=None):
                    result=runtime.submit('label이 A이고 reading이 30~40인 사람들을 시각화 해줘')
                self.assertEqual(result['status'],'incomplete',result)
                checkpoint=runtime.agent.get_state(runtime.config)
                current=deepcopy(checkpoint.values['recovery'])
                current['scope']['conditions']=current['scope']['conditions'][:1]
                current.pop('numeric_scope_version',None)
                current['model_seconds']=200.
                runtime.agent.update_state(runtime.config,{'recovery':current})
                with patch.object(runtime.recovery,'_next_local',return_value=None):
                    result=runtime.resume()
                self.assertEqual(result['status'],'exhausted',result)
                self.assertFalse(runtime.agent.get_state(runtime.config).next)
                repaired=runtime.inspect()['recovery']
                self.assertEqual(len(repaired['scope']['conditions']),3)
                self.assertEqual(repaired['stop_reason'],'model_time_budget')
                self.assertEqual(runtime.submit('label은 어떤 값들로 되어 있지 ?')['status'],'answered')
            finally:runtime.close()

    def test_schema_only_range_request_pushes_counts_to_database_without_inference(self):
        import duckdb
        from sqlglot import exp
        from core.analysis_sql import validate_query
        for dialect in ('mysql','databricks'):
            with self.subTest(dialect=dialect), tempfile.TemporaryDirectory() as root:
                queries=[]
                def factory(store):
                    def execute(envelope):
                        queries.append(envelope['query'])
                        tree=validate_query(envelope['query'],dialect=dialect)
                        # Execute the generated predicates, not an invented
                        # executor response based on the extracted intent.
                        tree=tree.transform(lambda node: exp.Table(this=exp.to_identifier('data'))
                            if isinstance(node,exp.Table) else node)
                        with duckdb.connect() as connection:
                            connection.register('data',DATA)
                            result=connection.execute(tree.sql(dialect='duckdb')).fetchdf()
                        info=store.register(result,source=TABLE,query=envelope['query'],
                            grain='aggregate',aggregation=envelope['query'],coverage='complete',predicate_known=False)
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                runtime=GraphAnalysisRuntime(root,'owner','schema-only',NoInference(),
                    connection_identity='fixture',remote_factory=factory,sql_dialect=dialect,intent_mode='contract_fixture')
                schema=runtime.datasets.register(DATA.iloc[:0],source=TABLE,
                    query='SELECT * FROM '+TABLE+' LIMIT 0',coverage='unknown',predicate_known=False)
                try:
                    result=runtime.submit('label이 A이고 reading이 30~40인 사람들을 시각화 해줘')
                    self.assertEqual(result['status'],'answered',result)
                    state=runtime.inspect()['recovery']
                    self.assertEqual(state['model_calls'],0)
                    self.assertEqual(len(queries),1)
                    self.assertNotIn('scope_error',state)
                    card=runtime.artifacts[state['artifact_ids'][0]]
                    self.assertEqual(card.kind,'histogram')
                    info=runtime.datasets.metadata[card.dataset_id]
                    frame=runtime.datasets.frames[info.id]
                    self.assertEqual(int(frame['__frequency'].sum()),3)
                    self.assertEqual(sorted(frame.reading.tolist()),[30,35,40])
                    self.assertEqual(runtime.datasets.metadata[schema.id].rows,0)
                    # Two independent COUNT results must not be confused with
                    # two competing raw populations when the filter changes.
                    counts=DATA.groupby('reading').size().reset_index(name='__frequency')
                    old=runtime.datasets.register(counts,source=TABLE,grain='aggregate',
                        aggregation='count',coverage='complete',predicate_known=False,
                        query='SELECT reading, COUNT(*) AS __frequency FROM '+TABLE+
                              ' WHERE reading IS NOT NULL GROUP BY reading')
                    runtime.context.selected_dataset_id=None
                    result=runtime.submit('label이 A과 B 이고 reading이 30~40인 사람들을 시각화 해줘')
                    self.assertEqual(result['status'],'answered',result)
                    state=runtime.inspect()['recovery']
                    self.assertEqual(state['model_calls'],0)
                    self.assertEqual(len(queries),2)
                    info=runtime.datasets.metadata[runtime.artifacts[state['artifact_ids'][0]].dataset_id]
                    self.assertEqual(int(runtime.datasets.frames[info.id]['__frequency'].sum()),4)
                    self.assertIn(old.id,runtime.datasets.metadata)
                    self.assertIn(' IN ',info.query)
                finally:runtime.close()

    def test_selected_schema_excludes_same_name_on_previous_source(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                selected=runtime.context.selected_dataset_id
                runtime.datasets.register(pd.DataFrame({'Reading':[1,2]}),source='other.lab.records',
                    coverage='complete',predicate_known=True)
                runtime.select_dataset(selected)
                scope=resolve_request_scope('reading이 30~40인',runtime.context)
                self.assertEqual(scope['columns'],['reading'])
                self.assertEqual(len(scope['conditions']),2)
                runtime.context.selected_dataset_id=None
                cold=resolve_request_scope('label이 A이고 reading이 30~40인',runtime.context)
                self.assertEqual(cold['sources'],[TABLE])
                self.assertEqual(cold['columns'],['label','reading'])
                self.assertEqual(len(cold['conditions']),3)
                explicit=resolve_request_scope('other.lab.records Reading이 1~2인',runtime.context)
                self.assertEqual(explicit['columns'],['Reading'])
            finally:runtime.close()

    def test_unobserved_explicit_literal_is_bound_without_sampling_rows(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                scope=resolve_request_scope('label이 never_observed이고 reading이 30~40인',runtime.context)
                self.assertEqual(scope['unresolved'],[])
                self.assertIn({'column':'label','op':'eq','value':'never_observed'},scope['conditions'])
                self.assertTrue(scope_matches('SELECT * FROM '+TABLE+
                    " WHERE label='never_observed' AND reading BETWEEN 30 AND 40",scope))
            finally:runtime.close()

    def test_identical_columns_on_multiple_unselected_sources_require_binding(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                runtime.datasets.register(DATA,source='other.lab.measurements',coverage='complete',predicate_known=True)
                runtime.context.selected_dataset_id=None
                scope=resolve_request_scope('label이 A이고 reading이 30~40인',runtime.context)
                self.assertIn('ambiguous_column_source',scope['unresolved'])
                self.assertFalse(scope_matches("SELECT * FROM "+TABLE+" WHERE label='A' AND reading BETWEEN 30 AND 40",scope))
                explicit=resolve_request_scope(TABLE+' label이 A이고 reading이 30~40인',runtime.context)
                self.assertEqual(explicit['unresolved'],[])
                self.assertEqual(explicit['sources'],[TABLE])
            finally:runtime.close()

    def test_recheck_uses_original_request_never_proposed_sql(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            try:
                full=resolve_request_scope('label이 A이고 reading이 30~40인',runtime.context)
                broken=deepcopy(full);broken['conditions']=broken['conditions'][:1]
                current={'request_id':'saved','request_text':'label이 A이고 reading이 30~40인',
                    'previous_scope':{},'scope':broken,'scope_error':'request_scope_mismatch'}
                self.assertTrue(runtime.recovery._recheck_numeric_scope(current))
                self.assertEqual(current['scope'],full)
                self.assertNotIn('scope_error',current)
                self.assertTrue(runtime.recovery._scope_valid(
                    "SELECT * FROM "+TABLE+" WHERE label='A' AND reading BETWEEN 30 AND 40",current))
                frequency="SELECT reading, COUNT(*) AS __frequency FROM "+TABLE+" WHERE label='A' AND reading BETWEEN 30 AND 40 AND reading IS NOT NULL GROUP BY reading"
                current['plan']={'query':frequency,'value_column':'reading'}
                self.assertTrue(runtime.recovery._scope_valid(frequency,current))
                self.assertNotIn('scope_error',current)
                self.assertFalse(runtime.recovery._scope_valid(
                    "SELECT * FROM "+TABLE+" WHERE label='A' AND reading BETWEEN 20 AND 40",current))
                self.assertFalse(runtime.recovery._recheck_numeric_scope(current))
                self.assertEqual(current['scope'],full)
            finally:runtime.close()

    def test_filtered_chart_and_followup_preserve_origin_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=self.runtime(root)
            selected=runtime.context.selected_dataset_id
            digest=stored_dataset_digest(runtime.datasets,selected)
            try:
                result=runtime.submit('현재 로딩된 데이터에서 reading이 30~40인 reading histogram을 그려줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                card=runtime.artifacts[state['artifact_ids'][0]]
                child=runtime.datasets.metadata[card.dataset_id]
                self.assertEqual(child.rows,4)
                self.assertEqual(card.columns,('reading',))
                self.assertTrue(card.image.startswith(b'\x89PNG'))
                self.assertEqual(state['model_calls'],0)
                self.assertEqual(stored_dataset_digest(runtime.datasets,selected),digest)
                self.assertEqual(runtime.context.selected_dataset_id,selected)
            finally:runtime.close()
            runtime=self.runtime(root)
            try:
                result=runtime.submit('같은 조건을 유지해서 histogram을 다시 보여줘')
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual(len(state['scope']['conditions']),2)
                self.assertEqual(runtime.datasets.metadata[runtime.artifacts[state['artifact_ids'][0]].dataset_id].rows,4)
                self.assertEqual(stored_dataset_digest(runtime.datasets,selected),digest)
            finally:runtime.close()


if __name__ == '__main__':
    unittest.main()
