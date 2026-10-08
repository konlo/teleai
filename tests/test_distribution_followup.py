"""Distribution journeys cross table/preview/restart boundaries without inference."""
from dataclasses import asdict
from datetime import datetime, timezone
import tempfile
import unittest
import pandas as pd
from langchain_core.messages import HumanMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.recovery import _column_listing_requested, _chart_kind
from tests.test_mysql_metadata_contract import NoInference
from utils.analysis_datasets import stored_dataset_digest


class DistributionFollowupTests(unittest.TestCase):
    def test_y_axis_followup_reuses_exact_frequency_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            source='lab.category_sensor'
            data=pd.DataFrame({'reading':['A']*12+['B']*3})
            schema=[self.context(source,data)]
            r=GraphAnalysisRuntime(root,'owner','axis',NoInference(),sql_dialect='mysql',
                reference_context_loader=lambda:schema,intent_mode='contract_fixture')
            raw=r.datasets.register(data,source=source,coverage='complete',predicate_known=True)
            r.select_dataset(raw.id)
            digest=stored_dataset_digest(r.datasets,raw.id)
            self.assertEqual(r.submit('reading 분포를 보여줘')['status'],'answered')
            before=r.artifacts[r.inspect()['recovery']['artifact_ids'][-1]]
            r.close()
            r=GraphAnalysisRuntime(root,'owner','axis',NoInference(),sql_dialect='mysql',
                reference_context_loader=lambda:schema,intent_mode='contract_fixture')
            try:
                result=r.submit('y축을 10으로해줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                after=r.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(after.dataset_id,before.dataset_id)
                self.assertEqual(after.render_spec['counts'],before.render_spec['counts'])
                self.assertEqual(after.render_spec['y_limits'],[0.0,10.0])
                self.assertNotEqual(after.image,before.image)
                self.assertIn('잘려',after.reason)
                self.assertEqual(state['model_calls'],0)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def context(self, source, data):
        return {'table': source, 'observed_at': datetime.now(timezone.utc).isoformat(),
                'columns': [{'name': c, 'dtype': str(data[c].dtype)} for c in data.columns]}

    def test_objective_wins_over_column_noun_but_schema_and_explanation_remain(self):
        for text in ['measurement column의 데이타 분포를 보여줘',
                     'measurement 컬럼 분포 보여주세요', 'show distribution of measurement column']:
            self.assertEqual(_chart_kind(text), 'histogram')
            self.assertFalse(_column_listing_requested(text))
        self.assertTrue(_column_listing_requested('table column들 다시 보여줘'))
        self.assertIsNone(_chart_kind('분포의 개념을 설명만 해줘'))
        self.assertIsNone(_chart_kind('measurement 분포를 표로 보여줘'))
        self.assertIsNone(_chart_kind('show distribution of measurement as a table'))

    def test_preview_table_switch_then_distribution_preserves_old_selection_and_restarts(self):
        for dialect in ('mysql', 'databricks'):
            with self.subTest(dialect=dialect), tempfile.TemporaryDirectory() as root:
                prefix = 'lab' if dialect == 'mysql' else 'catalog.lab'
                old_source, new_source = prefix+'.old_sensor', prefix+'.new_sensor'
                old = pd.DataFrame({'measurement':[500,600], 'group':['old']*2})
                new = pd.DataFrame({'measurement':[1,2,2,3]*5, 'group':['A','B']*10})
                schema = [self.context(old_source,old), self.context(new_source,new)]
                r = GraphAnalysisRuntime(root,'owner','journey',NoInference(),sql_dialect=dialect,
                    reference_context_loader=lambda:schema, summary_trigger_tokens=1,intent_mode='contract_fixture')
                try:
                    baseline=r.datasets.register(old,source=old_source,coverage='complete',predicate_known=True)
                    target=r.datasets.register(new,source=new_source,coverage='complete',predicate_known=True)
                    r.select_dataset(baseline.id)
                    hashes={i:stored_dataset_digest(r.datasets,i) for i in (baseline.id,target.id)}
                    self.assertEqual(r.submit('old_sensor measurement histogram 그려줘')['status'],'answered')
                    self.assertEqual(r.submit('new_sensor column list 보여줘')['status'],'answered')
                    self.assertEqual(r.submit('데이터 10row만 보여줘')['status'],'answered')
                    previous=r.inspect()['recovery']
                    self.assertEqual(previous['scope']['sources'],[new_source])
                    # Reproduce the older inconsistent checkpoint. The durable
                    # preview, not the selected old dataset or AI prose, wins.
                    previous['scope']['sources']=[old_source]
                    previous['scope']['conditions']=[{'column':'group','op':'eq','value':'old'}]
                    human=HumanMessage(id='saved-failure',content='measurement column의 데이타 분포를 보여줘')
                    r.agent.update_state(r.config,{'messages':[human],'recovery':{
                        'request_id':human.id,'request_text':human.content,'status':'working',
                        'chart':False,'metadata_kind':'columns','required_columns':['measurement'],
                        'required_sources':[],'kind':None,'operations':[],
                        'scope':previous['scope'],'confirmed_analysis':previous,
                        'model_calls':1,'model_seconds':0}},
                        as_node='ObservedSummarizationMiddleware.before_model')
                    r.close()
                    r=GraphAnalysisRuntime(root,'owner','journey',NoInference(),sql_dialect=dialect,
                        reference_context_loader=lambda:schema,summary_trigger_tokens=1,intent_mode='contract_fixture')
                    result=r.resume()
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery']
                    self.assertEqual(state['required_sources'],[new_source])
                    self.assertEqual(state['confirmed_analysis']['scope']['sources'],[new_source])
                    self.assertFalse(state['scope']['conditions'])
                    self.assertIsNone(state['metadata_kind'])
                    self.assertEqual(state['kind'],'histogram')
                    self.assertTrue(state['artifact_ids'])
                    chart=r.artifacts[state['artifact_ids'][-1]]
                    self.assertTrue(chart.image.startswith(b'\x89PNG'))
                    result_info=r.datasets.metadata[chart.dataset_id]
                    self.assertEqual(result_info.source,new_source)
                    self.assertEqual(int(r.datasets.frames[result_info.id]['__frequency'].sum()),20)
                    self.assertEqual(r.context.selected_dataset_id,baseline.id)
                    for i,h in hashes.items():self.assertEqual(stored_dataset_digest(r.datasets,i),h)
                    self.assertEqual(r.submit('group 컬럼의 분포를 보여줘')['status'],'answered')
                    self.assertEqual(r.inspect()['recovery']['kind'],'bar')
                finally:r.close()

    def test_sampled_preview_requires_whole_source_counts_not_ten_rows(self):
        with tempfile.TemporaryDirectory() as root:
            source='lab.wide_sensor'
            data=pd.DataFrame({'measurement':['','7','7','8']*50})
            schema=[self.context(source,data)];calls=[]
            def factory(store):
                def execute(envelope):
                    calls.append(envelope['query'])
                    self.assertEqual(envelope['source'],source)
                    self.assertIn('COUNT(*)',envelope['query'])
                    self.assertNotIn('LIMIT 10',envelope['query'])
                    counts=data.groupby('measurement').size().rename('__frequency').reset_index()
                    info=store.register(counts,source=source,coverage='complete',predicate_known=True,
                        grain='aggregate',query=envelope['query'],aggregation=envelope['query'])
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            r=GraphAnalysisRuntime(root,'owner','sample',NoInference(),sql_dialect='mysql',
                connection_identity='fixture',remote_factory=factory,reference_context_loader=lambda:schema,intent_mode='contract_fixture')
            try:
                sample=r.datasets.register(data.head(10),source=source,coverage='sampled',predicate_known=False,
                    query='SELECT * FROM lab.wide_sensor LIMIT 10')
                self.assertEqual(r.submit('wide_sensor 데이터 10row만 보여줘')['status'],'answered')
                result=r.submit('measurement column의 데이타 분포를 보여줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(state['kind'],'bar')
                chart=r.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(chart.render_spec['total_count'],200)
                self.assertEqual(len(calls),1)
                self.assertEqual(r.datasets.metadata[sample.id].rows,10)
                self.assertEqual(state['model_calls'],0)
            finally:r.close()
