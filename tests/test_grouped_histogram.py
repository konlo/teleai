"""Color/legend completion requires grouped data and an actual rendered asset."""
from dataclasses import asdict,replace
from pathlib import Path
import tempfile
import os
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import unittest
import pandas as pd
from langchain_core.messages import AIMessage,HumanMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_mysql_metadata_contract import NoInference
from utils.analysis_charts import histogram_from_counts
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_grouped_distribution import render

TABLE='lab.observations'
DATA=pd.DataFrame({'reading':[29,30,35,40,35,41,30,40],
                   'cohort':['A','A','A','A','B','A','B','B']})

class GroupedHistogramTests(unittest.TestCase):
    def setup_chart(self,r,coverage="complete"):
        raw=r.datasets.register(DATA,source=TABLE,coverage=coverage,predicate_known=True)
        query="SELECT reading, COUNT(*) AS __frequency FROM lab.observations WHERE reading IS NOT NULL AND reading BETWEEN 30 AND 40 AND cohort IN ('A','B') GROUP BY reading"
        info=r.datasets.register(pd.DataFrame({'reading':[30,35,40],'__frequency':[2,2,2]}),
            source=TABLE,grain='aggregate',coverage='complete',query=query,aggregation=query,predicate_known=True)
        card=histogram_from_counts(r.datasets,info.id,'reading','__frequency');r.artifacts[card.id]=card
        r.select_chart(card.id)
        return raw,info,card

    def test_style_followup_preserves_filters_renders_legend_reuses_and_restarts(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','grouped',NoInference(),sql_dialect='mysql',intent_mode='contract_fixture')
            raw,info,old=self.setup_chart(r);digest=stored_dataset_digest(r.datasets,raw.id)
            try:
                result=r.submit('legend를 넣어서 A과 B 에 따라서 색을 좀 넣어줘,')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery'];card=r.artifacts[state['artifact_ids'][0]]
                self.assertNotEqual(card.id,old.id)
                self.assertEqual(card.render_spec['legend_labels'],['A','B'])
                self.assertEqual(card.render_spec['group_totals'],[3,3])
                self.assertEqual(card.render_spec['total_count'],6)
                self.assertEqual(len(set(card.render_spec['colors'])),2)
                self.assertEqual(state['model_calls'],0)
                self.assertEqual(len(state['scope']['conditions']),3)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertFalse(r.recovery._valid_card(old,state,old.dataset_id))
                count=len(r.datasets.metadata)
                r.close();r=GraphAnalysisRuntime(root,'owner','grouped',NoInference(),sql_dialect='mysql',intent_mode='contract_fixture')
                result=r.submit('cohort별 색상 구분과 범례를 넣어줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(r.datasets.metadata),count)
                self.assertEqual(r.inspect()['recovery']['artifact_ids'],[card.id])
                result=r.submit('같은 조건을 유지해서 histogram을 다시 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.inspect()['recovery']['artifact_ids'],[card.id])
                self.assertEqual(len(r.datasets.metadata),count)
            finally:r.close()

    def test_remote_group_query_preserves_scope_without_loading_raw(self):
        with tempfile.TemporaryDirectory() as root:
            queries=[]
            def factory(store):
                def execute(envelope):
                    queries.append(envelope['query'])
                    self.assertIn('GROUP BY `reading`, `cohort`',envelope['query'])
                    self.assertIn("`cohort` IN ('A', 'B')",envelope['query'])
                    self.assertIn('`reading` >= 30',envelope['query'])
                    filtered=DATA[DATA.reading.between(30,40)]
                    frame=filtered.groupby(['reading','cohort']).size().reset_index(name='__frequency')
                    info=store.register(frame,source=TABLE,query=envelope['query'],grain='aggregate',
                        aggregation=envelope['query'],coverage='complete',predicate_known=True)
                    return {'status':'ready','dataset':asdict(info)}
                return execute
            r=GraphAnalysisRuntime(root,'owner','remote',NoInference(),sql_dialect='mysql',
                connection_identity='fixture',remote_factory=factory,intent_mode='contract_fixture')
            try:
                raw,_,_=self.setup_chart(r,coverage='sampled')
                result=r.submit('cohort에 따라서 색을 넣고 legend를 추가해줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),1)
                card=r.artifacts[r.inspect()['recovery']['artifact_ids'][0]]
                self.assertEqual(card.render_spec['total_count'],6)
                result=r.submit('cohort별 범례와 색상 구분을 해줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),1)
            finally:r.close()

    def test_ambiguous_grouping_asks_and_bad_frequency_never_renders(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','negative',NoInference(),intent_mode='contract_fixture')
            try:
                result=r.submit('legend를 넣어줘')
                self.assertFalse(r.inspect()['recovery'].get('artifact_ids'))
                self.assertIn('컬럼',result['text'])
                query='SELECT reading, cohort, SUM(amount) AS __frequency FROM lab.observations GROUP BY reading, cohort'
                info=r.datasets.register(pd.DataFrame({'reading':[30,30],'cohort':['A','A'],'__frequency':[1,2]}),
                    source=TABLE,query=query,grain='aggregate',aggregation=query,coverage='complete')
                with self.assertRaises(ValueError):render(r.datasets,info.id,'reading','cohort','__frequency')
            finally:r.close()

    def test_page_does_not_show_old_ungrouped_preview_after_new_chart(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{
                'TELLY_V1_STORAGE':root,'TELLY_DATA_BACKEND':'databricks'}),patch(
                'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=NoInference()):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=30).run()
            r=app.session_state['v1_runtime']
            r.recovery.intent_mode = 'contract_fixture'  # UI execution/rendering fixture, not an intent score.
            try:
                raw,info,old=self.setup_chart(r)
                r.submit('cohort별 범례와 색상 구분을 넣어줘')
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.image),2,'One historic chart and one grouped result, no stale standalone preview')
                self.assertFalse(any(h.value=='reading 분포' for h in app.subheader))
                self.assertIn('그룹별 COUNT',' '.join(c.value for c in app.caption))
            finally:r.close()
