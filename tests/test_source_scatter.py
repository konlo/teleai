"""Independent full-population oracle: never promote a preview to all rows."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.source_scatter import WEIGHT, plan_for, valid_card
from tests.test_mysql_metadata_contract import NoInference
from tests.test_row_preview import schema
from utils.analysis_datasets import stored_dataset_digest

DATA = pd.DataFrame({'reading':[1.,1.,2.,2.,2.,3.,None,4.],
                     'response':[9.,9.,4.,4.,4.,None,5.,8.]})


class SourceScatterTests(unittest.TestCase):
    def make(self, root, dialect, calls, *, coverage='complete'):
        source='lab.measurements' if dialect=='mysql' else 'catalog.lab.measurements'
        observed=[{'table':source,'observed_at':__import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
            'columns':[{'name':c,'dtype':'double'} for c in DATA.columns]}]
        def factory(store):
            def execute(envelope):
                calls.append(envelope['query'])
                expected=(DATA.dropna().groupby(['reading','response']).size()
                          .reset_index(name=WEIGHT))
                info=store.register(expected,source=source,query=envelope['query'],
                    coverage=coverage,grain='aggregate',predicate_known=False)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        return GraphAnalysisRuntime(root,'owner','full',NoInference(),sql_dialect=dialect,
            connection_identity='fixture',reference_context_loader=lambda:observed,
            remote_factory=factory,intent_mode='contract_fixture'),source

    def test_full_population_duplicates_nulls_reuse_restart_and_preserved_preview(self):
        for dialect in ('mysql','databricks'):
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as root:
                calls=[];r,source=self.make(root,dialect,calls)
                try:
                    preview=r.datasets.register(DATA.head(2),source=source,coverage='sampled',predicate_known=False)
                    r.select_dataset(preview.id);digest=stored_dataset_digest(r.datasets,preview.id)
                    result=r.submit(f'{source} 전체 데이터의 reading와 response scatter plot으로 그려줘')
                    self.assertEqual(result['status'],'answered',result)
                    current=r.inspect()['recovery']
                    card=r.artifacts[current['artifact_ids'][0]]
                    self.assertEqual(card.kind,'scatter')
                    self.assertEqual(card.render_spec['drawable_rows'],6)
                    self.assertEqual(card.render_spec['coordinate_count'],3)
                    self.assertEqual(card.columns,('reading','response'))
                    self.assertTrue(valid_card(r.context,current,card))
                    self.assertFalse(valid_card(r.context,current,replace(card,render_spec={**card.render_spec,'drawable_rows':8})))
                    self.assertEqual(r.context.selected_dataset_id,preview.id)
                    self.assertEqual(stored_dataset_digest(r.datasets,preview.id),digest)
                    self.assertEqual(len(calls),1);self.assertEqual(current['model_calls'],0)
                    self.assertIn('NULL 제외',result['text'])
                    r.close();r,source=self.make(root,dialect,calls)
                    self.assertEqual(r.submit(f'{source} 전체 데이터의 reading와 response scatter plot으로 그려줘')['status'],'answered')
                    self.assertEqual(len(calls),1)
                    self.assertEqual(r.inspect()['recovery']['artifact_ids'],[card.id])
                finally:r.close()

    def test_truncated_coordinates_cannot_complete_or_retry(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r,source=self.make(root,'mysql',calls,coverage='truncated')
            try:
                result=r.submit(f'{source} 전체 데이터의 reading와 response scatter plot으로 그려줘')
                self.assertEqual(result['status'],'blocked',result)
                self.assertIn('잘렸거나',result['text'])
                self.assertEqual(len(calls),1)
                self.assertEqual(len(r.artifacts),0)
            finally:r.close()

    def test_filters_do_not_enter_unfiltered_plan(self):
        from core.analysis_agent.source_scatter import target
        with tempfile.TemporaryDirectory() as root:
            r,source=self.make(root,'mysql',[])
            try:
                current={'request_text':'전체 reading response scatter', 'kind':'scatter','chart':True,
                    'required_sources':[source],'required_columns':['reading','response'],
                    'scope':{'conditions':[{'column':'reading','op':'ge','value':2}]}}
                self.assertIsNone(target(r.context,current))
                current['scope']={}
                current['request_text']='전체 reading response scatter 제목을 바꿔줘'
                self.assertIsNone(target(r.context,current))
                with self.assertRaises(ValueError):plan_for(r.context,source,'reading','absent_column')
            finally:r.close()

    def test_coordinate_limit_is_explicit_and_does_not_expand_other_sql(self):
        from core.analysis_agent.source_scatter import coordinate_query,fetch_limit
        for dialect,source in [('mysql','lab.measures'),('databricks','catalog.lab.measures')]:
            query=coordinate_query(source,'reading','response')
            self.assertEqual(fetch_limit(query,dialect,2,5),5)
            self.assertEqual(fetch_limit(query,dialect,2),2)
            filtered=coordinate_query(source,'reading','response',"`segment` IN ('A', 'B')")
            self.assertEqual(fetch_limit(filtered,dialect,2,5),5)
            self.assertEqual(fetch_limit(filtered+' LIMIT 10',dialect,2,5),2)
            self.assertEqual(fetch_limit(query.replace('AND','OR',1),dialect,2,5),2)
            nested=coordinate_query(source,'reading','response','`reading` IN (SELECT 1)')
            self.assertEqual(fetch_limit(nested,dialect,2,5),2)
            for altered in [f'SELECT * FROM {source}',query+' LIMIT 10',
                            query.replace('COUNT(*)','SUM(`reading`)'),
                            query.replace('IS NOT NULL','> 1',1)]:
                self.assertEqual(fetch_limit(altered,dialect,2,5),2)

    def test_failed_model_checkpoint_can_finish_grounded_whole_source_without_inference(self):
        from langchain_core.messages import HumanMessage
        with tempfile.TemporaryDirectory() as root:
            calls=[];r,source=self.make(root,'mysql',calls)
            try:
                human=HumanMessage(id='timed-out',content=f'{source} 전체 reading와 response scatter plot으로 그려줘')
                recovery,_=r.recovery._state({'messages':[human]})
                recovery.update(status='working',model_calls=3,model_seconds=200)
                r.agent.update_state(r.config,{'messages':[human],'recovery':recovery},
                    as_node='ObservedSummarizationMiddleware.before_model')
                r.close();r,source=self.make(root,'mysql',calls)
                result=r.resume()
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(calls),1)
                self.assertEqual(r.inspect()['recovery']['model_calls'],3)
            finally:r.close()


if __name__=='__main__':unittest.main()
