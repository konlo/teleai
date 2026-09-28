"""Approved metadata discovery survives restart without replacing protected raw data."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import tempfile
import unittest
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.semantic import semantic_metadata
from core.analysis_metadata_discovery import make_plan, stored_column_definitions
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_datasets import stored_dataset_digest

TARGET='catalog.schema.measurements'

class DiscoverMeaningModel(EvaluationModel):
    def _generate(self,messages,**kwargs):
        if 'ONLY the supplied external definitions' in str(messages[0].content):
            plan={'uncertain':False,'operation':'AVG','column':'metric_z','conditions':[],
                  'definition':'이번 서비스 이용 시간(분)'}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(plan)))])
        return super()._generate(messages,**kwargs)

class ColumnDiscoveryTests(unittest.TestCase):
    def test_metadata_denial_never_executes_or_replaces_original(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','denial',self.model(),
                remote_factory=lambda _: lambda request: self.fail('Denied SQL executed'),
                connection_identity='synthetic')
            try:
                raw=r.datasets.register(pd.DataFrame({'metric_z':[2.,4.]}),source=TARGET,
                    coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                proposal=r.submit('이번 서비스에 소비한 분량의 평균을 알려줘')
                self.assertEqual(proposal['status'],'awaiting_approval',proposal)
                result=r.respond(proposal['requests'][0]['id'],approved=False)
                self.assertNotEqual(result['status'],'answered',result)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(set(r.datasets.metadata),{raw.id})
                self.assertIsNone(stored_column_definitions(r.datasets,TARGET))
            finally:r.close()

    def model(self):
        plan=make_plan(TARGET)['metadata_plan']
        return DiscoverMeaningModel(calls=[
            {'name':'inspect_column_definitions','args':{'table':TARGET}},
            {'name':'query_databricks','args':plan}])

    def test_approval_restart_discovery_binding_and_original_preservation(self):
        calls=[]
        def factory(datasets):
            def run(envelope):
                calls.append(envelope['query'])
                frame=pd.DataFrame([{'table_catalog':'catalog','table_schema':'schema',
                    'table_name':'measurements','column_name':'metric_z','data_type':'DOUBLE',
                    'comment':'이번 서비스 이용 시간(분)'}])
                info=datasets.register(frame,source=envelope['source'],query=envelope['query'],
                    coverage='unknown',predicate_known=True,snapshot=datetime.now(timezone.utc).isoformat())
                return {'status':'ready','dataset':asdict(info)}
            return run
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','meaning',self.model(),remote_factory=factory,connection_identity='synthetic')
            raw=r.datasets.register(pd.DataFrame({'metric_z':[2.,4.,8.,10.]}),source=TARGET,coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id)
            proposal=r.submit('이번 서비스에 소비한 분량의 평균을 알려줘')
            self.assertEqual(proposal['status'],'awaiting_approval',proposal)
            self.assertEqual(calls,[])
            r.close()
            r=GraphAnalysisRuntime(root,'owner','meaning',self.model(),remote_factory=factory,connection_identity='synthetic')
            try:
                result=r.respond(proposal['requests'][0]['id'],approved=True)
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],6.)
                self.assertEqual(len(calls),1)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertEqual(semantic_metadata(r.context,raw.id)['columns'][0]['description'],'이번 서비스 이용 시간(분)')
            finally:r.close()

    def test_wrong_source_stale_full_page_and_modified_query_do_not_supply_definitions(self):
        for variant in ('wrong_source','stale','full_page','modified_query'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                r=GraphAnalysisRuntime(root,'owner','invalid',EvaluationModel())
                try:
                    plan=make_plan(TARGET)['metadata_plan']
                    row={'table_catalog':'catalog','table_schema':'schema','table_name':'measurements',
                         'column_name':'metric_z','data_type':'DOUBLE','comment':'meaning'}
                    if variant=='wrong_source':row['table_name']='other'
                    frame=pd.DataFrame([row]*(65 if variant=='full_page' else 1))
                    r.datasets.register(frame,source=plan['source'],query=plan['query']+(' ' if variant=='modified_query' else ''),
                        snapshot='2000-01-01T00:00:00Z' if variant=='stale' else datetime.now(timezone.utc).isoformat())
                    self.assertIsNone(stored_column_definitions(r.datasets,TARGET))
                finally:r.close()
