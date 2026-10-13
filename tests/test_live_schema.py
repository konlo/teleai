"""Actual tool/ledger/schema contracts with fake DB transport, not LLM scores."""
from dataclasses import asdict
from datetime import datetime,timezone
import tempfile
import unittest
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration,ChatResult
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from tests.test_llm_goal import GoalModel,goal
from tests.test_row_preview import schema
from utils.analysis_datasets import stored_dataset_digest

SOURCE='lab.observations'

class LiveSchemaModel(GoalModel):
    def _generate(self,messages,**kwargs):
        if any('goal_schema_v1' in str(m.content) for m in messages):
            return super()._generate(messages,**kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[{
            'name':'inspect_live_schema','args':{'source':SOURCE},'id':'live-schema'}]))])

class LiveSchemaTests(unittest.TestCase):
    def test_stale_schema_refreshes_once_without_replacing_original(self):
        calls=[]
        def factory(store):
            def execute(envelope):
                calls.append(envelope)
                frame=pd.DataFrame({'reading':pd.Series([],dtype='int64'),
                    'new_metric':pd.Series([],dtype='float64')})
                info=store.register(frame,source=SOURCE,query=envelope['query'],
                    coverage='complete',predicate_known=True,snapshot=datetime.now(timezone.utc).isoformat())
                return {'status':'ready','dataset':asdict(info)}
            return execute
        refs=schema(SOURCE)
        for item in refs:item.update(observed_at='2000-01-01T00:00:00Z',trained_at='2000-01-01T00:00:00Z')
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'schema-test','conversation',
                LiveSchemaModel(goals=[goal('metadata',{'kind':'columns'},sources=[SOURCE],columns=[])]),
                sql_dialect='mysql',remote_factory=factory,connection_identity='fixture',
                reference_context_loader=lambda:refs)
            try:
                original=r.datasets.register(pd.DataFrame({'unrelated':[1,2]}),source='lab.other',coverage='complete')
                r.select_dataset(original.id);digest=stored_dataset_digest(r.datasets,original.id)
                out=r.submit('observations 컬럼을 보여줘')
                self.assertEqual(out['status'],'answered',out)
                state=r.inspect()['recovery']
                self.assertEqual(state['metadata_evidence']['columns'],['reading','new_metric'])
                self.assertEqual(state['schema_subject']['table'],SOURCE)
                self.assertEqual(len(calls),1)
                self.assertIn('LIMIT 0',calls[0]['query'])
                self.assertEqual(r.context.selected_dataset_id,original.id)
                self.assertEqual(stored_dataset_digest(r.datasets,original.id),digest)
                self.assertEqual(len(state['remote_query_evidence']),1)
                self.assertEqual(next(iter(state['remote_query_evidence'].values()))['rows'],0)
                self.assertFalse(r.recovery._proposed_scope_valid({'name':'inspect_live_schema',
                    'args':{'source':'lab.wrong'}},state))
            finally:r.close()

    def test_manual_approval_does_not_offer_automatic_refresh(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'schema','manual',GoalModel(),sql_dialect='mysql',
                remote_factory=lambda datasets:lambda envelope:{},connection_identity='fixture',
                policy=RuntimePolicy(require_remote_approval=True))
            try:self.assertNotIn('inspect_live_schema',r.context.allowed_tool_names)
            finally:r.close()
