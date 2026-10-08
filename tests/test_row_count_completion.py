from dataclasses import asdict
import tempfile,unittest
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult,ChatGeneration
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.row_count import requested,exact_scalar
from tests.test_actual_agent_evaluation import EvaluationModel
from tests.test_row_preview import schema
from sqlglot import parse_one


class CounterModel(EvaluationModel):
    calls:int=0
    source:str='lab.observations'
    query:str=''
    def _generate(self,*args,**kwargs):
        self.calls+=1
        if self.calls>1:raise AssertionError('A completed exact COUNT needs no second inference')
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[{
            'id':'count-call','name':'query_databricks','args':{'source':self.source,
            'query':self.query or f'SELECT COUNT(*) AS n FROM {self.source}','reason':'전체 행 수 확인'}}]))])


class RowCountCompletionTests(unittest.TestCase):
    def test_actual_count_receipt_completes_and_failed_checkpoint_recovers_without_requery(self):
        for dialect,source in [('mysql','lab.observations'),('databricks','catalog.lab.observations')]:
            with self.subTest(dialect=dialect),tempfile.TemporaryDirectory() as root:
                model=CounterModel(source=source);calls=[]
                def factory(store):
                    def execute(e):
                        calls.append(e['query'])
                        info=store.register(pd.DataFrame({'n':[123]}),source=source,query=e['query'],
                            grain='aggregate',coverage='complete',predicate_known=False)
                        return {'status':'ready','dataset':asdict(info)}
                    return execute
                r=GraphAnalysisRuntime(root,'owner','count',model,sql_dialect=dialect,
                    reference_context_loader=lambda:schema(source),connection_identity='fixture',remote_factory=factory,intent_mode='contract_fixture')
                try:
                    result=r.submit(source+' 이 table row 수가 몇개야 ?')
                    self.assertEqual(result['status'],'answered',result)
                    self.assertIn('123',result['text']);self.assertEqual(model.calls,1)
                    recovery=r.inspect()['recovery'];recovery.update(status='working',whole_row_count=False,evidence_ids=[])
                    r.agent.update_state(r.config,{'recovery':recovery},as_node='ObservedSummarizationMiddleware.before_model')
                    self.assertEqual(r.resume()['status'],'answered')
                    self.assertEqual(model.calls,1);self.assertEqual(len(calls),1)
                finally:r.close()

    def test_exact_count_excludes_column_null_distinct_and_modified_totals(self):
        for query in ['SELECT COUNT(reading) FROM data','SELECT COUNT(DISTINCT reading) FROM data',
                      'SELECT COUNT(*)+1 FROM data','SELECT COUNT(*) FROM data GROUP BY reading']:
            self.assertFalse(exact_scalar(parse_one(query)))
        for query in ['SELECT COUNT(*) FROM data','SELECT COUNT(1) AS n FROM data']:
            self.assertTrue(exact_scalar(parse_one(query)))
        for text in ['row 수가 몇개야?','rows count','행 수 보여줘','number of rows in table']:
            self.assertTrue(requested(text))
        self.assertFalse(requested('education 고유값 개수'))


if __name__=='__main__':unittest.main()
