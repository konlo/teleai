"""Bounded source bins: oracle counts, execution receipts and immutable inputs."""
from dataclasses import asdict,replace
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd
import sqlglot

from core.analysis_agent import source_histogram as hist
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_load_plan import source_plan
from tests.test_llm_goal import GoalModel,goal
from tests.test_row_preview import schema
from utils.analysis_image_validation import validate_chart_image
from utils.analysis_datasets import DatasetStore,stored_dataset_digest

SOURCE='lab.observations'


def result(frame,bins=20,where='',dialect='databricks'):
    query=hist.query_for(SOURCE,'reading',bins,where,dialect)
    with duckdb.connect() as db:
        db.execute('CREATE SCHEMA lab');db.register('input',frame)
        db.execute('CREATE TABLE lab.observations AS SELECT * FROM input')
        output=db.execute(sqlglot.transpile(query,read=dialect,write='duckdb')[0]).fetchdf()
    store=DatasetStore()
    info=store.register(output,source=SOURCE,query=query,coverage='complete',
                        grain='aggregate',aggregation='count',predicate_known=False)
    return store,info


class SourceHistogramTests(unittest.TestCase):
    def test_million_unique_values_transfer_only_bins_in_both_dialects(self):
        frame=pd.DataFrame({'reading':np.arange(1_000_001,dtype=float)})
        for dialect in ('mysql','databricks'):
            with self.subTest(dialect=dialect):
                store,info=result(frame,dialect=dialect)
                desc,edges,counts,total=hist.values(store,info.id,'reading',dialect)
                self.assertEqual(info.rows,20);self.assertEqual(total,len(frame))
                self.assertEqual(counts.tolist(),np.histogram(frame.reading,bins=edges)[0].tolist())
                self.assertEqual(source_plan(SOURCE,info.query,dialect=dialect).grain,'aggregate')
                self.assertEqual(desc['source'],SOURCE)

    def test_skew_empty_bins_constant_nulls_and_filter_oracle(self):
        for values,where in [([0,0,0,1,10,100,None],''),([7,7,None],''),
                             (list(range(30)), 'reading >= 8 AND reading <= 17'),
                             (list(range(30)), 'reading < 2 OR reading > 27')]:
            with self.subTest(values=values,where=where):
                frame=pd.DataFrame({'reading':values})
                store,info=result(frame,bins=10,where=where)
                desc,edges,counts,total=hist.values(store,info.id,'reading')
                expected=frame.reading.dropna()
                if where=='reading >= 8 AND reading <= 17':expected=expected[expected.between(8,17)]
                if where=='reading < 2 OR reading > 27':expected=expected[(expected<2)|(expected>27)]
                self.assertEqual(counts.tolist(),np.histogram(expected,bins=edges)[0].tolist())
                self.assertEqual(total,len(expected));self.assertLessEqual(info.rows,10)
                card=hist.render(store,info.id,'reading',legend=True,palette='high_contrast')
                validate_chart_image(card.image)
                self.assertEqual(card.render_spec['legend_labels'],['reading'])

    def test_tampered_or_partial_sql_is_not_a_bin_receipt(self):
        query=hist.query_for(SOURCE,'reading')
        for malformed in (query+' LIMIT 5',query.replace('COUNT(*) AS','SUM(__bucket) AS'),
                          query.replace('COUNT(*) OVER ()','COUNT(*) OVER (PARTITION BY reading)'),
                          'not SQL',query.replace('MAX(__n)','SUM(__n)')):
            self.assertIsNone(hist.descriptor(malformed))

    def test_bad_frequencies_and_incomplete_coverage_never_render(self):
        for kind in ('sum','duplicate','partial','nonfinite','fraction','parent'):
            with self.subTest(kind=kind):
                store,info=result(pd.DataFrame({'reading':range(100)}))
                if kind=='partial':store.metadata[info.id]=replace(info,coverage='truncated')
                elif kind=='parent':store.metadata[info.id]=replace(info,parent_id='foreign')
                else:
                    frame=store.frames[info.id]
                    if kind=='sum':frame.loc[0,'__frequency']+=1
                    if kind=='duplicate':frame.loc[1,'reading']=frame.loc[0,'reading']
                    if kind=='nonfinite':frame.loc[0,'__hist_min']=np.inf
                    if kind=='fraction':
                        frame['__frequency']=frame['__frequency'].astype(float)
                        frame.loc[0,'__frequency']=1.5
                with self.assertRaises(ValueError):hist.render(store,info.id,'reading')
        store,info=result(pd.DataFrame({'reading':pd.Series([None,None],dtype=float)}))
        with self.assertRaises(ValueError):hist.render(store,info.id,'reading')

    def runtime(self,root,*,policy=None,failure=None):
        executed=[]
        def factory(store):
            def execute(envelope):
                executed.append(envelope)
                if failure:raise failure
                fixture=pd.DataFrame({'reading':range(101),'cohort':['A']*101})
                desc=hist.descriptor(envelope['query'],'mysql')
                memory,info=result(fixture,bins=desc['bins'],where=desc['where_sql'],dialect='mysql')
                saved=store.register(memory.frames[info.id],source=SOURCE,query=envelope['query'],
                    coverage='complete',grain='aggregate',aggregation='count',predicate_known=False,
                    snapshot=datetime.now(timezone.utc).isoformat())
                return {'status':'ready','dataset':asdict(saved)}
            return execute
        model=GoalModel(goals=[goal('chart',{'kind':'histogram'},columns=['reading'])])
        runtime=GraphAnalysisRuntime(root,'bins','conversation',model,sql_dialect='mysql',
            reference_context_loader=lambda:schema(SOURCE),remote_factory=factory,
            connection_identity='bin-fixture',policy=policy)
        raw=runtime.datasets.register(pd.DataFrame({'reading':[1,2]}),source=SOURCE,
                                      query='SELECT reading FROM lab.observations LIMIT 2',coverage='unknown')
        runtime.select_dataset(raw.id)
        return runtime,raw,executed

    def test_model_selected_action_finishes_with_receipt_and_preserves_preview(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,raw,executed=self.runtime(root)
            before=stored_dataset_digest(runtime.datasets,raw.id)
            try:
                output=runtime.submit('Show the entire source reading histogram')
                self.assertEqual(output['status'],'answered',output)
                current=runtime.inspect()['recovery']
                self.assertEqual(len(current['artifact_ids']),1)
                self.assertEqual(len(executed),1)
                self.assertEqual(len(current['remote_query_evidence']),1)
                self.assertEqual(current['action_cycle_history'][-1]['choices'][0]['tool'],'analyze_source_histogram')
                card=runtime.artifacts[current['artifact_ids'][0]]
                self.assertEqual(card.render_spec['total_count'],101)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),before)
                self.assertFalse(runtime.recovery._proposed_scope_valid({'name':'analyze_source_histogram',
                    'args':{'source':SOURCE,'column':'reading','where_sql':'reading > 10'}},current))
                runtime.model.calls=[{'name':'analyze_source_histogram','args':{
                    'source':SOURCE,'column':'reading'}}];runtime.model.position=0
                followup=runtime.submit('Show the entire source reading histogram again')
                self.assertEqual(followup['status'],'answered',followup)
                self.assertEqual(len(executed),1,'same SQL/connection receipt must be reused')
                self.assertIn('재사용했습니다',followup['text'])
                status=runtime.execution_status()
                self.assertEqual([q['status'] for q in status['queries']],['completed'])
                runtime.model.calls=[{'name':'show_chart','args':{'chart_id':card.id}}]
                runtime.model.position=0
                shown=runtime.submit('Display the saved entire-source histogram')
                self.assertEqual(shown['status'],'answered',shown)
                self.assertIn('저장된 이미지를 표시했습니다',shown['text'])
                self.assertEqual(len(executed),1)
                fresh=goal('chart',{'kind':'histogram'},columns=['reading'])
                fresh['fresh_source_required']=True;runtime.model.goals=[fresh]
                runtime.model.calls=[];runtime.model.position=0
                updated=runtime.submit('Refresh source and show entire reading histogram')
                self.assertEqual(updated['status'],'answered',updated)
                self.assertEqual(len(executed),2,'explicit fresh request submits once')
            finally:runtime.close()

    def test_manual_approval_mode_does_not_offer_composite_execution(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,_,executed=self.runtime(root,policy=RuntimePolicy(require_remote_approval=True))
            try:
                self.assertNotIn('analyze_source_histogram',runtime.context.allowed_tool_names)
                self.assertEqual(executed,[])
            finally:runtime.close()

    def test_missing_receipt_or_forged_card_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,_,_=self.runtime(root)
            try:
                self.assertEqual(runtime.submit('Show entire reading histogram')['status'],'answered')
                current=runtime.inspect()['recovery'];card=runtime.artifacts[current['artifact_ids'][0]]
                self.assertTrue(runtime.recovery._valid_card(card,current,card.dataset_id))
                altered=replace(card,render_spec={**card.render_spec,'total_count':999})
                self.assertFalse(runtime.recovery._valid_card(altered,current,card.dataset_id))
                runtime.context.remote_receipt_reader=lambda _:None
                self.assertFalse(runtime.recovery._valid_card(card,current,card.dataset_id))
            finally:runtime.close()

    def test_filtered_goal_matches_sql_and_bins(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,raw,executed=self.runtime(root)
            runtime.model.goals=[goal('chart',{'kind':'histogram','bins':10},columns=['reading'],
                conditions=[{'column':'reading','op':'ge','value':30},{'column':'reading','op':'le','value':40}])]
            try:
                answer=runtime.submit('Show reading 30 to 40 in ten histogram bins')
                self.assertEqual(answer['status'],'answered',answer)
                current=runtime.inspect()['recovery'];card=runtime.artifacts[current['artifact_ids'][0]]
                self.assertEqual(card.render_spec['total_count'],11)
                self.assertEqual(card.render_spec['bins'],10)
                desc=hist.descriptor(executed[0]['query'],'mysql')
                self.assertIn('30',desc['where_sql']);self.assertIn('40',desc['where_sql'])
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
            finally:runtime.close()

    def test_uncertain_submission_is_not_retried_or_reported_as_image(self):
        with tempfile.TemporaryDirectory() as root:
            runtime,_,executed=self.runtime(root,failure=OSError('lost after submission'))
            try:
                outcome=runtime.submit('Show entire reading histogram')
                self.assertNotEqual(outcome['status'],'answered')
                self.assertEqual(len(executed),1)
                self.assertEqual(len(runtime.ledger.uncertain()),1)
                self.assertFalse(runtime.artifacts)
            finally:runtime.close()


if __name__=='__main__':unittest.main()
