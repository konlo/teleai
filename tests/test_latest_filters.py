"""Same predicate, different stage: independent expected winners and counts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel
from tests import test_remote_latest as remote
from utils.analysis_latest import latest_distribution
from utils.analysis_datasets import stored_dataset_digest
from ui.analysis_chart_delivery import chart_references

FIXTURE=json.loads((Path(__file__).parent/'fixtures/latest_filters.json').read_text())
KEY,CLOCK,VALUE,FILTER=FIXTURE['columns']

def frame():
    return pd.DataFrame(FIXTURE['rows'],columns=FIXTURE['columns'])

def references():
    return [{'table':FIXTURE['source'],'observed_at':datetime.now(timezone.utc).isoformat(),
        'columns':[{'name':c,'dtype':str(frame()[c].dtype)} for c in FIXTURE['columns']]}]

def runtime(root):
    r=GraphAnalysisRuntime(root,'test','staged-latest',QuietModel())
    if not r.datasets.metadata:
        raw=r.datasets.register(frame(),source=FIXTURE['source'],coverage='complete',predicate_known=True)
        r.select_dataset(raw.id)
    return r

def counts(proof):
    return {row[VALUE]:row[proof['count_column']] for row in proof['counts']}

class LatestFilterTests(unittest.TestCase):
    def test_sql_nan_filter_matches_pandas_missing_semantics(self):
        import duckdb
        from utils.analysis_latest_filters import sql
        for op,expected in (('ne',1),('gt',0),('lt',1)):
            condition={'column':FILTER,'op':op,'value':FIXTURE['condition']['value']}
            for dialect in ('duckdb','databricks'):
                where=sql([condition],dialect=dialect,types={FILTER:'double'})
                from sqlglot import parse_one
                query=f'SELECT COUNT(*) FROM (SELECT CAST(\'NaN\' AS DOUBLE) AS `{FILTER}` UNION ALL SELECT 1.0) AS data WHERE '+where
                rendered=parse_one(query,read='databricks').sql(dialect='duckdb') if dialect=='databricks' else query.replace('`','"')
                with duckdb.connect() as conn:
                    self.assertEqual(conn.execute(rendered).fetchone()[0],expected)

    def test_numeric_histogram_bins_followup_keeps_filter_stage(self):
        import numpy as np
        for mode in ('local','remote'):
            for stage in ('before','after'):
                with self.subTest(mode=mode,stage=stage),tempfile.TemporaryDirectory() as root,patch.object(remote,'references',side_effect=references):
                    if mode=='remote':r,calls=remote.RemoteLatestTests().runtime(root,frame())
                    else:r=runtime(root)
                    try:
                        prefix=(FIXTURE['source']+' 테이블에서 ') if mode=='remote' else ''
                        r.submit(prefix+FIXTURE['numeric_prompt']+' '+FIXTURE[stage])
                        for bins in (4,6):
                            if bins==6:r.submit('6개 구간으로 바꿔줘')
                            proof=r.inspect()['recovery'].get('latest_selection_evidence')
                            self.assertTrue(proof)
                            expected,edges=np.histogram(FIXTURE['numeric_expected_'+stage],bins=bins)
                            self.assertEqual(proof['chart_spec']['counts'],expected.tolist())
                            self.assertEqual(proof['chart_spec']['edges'],edges.tolist())
                            self.assertEqual(proof['distribution']['row_selection']['filter_stage'],stage+'_selection')
                    finally:r.close()

    def test_explicit_stage_local_and_remote_graph(self):
        for mode in ('local','remote'):
            for stage in ('before','after'):
                with self.subTest(mode=mode,stage=stage),tempfile.TemporaryDirectory() as root:
                    if mode=='remote':
                        with patch.object(remote,'references',side_effect=references):
                            r,calls=remote.RemoteLatestTests().runtime(root,frame())
                        prompt=FIXTURE['source']+' 테이블에서 '+FIXTURE['prompt']
                    else:
                        r=runtime(root);calls=[];prompt=FIXTURE['prompt']
                    try:
                        outcome=r.submit(prompt+' '+FIXTURE[stage])
                        proof=r.inspect()['recovery'].get('latest_selection_evidence')
                        self.assertTrue(proof,outcome)
                        self.assertEqual(counts(proof),FIXTURE['expected_'+stage])
                        self.assertTrue(chart_references(r.events()[-1]))
                        self.assertEqual(len(calls),1 if mode=='remote' else 0)
                    finally:r.close()

    def test_stage_clarification_then_change_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            r=runtime(root)
            try:
                r.submit(FIXTURE['prompt'])
                self.assertFalse(chart_references(r.events()[-1]))
                raw=r.context.selected_dataset_id;digest=stored_dataset_digest(r.datasets,raw)
            finally:r.close()
            r=runtime(root)
            try:
                outcome=r.submit(FIXTURE['before'])
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertTrue(proof,outcome)
                self.assertEqual(counts(proof),FIXTURE['expected_before'])
            finally:r.close()
            r=runtime(root)
            try:
                outcome=r.submit(FIXTURE['after'])
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertTrue(proof,outcome)
                self.assertEqual(counts(proof),FIXTURE['expected_after'])
                self.assertEqual(stored_dataset_digest(r.datasets,raw),digest)
                self.assertEqual(r.context.selected_dataset_id,raw)
            finally:r.close()

    def test_tool_rejects_missing_stage_and_guard_rejects_dropped_filter(self):
        with tempfile.TemporaryDirectory() as root:
            r=runtime(root)
            try:
                raw=r.context.selected_dataset_id
                with self.assertRaises(ValueError):
                    latest_distribution(r.context,raw,[KEY],CLOCK,VALUE,conditions=[FIXTURE['condition']])
                r.submit(FIXTURE['prompt']+' '+FIXTURE['before'])
                state=r.inspect()['recovery'];spec=state['latest_per_key_spec']
                args={k:spec[k] for k in ('dataset_id','key_columns','order_column','value_column')}
                self.assertFalse(r.recovery._proposed_scope_valid({'name':'analyze_latest_distribution','args':args},state))
            finally:r.close()

    def test_large_filters_project_only_needed_columns_and_preserve_root(self):
        # 30,000 rows, 15,000 keys; expected counts scale directly from fixture.
        blocks=[]
        for i in range(5000):
            part=frame();part[KEY]=part[KEY]+str(i);blocks.append(part)
        large=pd.concat(blocks,ignore_index=True)
        large['irrelevant_payload']='x'*128
        with tempfile.TemporaryDirectory() as root:
            r=runtime(root)
            try:
                raw=r.datasets.register(large,source=FIXTURE['source'],coverage='complete',predicate_known=True)
                r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id)
                project=r.datasets.frames.project;batches=r.datasets.frames.batches
                def guarded_project(identity,columns):
                    if identity==raw.id:raise AssertionError('Full source projection')
                    return project(identity,columns)
                def guarded_batches(identity,columns,**kwargs):
                    if identity==raw.id:self.assertEqual(set(columns),set(FIXTURE['columns']))
                    return batches(identity,columns,**kwargs)
                with patch.object(r.datasets.frames,'project',side_effect=guarded_project),patch.object(r.datasets.frames,'batches',side_effect=guarded_batches):
                    for stage in ('before','after'):
                        proof=latest_distribution(r.context,raw.id,[KEY],CLOCK,VALUE,
                            conditions=[FIXTURE['condition']],filter_stage=stage+'_selection')
                        self.assertEqual(proof['status'],'ready',proof)
                        self.assertEqual(counts(proof),{k:v*5000 for k,v in FIXTURE['expected_'+stage].items()})
                        self.assertEqual(proof['execution_mode'],'bounded_local_sql')
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertNotIn(raw.id,r.datasets.frames.cache)
            finally:r.close()

    def test_remote_stage_change_does_not_reuse_wrong_receipt(self):
        with tempfile.TemporaryDirectory() as root,patch.object(remote,'references',side_effect=references):
            r,calls=remote.RemoteLatestTests().runtime(root,frame())
            try:
                prompt=FIXTURE['source']+' 테이블에서 '+FIXTURE['prompt']
                r.submit(prompt+' '+FIXTURE['before'])
                self.assertEqual(len(calls),1)
            finally:r.close()
            r,calls=remote.RemoteLatestTests().runtime(root,frame())
            try:
                outcome=r.submit(FIXTURE['after'])
                proof=r.inspect()['recovery'].get('latest_selection_evidence')
                self.assertTrue(proof,outcome)
                self.assertEqual(counts(proof),FIXTURE['expected_after'])
                self.assertEqual(len(calls),1)
                r.submit(prompt+' '+FIXTURE['after'])
                self.assertEqual(len(calls),1)
                state=r.inspect()['recovery'];spec=state['latest_per_key_spec']
                args={k:spec[k] for k in ('source','key_columns','order_column','value_column','conditions')}
                args.update(filter_stage='before_selection',result_dataset_id=proof['input_result_id'])
                from core.analysis_agent.latest_selection import accepted_remote
                self.assertFalse(accepted_remote(r.context,r.artifacts,spec,args,proof,state['remote_query_evidence']))
            finally:r.close()

    def test_filter_cannot_hide_ambiguous_winner_or_coerce_value(self):
        from utils.analysis_remote_latest import plan
        from tests.test_remote_latest import execute_fixture
        tied=pd.concat([frame(),frame().iloc[[1]].assign(**{FILTER:10})],ignore_index=True)
        with tempfile.TemporaryDirectory() as root,patch.object(remote,'references',side_effect=references):
            r,calls=remote.RemoteLatestTests().runtime(root,tied)
            try:
                outcome=r.submit(FIXTURE['source']+' 테이블에서 '+FIXTURE['prompt']+' '+FIXTURE['after'])
                self.assertEqual(r.inspect()['recovery'].get('latest_error_code'),'latest_order_tie',outcome)
                self.assertFalse(chart_references(r.events()[-1]))
                raw=r.datasets.register(tied,source=FIXTURE['source'],coverage='complete',predicate_known=True)
                proof=latest_distribution(r.context,raw.id,[KEY],CLOCK,VALUE,conditions=[FIXTURE['condition']],filter_stage='after_selection')
                self.assertEqual(proof['error_code'],'latest_order_tie')
                for condition in ({'column':FILTER,'op':'lt','value':'50'}, {'column':VALUE,'op':'eq','value':1}):
                    with self.assertRaises(ValueError):
                        latest_distribution(r.context,raw.id,[KEY],CLOCK,VALUE,conditions=[condition],filter_stage='before_selection')
                    with self.assertRaises(ValueError):
                        plan(r.context,FIXTURE['source'],[KEY],CLOCK,VALUE,conditions=[condition],filter_stage='before_selection')
                # Quoted input remains a literal, not SQL structure.
                condition={'column':VALUE,'op':'eq','value':"x' OR 1=1 --"}
                spec=plan(r.context,FIXTURE['source'],[KEY],CLOCK,VALUE,conditions=[condition],filter_stage='before_selection')
                output=execute_fixture(spec['remote_latest_plan']['query'],frame())
                self.assertEqual(output.iloc[0]['__selected_keys'],0)
            finally:r.close()

    def test_ambiguous_and_changed_source_followups_do_not_publish(self):
        from core.analysis_agent.latest_selection import continue_order
        for tail in ('',FIXTURE['before']+' '+FIXTURE['after']):
            with tempfile.TemporaryDirectory() as root:
                r=runtime(root)
                try:
                    r.submit(FIXTURE['prompt']+' '+tail)
                    self.assertFalse(chart_references(r.events()[-1]))
                    r.submit(FIXTURE['before'])
                    prior=r.inspect()['recovery']
                    other=r.datasets.register(frame(),source=FIXTURE['source']+'_other',coverage='complete',predicate_known=True)
                    r.select_dataset(other.id)
                    self.assertIsNone(continue_order(FIXTURE['after'],prior,r.context))
                finally:r.close()
