"""False positives and evaluator/product policy isolation for the closing batch."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from scripts.group_table_oracle import compare
from scripts.evaluate_analysis_agent import load_specs,load_grading,load_frames,reference_oracle
from tests.test_agent_sql_recovery_boundaries import JoinedProposalModel
from scripts.evaluate_spider2_teleai import evaluate_task


class BatchEvaluationContracts(unittest.TestCase):
    def test_group_oracle_rejects_wrong_order_missing_measure_and_wrong_group(self):
        grading=load_grading()['L1_051']
        spec=next(s for s in load_specs() if s['id']=='L1_051')
        expected=reference_oracle(spec,grading,load_frames())
        actual=expected.rename(columns={'고객수':'n','평균잔액':'m'})
        info=SimpleNamespace(query='SELECT job, COUNT(*) AS n, AVG(balance) AS m FROM data GROUP BY job ORDER BY m DESC')
        self.assertTrue(compare(actual,info,grading,expected))
        self.assertFalse(compare(actual.iloc[::-1],info,grading,expected))
        self.assertFalse(compare(actual.drop(columns='n'),info,grading,expected))
        altered=actual.copy();altered.loc[0,'m']+=100
        self.assertFalse(compare(altered,info,grading,expected))
        with self.assertRaises(ValueError):
            compare(actual,SimpleNamespace(query=info.query.replace('AVG(balance)','MAX(balance)')),grading,expected)

    def test_spider_proposal_does_not_inherit_auto_execute_policy(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'public.sqlite'
            with sqlite3.connect(path) as db:
                db.executescript('CREATE TABLE events(k INTEGER); CREATE TABLE labels(k INTEGER); INSERT INTO events VALUES(1);')
            # Source lists only physical tables referenced by the proposal.
            class SingleProposal(JoinedProposalModel):
                def _generate(self,messages,**kwargs):
                    result=super()._generate(messages,**kwargs)
                    result.generations[0].message.tool_calls[0]['args']['source']='events'
                    return result
            with patch('scripts.evaluate_spider2_teleai.database_path',return_value=path),patch('scripts.evaluate_spider2_teleai.task_document',return_value=''):
                result=evaluate_task(Path(root),{'instance_id':'local_fixture','db':'public','question':'events 전체 행 수를 SQL로 계산해줘'},
                    SingleProposal(query='SELECT COUNT(*) FROM events'),Path(root)/'predictions',benchmark_instruction=True)
            self.assertEqual(result['status'],'SQL_PROPOSED',result)
            self.assertEqual(result['remote_executions'],0)
            self.assertEqual(result['agent_status'],'awaiting_approval')

    def test_group_sort_preserves_metric_and_group_roles(self):
        from core.analysis_agent.eda_contract import group_sort
        metrics=[{'name':'n','aggregation':'count'},{'name':'m','aggregation':'mean','value_column':'measurement'}]
        self.assertEqual(group_sort('segment별 measurement 평균과 건수를 구하고 평균 내림차순으로',metrics,'segment'),
            {'sort':'metric_descending','sort_by':'m'})
        self.assertEqual(group_sort('segment별 measurement 평균과 건수를 구하고 segment 내림차순으로',metrics,'segment'),
            {'sort':'group_descending','sort_by':''})
        self.assertIsNone(group_sort('평균과 건수를 구해서 그 결과를 내림차순으로',metrics,'segment'))

        self.assertEqual(group_sort('age groups, average descending',metrics,'age'),
            {'sort':'metric_descending','sort_by':'m'})
