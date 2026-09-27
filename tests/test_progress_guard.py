"""Successful inspection is not automatically progress toward the user's goal."""
import json
import tempfile
import unittest
from unittest.mock import patch

from core.analysis_agent import progress_guard as guard
from tests import test_tool_repair_loop as repair_tests
GOOD=repair_tests.GOOD
from tests.test_actual_agent_evaluation import EvaluationModel

INSPECT={'name':'list_analysis_context','args':{}}


class ProgressGuardTests(unittest.TestCase):
    def test_duplicate_discovery_is_skipped_then_model_calculates(self):
        helper=repair_tests.ToolRepairTests()
        with tempfile.TemporaryDirectory() as root:
            model=EvaluationModel(calls=[INSPECT]*3+[GOOD]);r=helper.setup_runtime(root,model)
            try:
                progress=[];r.on_progress=progress.append
                with helper.planning_disabled(r):result=r.submit('reading 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                log=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                self.assertEqual(sum(e['event']=='tool_started' and e['tool']=='list_analysis_context' for e in log),2)
                self.assertTrue(any(e['event']=='tool_progress_duplicate_blocked' for e in log))
                self.assertTrue(any('같은 정보 조회' in text for text in progress))
                self.assertEqual(r.datasets.frames[r.inspect()['recovery']['evidence_ids'][-1]].iloc[0,0],9.)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()

    def test_completed_calculation_is_kept_while_missing_chart_is_replanned(self):
        helper=repair_tests.ToolRepairTests()
        chart={'name':'render_chart_spec','args':{'dataset_id':'$fixture','kind':'histogram','x':'reading'}}
        with tempfile.TemporaryDirectory() as root:
            model=EvaluationModel(calls=[GOOD]+[INSPECT]*3+[chart]);r=helper.setup_runtime(root,model)
            try:
                with helper.planning_disabled(r):result=r.submit('reading 평균을 계산하고 reading 히스토그램도 보여줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(len(state['evidence_ids']),1)
                self.assertTrue(r.artifacts[state['artifact_ids'][0]].image.startswith(b'\x89PNG'))
                log=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                blocked=next(e for e in log if e['event']=='tool_progress_duplicate_blocked')
                self.assertEqual(blocked['missing'],['chart'])
                self.assertEqual(sum(e['event']=='tool_started' and e['tool']=='local_analysis_sql' for e in log),1)
            finally:r.close()

    def test_changed_metadata_or_observation_allows_reinspection(self):
        helper=repair_tests.ToolRepairTests()
        with tempfile.TemporaryDirectory() as root:
            r=helper.setup_runtime(root,EvaluationModel())
            try:
                current={};call={'name':'inspect_dataset','args':{'dataset_id':r.context.selected_dataset_id}}
                first={'status':'ready','columns':['reading']}
                guard.observe(current,call,first,r.context);guard.observe(current,call,first,r.context)
                self.assertIn(guard.call_key(call),guard.stalled(current,r.context))
                r.context.reference_context.append({'table':'custom.observations','columns':[{'name':'new_field'}]})
                self.assertFalse(guard.stalled(current,r.context))
                guard.observe(current,call,first,r.context)
                self.assertFalse(guard.stalled(current,r.context))
                guard.observe(current,call,{'status':'ready','columns':['reading','new_field']},r.context)
                self.assertFalse(guard.stalled(current,r.context))
                altered={'name':'inspect_dataset','args':{**call['args'],'preview_rows':5}}
                guard.observe(current,altered,first,r.context)
                self.assertFalse(guard.stalled(current,r.context))
            finally:r.close()

    def test_restart_preserves_duplicate_guard_without_changing_request_scope(self):
        helper=repair_tests.ToolRepairTests();original=EvaluationModel._generate
        def interrupt(model,*args,**kwargs):
            if model.position>=3:raise ValueError('injected pause')
            return original(model,*args,**kwargs)
        with tempfile.TemporaryDirectory() as root:
            model=EvaluationModel(calls=[INSPECT]*3);r=helper.setup_runtime(root,model)
            with helper.planning_disabled(r),patch.object(EvaluationModel,'_generate',interrupt):
                result=r.submit('reading 평균을 알려줘')
            self.assertEqual(result['status'],'incomplete',result)
            rejected=r.inspect()['recovery']['progress_rejected'];r.close()
            model=EvaluationModel(calls=[GOOD]);r=helper.setup_runtime(root,model)
            try:
                self.assertEqual(r.inspect()['recovery']['progress_rejected'],rejected)
                with helper.planning_disabled(r):result=r.resume()
                self.assertEqual(result['status'],'answered',result)
            finally:r.close()

    def test_remote_and_failed_results_do_not_enter_discovery_cache(self):
        current={}
        guard.observe(current,{'name':'query_databricks','args':{}},{'status':'ready'},None)
        guard.observe(current,INSPECT,{'status':'needs_refresh'},None)
        self.assertFalse(current)
        text=guard.instruction({'calculation':True,'chart':True,'evidence_ids':['verified']},{'render_chart_spec'})
        self.assertIn('"verified": ["calculation"]',text)
        self.assertIn('"goal": "chart"',text)
        self.assertNotIn('query_databricks',text)

    def test_scope_metadata_change_invalidates_stall_and_new_request_starts_clean(self):
        from dataclasses import asdict, replace
        helper=repair_tests.ToolRepairTests()
        with tempfile.TemporaryDirectory() as root:
            r=helper.setup_runtime(root,EvaluationModel())
            try:
                current={};observation={'status':'ready'}
                guard.observe(current,INSPECT,observation,r.context)
                guard.observe(current,INSPECT,observation,r.context)
                self.assertTrue(guard.stalled(current,r.context))
                key=r.context.selected_dataset_id
                changed=asdict(replace(r.datasets.metadata[key],coverage='unknown'))
                with r.db.conn:
                    r.db.conn.execute('UPDATE assets SET metadata=? WHERE id=?',
                                      (json.dumps(changed),key))
                self.assertFalse(guard.stalled(current,r.context))
                self.assertFalse(guard.stalled({},r.context))
            finally:r.close()
