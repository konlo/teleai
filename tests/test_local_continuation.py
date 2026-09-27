"""A failed provider must not discard partial work or trigger remote replay."""
from contextlib import contextmanager
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import APITimeoutError, AuthenticationError
from tests import test_tool_repair_loop as helpers
from tests.test_actual_agent_evaluation import EvaluationModel
from utils.analysis_datasets import stored_dataset_digest

CHART={'name':'render_chart_spec','args':{'dataset_id':'$fixture','kind':'histogram','x':'reading'}}


class OutageModel(EvaluationModel):
    attempts: int = 0
    error_kind: str = 'timeout'
    def _generate(self,*args,**kwargs):
        self.attempts += 1
        if self.position >= len(self.calls):
            if self.error_kind == 'bug': raise ValueError('private-error')
            if self.error_kind == 'auth':
                raise AuthenticationError('private-error',response=httpx.Response(401,
                    request=httpx.Request('POST','https://example.invalid')),body=None)
            raise APITimeoutError(request=httpx.Request('POST','https://example.invalid'))
        return super()._generate(*args,**kwargs)


@contextmanager
def model_first(runtime, model):
    """Exercise inference failure, then the real grounded local planner."""
    original=runtime.recovery._next_local
    def plan(current,calls):
        return original(current,calls) if model.attempts > len(model.calls) else None
    with patch.object(runtime.recovery,'_next_local',side_effect=plan),patch.object(
            runtime.model_recovery,'sleep'):
        yield


class LocalContinuationTests(unittest.TestCase):
    def test_partial_chart_is_kept_and_missing_mean_finishes_without_user_resume(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel(calls=[CHART]);r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                progress=[];r.on_progress=progress.append
                digest=stored_dataset_digest(r.datasets,model.evaluation_dataset_id)
                with model_first(r,model):result=r.submit('reading 평균을 계산하고 reading 히스토그램도 보여줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertTrue(state['local_continuation']['error_id'])
                self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],9.)
                self.assertEqual(len(state['artifact_ids']),1)
                self.assertTrue(r.artifacts[state['artifact_ids'][0]].image.startswith(b'\x89PNG'))
                self.assertEqual(stored_dataset_digest(r.datasets,model.evaluation_dataset_id),digest)
                self.assertEqual(model.attempts,4)  # chart + three failed inferences; no new model calls
                self.assertEqual(state['model_calls'],4)
                events=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                self.assertEqual(sum(e['event']=='tool_started' and e['tool']=='render_chart_spec' for e in events),1)
                self.assertEqual(sum(e['event']=='automatic_local_continuation' for e in events),1)
                self.assertTrue(any('남은 작업을 이어' in t for t in progress))
                self.assertFalse(r.inspect()['requests'])
                self.assertFalse(r.agent.get_state(r.config).next)
            finally:r.close()

    def test_more_than_one_local_obligation_finishes_after_model_outage(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel();r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                with model_first(r,model):result=r.submit('reading 평균을 계산하고 reading 히스토그램도 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(model.attempts,3)
                self.assertEqual(len(r.inspect()['recovery']['artifact_ids']),1)
                self.assertEqual(len(r.inspect()['recovery']['evidence_ids']),1)
            finally:r.close()

    def test_filter_is_preserved_in_automatic_calculation(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel();r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                with model_first(r,model):result=r.submit('reading >= 10인 행의 reading 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.datasets.frames[r.inspect()['recovery']['evidence_ids'][-1]].iloc[0,0],15.)
                self.assertEqual(model.attempts,3)
            finally:r.close()

    def test_auth_and_programming_errors_do_not_enter_continuation(self):
        for kind in ['auth','bug']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                model=OutageModel(error_kind=kind);r=helpers.ToolRepairTests().setup_runtime(root,model)
                try:
                    with model_first(r,model):result=r.submit('reading 평균을 알려줘')
                    self.assertEqual(result['status'],'incomplete',result)
                    self.assertEqual(model.attempts,1)
                    self.assertNotIn('local_continuation',r.inspect()['recovery'])
                finally:r.close()

    def test_uncertain_remote_execution_blocks_automatic_continuation(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel();r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                with model_first(r,model),patch.object(r.ledger,'uncertain',return_value=[{'id':'uncertain'}]):
                    result=r.submit('reading 평균을 알려줘')
                self.assertEqual(result['status'],'incomplete',result)
                self.assertNotIn('local_continuation',r.inspect()['recovery'])
            finally:r.close()

    def test_no_verified_plan_preserves_failure_instead_of_claiming_success(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel();r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                with helpers.ToolRepairTests().planning_disabled(r),patch('core.analysis_agent.model_recovery.time.sleep'):
                    result=r.submit('reading 평균을 알려줘')
                self.assertEqual(result['status'],'incomplete',result)
                self.assertNotIn('local_continuation',r.inspect()['recovery'])
            finally:r.close()

    def test_local_failure_stops_without_model_or_remote_retry(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel(calls=[CHART]);r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                with model_first(r,model),patch('core.analysis_runtime_tools.local_query',
                        side_effect=TimeoutError('private local timeout')) as worker:
                    result=r.submit('reading 평균을 계산하고 reading 히스토그램도 보여줘')
                self.assertNotEqual(result['status'],'answered',result)
                self.assertEqual(r.inspect()['recovery']['stop_reason'],'local_continuation_unavailable')
                self.assertEqual(worker.call_count,1)
                self.assertEqual(model.attempts,4)
                self.assertEqual(len(r.inspect()['recovery']['artifact_ids']),1)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()

    def test_restart_keeps_local_only_mode_and_new_request_clears_it(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel(calls=[CHART]);r=helpers.ToolRepairTests().setup_runtime(root,model)
            with model_first(r,model),patch('core.analysis_runtime_tools.local_query',
                    side_effect=RuntimeError('synthetic process interruption')):
                result=r.submit('reading 평균을 계산하고 reading 히스토그램도 보여줘')
            self.assertEqual(result['status'],'incomplete',result)
            self.assertTrue(r.inspect()['recovery']['local_continuation'])
            r.close()
            model=OutageModel();r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                result=r.resume()
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(model.attempts,0)
                self.assertEqual(len(r.inspect()['recovery']['artifact_ids']),1)
                result=r.submit('reading 평균을 알려줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertNotIn('local_continuation',r.inspect()['recovery'])
            finally:r.close()

    def test_tool_budget_cannot_be_reset_by_provider_failure(self):
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel(calls=[CHART]);r=helpers.ToolRepairTests().setup_runtime(root,model)
            r.recovery.max_tool_calls=2
            try:
                with model_first(r,model):result=r.submit('reading 평균을 계산하고 reading 히스토그램도 보여줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(r.inspect()['recovery']['sent_calls']),2)
            finally:r.close()

    def test_prepared_chart_uses_output_dataset_not_explicit_raw_input(self):
        # Observed live serving decision: prepare_histogram includes dataset_id.
        prepare={'name':'prepare_histogram','args':{'source':'custom.observations',
            'column':'reading','dataset_id':'$fixture'}}
        with tempfile.TemporaryDirectory() as root:
            model=OutageModel(calls=[prepare]);r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                with model_first(r,model):
                    result=r.submit('먼저 reading 히스토그램을 만들고, 이어서 reading 평균도 계산해줘.')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(len(state['artifact_ids']),1)
                card=r.artifacts[state['artifact_ids'][0]]
                self.assertNotEqual(card.dataset_id,model.evaluation_dataset_id)
                self.assertEqual(r.datasets.metadata[card.dataset_id].parent_id,model.evaluation_dataset_id)
                self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],9.)
                self.assertEqual(model.attempts,4)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()

    def test_followup_filter_reset_restores_raw_for_mean_and_chart(self):
        with tempfile.TemporaryDirectory() as root:
            model=EvaluationModel();r=helpers.ToolRepairTests().setup_runtime(root,model)
            try:
                first=r.submit('reading < 10인 행의 reading 평균을 알려줘')
                self.assertEqual(first['status'],'answered',first)
                result=r.submit('현재 로딩된 원본 표본 전체에서 reading 평균을 계산하고 reading 히스토그램도 보여줘. 이전 reading 조건은 적용하지 말고 보유 데이터만 사용해.')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery']
                self.assertEqual(state['scope']['conditions'],[])
                self.assertEqual(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0],9.)
                self.assertEqual(len(state['artifact_ids']),1)
                self.assertEqual(model.position,0)
                self.assertFalse(r.inspect()['requests'])
            finally:r.close()
