"""Summary and pending-message failures must preserve history and exact grants."""
from core.analysis_agent.policy import RuntimePolicy
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import AuthenticationError, RateLimitError

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.memory import memory_middleware
from tests.test_actual_agent_evaluation import EvaluationModel
from tests.test_model_recovery import rate_error


class AuxiliaryModel(EvaluationModel):
    failures_left: int = 0
    summary_calls: int = 0
    classifier_calls: int = 0
    mode: str = 'status'
    fail_kind: str = 'classification'
    cooldown: str = '0'
    auth_failure: bool = False

    def _generate(self,messages,**kwargs):
        prompt=str(messages[0].content)
        kind=('summary' if prompt.startswith('이 분석 대화를 이어가기') else
              'classification' if prompt.startswith('사용자 메시지가 오직') else 'analysis')
        if kind=='summary':self.summary_calls+=1
        if kind=='classification':self.classifier_calls+=1
        if kind==self.fail_kind and (self.failures_left or self.auth_failure):
            self.failures_left=max(0,self.failures_left-1)
            if self.auth_failure:
                raise AuthenticationError('private-auth-content',response=httpx.Response(401,
                    request=httpx.Request('POST','https://example.invalid')),body=None)
            raise rate_error(self.cooldown)
        text=('요청과 확인한 결과를 유지합니다.' if kind=='summary' else
              self.mode if kind=='classification' and self.mode.startswith('invalid') else
              json.dumps({'action':self.mode}) if kind=='classification' else '이전 내용을 유지했습니다.')
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class AuxiliaryRecoveryTests(unittest.TestCase):
    def test_existing_failure_ledger_migrates_without_resetting_budget(self):
        from core.analysis_agent.assets import AssetDB
        from core.analysis_agent.model_recovery import ModelAttemptLedger
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','legacy')
            try:
                with db.conn:
                    db.conn.execute('CREATE TABLE model_recovery (request_id TEXT PRIMARY KEY, '
                        'failures INTEGER NOT NULL DEFAULT 0, failed_seconds REAL NOT NULL DEFAULT 0, '
                        'retries INTEGER NOT NULL DEFAULT 0, next_allowed_at REAL NOT NULL DEFAULT 0)')
                    db.conn.execute("INSERT INTO model_recovery VALUES ('old',2,7.5,1,12345)")
                ledger=ModelAttemptLedger(db)
                self.assertEqual(ledger.get('old'),{'failures':2,'failed_seconds':7.5,
                    'retries':1,'next_allowed_at':12345,'aux_calls':0,'aux_seconds':0.0})
                ledger.auxiliary_success('old',1.5)
                self.assertEqual(ModelAttemptLedger(db).get('old')['aux_calls'],1)
                self.assertEqual(ledger.get('old')['failures'],2)
            finally:db.close()

    def runtime(self, root, model, calls=None, **kwargs):
        calls=[] if calls is None else calls
        return GraphAnalysisRuntime(root,'owner','auxiliary',model,
            remote_factory=lambda _:lambda envelope:calls.append(envelope),
            connection_identity='synthetic-only',**kwargs, policy=RuntimePolicy(require_remote_approval=True),intent_mode='contract_fixture')

    def pending(self, r):
        return r.propose_query('fixture','SELECT 1','fixture verification')['requests'][0]

    def test_transient_status_classification_is_cached_across_restart(self):
        with tempfile.TemporaryDirectory() as root,patch('core.analysis_agent.model_recovery.time.sleep'):
            model=AuxiliaryModel(failures_left=1);calls=[];r=self.runtime(root,model,calls)
            grant=self.pending(r);before=r.agent.get_state(r.config).config
            try:
                result=r.submit('지금 어떤 승인을 기다리고 있어?')
                self.assertEqual(result['status'],'awaiting_approval',result)
                self.assertEqual(model.classifier_calls,2)
                self.assertEqual(r.ledger.get(grant['id'])['status'],'proposed')
                self.assertEqual(r.agent.get_state(r.config).config,before)
                ledger=r.inspect()['model_recovery']
                self.assertEqual(ledger['failures'],1);self.assertEqual(ledger['aux_calls'],1)
                self.assertEqual(calls,[])
            finally:r.close()
            model=AuxiliaryModel(auth_failure=True);r=self.runtime(root,model,calls)
            try:
                self.assertEqual(r.submit('지금 어떤 승인을 기다리고 있어?')['status'],'awaiting_approval')
                self.assertEqual(model.classifier_calls,0)
                self.assertEqual(r.inspect()['requests'][0]['id'],grant['id'])
                self.assertEqual(calls,[])
            finally:r.close()

    def test_malformed_unknown_and_model_failure_do_not_invalidate_approval(self):
        for mode in ('invalid-json','uncertain','approve','auth'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                model=AuxiliaryModel(mode=mode,auth_failure=mode=='auth');calls=[]
                r=self.runtime(root,model,calls);grant=self.pending(r)
                try:
                    before=r.agent.get_state(r.config).config
                    result=r.submit('그 요청 어떻게 되는 거야?')
                    self.assertEqual(result['status'],'awaiting_approval',result)
                    self.assertIn('유지',result['text'])
                    self.assertEqual(r.agent.get_state(r.config).config,before)
                    self.assertEqual(r.ledger.get(grant['id'])['status'],'proposed')
                    self.assertEqual(model.classifier_calls,1)
                    self.assertEqual(calls,[])
                    self.assertNotIn('private-auth-content',r.diagnostics.path.read_text())
                finally:r.close()

    def test_classification_cooldown_survives_restart(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.runtime(root,AuxiliaryModel(failures_left=1,cooldown='120'));grant=self.pending(r)
            try:
                self.assertEqual(r.submit('어떤 요청이야?')['error_type'],'RateLimitError')
            finally:r.close()
            model=AuxiliaryModel();r=self.runtime(root,model)
            try:
                result=r.submit('다른 표현으로 상태를 알려줘')
                self.assertEqual(result['error_type'],'ModelCoolingDown')
                self.assertEqual(model.classifier_calls,0)
                self.assertEqual(r.inspect()['requests'][0]['id'],grant['id'])
            finally:r.close()

    def test_status_variants_share_original_request_budget_and_preserve_grant(self):
        with tempfile.TemporaryDirectory() as root:
            model=AuxiliaryModel();r=self.runtime(root,model);grant=self.pending(r)
            try:
                base=r.inspect()['recovery']['model_calls']
                for n in range(12):result=r.submit(f'상태를 알려줘 {n}')
                self.assertEqual(result['status'],'awaiting_approval')
                self.assertEqual(result['error_type'],'ModelAttemptBudgetExceeded')
                self.assertEqual(model.classifier_calls,9-base)
                self.assertEqual(r.ledger.get(grant['id'])['status'],'proposed')
            finally:r.close()

    def test_classification_failure_does_not_disable_explicit_approval(self):
        from migration.test_approval_rollout import GraphApprovalTests
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            r=GraphApprovalTests().runtime(root,AuxiliaryModel(auth_failure=True),calls)
            try:
                grant=self.pending(r)
                self.assertEqual(r.submit('어떤 승인을 기다리는 거지?')['error_type'],'AuthenticationError')
                self.assertEqual(r.submit('승인해줘')['status'],'answered')
                self.assertEqual(len(calls),1)
                with self.assertRaises(PermissionError):r.respond(grant['id'],approved=True)
                self.assertEqual(len(calls),1)
            finally:r.close()

    def test_summary_cooldown_and_history_survive_graph_restart(self):
        from uuid import uuid4
        with tempfile.TemporaryDirectory() as root:
            model=AuxiliaryModel(fail_kind='summary')
            r=self.runtime(root,model,summary_trigger_tokens=100,summary_keep_messages=2)
            history=[HumanMessage(content='기존 원문 '*100,id=str(uuid4())),
                     AIMessage(content='확인했습니다.',id=str(uuid4()))]*4
            history=[m.model_copy(update={'id':str(uuid4())}) for m in history]
            r.agent.update_state(r.config,{'messages':history},as_node='model');r.resume();r.events()
            before=[m.id for m in r.events()]
            model.failures_left=1;model.cooldown='120'
            result=r.submit('앞선 내용을 유지해서 설명해줘')
            self.assertEqual(result['error_type'],'RateLimitError')
            retained=[m.id for m in r.events()]
            self.assertEqual(retained[:len(before)],before)
            r.close()
            model=AuxiliaryModel(fail_kind='summary')
            r=self.runtime(root,model,summary_trigger_tokens=100,summary_keep_messages=2)
            try:
                result=r.resume()
                self.assertEqual(result['error_type'],'ModelCoolingDown',result)
                self.assertEqual(model.summary_calls,0)
                self.assertEqual([m.id for m in r.events()],retained)
                self.assertEqual(r.inspect()['model_recovery']['failures'],1)
            finally:r.close()

    def summary_state(self, calls=0):
        return {'messages':[HumanMessage(content='private-history-value '*80),
            AIMessage(content='기존 대화'),HumanMessage(content='새로운 질문'),AIMessage(content='최근 답변')],
            'recovery':{'request_id':'summary-request','model_calls':calls,'model_seconds':0}}

    def test_summary_transient_retry_preserves_history_and_counts_attempts(self):
        with tempfile.TemporaryDirectory() as root,patch('core.analysis_agent.model_recovery.time.sleep'):
            model=AuxiliaryModel(fail_kind='summary',failures_left=1);r=self.runtime(root,model)
            try:
                middleware=memory_middleware(model,10,2,diagnostics=r.diagnostics,model_recovery=r.model_recovery)
                state=self.summary_state();original=list(state['messages'])
                result=middleware.before_model(state,None)
                r.model_attempts.sync(result['recovery'])
                self.assertEqual(result['recovery']['model_calls'],2)
                self.assertEqual(model.summary_calls,2)
                self.assertEqual(state['messages'],original)
                self.assertNotIn('private-history-value',r.diagnostics.path.read_text())
            finally:r.close()

    def test_summary_auth_error_never_uses_hidden_framework_retry(self):
        with tempfile.TemporaryDirectory() as root:
            model=AuxiliaryModel(fail_kind='summary',auth_failure=True);r=self.runtime(root,model)
            try:
                middleware=memory_middleware(model,10,2,model_recovery=r.model_recovery)
                state=self.summary_state();original=list(state['messages'])
                with self.assertRaises(AuthenticationError):middleware.before_model(state,None)
                self.assertEqual(model.summary_calls,1)
                self.assertEqual(state['messages'],original)
            finally:r.close()

    def test_summary_reserves_last_analysis_attempt_even_when_retrying(self):
        with tempfile.TemporaryDirectory() as root,patch('core.analysis_agent.model_recovery.time.sleep'):
            model=AuxiliaryModel(fail_kind='summary',failures_left=1);r=self.runtime(root,model)
            try:
                middleware=memory_middleware(model,10,2,model_recovery=r.model_recovery)
                with self.assertRaises(RateLimitError):middleware.before_model(self.summary_state(calls=8),None)
                self.assertEqual(model.summary_calls,1)
                self.assertEqual(r.model_attempts.get('summary-request')['retries'],0)
            finally:r.close()

    def test_graph_summary_recovers_without_erasing_visible_transcript(self):
        with tempfile.TemporaryDirectory() as root,patch('core.analysis_agent.model_recovery.time.sleep'):
            model=AuxiliaryModel(fail_kind='summary');r=self.runtime(root,model,summary_trigger_tokens=100,summary_keep_messages=2)
            try:
                history=[HumanMessage(content='확정한 조건 '*80),AIMessage(content='확인했습니다.')]*5
                # Give each retained turn a distinct message ID.
                from uuid import uuid4
                history=[message.model_copy(update={'id':str(uuid4())}) for message in history]
                r.agent.update_state(r.config,{'messages':history},as_node='model');r.resume();r.events()
                before=len(r.events());model.failures_left=1;model.summary_calls=0
                result=r.submit('앞선 내용을 유지해서 설명해줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(model.summary_calls,2)
                self.assertEqual(r.inspect()['recovery']['model_calls'],3)
                self.assertEqual(len(r.events()),before+2)
                self.assertEqual(r.inspect()['model_recovery']['retries'],1)
            finally:r.close()
