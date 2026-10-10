"""Repeat planning must not repeat a validated, unchanged subject reading."""
from copy import deepcopy
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from core.analysis_agent.assets import AssetDB
from core.analysis_agent.diagnostics import Diagnostics
from core.analysis_agent.model_recovery import ModelAttemptLedger, ModelRecoveryMiddleware
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.subject_identity import read
from core.analysis_agent.support_report import summarize, brief


class SubjectDecisionReuseTests(unittest.TestCase):
    def setUp(self):
        root=tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        self.db=AssetDB(root.name,'owner','subjects');self.addCleanup(self.db.close)
        self.ledger=ModelAttemptLedger(self.db)
        self.diagnostics=Diagnostics(self.db.directory)
        self.diagnostics.run_id='a'*32;self.diagnostics.emit('run_started')
        self.model=Mock()
        self.model.invoke.return_value=SimpleNamespace(content=json.dumps({'candidate_roles':{
            'new_source':'requested_table','rows':'other'}}))
        self.interpreter=SimpleNamespace(selection_model=self.model,
            model_recovery=ModelRecoveryMiddleware(self.ledger,self.diagnostics,RuntimePolicy()),
            diagnostics=self.diagnostics,budget=SimpleNamespace(wrap_model_call=lambda req,fn:fn(req)))
        self.current={'request_id':'request','request_text':'new_source rows 보여줘'}
        self.data={'namespace':'namespace','tables':[{'table':'namespace.new_source',
            'columns':[{'name':'measurement','dtype':'int'}]}]}
        self.patch=patch('core.analysis_agent.model_roles.json_role',return_value=None)
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_goal_repair_feedback_reuses_verified_subject_without_new_inference(self):
        first=read(self.interpreter,self.current,self.data)
        self.data.update(failed_selection={'capabilities':['metadata']},
                         failed_goal_validation='Output obligation needs rows')
        second=read(self.interpreter,self.current,self.data)
        self.assertEqual(first,second)
        self.model.invoke.assert_called_once()
        self.assertEqual(self.ledger.get('request')['aux_calls'],1)
        report=summarize(self.diagnostics.path)
        self.assertEqual(report['inference_roles'],{'subject_identity':1})
        self.assertEqual(report['subject_decisions_reused'],1)
        self.assertIn('subject_identity',brief(report))

    def test_cached_result_and_scope_do_not_alias_mutable_caller_data(self):
        self.current['request_text']='fresh_schema table list'
        self.model.invoke.return_value=SimpleNamespace(content=json.dumps({'candidate_roles':{
            'fresh_schema':'requested_schema','table':'other','list':'other'}}))
        result=read(self.interpreter,self.current,self.data)
        self.assertEqual(result,[])
        self.data['literal_inventory_scope']['schema']='WRONG'
        second=read(self.interpreter,self.current,self.data)
        second.append({'name':'WRONG','quote':'WRONG'})
        third=read(self.interpreter,self.current,self.data)
        self.assertEqual(third,[])
        self.assertEqual(self.data['literal_inventory_scope']['schema'],'fresh_schema')
        self.model.invoke.assert_called_once()

    def test_new_request_or_changed_subject_evidence_requires_another_model_read(self):
        changes=[lambda current,data:current.update(request_id='new'),
                 lambda current,data:current.update(request_text='new_source rows 다시 보여줘'),
                 lambda current,data:data.update(namespace='different'),
                 lambda current,data:data.update(selected_dataset={'source':'different.table'}),
                 lambda current,data:data['tables'][0]['columns'].append({'name':'new_field'}),
                 lambda current,data:data.update(literal_source_mentions=[{
                     'source':'namespace.new_source','quote':'new_source'}])]
        for change in changes:
            with self.subTest(change=change):
                self.interpreter._subject_identity_cache=None
                current=deepcopy(self.current);data=deepcopy(self.data)
                before=self.model.invoke.call_count
                read(self.interpreter,current,data)
                change(current,data)
                read(self.interpreter,current,data)
                self.assertEqual(self.model.invoke.call_count-before,2)
                # Each subtest has an independent inference budget.
                self.current['request_id']+='x'

    def test_invalid_decisions_are_not_reused(self):
        self.model.invoke.return_value=SimpleNamespace(content='{}')
        with self.assertRaises(ValueError):read(self.interpreter,self.current,self.data)
        self.assertIsNone(getattr(self.interpreter,'_subject_identity_cache',None))
        self.model.invoke.return_value=SimpleNamespace(content=json.dumps({'candidate_roles':{
            'new_source':'requested_table','rows':'other'}}))
        self.assertEqual(read(self.interpreter,self.current,self.data),[
            {'name':'new_source','quote':'new_source'}])
        self.assertEqual(self.model.invoke.call_count,3)

    def test_provider_failure_never_creates_a_reusable_decision(self):
        self.model.invoke.side_effect=RuntimeError('provider failed')
        with self.assertRaises(RuntimeError):read(self.interpreter,self.current,self.data)
        self.assertIsNone(getattr(self.interpreter,'_subject_identity_cache',None))
        self.model.invoke.side_effect=None
        read(self.interpreter,self.current,self.data)
        self.assertEqual(self.model.invoke.call_count,2)

    def test_new_model_instance_never_reuses_the_previous_models_decision(self):
        read(self.interpreter,self.current,self.data)
        other=Mock();other.invoke.return_value=self.model.invoke.return_value
        self.interpreter.selection_model=other
        read(self.interpreter,self.current,self.data)
        other.invoke.assert_called_once()

    def test_cached_read_does_not_reset_or_increase_exhausted_request_budget(self):
        first=read(self.interpreter,self.current,self.data)
        for _ in range(8):self.ledger.auxiliary_success('request',0.)
        self.assertEqual(read(self.interpreter,self.current,self.data),first)
        self.assertEqual(self.ledger.get('request')['aux_calls'],9)
        from core.analysis_agent.model_recovery import ModelAttemptBudgetExceeded
        with self.assertRaises(ModelAttemptBudgetExceeded):
            self.interpreter.model_recovery.auxiliary_call(self.current,Mock())
        self.model.invoke.assert_called_once()
