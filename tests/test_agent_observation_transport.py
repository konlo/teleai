"""Production receipt transport and restart recovery; no network or model."""
from core.analysis_agent.policy import RuntimePolicy
from datetime import date, datetime, time, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.analysis_tool_contract import json_tool_value, normalize_tool_result
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.approvals import ApprovalLedger
from scripts.evaluate_analysis_statistics import ForbiddenModel


class ObservationTransportTests(unittest.TestCase):
    def test_typed_values_are_strict_json_without_precision_loss(self):
        result=normalize_tool_result({'status':'ready','preview':[{
            'day':date(2025,1,2),'stamp':datetime(2025,1,2,tzinfo=timezone.utc),
            'time':time(12,30),'money':Decimal('123456789012345678.123456789'),
            'missing':pd.NA,'nat':pd.NaT,'nan':float('nan'),'infinity':np.inf,
            'integer':np.int64(7),'binary':b'\x00\xff'}]})
        decoded=json.loads(json.dumps(result,allow_nan=False))['preview'][0]
        self.assertEqual(decoded['money'],'123456789012345678.123456789')
        self.assertEqual(decoded['stamp'],'2025-01-02T00:00:00+00:00')
        self.assertEqual(decoded['day'],'2025-01-02')
        self.assertEqual(decoded['integer'],7)
        self.assertTrue(all(decoded[k] is None for k in ('missing','nat','nan','infinity')))
        self.assertEqual(decoded['binary'],{'encoding':'base64','value':'AP8='})
        with self.assertRaises(TypeError):json_tool_value(object())

    def test_ledger_first_return_equals_replay_without_second_execution(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=ApprovalLedger(Path(root)/'ledger.db')
            envelope=ledger.envelope('items','SELECT * FROM items LIMIT 2','preview','identity')
            ledger.propose('call',envelope);ledger.decide('call',True)
            calls=[]
            def execute(_):
                calls.append(1)
                return {'status':'ready','preview':[{'value':Decimal('0.100'),'date':date(2025,1,1)}]}
            first=ledger.execute('call',envelope,execute)
            self.assertEqual(first,ledger.execute('call',envelope,execute))
            self.assertEqual(len(calls),1)
            json.dumps(first,allow_nan=False)

    def runtime(self,root,calls):
        def factory(datasets):
            def execute(envelope):
                calls.append(envelope['query'])
                from dataclasses import asdict
                frame=pd.DataFrame({'observed_date':[date(2025,1,2)],'metric':[Decimal('1.250')]})
                info=datasets.register(frame,source=envelope['source'],query=envelope['query'],
                                       coverage='unknown',grain='raw')
                return {'status':'ready','dataset':asdict(info),'preview':frame.to_dict('records')}
            return execute
        return GraphAnalysisRuntime(root,'test','typed',ForbiddenModel(),
                                    connection_identity='test',remote_factory=factory, policy=RuntimePolicy(require_remote_approval=True),intent_mode='contract_fixture')

    def test_typed_load_completes_without_model_and_preserves_raw_types(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,calls)
            try:
                proposal=r.propose_query('items','SELECT * FROM items LIMIT 2','Load preview')
                outcome=r.respond(proposal['requests'][0]['id'],approved=True)
                self.assertEqual(outcome['status'],'answered',outcome)
                raw=r.inspect()['selected_dataset']['id']
                self.assertEqual(r.datasets.frames[raw].iloc[0]['observed_date'],date(2025,1,2))
                self.assertEqual(r.datasets.frames[raw].iloc[0]['metric'],Decimal('1.250'))
                self.assertEqual(len(calls),1)
            finally:r.close()

    def test_legacy_broken_observation_recovers_after_restart_without_reload(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,calls)
            try:
                proposal=r.propose_query('items','SELECT * FROM items LIMIT 2','Load preview')
                # Reproduce the former LangChain repr fallback after the ledger
                # has durably recorded a successful remote execution.
                with patch('core.analysis_agent.runtime.normalize_tool_result',side_effect=str):
                    failed=r.respond(proposal['requests'][0]['id'],approved=True)
                self.assertEqual(failed['status'],'incomplete')
                self.assertEqual(r.ledger.get(proposal['requests'][0]['id'])['status'],'completed')
            finally:r.close()
            r=self.runtime(root,calls)
            try:
                outcome=r.resume()
                self.assertEqual(outcome['status'],'answered',outcome)
                self.assertIsNotNone(r.inspect()['selected_dataset'])
                self.assertEqual(len(calls),1)
                self.assertEqual(len(r.datasets.metadata),1)
            finally:r.close()

    def test_completed_receipt_with_changed_call_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[];r=self.runtime(root,calls)
            try:
                proposal=r.propose_query('items','SELECT * FROM items LIMIT 2','Load preview')
                with patch('core.analysis_agent.runtime.normalize_tool_result',side_effect=str):
                    r.respond(proposal['requests'][0]['id'],approved=True)
                checkpoint=r.agent.get_state(r.config)
                for message in checkpoint.values['messages']:
                    if getattr(message,'tool_calls',[]):
                        message.tool_calls[0]['args']['query']='SELECT * FROM items LIMIT 999'
                self.assertFalse(r._recover_completed_observations(checkpoint))
                self.assertEqual(len(calls),1)
            finally:r.close()


if __name__=='__main__':unittest.main()
