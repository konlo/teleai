"""Default automatic read policy, durable execution and old checkpoint migration."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.approvals import ApprovalLedger
from tests.test_remote_result_completion import DeferredReplyModel, FIXTURE


class AutomaticReadTests(unittest.TestCase):
    def runtime(self, root, *, manual=False, error=None, rows=None, model=None):
        executions=[]
        def factory(datasets):
            def execute(envelope):
                executions.append(envelope['query'])
                if error:raise error
                frame=pd.DataFrame(FIXTURE['rows'] if rows is None else rows,
                                   columns=['table_name','table_type'])
                info=datasets.register(frame,source=envelope['source'],query=envelope['query'],
                    coverage='complete',predicate_known=True,snapshot='auto-fixture')
                return {'status':'ready','dataset':asdict(info)}
            return execute
        runtime=GraphAnalysisRuntime(root,'auto','conversation',model or DeferredReplyModel(),
            connection_identity='fixture',remote_factory=factory,
            policy=RuntimePolicy(require_remote_approval=manual))
        self.addCleanup(runtime.close)
        return runtime,executions

    def test_policy_defaults_and_explicit_manual(self):
        self.assertFalse(RuntimePolicy().require_remote_approval)
        with patch.dict(os.environ,{'TELLY_REQUIRE_REMOTE_APPROVAL':'false'}):
            self.assertFalse(RuntimePolicy.from_env().require_remote_approval)
        with patch.dict(os.environ,{'TELLY_REQUIRE_REMOTE_APPROVAL':'true'}):
            self.assertTrue(RuntimePolicy.from_env().require_remote_approval)
        with patch.dict(os.environ,{'TELLY_REQUIRE_REMOTE_APPROVAL':'typo'}):
            with self.assertRaises(ValueError):RuntimePolicy.from_env()

    def test_model_query_finishes_without_interrupt_or_fictional_user_approval(self):
        with tempfile.TemporaryDirectory() as root:
            r,calls=self.runtime(root)
            result=r.submit(FIXTURE['prompt'])
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual(calls,[FIXTURE['query']])
            self.assertFalse(r.inspect()['requests'])
            self.assertIn('sample_events',result['text'].replace('\\_', '_'))
            self.assertNotIn('승인',result['text'])
            self.assertFalse(any('승인했습니다' in str(m.content) for m in r.events()))
            with r.ledger.connect() as db:
                key=db.execute('SELECT id FROM requests').fetchone()[0]
            receipt=r.ledger.get(key)
            envelope={k:receipt[k] for k in ('source','query','reason','connection')}
            self.assertEqual(r.ledger.execute(key,envelope,lambda _:self.fail('duplicate SQL')),receipt['result'])

    def test_controller_query_auto_and_empty_results(self):
        with tempfile.TemporaryDirectory() as root:
            r,calls=self.runtime(root,rows=[])
            result=r.propose_query(FIXTURE['source'],FIXTURE['query'],FIXTURE['reason'])
            self.assertEqual(result['status'],'answered',result)
            self.assertIn('0행',result['text'])
            self.assertEqual(len(calls),1)

    def test_unknown_submission_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            r,calls=self.runtime(root,error=TimeoutError('synthetic'))
            result=r.submit(FIXTURE['prompt'])
            self.assertEqual(result['status'],'blocked',result)
            self.assertEqual(len(calls),1)
            self.assertTrue(r.ledger.uncertain())
            with self.assertRaises(PermissionError):r.resume()
            self.assertEqual(len(calls),1)

    def test_manual_checkpoint_resumes_under_auto_policy_once(self):
        with tempfile.TemporaryDirectory() as root:
            old,calls=self.runtime(root,manual=True)
            proposal=old.submit(FIXTURE['prompt'])
            self.assertEqual(proposal['status'],'awaiting_approval')
            self.assertFalse(calls)
            old.close()
            r,calls=self.runtime(root)
            result=r.resume()
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual(len(calls),1)
            self.assertFalse(r.inspect()['requests'])
            r.close()
            again,calls=self.runtime(root)
            self.assertFalse(calls)
            self.assertEqual(again.inspect()['state'],'idle')

    def test_tool_level_controller_checkpoint_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            old,calls=self.runtime(root,manual=True)
            old.propose_query(FIXTURE['source'],FIXTURE['query'],FIXTURE['reason'])
            self.assertFalse(calls)
            old.close()
            r,calls=self.runtime(root)
            result=r.resume()
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual(len(calls),1)

    def test_cancelled_legacy_query_never_auto_authorized(self):
        with tempfile.TemporaryDirectory() as root:
            old,_=self.runtime(root,manual=True)
            proposal=old.submit(FIXTURE['prompt'])
            old.ledger.decide(proposal['requests'][0]['id'],False)
            old.close()
            r,calls=self.runtime(root)
            result=r.resume()
            self.assertEqual(result['status'],'blocked',result)
            self.assertFalse(calls)

    def test_connection_change_invalidates_legacy_query(self):
        with tempfile.TemporaryDirectory() as root:
            old,_=self.runtime(root,manual=True)
            old.submit(FIXTURE['prompt']);old.close()
            r,calls=self.runtime(root)
            r.connection_identity='other'
            result=r.resume()
            self.assertNotEqual(result['status'],'answered',result)
            self.assertFalse(calls)

    def test_writes_are_rejected_before_executor(self):
        with tempfile.TemporaryDirectory() as root:
            r,calls=self.runtime(root)
            for query in ['DROP TABLE '+FIXTURE['source'],'DELETE FROM '+FIXTURE['source'],
                          FIXTURE['query']+'; DELETE FROM '+FIXTURE['source']]:
                with self.assertRaises((ValueError,PermissionError)):
                    r.propose_query(FIXTURE['source'],query,'invalid')
            self.assertFalse(calls)

    def test_automatic_grant_does_not_revive_terminal_states(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=ApprovalLedger(Path(root)/'ledger.sqlite')
            envelope=ledger.envelope(FIXTURE['source'],FIXTURE['query'],'test','fixture')
            for status in ['rejected','invalidated','unknown','submitting','failed']:
                ledger.propose(status,envelope)
                with ledger.connect() as db:db.execute('UPDATE requests SET status=? WHERE id=?',(status,status))
                self.assertFalse(ledger.authorize_automatic(status,envelope))
                with self.assertRaises(PermissionError):ledger.execute(status,envelope,lambda _:self.fail(status))

    def test_auto_histogram_then_reuse_preserves_loaded_data(self):
        from migration.test_recovery_journey import SOURCE, COLUMN, QUERY, FIXTURE as HISTOGRAM, JourneyModel
        calls=[]
        def factory(datasets):
            def execute(envelope):
                calls.append(envelope['query'])
                frame=pd.DataFrame(HISTOGRAM['rows']).groupby(COLUMN).size().reset_index(name='frequency')
                info=datasets.register(frame,source=SOURCE,query=QUERY,coverage='complete',
                    grain='aggregate',aggregation=QUERY)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'auto','histogram',JourneyModel(),connection_identity='fixture',remote_factory=factory)
            self.addCleanup(r.close)
            first=r.submit(f'{COLUMN} histogram을 보여줘')
            self.assertEqual(first['status'],'answered',first)
            self.assertEqual(len(calls),1)
            artifacts=r.inspect()['recovery']['artifact_ids']
            self.assertTrue(r.artifacts[artifacts[0]].image.startswith(b'\x89PNG'))
            snapshots={key:info.snapshot for key,info in r.datasets.metadata.items()}
            followup=r.submit(f'{COLUMN} histogram을 다시 보여줘')
            self.assertEqual(followup['status'],'answered',followup)
            self.assertEqual(len(calls),1)
            self.assertTrue(all(r.datasets.metadata[key].snapshot==value for key,value in snapshots.items()))

    def test_zero_row_schema_to_histogram_uses_one_aggregate_without_model(self):
        from scripts.evaluate_analysis_statistics import ForbiddenModel
        source='fixture_2026.custom_records'
        column='reading'
        calls=[]
        def factory(datasets):
            def execute(envelope):
                query=envelope['query']
                calls.append(query)
                if 'LIMIT 0' in query.upper():
                    frame=pd.DataFrame({column:pd.Series(dtype='int64'),
                                        'category':pd.Series(dtype='object')})
                    info=datasets.register(frame,source=source,query=query,
                        coverage='complete',predicate_known=True)
                else:
                    self.assertIn('COUNT(*)',query)
                    self.assertIn('GROUP BY',query)
                    self.assertNotIn('SELECT *',query.upper())
                    frame=pd.DataFrame({column:[10,20,30],'__frequency':[2,3,1]})
                    info=datasets.register(frame,source=source,query=query,
                        coverage='complete',predicate_known=True,
                        grain='aggregate',aggregation=query)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'auto','schema-histogram',ForbiddenModel(),
                connection_identity='fixture',remote_factory=factory)
            self.addCleanup(runtime.close)
            schema=runtime.propose_query(source,f'SELECT * FROM {source} LIMIT 0',
                '현재 컬럼 확인')
            self.assertEqual(schema['status'],'answered',schema)
            schema_ids=set(runtime.datasets.metadata)
            unbound={'chart':True,'kind':'histogram','required_sources':[],
                     'required_columns':[column],'scope':{}}
            inferred=runtime.recovery._next_local(unbound,{})
            self.assertEqual(inferred['name'],'prepare_histogram')
            self.assertEqual(unbound['required_sources'],[source])
            result=runtime.submit(f'그러면 {column}의 히스토그램을 그려줘')
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual(len(calls),2)
            self.assertEqual(runtime.inspect()['recovery']['model_calls'],0)
            self.assertTrue(schema_ids.issubset(runtime.datasets.metadata))
            self.assertEqual(runtime.datasets.metadata[next(iter(schema_ids))].rows,0)
            card=runtime.artifacts[runtime.inspect()['chart_ids'][0]]
            self.assertEqual(card.kind,'histogram')
            self.assertTrue(card.image.startswith(b'\x89PNG\r\n\x1a\n'))
            again=runtime.submit(f'{column} 히스토그램을 다시 보여줘')
            self.assertEqual(again['status'],'answered',again)
            self.assertEqual(len(calls),2)

    def test_timed_out_histogram_checkpoint_resumes_from_verified_schema(self):
        from scripts.evaluate_analysis_statistics import ForbiddenModel
        source='fixture_2026.sensor_readings'
        calls=[]
        def factory(datasets):
            def execute(envelope):
                query=envelope['query']
                calls.append(query)
                frame=(pd.DataFrame({'reading':pd.Series(dtype='int64')})
                       if 'LIMIT 0' in query.upper() else
                       pd.DataFrame({'reading':[1,2],'__frequency':[3,4]}))
                aggregate={} if 'LIMIT 0' in query.upper() else {
                    'grain':'aggregate','aggregation':query}
                info=datasets.register(frame,source=source,query=query,
                    coverage='complete',predicate_known=True,**aggregate)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'auto','resume-histogram',ForbiddenModel(),
                connection_identity='fixture',remote_factory=factory)
            self.addCleanup(runtime.close)
            runtime.propose_query(source,f'SELECT * FROM {source} LIMIT 0',
                '현재 컬럼 확인')
            with patch.object(runtime.recovery,'_next_local',return_value=None):
                failed=runtime.submit('그러면 reading의 히스토그램을 그려줘')
            self.assertEqual(failed['status'],'incomplete')
            self.assertEqual(runtime.agent.get_state(runtime.config).next,('model',))
            checkpoint=runtime.agent.get_state(runtime.config)
            recovery=dict(checkpoint.values['recovery'])
            recovery['model_seconds']=runtime.policy.turn_slo_seconds+1
            runtime.agent.update_state(runtime.config,{'recovery':recovery})
            result=runtime.resume()
            self.assertEqual(result['status'],'answered',result)
            self.assertEqual(len(calls),2)
            self.assertEqual(len(runtime.inspect()['chart_ids']),1)
            self.assertTrue(runtime.artifacts[runtime.inspect()['chart_ids'][0]].image.startswith(b'\x89PNG'))

    def test_page_executes_without_approval_button_and_rerun_does_not_reload(self):
        self._assert_page_execution()

    def test_page_automatically_resumes_legacy_pending_approval(self):
        self._assert_page_execution(initial_manual=True)

    def _assert_page_execution(self, initial_manual=False):
        from streamlit.testing.v1 import AppTest
        executions=[]
        def factory(config,datasets,**kwargs):
            def execute(envelope):
                executions.append(envelope['query'])
                info=datasets.register(pd.DataFrame(FIXTURE['rows']),source=envelope['source'],
                    query=envelope['query'],coverage='complete',predicate_known=True,snapshot='ui')
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{
                'TELLY_V1_STORAGE':root,'TELLY_REQUIRE_REMOTE_APPROVAL':'true' if initial_manual else 'false'}), \
                patch('core.analysis_agent.model_provider.build_analysis_chat_model',return_value=DeferredReplyModel()), \
                patch('core.analysis_agent.databricks.make_executor',side_effect=factory):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=20).run()
            try:
                app.chat_input[0].set_value(FIXTURE['prompt']).run()
                self.assertFalse(app.exception)
                if initial_manual:
                    self.assertEqual(executions,[])
                    os.environ['TELLY_REQUIRE_REMOTE_APPROVAL']='false'
                    app.run()
                    self.assertFalse(app.exception)
                self.assertFalse(any(str(b.key).startswith('yes-') for b in app.button))
                self.assertEqual(len(executions),1)
                rendered='\n'.join(m.value for m in app.markdown).replace('\\_','_')
                self.assertIn('sample_events',rendered)
                app.run()
                self.assertEqual(len(executions),1)
            finally:app.session_state['v1_runtime'].close()


if __name__=='__main__':unittest.main()
