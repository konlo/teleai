"""Production backend selection, dependency and durable-state boundaries."""
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_agent.backends import load_data_backend
from core.analysis_agent.databricks import ConnectionConfig
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_mysql_metadata_contract import NoInference


CONNECTION = {'TELLY_DATA_BACKEND':'databricks', 'DATABRICKS_HOST':'example.invalid',
              'DATABRICKS_HTTP_PATH':'/sql/fixture', 'DATABRICKS_TOKEN':'fixture-secret',
              'DATABRICKS_CATALOG':'eval_catalog', 'DATABRICKS_SCHEMA':'lab'}
TABLE = 'eval_catalog.lab.observations'


class BackendIsolationTests(unittest.TestCase):
    def test_fresh_install_discovers_names_then_probes_schema_without_cached_profiles(self):
        executions=[]
        def factory(datasets):
            def execute(envelope):
                executions.append(envelope['query'])
                if 'information_schema.tables' in envelope['source']:
                    frame=pd.DataFrame([dict(table_catalog='eval_catalog',table_schema='lab',
                        table_name='observations',table_type='MANAGED')])
                else:
                    self.assertEqual(envelope['source'],TABLE)
                    self.assertEqual(envelope['query'],'SELECT * FROM `eval_catalog`.`lab`.`observations` LIMIT 0')
                    frame=pd.DataFrame({'reading':pd.Series([],dtype='float64')})
                info=datasets.register(frame,source=envelope['source'],query=envelope['query'],
                    snapshot=datetime.now(timezone.utc).isoformat(),coverage='unknown',predicate_known=True)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','fresh-install',NoInference(),sql_dialect='databricks',
                source_namespace='eval_catalog',reference_context_loader=lambda:[],
                remote_factory=factory,connection_identity='synthetic',intent_mode='contract_fixture')
            try:
                listing=runtime.submit('테이블 목록을 보여줘')
                self.assertEqual(listing['status'],'answered',listing)
                self.assertIn('observations',listing['text'])
                # Inventory proves the name only; the next request requires real schema.
                runtime.inspect()
                from core.analysis_catalog import resolve_table_context
                self.assertEqual(resolve_table_context(runtime.context.reference_context,
                    runtime.datasets,TABLE)['status'],'needs_refresh')
                columns=runtime.submit('observations 컬럼들을 보여줘')
                self.assertEqual(columns['status'],'answered',columns)
                self.assertIn('reading',columns['text'])
                self.assertEqual(len(executions),2)
            finally:runtime.close()

    def test_inventory_cannot_invent_names_or_reuse_stale_metadata(self):
        from core.analysis_catalog import discovered_reference_context
        query='SELECT table_catalog, table_schema, table_name FROM eval_catalog.information_schema.tables LIMIT 100'
        for variant in ['valid','computed_name','stale','wrong_catalog','wrong_source','unbounded']:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                runtime=GraphAnalysisRuntime(root,'owner','inventory',NoInference(),intent_mode='contract_fixture')
                try:
                    actual=query
                    if variant=='computed_name':actual=query.replace('table_name',"'observations' AS table_name")
                    if variant=='wrong_source':actual=query.replace('information_schema.tables','lab.another')
                    if variant=='unbounded':actual=query.replace(' LIMIT 100','')
                    runtime.datasets.register(pd.DataFrame([dict(table_catalog='another' if variant=='wrong_catalog'
                        else 'eval_catalog',table_schema='lab',table_name='observations')]),
                        source='eval_catalog.information_schema.tables',query=actual,
                        snapshot='2000-01-01T00:00:00Z' if variant=='stale' else datetime.now(timezone.utc).isoformat())
                    result=discovered_reference_context(runtime.datasets)
                    self.assertEqual(len(result),1 if variant=='valid' else 0)
                    if result:self.assertEqual(result[0]['columns'],[])
                finally:runtime.close()

    def test_controller_sql_is_not_replanned_from_a_table_list_reason(self):
        source='eval_catalog.information_schema.tables'
        query=("SELECT table_name, table_type FROM eval_catalog.information_schema.tables "
               "WHERE table_schema = 'lab' ORDER BY table_name LIMIT 10")
        executions=[]
        def factory(datasets):
            def execute(envelope):
                executions.append(envelope['query'])
                self.assertEqual(envelope['query'],query)
                info=datasets.register(pd.DataFrame([{'table_name':'observations','table_type':'MANAGED'}]),
                    source=source,query=query,coverage='unknown',predicate_known=True)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','exact-sql',NoInference(),sql_dialect='databricks',
                remote_factory=factory,connection_identity='synthetic',
                reference_context_loader=lambda:[{'table':TABLE,'columns':[{'name':'reading','dtype':'double'}]}],intent_mode='contract_fixture')
            try:
                result=runtime.propose_query(source,query,'설정한 namespace의 실제 테이블 목록을 확인합니다.')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(executions,[query])
                self.assertEqual(runtime.inspect()['recovery']['expected_load_query'],query)
            finally:runtime.close()

    def test_databricks_selection_does_not_import_mysql_even_when_unavailable(self):
        code = '''
import importlib.abc,sys
from pathlib import Path
class BlockMySQL(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='mysql' or fullname.startswith('mysql.') or fullname=='core.analysis_agent.mysql':
   raise AssertionError('Databricks imported MySQL')
sys.meta_path.insert(0,BlockMySQL())
from core.analysis_agent.backends import load_data_backend
backend=load_data_backend(Path.cwd())
from core.analysis_agent.runtime import GraphAnalysisRuntime
assert backend.dialect=='databricks'
assert backend.executor_factory.__module__=='core.analysis_agent.databricks'
assert not any(k=='mysql' or k.startswith('mysql.') for k in sys.modules)
assert 'fixture-secret' not in repr(backend)
print('PASS')
'''
        environment = {**os.environ, **CONNECTION, 'TELLY_MYSQL_OPTION_FILE':'/does/not/exist'}
        result = subprocess.run([sys.executable, '-c', code], env=environment,
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_invalid_or_missing_selected_config_never_falls_back(self):
        with patch.dict(os.environ, CONNECTION):
            with self.assertRaises(ValueError):
                load_data_backend('.', name='typo')
            with patch.dict(os.environ, {'DATABRICKS_TOKEN':'', 'DATABRICKS_ACCESS_TOKEN':''}):
                with self.assertRaises(ValueError):
                    load_data_backend('.', name='databricks')
            with patch.dict(os.environ, {'DATABRICKS_TOKEN':'', 'DATABRICKS_ACCESS_TOKEN':'alias-token'}):
                self.assertEqual(ConnectionConfig.from_env().access_token, 'alias-token')

    def test_mysql_selection_does_not_import_databricks_adapter(self):
        if importlib.util.find_spec('mysql') is None or importlib.util.find_spec('mysql.connector') is None:
            self.skipTest('optional MySQL evaluation driver is not installed')
        code='''
import importlib.abc,sys
from pathlib import Path
class BlockDatabricks(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='core.analysis_agent.databricks' or fullname=='databricks' or fullname.startswith('databricks.'):
   raise AssertionError('MySQL imported Databricks adapter')
sys.meta_path.insert(0,BlockDatabricks())
from core.analysis_agent.backends import load_data_backend
backend=load_data_backend(Path.cwd())
from core.analysis_agent.runtime import GraphAnalysisRuntime
assert backend.dialect=='mysql'
assert backend.executor_factory.__module__=='core.analysis_agent.mysql'
print('PASS')
'''
        with tempfile.TemporaryDirectory() as temporary:
            credentials=Path(temporary)/'reader.cnf'
            credentials.write_text('[client]\nuser=fixture\npassword=fixture\nsocket=/missing\n')
            credentials.chmod(0o600)
            environment={**os.environ,'TELLY_DATA_BACKEND':'mysql','TELLY_MYSQL_OPTION_FILE':str(credentials),
                'TELLY_MYSQL_DATABASE':'lab','DATABRICKS_TOKEN':''}
            result=subprocess.run([sys.executable,'-c',code],env=environment,
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)


    def test_same_checkpoint_cannot_be_reopened_with_another_engine(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'one', NoInference(), sql_dialect='databricks',intent_mode='contract_fixture')
            original = runtime.datasets.register(pd.DataFrame({'reading':[1, 2]}), source=TABLE,
                coverage='complete', predicate_known=True)
            runtime.close()
            with self.assertRaisesRegex(ValueError, '다른 데이터 backend'):
                GraphAnalysisRuntime(root, 'owner', 'one', NoInference(), sql_dialect='mysql',intent_mode='contract_fixture')
            reopened = GraphAnalysisRuntime(root, 'owner', 'one', NoInference(), sql_dialect='databricks',intent_mode='contract_fixture')
            try:
                self.assertIn(original.id, reopened.datasets.metadata)
            finally:
                reopened.close()

    def test_databricks_page_lists_tables_then_columns_without_mysql_import(self):
        from streamlit.testing.v1 import AppTest
        reference = [{'table':TABLE, 'observed_at':datetime.now(timezone.utc).isoformat(),
                      'columns':[{'name':'reading','dtype':'double'}]}]
        executions = []
        def factory(config, datasets, **kwargs):
            def execute(envelope):
                executions.append(envelope['query'])
                self.assertIn('`eval_catalog`.information_schema.tables', envelope['query'])
                self.assertNotIn('teleai_default', envelope['query'])
                frame = pd.DataFrame([dict(table_catalog='eval_catalog',table_schema='lab',
                    table_name='observations',table_type='MANAGED')])
                info = datasets.register(frame, source=envelope['source'],query=envelope['query'],
                    snapshot=datetime.now(timezone.utc).isoformat(),coverage='unknown',predicate_known=True)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {
                **CONNECTION, 'TELLY_V1_STORAGE':root,'TELLY_MYSQL_OPTION_FILE':'/missing'}), \
                patch('core.analysis_agent.model_provider.build_analysis_chat_model',return_value=NoInference()), \
                patch('core.analysis_agent.databricks.make_executor',side_effect=factory), \
                patch('core.analysis_catalog.load_saved_reference_context',return_value=reference):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'main.py'),default_timeout=20).run()
            app.session_state['v1_runtime'].recovery.intent_mode = 'contract_fixture'  # Render/receipt fixture only.
            self.assertFalse(app.exception)
            app.chat_input[0].set_value('eval_catalog lab 테이블 목록을 보여줘').run()
            self.assertFalse(app.exception)
            self.assertIn('observations', '\n'.join(m.value for m in app.markdown))
            app.chat_input[0].set_value('observations 컬럼들을 보여줘').run()
            self.assertFalse(app.exception)
            self.assertIn('reading', '\n'.join(m.value for m in app.markdown))
            runtime=app.session_state['v1_runtime']
            runtime.recovery.intent_mode = 'contract_fixture'  # UI execution/rendering fixture, not an intent score.
            self.assertEqual(runtime.context.sql_dialect,'databricks')
            self.assertEqual(len(executions),1)
            self.assertEqual(runtime.inspect()['recovery']['status'],'complete')
            self.assertNotIn('mysql_eval',str(runtime.db.directory))
            runtime.close()
