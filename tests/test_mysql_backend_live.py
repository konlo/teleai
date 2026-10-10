"""Opt-in real-MySQL checks: TELLY_TEST_MYSQL=1 python -m pytest this file."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.getenv('TELLY_TEST_MYSQL') == '1', 'local MySQL evaluation DB required')
class MySQLBackendLiveTests(unittest.TestCase):
    def setUp(self):
        from core.analysis_agent.mysql import MySQLConfig
        self.root = Path(__file__).resolve().parents[1]
        self.config = MySQLConfig.from_env(self.root)

    def test_read_only_execution_and_dynamic_schema(self):
        from core.analysis_agent.assets import AssetDB, PersistentDatasets
        from core.analysis_agent.mysql import make_executor, reference_context
        observed = reference_context(self.config)
        self.assertTrue(observed)
        table = observed[0]['table']
        database, name = table.split('.')
        with tempfile.TemporaryDirectory() as temporary:
            db = AssetDB(temporary, 'mysql-live-test', 'one', max_scope_bytes=64*1024*1024)
            datasets = PersistentDatasets(db, 64*1024*1024,
                max_columns=256, max_frame_bytes=64*1024*1024)
            execute = make_executor(self.config, datasets, max_rows=100)
            def run(source, query):
                return execute({'connection':self.config.identity(),
                                'source':source, 'query':query})
            schema = run(table, f'SELECT * FROM `{database}`.`{name}` LIMIT 0')
            self.assertEqual(schema['status'], 'ready')
            self.assertEqual(schema['dataset']['rows'], 0)
            self.assertEqual(len(schema['dataset']['columns']), len(observed[0]['columns']))
            listing = run('information_schema.tables',
                "SELECT table_schema, table_name, table_type FROM information_schema.tables "
                f"WHERE table_schema = '{database}' ORDER BY table_name LIMIT 100")
            self.assertEqual(listing['dataset']['rows'], len(observed))
            count = run(table, f'SELECT COUNT(*) AS n FROM `{database}`.`{name}`')
            self.assertEqual(count['status'], 'ready')
            self.assertGreaterEqual(int(count['preview'][0]['n']), 0)
            if int(count['preview'][0]['n']) > 2:
                bounded = make_executor(self.config, datasets, max_rows=2)
                result = bounded({'connection':self.config.identity(), 'source':table,
                    'query':f'SELECT * FROM `{database}`.`{name}`'})
                self.assertEqual(result['dataset']['rows'], 2)
                self.assertEqual(result['dataset']['coverage'], 'truncated')
            with self.assertRaises(ValueError):
                run(table, f'DELETE FROM `{database}`.`{name}`')
            with self.assertRaises(ValueError):
                run(table, 'SELECT * FROM mysql.user')
            with self.assertRaises(ValueError):
                run('other.table', f'SELECT * FROM `{database}`.`{name}` LIMIT 1')

    def test_streamlit_agent_lists_live_tables(self):
        from streamlit.testing.v1 import AppTest
        from core.analysis_agent.mysql import reference_context
        expected = [item['table'].rsplit('.', 1)[-1] for item in reference_context(self.config)]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'TELLY_DATA_BACKEND':'mysql', 'TELLY_V1_STORAGE':temporary}):
            app = AppTest.from_file(str(self.root/'ui/analysis_page.py'), default_timeout=45).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value(f'{self.config.database}에는 어떤 테이블이 있지?').run()
            self.assertFalse(app.exception)
            answers = '\n'.join(item.value for item in app.markdown)
            self.assertIn('조회가 완료되었습니다', answers)
            for name in expected:
                self.assertIn(name.replace('_', '\\_'), answers)
            # Exact form of the reported follow-up, with a live discovered table.
            app.chat_input[0].set_value(f'{expected[0]} 컬럼들을 보여줘').run()
            self.assertFalse(app.exception)
            runtime = app.session_state['v1_runtime']
            self.assertEqual(runtime.inspect()['recovery']['status'], 'complete')
            proof=runtime.inspect()['recovery']['metadata_evidence']
            self.assertIn(proof['kind'], {'columns','dtypes'})
            wanted=next(item for item in reference_context(self.config) if item['table'].rsplit('.',1)[-1]==expected[0])
            self.assertEqual(proof['table'],wanted['table'])
            self.assertEqual(proof['columns'],[column['name'] for column in wanted['columns']])
            answers='\n'.join(item.value for item in app.markdown)
            for column in proof['columns']:
                self.assertIn(column,answers.replace('\\_','_'))
            self.assertFalse(runtime.inspect()['recovery']['artifact_ids'])
            self.assertGreaterEqual(runtime.inspect()['recovery']['model_calls'], 1)

    def test_live_mysql_column_definition_tool_executes_and_reuses_metadata(self):
        from core.analysis_agent.assets import AssetDB, PersistentDatasets
        from core.analysis_agent.mysql import make_executor, reference_context
        from core.analysis_tool_contract import AnalysisToolContext
        from core.analysis_metadata_discovery import inspect_column_definitions
        observed = reference_context(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            db = AssetDB(temporary, 'metadata-live', 'one', max_scope_bytes=64*1024*1024)
            datasets = PersistentDatasets(db, 64*1024*1024,
                max_columns=256, max_frame_bytes=64*1024*1024)
            context = AnalysisToolContext(datasets, None, observed, lambda **kw: None,
                                          sql_dialect='mysql')
            planned = inspect_column_definitions(context, observed[0]['table'])
            self.assertEqual(planned['status'], 'planned', planned)
            execute = make_executor(self.config, datasets, max_rows=100)
            execute({'connection':self.config.identity(), **planned['metadata_plan']})
            result = inspect_column_definitions(context, observed[0]['table'])
            self.assertEqual(result['status'], 'ready', result)
            self.assertEqual([c['name'] for c in result['table_context']['columns']],
                             [c['name'] for c in observed[0]['columns']])

    def test_live_mysql_latest_histogram_and_relationship_metadata_use_mysql_sql(self):
        from core.analysis_agent.assets import AssetDB, PersistentDatasets
        from core.analysis_agent.mysql import make_executor, reference_context
        from core.analysis_tool_contract import AnalysisToolContext
        from core.analysis_relationships import inspect_relationships
        from utils.analysis_remote_latest import prepare
        import re
        observed=reference_context(self.config)
        item=next(c for c in observed if sum(bool(re.search(r'int|double|float|decimal',x['dtype']))
                   for x in c['columns'])>=3)
        numeric=[c['name'] for c in item['columns'] if re.search(r'int|double|float|decimal',c['dtype'])]
        with tempfile.TemporaryDirectory() as temporary:
            db=AssetDB(temporary,'mysql-latest','one',max_scope_bytes=64*1024*1024)
            datasets=PersistentDatasets(db,64*1024*1024,max_columns=256,max_frame_bytes=64*1024*1024)
            context=AnalysisToolContext(datasets,{},observed,lambda **kw:None,sql_dialect='mysql')
            execute=make_executor(self.config,datasets,max_rows=100)
            relations=inspect_relationships(context,item['table'])
            self.assertEqual(relations['status'],'planned')
            execute({'connection':self.config.identity(),**relations['metadata_plan']})
            self.assertEqual(inspect_relationships(context,item['table'])['status'],'ready')
            options=dict(categorical=False,bins=5,
                conditions=[{'column':numeric[0],'op':'le','value':10}],filter_stage='before_selection')
            planned=prepare(context,item['table'],[numeric[0]],numeric[1],numeric[2],**options)
            self.assertEqual(planned['status'],'planned',planned)
            query=planned['remote_latest_plan']['query']
            self.assertNotIn('EXPLODE',query)
            self.assertNotIn('AS STRING',query)
            result=execute({'connection':self.config.identity(),**{
                k:planned['remote_latest_plan'][k] for k in ('source','query','reason')}})
            rendered=prepare(context,item['table'],[numeric[0]],numeric[1],numeric[2],
                             result['dataset']['id'],**options)
            # Dynamically observed tables can have no qualifying rows, NULLs
            # or latest-order ties. Those are legitimate policy outcomes.
            if rendered['status']=='needs_context':
                self.assertIn(rendered.get('error_code'),{'latest_empty','latest_null_policy','latest_order_tie'})
                self.assertFalse(context.artifacts)
                return
            self.assertEqual(rendered['status'],'ready',rendered)
            self.assertTrue(context.artifacts[rendered['cards'][0]['id']].image.startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
