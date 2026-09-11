"""Runtime-independent tool and UI controller contracts."""
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from core.analysis_loop import AnalysisSession
from core.analysis_runtime import CurrentAnalysisRuntime
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore


class RuntimeBoundaryTests(unittest.TestCase):
    def test_tools_run_without_any_session_and_keep_artifact_identity(self):
        fixture = json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
        store = DatasetStore()
        info = store.register(pd.DataFrame(fixture['rows']), source=fixture['source'],
                              coverage='complete', predicate_known=True)
        propose = Mock(return_value={'status': 'awaiting_approval'})
        context = AnalysisToolContext(store, {}, [], propose)
        tools = {t.name: t.run for t in build_analysis_tools(context)}
        result = tools['local_analysis_sql'](info.id, 'SELECT COUNT(*) AS n FROM data')
        self.assertEqual(result['preview'][0]['n'], len(fixture['rows']))
        cards = tools['recommend_chart_images'](info.id)
        self.assertTrue(cards['cards'])
        self.assertTrue(all(context.artifacts[c['id']].dataset_id == info.id for c in cards['cards']))
        propose.assert_not_called()
        tools['propose_databricks_query'](source=fixture['source'], query='SELECT 1', reason='test')
        propose.assert_called_once_with(source=fixture['source'], query='SELECT 1', reason='test')

    def test_local_sql_cannot_silently_treat_sample_as_population(self):
        store=DatasetStore()
        info=store.register(pd.DataFrame({'value':[1,2]}),source='synthetic',coverage='truncated')
        context=AnalysisToolContext(store,{},[],Mock())
        analyze=next(t.run for t in build_analysis_tools(context) if t.name=='local_analysis_sql')
        self.assertEqual(analyze(info.id,'SELECT AVG(value) FROM data')['status'],'needs_data')
        self.assertEqual(analyze(info.id,'SELECT AVG(value) FROM data',current_result_only=True)['status'],'ready')

    def test_sessions_do_not_share_context(self):
        contexts = [AnalysisToolContext(DatasetStore(), {}, [], Mock()) for _ in range(2)]
        contexts[0].reference_context.append({'table': 'first'})
        for index, context in enumerate(contexts):
            catalog = next(t.run for t in build_analysis_tools(context) if t.name=='list_analysis_context')()
            self.assertEqual(len(catalog['available_tables']), 1 if index==0 else 0)

    def test_invalid_proposal_preserves_existing_approval_and_cancel_is_local(self):
        session = AnalysisSession('test', '', [])
        runtime = CurrentAnalysisRuntime(session)
        request_id = runtime.propose_table('catalog.schema.events')
        with self.assertRaises(ValueError):
            runtime.propose_table('bad..table')
        self.assertEqual(runtime.inspect()['requests'][0]['id'], request_id)
        runtime.cancel(request_id)
        self.assertEqual(runtime.inspect()['requests'], [])
        self.assertEqual(runtime.inspect()['state'], 'idle')
        events = runtime.events()
        events[0]['content'] = 'modified outside runtime'
        self.assertNotEqual(session.history[0]['content'], events[0]['content'])

    def test_controller_approval_reaches_executor_once(self):
        session = AnalysisSession('test', '', [])
        runtime = CurrentAnalysisRuntime(session)
        request_id = runtime.propose_table('events')
        execute = Mock(return_value={'rows': 1})
        model = Mock(return_value={'role': 'assistant', 'content': '완료'})
        runtime.respond(request_id, approved=True, execute=execute, model=model)
        with self.assertRaises(ValueError):
            runtime.respond(request_id, approved=True, execute=execute, model=model)
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[0].status, 'executing')
