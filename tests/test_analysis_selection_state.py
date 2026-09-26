"""Explicit active dataset selection survives restart and bounds chart reuse."""
import tempfile
import unittest
from dataclasses import asdict

import pandas as pd

from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_runtime_tools import build_analysis_tools
from migration.test_persistent_runtime import QuietModel
from migration.test_approval_rollout import NoUnexpectedModelCall


class SelectionStateTests(unittest.TestCase):
    def test_approved_aggregate_preserves_selected_raw_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            def factory(datasets):
                def execute(envelope):
                    info = datasets.register(pd.DataFrame({'n':[3]}),
                        source=envelope['source'], query=envelope['query'],
                        grain='aggregate', aggregation=envelope['query'],
                        coverage='complete', predicate_known=False)
                    return {'status':'ready', 'dataset':asdict(info), 'preview':[{'n':3}]}
                return execute
            runtime = GraphAnalysisRuntime(directory, 'owner', 'aggregate-selection',
                NoUnexpectedModelCall(), connection_identity='test', remote_factory=factory)
            try:
                raw = runtime.datasets.register(pd.DataFrame({'measure':[1, 2, 3]}),
                    source='synthetic.events', coverage='complete', predicate_known=True)
                runtime.select_dataset(raw.id)
                pending = runtime.propose_query('synthetic.events',
                    'SELECT COUNT(*) AS n FROM synthetic.events', '행 수 확인')['requests'][0]
                outcome = runtime.respond(pending['id'], approved=True)
                self.assertEqual(outcome['status'], 'answered', outcome)
                self.assertEqual(runtime.inspect()['selected_dataset']['id'], raw.id)
                self.assertEqual(len(runtime.datasets.metadata), 2)
            finally:
                runtime.close()

    def test_selection_persists_and_source_only_chart_uses_selected_root(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = GraphAnalysisRuntime(directory, 'owner', 'selection', QuietModel())
            first = runtime.datasets.register(pd.DataFrame({'measure':[1, 2, 2]}),
                source='synthetic.events', snapshot='v1',
                coverage='complete', predicate_known=True)
            second = runtime.datasets.register(pd.DataFrame({'measure':[10, 20, 20]}),
                source='synthetic.events', snapshot='v2',
                coverage='complete', predicate_known=True)
            self.assertEqual(first.role, 'root')
            self.assertEqual(runtime.select_dataset(first.id)['root_id'], first.id)
            with self.assertRaises(KeyError):
                runtime.select_dataset('not-loaded')
            self.assertEqual(runtime.inspect()['selected_dataset']['id'], first.id)
            runtime.close()

            restored = GraphAnalysisRuntime(directory, 'owner', 'selection', QuietModel())
            try:
                self.assertEqual(restored.inspect()['selected_dataset']['id'], first.id)
                tools = {tool.name: tool.run for tool in build_analysis_tools(restored.context)}
                first_chart = tools['prepare_histogram']('synthetic.events', 'measure')
                self.assertEqual(first_chart['status'], 'ready')
                self.assertEqual(set(restored.datasets.frames[first_chart['loaded_dataset']]['measure']),
                                 {1, 2})
                restored.select_dataset(second.id)
                second_chart = tools['prepare_histogram']('synthetic.events', 'measure')
                self.assertEqual(second_chart['status'], 'ready')
                self.assertEqual(set(restored.datasets.frames[second_chart['loaded_dataset']]['measure']),
                                 {10, 20})
                self.assertNotEqual(first_chart['cards'][0]['id'], second_chart['cards'][0]['id'])
            finally:
                restored.close()


if __name__ == '__main__':
    unittest.main()
