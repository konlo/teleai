"""Independent oracle for latest-per-key semantics, lineage, and delivery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from migration.test_persistent_runtime import QuietModel
from utils.analysis_datasets import DatasetStore, stored_dataset_digest
from utils.analysis_latest import latest_distribution
from ui.analysis_chart_delivery import chart_references

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/latest_per_key.json').read_text())
KEY, VALUE, CLOCK, INGEST = FIXTURE['columns']


def source_frame():
    frame = pd.DataFrame(FIXTURE['rows'], columns=FIXTURE['columns'])
    for column in (CLOCK, INGEST):
        frame[column] = pd.to_datetime(frame[column])
    return frame


class LatestDistributionTests(unittest.TestCase):
    def context(self, frame=None, **metadata):
        store = DatasetStore()
        raw = store.register(source_frame() if frame is None else frame,
            source=FIXTURE['source'], coverage='complete', predicate_known=True, **metadata)
        return AnalysisToolContext(store, {}, [], lambda **kwargs: None), raw

    def run_latest(self, context, raw, **kwargs):
        return latest_distribution(context, raw.id, [KEY], CLOCK, VALUE, **kwargs)

    def test_latest_is_not_physical_last_or_group_by_product_max(self):
        for seed in (1, 7, 29):
            with self.subTest(seed=seed):
                context, raw = self.context(source_frame().sample(frac=1, random_state=seed))
                before = stored_dataset_digest(context.datasets, raw.id)
                result = self.run_latest(context, raw)
                self.assertEqual(result['status'], 'ready')
                selected = context.datasets.frames[result['dataset']['id']]
                self.assertEqual(dict(zip(selected[KEY], selected[VALUE])), FIXTURE['expected_latest'])
                counts = {r[VALUE]: r[result['count_column']] for r in result['counts']}
                self.assertEqual(counts, FIXTURE['expected_counts'])
                self.assertEqual(sum(counts.values()), selected[KEY].nunique())
                self.assertEqual(stored_dataset_digest(context.datasets, raw.id), before)

    def test_selection_scope_survives_downstream_sql_without_flag(self):
        context, raw = self.context()
        result = self.run_latest(context, raw)
        tools = {tool.name: tool.run for tool in build_analysis_tools(context)}
        for identity, query, expected in (
            (result['dataset']['id'], 'SELECT COUNT(*) AS n FROM data', 5),
            (result['distribution']['id'], f'SELECT SUM({result["count_column"]}) AS n FROM data', 5),
            (raw.id, 'SELECT COUNT(*) AS n FROM data', 10),
        ):
            with self.subTest(identity=identity):
                output = tools['local_analysis_sql'](dataset_id=identity, query=query)
                self.assertEqual(output['status'], 'ready', output)
                self.assertEqual(output['selected_dataset_id'], identity)
                self.assertEqual(context.datasets.frames[output['dataset']['id']].iloc[0, 0], expected)
                self.assertEqual(output['dataset']['row_selection'], context.datasets.metadata[identity].row_selection)

    def test_ties_and_nulls_never_publish_arbitrary_chart(self):
        source = source_frame()
        duplicate = source.sort_values(CLOCK).drop_duplicates(KEY, keep='last').iloc[[0]].copy()
        duplicate[VALUE] = 'different'
        cases = [('tie', pd.concat([source, duplicate], ignore_index=True), 'latest_order_tie')]
        for column in (KEY, CLOCK, VALUE):
            missing = source.copy()
            missing.loc[0, column] = None
            cases.append((column, missing, 'latest_null_policy'))
        for name, frame, error in cases:
            with self.subTest(case=name):
                context, raw = self.context(frame)
                result = self.run_latest(context, raw)
                self.assertEqual(result['status'], 'needs_context')
                self.assertEqual(result['error_code'], error)
                self.assertEqual(len(context.datasets.metadata), 1)
                self.assertFalse(context.artifacts)
        older = source.sort_values(CLOCK).iloc[[0]]
        context, raw = self.context(pd.concat([source, older], ignore_index=True))
        self.assertEqual(self.run_latest(context, raw)['status'], 'ready')

    def test_tie_break_and_numeric_identifiers(self):
        frame = source_frame()
        newest = frame.sort_values(CLOCK).drop_duplicates(KEY, keep='last').iloc[[0]].copy()
        newest[INGEST] += pd.Timedelta(days=100)
        newest[VALUE] = 'replacement'
        frame = pd.concat([frame, newest], ignore_index=True)
        context, raw = self.context(frame)
        result = self.run_latest(context, raw, tie_break_columns=[INGEST])
        self.assertEqual(result['status'], 'ready')
        self.assertIn('replacement', context.datasets.frames[result['dataset']['id']][VALUE].tolist())
        numeric = source_frame()
        numeric[VALUE] = numeric[VALUE].map({'P100': 100, 'P200': 200, 'P300': 300})
        context, raw = self.context(numeric)
        result = self.run_latest(context, raw, categorical=True)
        self.assertEqual(result['cards'][0]['kind'], 'bar')
        result = self.run_latest(context, raw, categorical=False)
        self.assertEqual(result['cards'][0]['kind'], 'histogram')

    def test_window_aliases_and_ctes_bind_only_physical_columns(self):
        context, raw = self.context()
        run = {t.name: t.run for t in build_analysis_tools(context)}['local_analysis_sql']
        ranked = f'SELECT {KEY}, {VALUE}, ROW_NUMBER() OVER (PARTITION BY {KEY} ORDER BY {CLOCK} DESC) AS rank FROM data'
        queries = [
            f'SELECT {VALUE}, COUNT(*) AS n FROM ({ranked}) q WHERE rank=1 GROUP BY {VALUE}',
            f'WITH ranked AS ({ranked}) SELECT {VALUE}, COUNT(*) AS n FROM ranked WHERE rank=1 GROUP BY {VALUE}',
        ]
        for query in queries:
            result = run(dataset_id=raw.id, query=query)
            self.assertEqual(result['status'], 'ready', result)
            output = context.datasets.frames[result['dataset']['id']]
            self.assertEqual(dict(zip(output[VALUE], output['n'])), FIXTURE['expected_counts'])
        for query in (
            'SELECT missing FROM (SELECT missing FROM data) q',
            'SELECT * FROM (SELECT * FROM external_table) q',
        ):
            with self.assertRaises(Exception): run(dataset_id=raw.id, query=query)

    def runtime(self, root, frame=None):
        runtime = GraphAnalysisRuntime(root, 'test', 'latest', QuietModel(),intent_mode='contract_fixture')
        if not runtime.datasets.metadata:
            raw = runtime.datasets.register(source_frame() if frame is None else frame,
                source=FIXTURE['source'], coverage='complete', predicate_known=True)
            runtime.select_dataset(raw.id)
        return runtime

    def test_ambiguous_order_resolves_after_restart_with_verified_attachment(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            try:
                result = runtime.submit(FIXTURE['cases'][0]['prompt'])
                self.assertIn('어느 컬럼', result['text'])
                self.assertFalse(chart_references(runtime.events()[-1]))
                raw = runtime.context.selected_dataset_id
            finally: runtime.close()
            runtime = self.runtime(root)
            try:
                result = runtime.submit(f'{CLOCK} 기준으로 해줘')
                self.assertEqual(result['status'], 'answered', result)
                proof = runtime.inspect()['recovery']['latest_selection_evidence']
                self.assertEqual(proof['selected_keys'], 5)
                self.assertTrue(chart_references(runtime.events()[-1]))
                selected = proof['dataset']['id']
            finally: runtime.close()
            runtime = self.runtime(root)
            try:
                self.assertEqual(runtime.datasets.metadata[selected].row_selection['input_dataset_id'], raw)
                self.assertEqual(len(runtime.datasets.frames[raw]), 10)
            finally: runtime.close()

    def test_wrong_plan_cannot_meet_completion_and_scope_is_not_silently_changed(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self.runtime(root)
            try:
                with patch.object(runtime.recovery, '_next_local', return_value=None):
                    result = runtime.submit(FIXTURE['cases'][1]['prompt'])
                self.assertNotEqual(result['status'], 'answered', result)
                self.assertFalse(chart_references(runtime.events()[-1]))
                result = runtime.submit(FIXTURE['cases'][1]['prompt'] + ' DB에서 다시 조회해줘')
                self.assertFalse(chart_references(runtime.events()[-1]))
                self.assertIsNone(runtime.inspect()['recovery']['latest_selection_evidence'])
                self.assertIn('새 데이터 조회', result['text'])
                result = runtime.submit(FIXTURE['cases'][1]['prompt'] + ' 추가로 평균도 알려줘')
                self.assertFalse(runtime.events()[-1].additional_kwargs['analysis_complete'])
                self.assertFalse(chart_references(runtime.events()[-1]))
                self.assertIn('추가 통계', result['text'])
            finally: runtime.close()


if __name__ == '__main__':
    unittest.main()
