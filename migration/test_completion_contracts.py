"""Production graph tests for false completion, budgets and cached plans."""
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.recovery import RecoveryMiddleware
from migration.test_persistent_runtime import QuietModel
from migration.test_recovery_journey import PlanOnlyModel, SOURCE, COLUMN, FIXTURE


class ScriptModel(QuietModel):
    calls: list = []
    position: int = 0
    dataset_id: str = ''
    answer: str = '평균은 999입니다.'
    repeat: bool = False

    def _generate(self, messages, **kwargs):
        if self.calls and (self.repeat or self.position < len(self.calls)):
            call = self.calls[min(self.position, len(self.calls) - 1)]
            args = {k: self.dataset_id if v == '$dataset' else v for k, v in call['args'].items()}
            message = AIMessage(content='', tool_calls=[dict(name=call['name'], args=args, id=str(uuid4()))])
        else: message = AIMessage(content=self.answer)
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class CompletionTests(unittest.TestCase):
    @staticmethod
    def scope_reference():
        frame = pd.DataFrame(FIXTURE['rows'])
        return [{'table': SOURCE, 'columns': [
            {'name': column, 'dtype': str(frame[column].dtype),
             'top_values': [{'value': value} for value in frame[column].dropna().unique()[:16]]}
            for column in frame.columns]}]

    def test_fabricated_mean_without_calculation_never_completes(self):
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'owner', 'fake-number', ScriptModel())
            result = r.submit('보유 데이터의 평균을 계산해줘')
            self.assertEqual(result['status'], 'exhausted')
            self.assertNotIn('999', result['text'])
            self.assertEqual(r.inspect()['recovery']['attempts'], 2)
            r.close()

    def test_actual_calculation_replaces_fabricated_final_number(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name': 'local_analysis_sql', 'args': {
                'dataset_id': '$dataset', 'query': f'SELECT AVG({COLUMN}) AS result FROM data'}}])
            r = GraphAnalysisRuntime(root, 'owner', 'real-number', model)
            frame = pd.DataFrame(FIXTURE['rows'])
            model.dataset_id = r.datasets.register(frame, source=SOURCE, coverage='complete', predicate_known=True).id
            result = r.submit(f'{COLUMN} 평균을 계산해줘')
            self.assertEqual(result['status'], 'answered', result)
            self.assertNotIn('999', result['text'])
            self.assertIn(str(frame[COLUMN].mean()), result['text'])
            self.assertTrue(r.inspect()['recovery']['evidence_ids'])
            r.close()

    def test_metadata_and_previous_turn_result_are_not_new_calculation(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name': 'inspect_table_context', 'args': {'table': SOURCE}}])
            r = GraphAnalysisRuntime(root, 'owner', 'metadata-number', model)
            r.context.reference_context = [{'table': SOURCE, 'columns': [{'name': COLUMN}]}]
            result = r.submit(f'{COLUMN} 평균')
            self.assertEqual(result['status'], 'exhausted')
            self.assertFalse(r.inspect()['recovery']['evidence_ids'])
            r.close()

    def test_identical_bad_tool_calls_stop_before_graph_recursion(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name': 'inspect_dataset', 'args': {'dataset_id': 'missing'}}], repeat=True)
            r = GraphAnalysisRuntime(root, 'owner', 'bad-loop', model)
            result = r.submit('histogram')
            self.assertEqual(result['status'], 'exhausted', result)
            self.assertEqual(r.inspect()['recovery']['stop_reason'], 'repeated_failed_tool')
            self.assertEqual(model.position, 2)
            self.assertEqual(r.inspect()['state'], 'idle')
            r.close()

    def test_successful_but_irrelevant_tool_loop_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name': 'list_analysis_context', 'args': {}}], repeat=True)
            r = GraphAnalysisRuntime(root, 'owner', 'metadata-loop', model)
            result = r.submit('histogram')
            self.assertEqual(result['status'], 'exhausted', result)
            self.assertEqual(r.inspect()['recovery']['stop_reason'], 'model_call_budget')
            self.assertEqual(model.position, 10)
            r.close()

    def test_complete_cached_plan_ends_without_model_or_remote_call(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            model = PlanOnlyModel()
            r = GraphAnalysisRuntime(root, 'owner', 'cache', model, connection_identity='test',
                remote_factory=lambda _: lambda request: calls.append(request))
            legacy_source = '.'.join('`' + part + '`' for part in SOURCE.split('.'))
            r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=legacy_source, coverage='complete', predicate_known=True)
            result = r.submit(f'{COLUMN} histogram')
            self.assertEqual(result['status'], 'answered', result)
            self.assertEqual(r.inspect()['recovery']['model_calls'], 0)
            self.assertEqual(calls, [])
            self.assertEqual(r.inspect()['requests'], [])
            self.assertTrue(r.inspect()['recovery']['artifact_ids'])
            r.close()

    def test_repeated_histogram_reuses_verified_png_without_model_call(self):
        from utils.analysis_charts import histogram_from_counts
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'owner', 'cached-chart', QuietModel())
            frame = pd.DataFrame(FIXTURE['rows']).groupby(COLUMN).size().reset_index(name='__frequency')
            query = (f'SELECT {COLUMN}, COUNT(*) AS __frequency FROM {SOURCE} '
                     f'WHERE {COLUMN} IS NOT NULL GROUP BY {COLUMN}')
            info = r.datasets.register(frame, source=SOURCE, coverage='complete',
                grain='aggregate', aggregation=query, query=query, predicate_known=True)
            card = histogram_from_counts(r.datasets, info.id, COLUMN, '__frequency')
            r.artifacts[card.id] = card

            result = r.submit(f'{COLUMN} histogram을 보여줘')

            self.assertEqual(result['status'], 'answered', result)
            state = r.inspect()['recovery']
            self.assertEqual(state['model_calls'], 0)
            self.assertEqual(state['artifact_ids'], [card.id])
            self.assertTrue(any('이미지를 생성했습니다' in str(m.content) for m in r.events()))
            r.close()

    def test_model_time_budget_is_persistent_not_reset_by_approval(self):
        with tempfile.TemporaryDirectory() as root:
            r = GraphAnalysisRuntime(root, 'owner', 'budget', QuietModel())
            guard = RecoveryMiddleware(r.artifacts, r.diagnostics, max_model_seconds=10)
            state = {'messages': [HumanMessage(content='histogram', id='request')],
                     'recovery': {'request_id': 'request', 'chart': True, 'model_seconds': 10.5}}
            result = guard.before_step(state)
            self.assertEqual(result['jump_to'], 'end')
            self.assertEqual(result['recovery']['stop_reason'], 'model_time_budget')
            r.close()

    def test_bad_sql_observation_allows_corrected_sql_without_restart(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[
                {'name': 'local_analysis_sql', 'args': {'dataset_id': '$dataset', 'query': 'SELECT AVG(missing_column) FROM data'}},
                {'name': 'local_analysis_sql', 'args': {'dataset_id': '$dataset', 'query': f'SELECT AVG({COLUMN}) AS result FROM data'}}])
            r = GraphAnalysisRuntime(root, 'owner', 'sql-repair', model)
            model.dataset_id = r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=SOURCE,
                coverage='complete', predicate_known=True).id
            result = r.submit(f'{COLUMN} 평균')
            self.assertEqual(result['status'], 'answered', result)
            errors = [json.loads(m.content) for m in r.events() if isinstance(m, ToolMessage) and m.name == 'local_analysis_sql']
            self.assertEqual(errors[0]['error_code'], 'local_sql_error')
            self.assertEqual(errors[1]['status'], 'ready')
            self.assertEqual(model.position, 2)
            r.close()

    def test_new_turn_does_not_reuse_previous_numeric_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name': 'local_analysis_sql', 'args': {
                'dataset_id': '$dataset', 'query': f'SELECT AVG({COLUMN}) AS result FROM data'}}])
            r = GraphAnalysisRuntime(root, 'owner', 'new-turn', model)
            model.dataset_id = r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=SOURCE,
                coverage='complete', predicate_known=True).id
            self.assertEqual(r.submit(f'{COLUMN} 평균')['status'], 'answered')
            result = r.submit(f'{COLUMN} 중앙값')
            self.assertEqual(result['status'], 'exhausted')
            self.assertFalse(r.inspect()['recovery']['evidence_ids'])
            r.close()

    def test_four_turn_fixture_keeps_expected_operation_and_current_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name': 'local_analysis_sql', 'args': {
                'dataset_id': '$dataset', 'query': turn['sql']}} for turn in FIXTURE['turns']])
            r = GraphAnalysisRuntime(root, 'owner', 'followup', model)
            model.dataset_id = r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=SOURCE,
                coverage='complete', predicate_known=True).id
            request_ids = []
            for turn in FIXTURE['turns']:
                outcome = r.submit(turn['prompt'])
                self.assertEqual(outcome['status'], 'answered', outcome)
                state = r.inspect()['recovery']
                self.assertEqual(len(state['evidence_ids']), 1)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][0]].iloc[0, 0]), turn['expected'])
                request_ids.append(state['request_id'])
            self.assertEqual(len(set(request_ids)), 4)
            r.close()

    def test_scope_columns_applied_before_sql_are_completion_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            row = FIXTURE['rows'][0]
            filter_column = next(k for k, v in row.items() if isinstance(v, str))
            model = ScriptModel(calls=[{'name':'local_analysis_sql', 'args':{
                'dataset_id':'$dataset', 'query':f'SELECT AVG({COLUMN}) AS result FROM data',
                'requested_conditions':[{'column':filter_column, 'op':'eq', 'value':row[filter_column]}]}}])
            r = GraphAnalysisRuntime(root, 'owner', 'scope-evidence', model)
            model.dataset_id = r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=SOURCE,
                coverage='complete', predicate_known=True).id
            result = r.submit(f'{filter_column} {row[filter_column]}의 {COLUMN} 평균')
            self.assertEqual(result['status'], 'answered', result)
            self.assertIn(filter_column, result['text'])
            r.close()

    def test_exact_filtered_dataset_is_counted_without_second_model_call(self):
        with tempfile.TemporaryDirectory() as root:
            frame = pd.DataFrame(FIXTURE['rows'])
            category = next(k for k, value in FIXTURE['rows'][0].items() if isinstance(value, str))
            value = FIXTURE['rows'][0][category]
            model = ScriptModel(calls=[{'name':'use_dataset', 'args':{
                'dataset_id':'$dataset', 'columns':list(frame.columns),
                'conditions':[{'column':category, 'op':'eq', 'value':value}]}}])
            r = GraphAnalysisRuntime(root, 'owner', 'filtered-count-recovery', model)
            model.dataset_id = r.datasets.register(frame, source=SOURCE,
                coverage='complete', predicate_known=True).id
            # Two equivalent source frames make the initial deterministic
            # choice ambiguous, so the model materializes one scoped child.
            r.datasets.register(frame.copy(), source=SOURCE,
                coverage='complete', predicate_known=True)

            result = r.submit(f"{category} {value}인 '사람'들의 수를 계산해줘")

            self.assertEqual(result['status'], 'answered', result)
            self.assertEqual(model.position, 1, 'exact filtered child should avoid a second model call')
            state = r.inspect()['recovery']
            calculated = r.datasets.frames[state['evidence_ids'][-1]]
            self.assertEqual(int(calculated.iloc[0, 0]), int(frame[category].eq(value).sum()))
            r.close()

    def test_column_count_uses_metadata_without_loading_or_numeric_sql(self):
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name':'inspect_table_context','args':{'table':SOURCE}}])
            r = GraphAnalysisRuntime(root, 'owner', 'column-count', model)
            columns = [{'name':c} for c in FIXTURE['rows'][0]]
            r.context.reference_context = [{'table':SOURCE, 'columns':columns}]
            result = r.submit(f'{SOURCE} 테이블의 컬럼 전체 목록과 총 컬럼 개수를 알려줘')
            self.assertEqual(result['status'], 'answered', result)
            self.assertIn(f'{len(columns)}개', result['text'])
            self.assertNotIn('999', result['text'])
            self.assertFalse(r.datasets.metadata)
            r.close()

    def test_value_counts_and_missing_values_cannot_complete_from_schema(self):
        category = next(k for k, v in FIXTURE['rows'][0].items() if isinstance(v, str))
        for prompt in (f'{category} 컬럼의 고유값 개수를 알려줘', '각 컬럼의 결측값 개수를 알려줘'):
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as root:
                model = ScriptModel(calls=[{'name':'inspect_table_context', 'args':{'table':SOURCE}}])
                r = GraphAnalysisRuntime(root, 'owner', 'value-stats', model)
                r.context.reference_context = [{'table':SOURCE, 'columns':[{'name':c} for c in FIXTURE['rows'][0]]}]
                result = r.submit(prompt)
                self.assertEqual(result['status'], 'exhausted', result)
                self.assertIsNone(r.inspect()['recovery']['metadata_kind'])
                r.close()

    def test_categorical_type_error_is_repaired_instead_of_returning_zero(self):
        category = next(k for k, v in FIXTURE['rows'][0].items() if isinstance(v, str))
        value = FIXTURE['rows'][0][category]
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[{'name':'local_analysis_sql', 'args':{
                'dataset_id':'$dataset', 'query':'SELECT COUNT(*) AS n FROM data',
                'requested_conditions':[{'column':category, 'op':'eq', 'value':v}]}} for v in (True, value)])
            r = GraphAnalysisRuntime(root, 'owner', 'categorical-type', model)
            frame = pd.DataFrame(FIXTURE['rows'])
            model.dataset_id = r.datasets.register(frame, source=SOURCE, coverage='complete', predicate_known=True).id
            # Keep this test on the model/tool validation path. A single
            # unambiguous dataset now uses the deterministic filtered-count path.
            r.datasets.register(frame, source='fixture.other', coverage='complete', predicate_known=True)
            result = r.submit(f'{category} {value}의 인원수를 알려줘')
            observations = [json.loads(m.content) for m in r.events() if isinstance(m, ToolMessage)]
            self.assertEqual(len(observations), 1, 'wrong typed scope must be rejected before local execution')
            self.assertEqual(observations[0]['preview'][0]['n'], int(frame[category].eq(value).sum()))
            self.assertEqual(result['status'], 'answered', result)
            self.assertEqual(model.position, 2)
            r.close()

    def test_wrong_month_result_is_rejected_then_correct_scope_completes(self):
        wrong = FIXTURE['turns'][0]['sql'].replace('2026-08', '2026-07')
        correct = FIXTURE['turns'][0]['sql']
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[
                {'name':'local_analysis_sql', 'args':{'dataset_id':'$dataset', 'query':wrong}},
                {'name':'local_analysis_sql', 'args':{'dataset_id':'$dataset', 'query':correct}}])
            r = GraphAnalysisRuntime(root, 'owner', 'scope-repair', model)
            model.dataset_id = r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=SOURCE,
                coverage='complete', predicate_known=True).id
            outcome = r.submit(FIXTURE['turns'][0]['prompt'])
            state = r.inspect()['recovery']
            self.assertEqual(outcome['status'], 'answered', outcome)
            self.assertEqual(float(r.datasets.frames[state['evidence_ids'][0]].iloc[0, 0]),
                             FIXTURE['turns'][0]['expected'])
            self.assertEqual(state['scope']['conditions'],
                             [{'column':'period', 'op':'eq', 'value':'2026-08'}])
            self.assertGreaterEqual(model.position, 2)
            local_calls = [m for m in r.events()
                if isinstance(m, ToolMessage) and m.name == 'local_analysis_sql']
            self.assertEqual(len(local_calls), 1, 'wrong-scope local calculation must be rejected before execution')
            self.assertTrue(any(event.get('event') == 'request_scope_rejected'
                for event in (json.loads(line) for line in r.diagnostics.path.read_text().splitlines())))
            r.close()

    def test_followup_cannot_silently_drop_inherited_month(self):
        first, followup = FIXTURE['turns'][:2]
        missing_month = followup['sql'].replace("period = '2026-08' AND ", '')
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[
                {'name':'local_analysis_sql', 'args':{'dataset_id':'$dataset', 'query':first['sql']}},
                {'name':'local_analysis_sql', 'args':{'dataset_id':'$dataset', 'query':missing_month}}])
            r = GraphAnalysisRuntime(root, 'owner', 'scope-followup', model)
            model.dataset_id = r.datasets.register(pd.DataFrame(FIXTURE['rows']), source=SOURCE,
                coverage='complete', predicate_known=True).id
            self.assertEqual(r.submit(first['prompt'])['status'], 'answered')
            outcome = r.submit(followup['prompt'])
            state = r.inspect()['recovery']
            self.assertNotEqual(outcome['status'], 'answered', outcome)
            self.assertEqual(state['evidence_ids'], [])
            self.assertEqual({(c['column'], c['value']) for c in state['scope']['conditions']},
                             {('period', '2026-08'), ('segment', 'A')})
            self.assertEqual(state['stop_reason'], 'request_scope_mismatch')
            r.close()

    def test_wrong_remote_scope_never_creates_approval_before_corrected_query(self):
        wrong = "SELECT COUNT(*) FROM fixture.synthetic_events WHERE period='2026-07'"
        correct = "SELECT COUNT(*) FROM fixture.synthetic_events WHERE period='2026-08'"
        with tempfile.TemporaryDirectory() as root:
            executions = []
            model = ScriptModel(calls=[
                {'name':'query_databricks', 'args':{'source':SOURCE, 'query':wrong, 'reason':'count'}},
                {'name':'query_databricks', 'args':{'source':SOURCE, 'query':correct, 'reason':'count'}}])
            r = GraphAnalysisRuntime(root, 'owner', 'remote-scope', model,
                connection_identity='test', remote_factory=lambda _: lambda envelope: executions.append(envelope))
            r.context.reference_context = self.scope_reference()
            outcome = r.submit('2026-08의 개수를 알려줘')
            self.assertEqual(outcome['status'], 'awaiting_approval', outcome)
            self.assertEqual(len(r.inspect()['requests']), 1)
            self.assertEqual(r.inspect()['requests'][0]['query'], correct)
            self.assertEqual(executions, [])
            self.assertNotIn(wrong, [item['query'] for item in r.inspect()['requests']])
            r.close()

    def test_wrong_filtered_histogram_is_rejected_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as root:
            executions = []
            model = ScriptModel(calls=[
                {'name':'prepare_histogram', 'args':{'source':SOURCE, 'column':COLUMN,
                    'where_sql':"period='2026-07'"}},
                {'name':'prepare_histogram', 'args':{'source':SOURCE, 'column':COLUMN,
                    'where_sql':"period='2026-08'"}}])
            r = GraphAnalysisRuntime(root, 'owner', 'chart-scope', model,
                connection_identity='test', remote_factory=lambda _: lambda envelope: executions.append(envelope))
            r.context.reference_context = self.scope_reference()
            outcome = r.submit(f'2026-08의 {COLUMN} histogram')
            self.assertEqual(outcome['status'], 'awaiting_approval', outcome)
            prepare_observations = [m for m in r.events()
                if isinstance(m, ToolMessage) and m.name == 'prepare_histogram']
            self.assertEqual(len(prepare_observations), 1)
            pending = r.inspect()['requests']
            self.assertEqual(len(pending), 1)
            self.assertIn('2026-08', pending[0]['query'])
            self.assertNotIn('2026-07', pending[0]['query'])
            self.assertEqual(executions, [])
            r.close()

    def test_request_scope_survives_restart_while_waiting_for_approval(self):
        query = "SELECT COUNT(*) FROM fixture.synthetic_events WHERE period='2026-08'"
        with tempfile.TemporaryDirectory() as root:
            executions = []
            model = ScriptModel(calls=[{'name':'query_databricks', 'args':{
                'source':SOURCE, 'query':query, 'reason':'count'}}])
            r = GraphAnalysisRuntime(root, 'owner', 'scope-restart', model,
                connection_identity='test', remote_factory=lambda _: lambda envelope: executions.append(envelope))
            r.context.reference_context = self.scope_reference()
            self.assertEqual(r.submit('2026-08의 개수를 알려줘')['status'], 'awaiting_approval')
            expected_scope = r.inspect()['recovery']['scope']
            r.close()
            reopened = GraphAnalysisRuntime(root, 'owner', 'scope-restart', QuietModel(),
                connection_identity='test', remote_factory=lambda _: lambda envelope: executions.append(envelope))
            reopened.context.reference_context = self.scope_reference()
            self.assertEqual(reopened.inspect()['state'], 'awaiting_approval')
            self.assertEqual(reopened.inspect()['recovery']['scope'], expected_scope)
            self.assertEqual(executions, [])
            reopened.close()

    def test_grounded_affirmative_filter_rejects_model_added_category(self):
        frame = pd.DataFrame({'id':[1, 2, 3], 'service_flag':['yes', 'no', 'yes'],
                              'status':['open', 'closed', 'closed']})
        source = 'fixture.flags'
        reference = [{'table':source, 'columns':[
            {'name':'id', 'top_values':[]},
            {'name':'service_flag', 'aliases':['서비스 사용'],
             'top_values':[{'value':'yes'}, {'value':'no'}]},
            {'name':'status', 'aliases':['상태'],
             'top_values':[{'value':'open'}, {'value':'closed'}]}]}]
        wrong_conditions = [
            {'column':'service_flag', 'op':'eq', 'value':'yes'},
            {'column':'status', 'op':'eq', 'value':'closed'}]
        correct_conditions = [{'column':'service_flag', 'op':'eq', 'value':'yes'}]
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel(calls=[
                {'name':'local_analysis_sql', 'args':{'dataset_id':'$dataset', 'query':'COUNT(id)',
                    'requested_conditions':wrong_conditions}},
                {'name':'local_analysis_sql', 'args':{'dataset_id':'$dataset', 'query':'COUNT(id)',
                    'requested_conditions':correct_conditions}}])
            r = GraphAnalysisRuntime(root, 'owner', 'boolean-scope', model)
            r.context.reference_context = reference
            model.dataset_id = r.datasets.register(frame, source=source,
                coverage='complete', predicate_known=True).id
            r.datasets.register(frame, source='fixture.other',
                coverage='complete', predicate_known=True)
            outcome = r.submit('서비스 사용 이력이 있는 사람이 몇 명이야?')
            state = r.inspect()['recovery']
            self.assertEqual(outcome['status'], 'answered', outcome)
            self.assertEqual(state['scope']['conditions'], correct_conditions)
            self.assertEqual(float(r.datasets.frames[state['evidence_ids'][0]].iloc[0, 0]),
                             float(frame['service_flag'].eq('yes').sum()))
            self.assertGreaterEqual(model.position, 2)
            local_calls = [m for m in r.events()
                if isinstance(m, ToolMessage) and m.name == 'local_analysis_sql']
            self.assertEqual(len(local_calls), 1, 'model-added category must be rejected before execution')
            r.close()

    def test_grounded_filtered_count_has_deterministic_local_fallback(self):
        frame = pd.DataFrame({'id':[1, 2, 3], 'service_flag':['yes', 'no', 'yes']})
        reference = [{'table':'fixture.flags', 'columns':[
            {'name':'id', 'top_values':[]},
            {'name':'service_flag', 'aliases':['서비스 사용'],
             'top_values':[{'value':'yes'}, {'value':'no'}]}]}]
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel()
            r = GraphAnalysisRuntime(root, 'owner', 'count-fallback', model)
            r.context.reference_context = reference
            r.datasets.register(frame, source='fixture.flags',
                coverage='complete', predicate_known=True)
            outcome = r.submit('서비스 사용 이력이 있는 사람이 몇 명이야?')
            state = r.inspect()['recovery']
            self.assertEqual(outcome['status'], 'answered', outcome)
            self.assertEqual(model.position, 0, 'deterministic count should not need a model call')
            self.assertEqual(float(r.datasets.frames[state['evidence_ids'][0]].iloc[0, 0]), 2.0)
            calls = [m for m in r.events()
                if isinstance(m, ToolMessage) and m.name == 'local_analysis_sql']
            self.assertEqual(len(calls), 1)
            r.close()

    def test_grounded_ratio_has_deterministic_percent_fallback(self):
        frame = pd.DataFrame({'duration':[400, 600, 700],
                              'converted':['yes', 'yes', 'no']})
        reference = [{'table':'fixture.calls', 'columns':[
            {'name':'duration', 'aliases':['상담 시간'], 'top_values':[]},
            {'name':'converted', 'aliases':['예금 가입'],
             'top_values':[{'value':'yes'}, {'value':'no'}]}]}]
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel()
            r = GraphAnalysisRuntime(root, 'owner', 'ratio-fallback', model)
            r.context.reference_context = reference
            r.datasets.register(frame, source='fixture.calls',
                coverage='complete', predicate_known=True)
            outcome = r.submit(
                "상담 시간(duration)이 500초를 초과한 고객들의 예금 가입(converted='yes') 비율을 구해줘")
            state = r.inspect()['recovery']
            self.assertEqual(outcome['status'], 'answered', outcome)
            self.assertEqual(model.position, 0)
            result = float(r.datasets.frames[state['evidence_ids'][0]].iloc[0, 0])
            self.assertAlmostEqual(result, 50.0)
            self.assertIn('비율(%): 50.0', outcome['text'])
            self.assertEqual(state['scope']['conditions'],
                             [{'column':'duration', 'op':'gt', 'value':500}])
            self.assertEqual(state['scope']['measure_conditions'],
                             [{'column':'converted', 'op':'eq', 'value':'yes'}])
            r.close()

    def test_grounded_count_and_ratio_obligations_share_one_local_result(self):
        frame = pd.DataFrame({'campaign':[5, 11, 12],
                              'converted':['yes', 'yes', 'no']})
        reference = [{'table':'fixture.calls', 'columns':[
            {'name':'campaign', 'aliases':['접촉 횟수'], 'top_values':[]},
            {'name':'converted', 'aliases':['가입'],
             'top_values':[{'value':'yes'}, {'value':'no'}]}]}]
        with tempfile.TemporaryDirectory() as root:
            model = ScriptModel()
            r = GraphAnalysisRuntime(root, 'owner', 'count-ratio-fallback', model)
            r.context.reference_context = reference
            r.datasets.register(frame, source='fixture.calls',
                coverage='complete', predicate_known=True)
            outcome = r.submit('접촉 횟수(campaign)가 10회를 초과한 고객 인원수와 가입률을 구해줘')
            state = r.inspect()['recovery']
            self.assertEqual(outcome['status'], 'answered', outcome)
            result = r.datasets.frames[state['evidence_ids'][0]]
            self.assertEqual(int(result['count'].iloc[0]), 2)
            self.assertAlmostEqual(float(result['percent'].iloc[0]), 50.0)
            self.assertEqual(model.position, 0)
            r.close()


if __name__ == '__main__': unittest.main()
