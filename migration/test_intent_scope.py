"""Request scope is grounded independently from the model's executed SQL."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest

import pandas as pd

from core.analysis_agent.intent_scope import resolve_request_scope, scope_matches, measure_scope_matches
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import Condition, DatasetStore


FIXTURE = json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())


class IntentScopeTests(unittest.TestCase):
    def context(self, *, with_rows=True, reference=None):
        store = DatasetStore()
        if with_rows:
            store.register(pd.DataFrame(FIXTURE['rows']), source=FIXTURE['source'],
                           coverage='complete', predicate_known=True)
        return AnalysisToolContext(store, {}, reference or [], lambda **_: None)

    def test_fixture_followups_keep_and_change_grounded_conditions(self):
        context = self.context()
        scope = None
        for turn in FIXTURE['turns']:
            scope = resolve_request_scope(turn['prompt'], context, previous=scope)
            self.assertEqual(scope['unresolved'], [], (turn, scope))
            self.assertTrue(scope_matches(turn['sql'], scope), (turn, scope))
        self.assertTrue(any(c['value'] == '2026-07' for c in scope['conditions']))

    def test_wrong_month_is_rejected_while_actual_correct_result_is_accepted(self):
        context = self.context()
        scope = resolve_request_scope(FIXTURE['turns'][0]['prompt'], context)
        analyze = next(tool.run for tool in build_analysis_tools(context) if tool.name == 'local_analysis_sql')
        raw = next(iter(context.datasets.metadata))
        correct = analyze(raw, FIXTURE['turns'][0]['sql'])
        wrong = analyze(raw, FIXTURE['turns'][0]['sql'].replace('2026-08', '2026-07'))
        self.assertTrue(scope_matches(context.datasets.metadata[correct['dataset']['id']], scope))
        self.assertFalse(scope_matches(context.datasets.metadata[wrong['dataset']['id']], scope))

    def test_columns_aliases_and_values_come_from_external_context(self):
        reference = [{'table':'fixture.external', 'columns':[
            {'name':'report_month', 'dtype':'string', 'aliases':['월'],
             'top_values':[{'value':'2026-08','count':7}]},
            {'name':'cohort_code', 'dtype':'string', 'aliases':['그룹'],
             'top_values':[{'value':'X','count':4}]}]}]
        context = self.context(with_rows=False, reference=reference)
        scope = resolve_request_scope('2026-08의 그룹 X만 계산해줘', context)
        self.assertEqual(scope['unresolved'], [])
        self.assertTrue(scope_matches("SELECT COUNT(*) FROM fixture.external WHERE report_month='2026-08' AND cohort_code='X'", scope))
        self.assertFalse(scope_matches("SELECT COUNT(*) FROM fixture.external WHERE report_month='2026-08' AND cohort_code='Y'", scope))

    def test_explicit_comparisons_and_column_replacement(self):
        context = self.context()
        scope = resolve_request_scope('value >= 10이고 value < 90인 자료를 계산해줘', context)
        self.assertEqual(scope['unresolved'], [])
        self.assertTrue(scope_matches('SELECT COUNT(*) FROM data WHERE value >= 10 AND value < 90', scope))
        updated = resolve_request_scope('같은 조건에서 value >= 20으로 바꿔줘', context, previous=scope)
        self.assertTrue(scope_matches('SELECT COUNT(*) FROM data WHERE value >= 20', updated))
        fresh = resolve_request_scope('value 평균을 계산해줘', context, previous=scope)
        self.assertEqual(fresh['conditions'], [])

    def test_month_reference_rolls_back_year_and_retains_other_columns(self):
        previous = {'conditions':[asdict(Condition('month_key','eq','2026-01')),
                                  asdict(Condition('group_key','eq','X'))], 'unresolved':[]}
        scope = resolve_request_scope('같은 그룹의 이전 달 중앙값', self.context(with_rows=False), previous=previous)
        self.assertEqual(scope['unresolved'], [])
        self.assertTrue(scope_matches("SELECT MEDIAN(value) FROM data WHERE month_key='2025-12' AND group_key='X'", scope))

    def test_ambiguous_date_column_or_missing_month_reference_stays_unresolved(self):
        columns = [{'name':name,'dtype':'str','top_values':[{'value':'2026-08'}]} for name in ('begin_month','end_month')]
        context = self.context(with_rows=False, reference=[{'table':'fixture.ambiguous','columns':columns}])
        scope = resolve_request_scope('2026-08의 평균', context)
        self.assertIn('ambiguous_date_column', scope['unresolved'])
        self.assertFalse(scope_matches("SELECT AVG(value) FROM data WHERE begin_month='2026-08'", scope))
        previous = resolve_request_scope('같은 그룹의 이전 달', context)
        self.assertIn('ambiguous_previous_month', previous['unresolved'])

    def test_grounded_or_is_preserved_and_unknown_sql_is_rejected(self):
        context = self.context()
        scope = resolve_request_scope("segment = 'A' 또는 period = '2026-08'인 자료", context)
        self.assertEqual(scope['unresolved'], [])
        self.assertTrue(scope_matches("SELECT * FROM data WHERE segment='A' OR period='2026-08'", scope))
        self.assertFalse(scope_matches("SELECT * FROM data WHERE segment='A' AND period='2026-08'", scope))
        known = resolve_request_scope(FIXTURE['turns'][0]['prompt'], context)
        self.assertFalse(scope_matches("SELECT AVG(value) FROM data WHERE period='2026-08' OR value>10", known))

    def test_common_condition_plus_cross_column_or_and_value_list(self):
        reference=[{'table':'fixture.people','columns':[
            {'name':'default','aliases':['연체'],'top_values':[{'value':'yes'},{'value':'no'}]},
            {'name':'housing','aliases':['집 대출'],'top_values':[{'value':'yes'},{'value':'no'}]},
            {'name':'loan','aliases':['신용 대출'],'top_values':[{'value':'yes'},{'value':'no'}]},
            {'name':'month','aliases':['월'],'top_values':[]}]}]
        context=self.context(with_rows=False,reference=reference)
        scope=resolve_request_scope("연체(default='yes') 이력이 있으면서 집 대출이나 신용 대출 중 하나라도 있는 고객",context)
        self.assertEqual(scope['unresolved'],[])
        self.assertEqual(scope['conditions'],[{'column':'default','op':'eq','value':'yes'}])
        self.assertEqual({item['column'] for item in scope['any_conditions']},{'housing','loan'})
        self.assertTrue(scope_matches("SELECT * FROM fixture.people WHERE default='yes' AND (housing='yes' OR loan='yes')",scope))
        self.assertFalse(scope_matches("SELECT * FROM fixture.people WHERE default='yes' AND housing='yes' AND loan='yes'",scope))
        months=resolve_request_scope("3, 4, 5월('mar', 'apr', 'may') 고객",context)
        self.assertEqual(months['conditions'],[{'column':'month','op':'in','value':['mar','apr','may']}])
        self.assertTrue(scope_matches("SELECT * FROM fixture.people WHERE month IN ('mar','apr','may')",months))

    def test_unsupported_date_relation_does_not_turn_into_equality(self):
        context = self.context()
        scope = resolve_request_scope('2026-08 이전의 value 평균', context)
        self.assertIn('unsupported_date_relation', scope['unresolved'])
        self.assertFalse(scope_matches("SELECT AVG(value) FROM data WHERE period='2026-08'", scope))

    def test_extra_or_missing_filters_are_not_equivalent(self):
        context = self.context()
        scope = resolve_request_scope(FIXTURE['turns'][0]['prompt'], context)
        self.assertFalse(scope_matches('SELECT AVG(value) FROM data', scope))
        self.assertFalse(scope_matches("SELECT AVG(value) FROM data WHERE period='2026-08' AND segment='A'", scope))

    def test_large_raw_frames_are_not_loaded_to_resolve_dates(self):
        context = self.context()
        from dataclasses import replace
        key, info = next(iter(context.datasets.metadata.items()))
        context.datasets.metadata[key] = replace(info, rows=1_000_000)
        class NoRead(dict):
            def __getitem__(self, key): raise AssertionError('large frame was read')
        context.datasets.frames = NoRead()
        scope = resolve_request_scope(FIXTURE['turns'][0]['prompt'], context)
        self.assertIn('ungrounded_date_column', scope['unresolved'])

    def test_boolean_possession_words_require_grounded_binary_values(self):
        reference = [{'table':'fixture.flags', 'columns':[
            {'name':'service_flag', 'aliases':['서비스 사용'],
             'top_values':[{'value':'yes'}, {'value':'no'}]},
            {'name':'status', 'aliases':['상태'],
             'top_values':[{'value':'open'}, {'value':'closed'}]}]}]
        context = self.context(with_rows=False, reference=reference)
        positive = resolve_request_scope('서비스 사용 이력이 있는 사람 수', context)
        negative = resolve_request_scope('서비스 사용 이력이 없는 사람 수', context)
        unknown = resolve_request_scope('상태 이력이 있는 사람 수', context)
        self.assertTrue(scope_matches("SELECT COUNT(*) FROM data WHERE service_flag='yes'", positive))
        self.assertTrue(scope_matches("SELECT COUNT(*) FROM data WHERE service_flag='no'", negative))
        self.assertEqual(unknown['conditions'], [])

    def test_ratio_separates_population_from_numerator_condition(self):
        reference = [{'table':'fixture.calls', 'columns':[
            {'name':'duration', 'aliases':['상담 시간'], 'top_values':[]},
            {'name':'converted', 'aliases':['예금 가입'],
             'top_values':[{'value':'yes'}, {'value':'no'}]}]}]
        context = self.context(with_rows=False, reference=reference)
        scope = resolve_request_scope(
            "상담 시간(duration)이 500초를 초과한 고객들의 예금 가입(converted='yes') 비율", context)
        self.assertEqual(scope['conditions'], [{'column':'duration', 'op':'gt', 'value':500}])
        self.assertEqual(scope['measure_conditions'],
                         [{'column':'converted', 'op':'eq', 'value':'yes'}])
        self.assertEqual(scope['ratio'], {'column':'converted'})
        self.assertTrue(scope_matches("SELECT * FROM data WHERE duration > 500", scope))
        self.assertTrue(measure_scope_matches(
            "SELECT SUM(CASE WHEN converted='yes' THEN 1 ELSE 0 END)/COUNT(*) FROM data", scope))
        self.assertFalse(measure_scope_matches(
            "SELECT SUM(CASE WHEN converted='no' THEN 1 ELSE 0 END)/COUNT(*) FROM data", scope))

        possession = resolve_request_scope(
            "상담 시간이 500초를 초과한 고객의 예금 가입 보유율", context)
        self.assertEqual(possession['ratio'], {'column':'converted'})

    def test_ratio_grounds_parenthetical_population_and_binary_numerator(self):
        reference = [{'table':'fixture.passengers', 'columns':[
            {'name':'group', 'aliases':['여성', '남성'],
             'top_values':[{'value':'female'}, {'value':'male'}]},
            {'name':'survived', 'aliases':['생존율'],
             'top_values':[{'value':0}, {'value':1}]}]}]
        context = self.context(with_rows=False, reference=reference)
        scope = resolve_request_scope('여성(female) 승객들의 생존율(%)을 계산해줘', context)
        self.assertEqual(scope['conditions'], [{'column':'group', 'op':'eq', 'value':'female'}])
        self.assertEqual(scope['measure_conditions'],
                         [{'column':'survived', 'op':'eq', 'value':1}])
        self.assertEqual(scope['ratio'], {'column':'survived', 'aggregation':'mean_zero_one'})
        self.assertTrue(measure_scope_matches(
            'SELECT 100.0 * AVG(survived) FROM data', scope))

    def test_bounded_human_label_before_parenthetical_value_is_grounded(self):
        reference = [{'table':'fixture.people', 'columns':[
            {'name':'marital', 'aliases':['혼인 상태'],
             'top_values':[{'value':'married'}, {'value':'divorced'}]}]}]
        context = self.context(with_rows=False, reference=reference)
        scope = resolve_request_scope("혼인 상태가 이혼(divorced)인 사람 수", context)
        self.assertEqual(scope['conditions'],
                         [{'column':'marital', 'op':'eq', 'value':'divorced'}])

    def test_grounded_decade_parenthetical_category_and_affirmative_suffix_combine(self):
        reference = [{'table':'fixture.customers', 'columns':[
            {'name':'age', 'aliases':['20대'], 'top_values':[]},
            {'name':'marital', 'aliases':['미혼'],
             'top_values':[{'value':'single'}, {'value':'married'}]},
            {'name':'subscribed', 'aliases':['가입'],
             'top_values':[{'value':'yes'}, {'value':'no'}]}]}]
        context = self.context(with_rows=False, reference=reference)
        scope = resolve_request_scope("미혼(single)인 20대(20~29세) 중 가입한 사람 수", context)
        self.assertEqual({(c['column'], c['op'], c['value']) for c in scope['conditions']}, {
            ('marital', 'eq', 'single'), ('age', 'ge', 20), ('age', 'le', 29),
            ('subscribed', 'eq', 'yes')})


if __name__ == '__main__':
    unittest.main()
