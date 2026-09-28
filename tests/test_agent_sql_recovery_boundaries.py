"""Read-only SQL proposal boundaries independent of a remote warehouse or LLM."""
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import tempfile
import unittest

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from uuid import uuid4

from core.analysis_agent.intent_scope import resolve_request_scope, scope_matches
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.sql_preflight import known_column_error
from scripts.evaluate_spider2_teleai import check_sqlite_candidate, schema_context


class CapturingModel(BaseChatModel):
    seen: list = []

    @property
    def _llm_type(self):
        return 'capture-evaluation-boundary'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='No result.'))])


class RepairingSqlModel(BaseChatModel):
    calls: int = 0
    saw_feedback: bool = False

    @property
    def _llm_type(self):
        return 'scripted-sql-preflight-repair'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.calls += 1
        self.saw_feedback = self.saw_feedback or any(
            isinstance(message, SystemMessage) and 'dialect_error' in str(message.content)
            for message in messages)
        query = ('SELECT ST_Y(value) FROM items' if self.calls == 1 else
                 'SELECT COUNT(*) FROM items')
        call = {'name': 'query_databricks', 'id': str(uuid4()),
                'args': {'source': 'items', 'query': query, 'reason': 'Read-only count'}}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='', tool_calls=[call]))])


class JoinedProposalModel(BaseChatModel):
    query: str

    @property
    def _llm_type(self):
        return 'scripted-qualified-join-proposal'

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        call = {'name': 'query_databricks', 'id': str(uuid4()),
                'args': {'source': 'events | labels', 'query': self.query,
                         'reason': 'Read-only joined count'}}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='', tool_calls=[call]))])


class JoinedScopeContractTests(unittest.TestCase):
    def test_qualified_inner_join_or_requires_both_roles(self):
        requested = {'conditions': [], 'any_conditions': [
            {'column': 'left_role.label', 'op': 'eq', 'value': 'target'},
            {'column': 'right_role.label', 'op': 'eq', 'value': 'target'}],
            'join_edges': [
                ['e.left_key', 'left_role.key'],
                ['e.right_key', 'right_role.key']],
            'unresolved': []}
        source = ('FROM events AS e '
                  'JOIN labels AS left_role ON e.left_key = left_role.key '
                  'JOIN labels AS right_role ON e.right_key = right_role.key')
        valid = ("SELECT COUNT(*) " + source +
                 " WHERE left_role.label = 'target' OR right_role.label = 'target'")
        self.assertTrue(scope_matches(valid, requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace(' OR ', ' AND '), requested,
                                       dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace('right_role.label =', 'left_role.label ='),
                                       requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace('right_role.label =', 'label ='),
                                       requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace('right_role.label =', 'absent.label ='),
                                       requested, dialect='sqlite'))

    def test_join_conditions_cannot_hide_extra_filters(self):
        requested = {'conditions': [], 'any_conditions': [
            {'column': 'a.label', 'op': 'eq', 'value': 'target'},
            {'column': 'b.label', 'op': 'eq', 'value': 'target'}],
            'join_edges': [['a.key', 'e.left_key'], ['b.key', 'e.right_key']],
            'unresolved': []}
        valid = ("SELECT COUNT(*) FROM events e "
                 "JOIN labels a ON e.left_key = a.key "
                 "JOIN labels b ON e.right_key = b.key "
                 "WHERE a.label = 'target' OR b.label = 'target'")
        self.assertTrue(scope_matches(valid, requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace('e.left_key = a.key',
                 "e.left_key = a.key AND a.kind = 'active'"), requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace('JOIN labels a', 'LEFT JOIN labels a'),
                                       requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid.replace('e.left_key = a.key',
                 'e.right_key = a.key'), requested, dialect='sqlite'))
        self.assertFalse(scope_matches(valid, {**requested, 'join_edges': []},
                                       dialect='sqlite'))
        self.assertFalse(scope_matches(valid, {**requested,
                'unresolved': ['unsupported_disjunction']}, dialect='sqlite'))

    def test_linear_cte_preserves_raw_joined_population(self):
        requested = {'conditions': [], 'any_conditions': [
            {'column': 'a.label', 'op': 'eq', 'value': 'target'},
            {'column': 'b.label', 'op': 'eq', 'value': 'target'}],
            'join_edges': [['a.key', 'e.left_key'], ['b.key', 'e.right_key']],
            'unresolved': []}
        query = ("WITH raw AS (SELECT e.left_key, a.label AS first_label "
                 "FROM events e JOIN labels a ON e.left_key = a.key "
                 "JOIN labels b ON e.right_key = b.key "
                 "WHERE a.label = 'target' OR b.label = 'target'), "
                 "projected AS (SELECT left_key FROM raw) "
                 "SELECT COUNT(*) FROM projected")
        self.assertTrue(scope_matches(query, requested, dialect='sqlite'))
        self.assertFalse(scope_matches(query.replace('FROM raw)',
                 "FROM raw WHERE first_label = 'other')"), requested, dialect='sqlite'))

    def test_explicit_qualified_request_preserves_two_join_roles(self):
        with tempfile.TemporaryDirectory() as root:
            model = CapturingModel()
            stamp = datetime.now(timezone.utc).isoformat()
            context = [
                {'table': 'events', 'observed_at': stamp,
                 'training_status': 'runtime_schema', 'columns': [
                     {'name': 'left_key', 'dtype': 'INTEGER'},
                     {'name': 'right_key', 'dtype': 'INTEGER'}]},
                {'table': 'labels', 'observed_at': stamp,
                 'training_status': 'runtime_schema', 'columns': [
                     {'name': 'label', 'dtype': 'TEXT'},
                     {'name': 'key', 'dtype': 'INTEGER'}]},
            ]
            runtime = GraphAnalysisRuntime(root, 'evaluation', 'qualified-scope', model,
                reference_context_loader=lambda: context)
            try:
                request = ("join e.left_key = a.key and e.right_key = b.key; "
                           "a.label = 'target' OR b.label = 'target'")
                scope = resolve_request_scope(request, runtime.context)
                self.assertEqual(scope['unresolved'], [], scope)
                self.assertEqual(scope['any_conditions'], [
                    {'column': 'a.label', 'op': 'eq', 'value': 'target'},
                    {'column': 'b.label', 'op': 'eq', 'value': 'target'}])
                self.assertEqual(scope['join_edges'], [
                    ['a.key', 'e.left_key'], ['b.key', 'e.right_key']])
                sql = ("SELECT COUNT(*) FROM events e "
                       "JOIN labels a ON e.left_key = a.key "
                       "JOIN labels b ON e.right_key = b.key "
                       "WHERE a.label = 'target' OR b.label = 'target'")
                self.assertTrue(scope_matches(sql, scope, dialect='sqlite'))
            finally:
                runtime.close()

    def test_actual_graph_stages_only_the_matching_joined_or_query(self):
        stamp = datetime.now(timezone.utc).isoformat()
        context = [
            {'table': 'events', 'observed_at': stamp, 'columns': [
                {'name': 'left_key'}, {'name': 'right_key'}]},
            {'table': 'labels', 'observed_at': stamp, 'columns': [
                {'name': 'key'}, {'name': 'label'}]},
        ]
        base = ('SELECT COUNT(*) FROM events e '
                'JOIN labels a ON e.left_key = a.key '
                'JOIN labels b ON e.right_key = b.key '
                "WHERE a.label = 'target' OR b.label = 'target'")
        for label, query, expected in (
                ('correct', base, 'awaiting_approval'),
                ('wrong_boolean', base.replace(' OR ', ' AND '), 'exhausted')):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                remote_calls = []
                runtime = GraphAnalysisRuntime(root, 'owner', label,
                    JoinedProposalModel(query=query),
                    connection_identity='test-connection',
                    remote_factory=lambda _: lambda envelope: remote_calls.append(envelope),
                    reference_context_loader=lambda: context,
                    sql_dialect='sqlite')
                try:
                    result = runtime.submit(
                        "Count records joining e.left_key = a.key and e.right_key = b.key "
                        "where a.label = 'target' OR b.label = 'target'")
                    self.assertEqual(result['status'], expected, result)
                    self.assertEqual(bool(result.get('requests')), expected == 'awaiting_approval')
                    self.assertEqual(remote_calls, [])
                finally:
                    runtime.close()
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'owner', 'missing-relationship',
                JoinedProposalModel(query=base), connection_identity='test-connection',
                remote_factory=lambda _: lambda envelope: self.fail('remote execution'),
                reference_context_loader=lambda: context, sql_dialect='sqlite')
            try:
                result = runtime.submit(
                    "Count records where a.label = 'target' OR b.label = 'target'")
                self.assertEqual(result['status'], 'blocked', result)
                self.assertFalse(result.get('requests'))
            finally:
                runtime.close()

    def test_current_schema_catches_wrong_qualified_column_inside_cte(self):
        stamp = datetime.now(timezone.utc).isoformat()
        context = [
            {'table': 'records', 'observed_at': stamp, 'columns': [
                {'name': 'left_key'}, {'name': 'right_key'}]},
            {'table': 'lookup', 'observed_at': stamp, 'columns': [
                {'name': 'key'}, {'name': 'label'}, {'name': 'location'}]},
        ]
        query = ('WITH chosen AS (SELECT a.missing FROM records r '
                 'JOIN lookup a ON r.left_key = a.key) SELECT * FROM chosen')
        error = known_column_error(query, context, dialect='sqlite')
        self.assertEqual(error['error_code'], 'unknown_column')
        self.assertIn('lookup', error['message'])
        self.assertIsNone(known_column_error(query.replace('a.missing', 'a.label'),
                                             context, dialect='sqlite'))
        self.assertIsNone(known_column_error(
            'WITH chosen AS (SELECT label AS renamed FROM lookup) SELECT renamed FROM chosen',
            context, dialect='sqlite'))
        stale = [{**item, 'observed_at': '2000-01-01T00:00:00+00:00'}
                 for item in context]
        self.assertIsNone(known_column_error(query, stale, dialect='sqlite'))


class PublicSqliteProbeTests(unittest.TestCase):
    def test_real_execution_catches_missing_function_and_preserves_database(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'fixture.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE items(value INTEGER)')
                db.execute('INSERT INTO items(value) VALUES (7)')
            query = 'SELECT ST_Y(value) FROM items'
            probe = check_sqlite_candidate(path, query)
            self.assertEqual(probe['status'], 'dialect_error')
            self.assertIn('no such function', probe['error'])
            self.assertEqual(check_sqlite_candidate(
                path, 'SELECT COUNT(*) FROM items')['status'], 'executable')
            self.assertEqual(check_sqlite_candidate(
                path, 'DELETE FROM items')['status'], 'sql_runtime_error')
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM items').fetchone(), (1,))

    def test_public_schema_examples_show_complex_value_encoding_without_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'fixture.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE lookup(item_key INTEGER, label JSONB, location POINT)')
                db.execute("INSERT INTO lookup VALUES (1, '{\"en\": \"alpha\"}', '(11.0,22.0)')")
            context = schema_context(path)
            columns = {item['name']: item for item in context[0]['columns']}
            self.assertEqual(columns['label']['top_values'], ['{"en": "alpha"}'])
            self.assertEqual(columns['location']['top_values'], ['(11.0,22.0)'])
            self.assertNotIn('top_values', columns['item_key'])

    def test_reference_document_is_not_a_user_request(self):
        with tempfile.TemporaryDirectory() as root:
            model = CapturingModel()
            runtime = GraphAnalysisRuntime(root, 'evaluation', 'separate-document', model,
                agent_instructions='Read-only proposal instructions.',
                reference_document='SUPPLIED_DOCUMENT_ONLY OR irrelevant_text')
            try:
                runtime.submit('What does the current table contain?')
                self.assertTrue(model.seen)
                seen = [message for batch in model.seen for message in batch]
                self.assertTrue(any(isinstance(message, SystemMessage)
                                    and 'SUPPLIED_DOCUMENT_ONLY' in str(message.content)
                                    for message in seen))
                self.assertFalse(any(isinstance(message, HumanMessage)
                                     and 'SUPPLIED_DOCUMENT_ONLY' in str(message.content)
                                     for message in seen))
                self.assertFalse(runtime.inspect()['requests'])
            finally:
                runtime.close()

    def test_rejected_scope_proposal_returns_reason_and_stops_identical_retry(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'evaluation', 'scope-feedback', CapturingModel())
            try:
                current, _ = runtime.recovery._state({'messages': [HumanMessage(
                    content='Use either category A or category B')]})
                current['scope'] = {'conditions': [], 'any_conditions': [],
                    'columns': [], 'unresolved': ['unsupported_disjunction']}
                current['scope_error'] = 'request_scope_unresolved'
                proposal = AIMessage(content='', tool_calls=[{'name': 'query_databricks',
                    'args': {'source': 'events', 'query': 'SELECT * FROM events',
                             'reason': 'count'}, 'id': 'attempt-1'}])
                first = runtime.recovery._reject_scope_call(current, proposal)
                feedback = first['messages'][-1].content
                self.assertIn('request_scope_unresolved', feedback)
                self.assertIn('unsupported_disjunction', feedback)
                self.assertIn('sql_dialect', feedback)
                second = runtime.recovery._reject_scope_call(current, proposal)
                self.assertEqual(second['recovery']['status'], 'blocked')
                self.assertEqual(second['recovery']['stop_reason'], 'request_scope_unresolved')
            finally:
                runtime.close()

    def test_preflight_reports_physical_source_instead_of_guessing_label(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, 'evaluation', 'source-hint', CapturingModel())
            try:
                error = runtime.recovery._proposal_preflight_error({
                    'name': 'query_databricks', 'args': {
                        'source': 'tool_name',
                        'query': 'SELECT COUNT(*) FROM events e JOIN labels l ON e.key = l.key'}})
                self.assertEqual(error['error_code'], 'invalid_source_or_sql')
                self.assertEqual(error['expected_source'], 'events | labels')
            finally:
                runtime.close()

    def test_stale_table_context_cannot_authorize_new_sql(self):
        with tempfile.TemporaryDirectory() as root:
            stale = [{'table': 'catalog.schema.items',
                      'observed_at': '2000-01-01T00:00:00+00:00',
                      'columns': [{'name': 'value', 'dtype': 'INTEGER'}]}]
            runtime = GraphAnalysisRuntime(root, 'evaluation', 'stale-preflight',
                CapturingModel(), reference_context_loader=lambda: stale)
            try:
                error = runtime.recovery._proposal_preflight_error({
                    'name':'query_databricks', 'args':{
                        'source':'items', 'query':'SELECT value FROM items'}})
                self.assertEqual(error['error_code'], 'schema_stale')
                self.assertIn('LIMIT 0', error['refresh_query'])
                probe = runtime.recovery._proposal_preflight_error({
                    'name':'query_databricks', 'args':{
                        'source':'items', 'query':'SELECT * FROM items LIMIT 0'}})
                self.assertIsNone(probe)
                runtime.datasets.register(pd.DataFrame({'value': pd.Series([], dtype='int64')}),
                    source='catalog.schema.items',
                    query='SELECT * FROM catalog.schema.items LIMIT 0',
                    coverage='complete', predicate_known=True)
                refreshed = runtime.recovery._proposal_preflight_error({
                    'name':'query_databricks', 'args':{
                        'source':'items', 'query':'SELECT value FROM items'}})
                self.assertIsNone(refreshed)
                missing = runtime.recovery._proposal_preflight_error({
                    'name':'query_databricks', 'args':{
                        'source':'items', 'query':'SELECT missing FROM items'}})
                self.assertEqual(missing['error_code'], 'unknown_column')
            finally:
                runtime.close()

    def test_dialect_failure_repairs_before_approval_without_executing(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'fixture.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE items(value INTEGER)')
                db.execute('INSERT INTO items(value) VALUES (7)')
            model = RepairingSqlModel()
            remote_calls = []
            runtime = GraphAnalysisRuntime(root, 'evaluation', 'repair-before-approval', model,
                connection_identity='public-sqlite-test',
                remote_factory=lambda _: lambda envelope: remote_calls.append(envelope),
                sql_dialect='sqlite',
                proposal_validator=lambda query: check_sqlite_candidate(path, query),
                agent_instructions='Stage one SQLite SELECT with query_databricks.')
            try:
                result = runtime.submit('Count records in items.')
                self.assertEqual(result['status'], 'awaiting_approval', result)
                self.assertEqual(model.calls, 2)
                self.assertTrue(model.saw_feedback)
                self.assertEqual(result['requests'][0]['query'], 'SELECT COUNT(*) FROM items')
                self.assertEqual(remote_calls, [])
                self.assertEqual(runtime.ledger.uncertain(), [])
            finally:
                runtime.close()


if __name__ == '__main__':
    unittest.main()
