"""Checkpointed completion evidence and bounded, approval-safe recovery."""
from copy import deepcopy
import json
import re
import time
from typing import NotRequired
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, RemoveMessage

from core.analysis_agent.failure_messages import remote_failure_message
from core.analysis_agent.memory import latest_user_request
from core.analysis_agent.intent_scope import resolve_request_scope, scope_matches, measure_scope_matches


def _dtype_family(value):
    """Classify common pandas/SQL dtype labels without table-specific facts."""
    dtype = str(value or '').strip().casefold()
    if re.search(r'(?:^|\W)(?:bool|boolean)(?:\W|$)', dtype):
        return 'categorical'
    if re.search(r'(?:int|float|double|decimal|numeric|number|real|long|short|byte)', dtype):
        return 'numeric'
    if re.search(r'(?:object|string|category|categorical|varchar|char|text|enum)', dtype):
        return 'categorical'
    return 'other'


class RecoveryState(AgentState):
    recovery: NotRequired[dict]


class RecoveryPlanningMiddleware(AgentMiddleware):
    """Cheap local transitions before summarization. Remote calls still pass HITL."""
    def __init__(self, recovery):
        self.recovery = recovery

    @hook_config(can_jump_to=['tools', 'end'])
    def before_model(self, state, runtime):
        return self.recovery.before_step(state)


class RecoveryMiddleware(AgentMiddleware):
    state_schema = RecoveryState

    def __init__(self, artifacts, diagnostics, max_attempts=2, context=None,
                 transcript=None, max_model_calls=10, max_tool_calls=12,
                 max_model_seconds=180):
        self.artifacts, self.diagnostics = artifacts, diagnostics
        self.max_attempts, self.context, self.transcript = max_attempts, context, transcript
        self.max_model_calls, self.max_tool_calls = max_model_calls, max_tool_calls
        self.max_model_seconds = max_model_seconds

    def _state(self, state):
        messages = state.get('messages', [])
        human = latest_user_request(messages)
        if human is None and self.transcript is not None:
            human = latest_user_request(self.transcript.messages())
        current = deepcopy(state.get('recovery') or {})
        request_kind = human.additional_kwargs.get('request_kind') if human else None
        text = str(human.content) if human else ''
        current_loaded_reference = bool(human and re.search(
            r'(?:현재|지금|이|그)\s*(?:로딩된|보유한|저장된)\b|현재\s*결과|보유\s*데이터', text))
        # Controller-authored table previews are data-loading requests.  They
        # are not analysis prompts even when their explanatory copy contains
        # words such as "통계".  Keep a narrow legacy match so approvals that
        # were already checkpointed before request_kind was introduced can be
        # recovered after a restart without another remote query.
        legacy_load = bool(human and re.fullmatch(
            r'데이터 구조와 예시를 살펴볼 최대 [\d,]+행입니다\. 전체 통계용 데이터가 아닙니다\.',
            str(human.content).strip()))
        data_load = request_kind == 'remote_load' or legacy_load
        if human and current.get('request_id') != human.id:
            previous = current
            fresh_source_required = bool(re.search(
                r'최신|새로\s*(?:갱신|업데이트|변경)된|현재\s*(?:원본|테이블|DB|데이터베이스)|'
                r'지금\s*(?:원본|테이블|DB|데이터베이스)|오늘\s*기준|방금\s*갱신', text, re.I)) and not current_loaded_reference
            objective_text = re.sub(r'(?:평균|중앙값|합계|최소값|최대값)\s*(?:말고|대신|아닌|아니라)', '', text)
            if re.search(r'히스토그램|\bhistogram', text, re.I): kind = 'histogram'
            elif re.search(r'산점도|산포도|\bscatter(?:\s*plot)?\b', text, re.I): kind = 'scatter'
            elif re.search(r'박스\s*플롯|상자\s*수염|상자\s*그림|\bbox\s*plot\b|\bboxplot\b', text, re.I): kind = 'boxplot'
            elif re.search(r'막대|바\s*차트|\bbar(?:\s*chart)?\b', text, re.I): kind = 'bar'
            elif re.search(r'선\s*(?:그래프|차트)|꺾은선|\bline(?:\s*chart)?\b', text, re.I): kind = 'line'
            else: kind = None
            chart = bool(kind or re.search(r'차트|시각화|그래프|\bchart|\bplot', text, re.I))
            chart_spec_requested = bool(chart and (
                kind in {'bar', 'line', 'scatter', 'boxplot'} or
                re.search(r'제목|축\s*라벨|정렬|상위\s*\d+|top\s*\d+|\bbins?\b|\d+\s*(?:개\s*)?구간|가로|세로', text, re.I)))
            profile_kind = None
            if not chart and re.search(r'결측|누락|\b(?:null|missing|nan)\b', text, re.I):
                profile_kind = 'missing'
            elif not chart and re.search(r'고유값|유니크|\b(?:distinct|unique)\b', text, re.I):
                profile_kind = 'distinct'
            elif not chart and re.search(r'기초\s*통계|요약\s*통계|데이터\s*프로파일|\b(?:describe|profile)\b', text, re.I):
                profile_kind = 'summary'
            count_request = not chart and not profile_kind and bool(re.search(
                r'건수|인원\s*수|명수|빈도|결측|고유값|(?:사람|고객|신청자|가입자|사용자|행)[\'\"]*(?:들)?(?:의)?\s*수',
                text))
            calculation = bool(re.search(
                r'평균|중앙값|합계|총합|최솟값|최댓값|최소값|최대값|표준편차|상관|비율|성공률|개수|몇\s*(?:명|개|건)|계산|통계|'
                r'[가-힣A-Za-z]+[율률]|\b(?:mean|average|avg|count|sum|median|std|correlation|ratio|rate|percentage|percent)\b', text, re.I)) or count_request
            if re.search(r'(?:이란|개념|정의|뜻).*(?:설명|알려)|(?:이란|란)\s*(?:무엇|뭐)', text):
                calculation = False
            names, sources = set(), set()
            if self.context:
                from core.analysis_catalog import grounded_reference_columns
                for table in self.context.reference_context:
                    sources.add(table.get('table', ''))
                    names.update(c['name'] for c in grounded_reference_columns(table, self.context.datasets) if c.get('name'))
                for info in self.context.datasets.metadata.values():
                    names.update(info.columns)
                    sources.add(info.source)
            def mentioned(name):
                return bool(name and re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', text))
            mentioned_columns = [name for name in names if mentioned(name)]
            mentioned_columns.sort(key=lambda name: (text.find(name) if text.find(name) >= 0 else len(text), name))
            operations = []
            for pattern, operation in [(r'평균|\b(?:mean|average|avg)\b', 'AVG'),
                    (r'합계|총합|\bsum\b', 'SUM'), (r'중앙값|\bmedian\b', 'MEDIAN'),
                    (r'개수|몇\s*(?:명|개|건)|\bcount\b', 'COUNT'),
                    (r'최솟값|최소값|최저령|\bmin\b', 'MIN'),
                    (r'최댓값|최대값|최고령|\bmax\b', 'MAX')]:
                if re.search(pattern, objective_text, re.I): operations.append(operation)
            if re.search(r'상관(?:계수|관계)?|피어슨|\b(?:correlation|pearson)\b', objective_text, re.I):
                operations.append('CORR')
            if re.search(r'비율|성공률|생존율|전환율|[가-힣A-Za-z]+[율률]|\b(?:ratio|rate|percentage|percent)\b',
                         objective_text, re.I): operations.append('RATIO')
            if count_request and 'COUNT' not in operations: operations.append('COUNT')
            if profile_kind:
                calculation, operations = False, []
            metadata_kind = None
            column_words = bool(re.search(r'컬럼|필드|\bcolumns?\b|\bfields?\b', text, re.I))
            if (not chart and not operations and column_words and
                    re.search(r'데이터\s*타입|자료형|\bdtypes?\b|\bdata\s*types?\b', text, re.I)):
                metadata_kind, calculation = 'dtypes', False
            elif (not chart and not operations and column_words and
                    re.search(r'수치형|숫자형|\bnumeric\b', text, re.I)):
                metadata_kind, calculation = 'numeric_columns', False
            elif (not chart and not operations and column_words and
                    re.search(r'문자열|범주형|카테고리형|\bcategorical\b|\bstring\b', text, re.I)):
                metadata_kind, calculation = 'categorical_columns', False
            elif (not chart and set(operations) <= {'COUNT'} and
                    column_words and
                    not re.search(r'고유|결측|중복|누락|빈도|\bnull\b|\bdistinct\b|\bmissing\b|\bunique\b', text, re.I) and
                    re.search(r'목록|개수|구조|이름|어떤|몇|전체|\blist\b|\bschema\b', text, re.I)):
                metadata_kind, calculation, operations = 'columns', False, []
            current = dict(request_id=human.id, attempts=0, chart=chart, kind=kind,
                calculation=calculation, operations=operations, metadata_kind=metadata_kind,
                profile_kind=profile_kind, chart_spec_requested=chart_spec_requested,
                data_load=data_load,
                expected_load_source=human.additional_kwargs.get('source','') if data_load else '',
                expected_load_query=human.additional_kwargs.get('query','') if data_load else '',
                whole_row_count=bool(re.search(
                    r'전체\s*(?:행|레코드|데이터)(?:의)?\s*(?:수|개수)|총\s*(?:데이터\s*)?(?:행|레코드)\s*(?:수|개수)', text)),
                current_result_only=bool(current_loaded_reference or re.search(
                    r'(?:현재|보유|지금|이|그)\s*(?:로딩된\s*)?(?:[\d,]+\s*행\s*)?(?:결과|표본|샘플|데이터)|'
                    r'표본|샘플|일부\s*데이터', text)),
                fresh_source_required=fresh_source_required,
                request_started_at=time.time(),
                required_columns=mentioned_columns,
                required_sources=sorted(s for s in sources if mentioned(s) or mentioned(s.split('.')[-1])),
                columns=[], failed={}, status='working')
            current['previous_scope'] = previous.get('scope', {})
            current['scope'] = resolve_request_scope(text, self.context, current['previous_scope'])
            if data_load:
                # The exact source and query are already bound to the approval
                # envelope.  Natural-language scope extraction from the reason
                # text would invent an analysis obligation.
                current.update(chart=False,kind=None,calculation=False,operations=[],metadata_kind=None,profile_kind=None,
                    chart_spec_requested=False,whole_row_count=False,current_result_only=False,fresh_source_required=False,
                    required_columns=[])
                current['scope']={'conditions':[],'any_conditions':[],
                    'measure_conditions':[],'ratio':None,'unresolved':[],'columns':[]}
                if current['expected_load_source']:
                    current['required_sources']=[current['expected_load_source']]
            if (previous.get('status') == 'complete' and not metadata_kind and
                    re.search(r'그중|같은|아까|앞선|이어서|말고|대신|바꿔', text) and
                    (calculation or chart or re.search(r'그중|말고|대신|바꿔|조건', text))):
                # Keep the operation obligation for an elliptical follow-up, but
                # require fresh evidence. Explicit replacement operations win.
                current['calculation'] = calculation or previous.get('calculation', False)
                current['chart'] = chart or (previous.get('chart', False) and not calculation)
                if not operations and current['calculation']: current['operations'] = previous.get('operations', [])
                if not current['required_columns']: current['required_columns'] = previous.get('required_columns', [])
                if not current['required_sources']: current['required_sources'] = previous.get('required_sources', [])
                if current['chart'] and not kind: current['kind'] = previous.get('kind')
                if current['chart'] and re.search(r'제목|축|라벨|정렬|상위|top|bin|구간|가로|세로|바꿔|수정', text, re.I):
                    current['chart_spec_requested'] = True
        upgraded_current_result = bool(human and current.get('request_id') == human.id
            and current_loaded_reference and not current.get('current_result_only'))
        if upgraded_current_result:
            # Classification rules can improve while a checkpoint is paused.
            # Reconcile the saved request from its original user text so a
            # restart can adopt already-persisted local evidence without a new
            # model or remote query.
            current['current_result_only'] = True
            current['fresh_source_required'] = False
        elif data_load and not current.get('data_load'):
            # Upgrade an in-flight legacy checkpoint.  Its successful tool
            # observation may already be marked processed, so the loop below
            # is also allowed to reconsider that single observation.
            current.update(data_load=True,chart=False,kind=None,calculation=False,
                operations=[],metadata_kind=None,profile_kind=None,chart_spec_requested=False,whole_row_count=False,
                current_result_only=False,fresh_source_required=False)
            current['scope']={'conditions':[],'any_conditions':[],
                'measure_conditions':[],'ratio':None,'unresolved':[],'columns':[]}
        if human and ('scope' not in current or current['scope'].get('unresolved')):
            # Metadata discovery may resolve an ambiguous column. Re-read only
            # the original human request, never model-generated SQL or summaries.
            current['scope'] = resolve_request_scope(str(human.content), self.context, current.get('previous_scope'))
        for key, default in [('processed', []), ('sent_calls', []), ('failed_signatures', {}),
                ('evidence_ids', []), ('artifact_ids', []), ('failed', {}), ('attempts', 0),
                ('model_calls', 0), ('model_seconds', 0.0), ('columns', [])]:
            current.setdefault(key, default)
        start = next((i for i, m in enumerate(messages) if human and m.id == human.id), 0)
        # Tool-call de-duplication is request scoped.  Reusing the same safe
        # inspection in a later user turn must not be blocked by an identical
        # call that completed an earlier turn in the conversation.
        request_messages = messages[start:]
        calls = {c['id']: c for m in request_messages if isinstance(m, AIMessage) for c in m.tool_calls}
        if human and human.additional_kwargs.get('selected_card') in self.artifacts:
            card = self.artifacts[human.additional_kwargs['selected_card']]
            if self._valid_card(card, current, card.dataset_id): current['artifact_ids'] = [card.id]
        for message in request_messages:
            if not isinstance(message, ToolMessage):
                continue
            call = calls.get(message.tool_call_id, {})
            name, arguments = message.name or call.get('name'), call.get('args', {})
            try:
                observation = json.loads(message.content)
            except (ValueError, TypeError):
                observation = {'status': 'error', 'error_code': 'invalid_tool_observation'}
                if name == 'query_databricks' and 'rejected' in str(message.content).lower():
                    current['remote_rejected'] = True
            if not isinstance(observation, dict): continue
            reconsider_load = bool(current.get('data_load') and not current.get('load_evidence_id')
                and name == 'query_databricks')
            reconsider_calculation = bool(upgraded_current_result and not current.get('evidence_ids')
                and name == 'local_analysis_sql' and observation.get('status') == 'ready')
            if message.tool_call_id in current['processed'] and not reconsider_load and not reconsider_calculation:
                continue
            if message.tool_call_id not in current['processed']:
                current['processed'].append(message.tool_call_id)
            failed = observation.get('status') in {'error', 'needs_data', 'needs_context', 'needs_refresh', 'unavailable', 'no_valid_chart'}
            if failed:
                current['failed'][name or message.tool_call_id] = observation
                signature = self._signature(call or {'name': name, 'args': {}})
                current['failed_signatures'][signature] = current['failed_signatures'].get(signature, 0) + 1
            else:
                current['failed'].pop(name, None)
                if name == 'inspect_table_context' and not current.get('chart') and not current.get('calculation'):
                    current['failed'].pop('inspect_dataset', None)
                    context = observation.get('table_context', {})
                    if current.get('metadata_kind') in {'columns', 'dtypes', 'numeric_columns', 'categorical_columns'} and isinstance(context.get('columns'), list):
                        expected = current.get('required_sources', [])
                        if not expected or self._source_key(context.get('table','')) in {self._source_key(s) for s in expected}:
                            schema = [{'name':c.get('name'), 'dtype':str(c.get('dtype') or '')}
                                      for c in context['columns'] if c.get('name')]
                            kind = current.get('metadata_kind')
                            # Exhaustive type/subset answers require a dtype for
                            # every column; an incomplete snapshot is not proof.
                            if kind == 'columns' or (schema and all(c['dtype'] for c in schema)):
                                selected = ([c['name'] for c in schema if _dtype_family(c['dtype']) == 'numeric']
                                            if kind == 'numeric_columns' else
                                            [c['name'] for c in schema if _dtype_family(c['dtype']) == 'categorical']
                                            if kind == 'categorical_columns' else [])
                                current['metadata_evidence'] = {'table':context.get('table'),
                                    'kind':kind, 'columns':[c['name'] for c in schema],
                                    'schema':schema, 'selected_columns':selected,
                                    'authority':observation.get('authority'),
                                    'scope':observation.get('scope'),
                                    'schema_changed':observation.get('schema_changed', False)}
            if name == 'prepare_histogram' and observation.get('histogram_plan'):
                plan = observation['histogram_plan']
                if not self._scope_valid(plan.get('query', ''), current, plan.get('value_column')):
                    current['failed'][name] = {'status':'error', 'error_code':'request_scope_mismatch'}
                    continue
                current.update(chart=True, kind='histogram', plan=observation['histogram_plan'],
                    plan_call=message.tool_call_id, columns=[observation['histogram_plan']['value_column']])
                current['artifact_ids'] = []
                current.pop('loaded_dataset', None)
                if observation.get('status') == 'ready' and observation.get('loaded_dataset'):
                    current.update(query_seen=message.tool_call_id, loaded_dataset=observation['loaded_dataset'])
            if name == 'query_databricks' and current.get('plan') and arguments.get('query') == current['plan']['query']:
                current['query_seen'] = current['plan_call']
                if observation.get('status') == 'ready':
                    current['loaded_dataset'] = observation['dataset']['id']
            if name == 'query_databricks' and current.get('data_load') and observation.get('status') == 'ready':
                dataset_id = observation.get('dataset', {}).get('id')
                info = self.context.datasets.metadata.get(dataset_id) if self.context and dataset_id else None
                expected_query=current.get('expected_load_query')
                expected_source=current.get('expected_load_source')
                query_matches=not expected_query or arguments.get('query') == expected_query
                source_matches=not expected_source or self._source_key(arguments.get('source','')) == self._source_key(expected_source)
                if info is not None and query_matches and source_matches and self._source_matches(info,current):
                    current.update(load_evidence_id=dataset_id,loaded_dataset=dataset_id,
                        columns=list(info.columns),status='working')
                    current['failed'].pop(name,None)
                    self.diagnostics.emit('data_load_evidence',request_id=current.get('request_id'),
                        dataset_id=dataset_id,source=info.source,rows=info.rows,columns=len(info.columns))
            if name == 'profile_dataset' and observation.get('status') == 'ready':
                dataset_id = arguments.get('dataset_id') or observation.get('dataset_id')
                info = self.context.datasets.metadata.get(dataset_id) if self.context and dataset_id else None
                profiled = {column.get('name') for column in observation.get('profile', {}).get('columns', [])}
                if (info is not None and self._source_matches(info, current)
                        and self._fresh_for_request(info, current)
                        and set(current.get('required_columns', [])).issubset(profiled)
                        and (current.get('current_result_only')
                             or (info.coverage == 'complete' and info.predicate_known))
                        and self._scope_valid(info, current)):
                    current['profile_evidence'] = observation
                    current['failed'].pop(name, None)
            if name in {'local_analysis_sql', 'query_databricks'} and observation.get('status') == 'ready':
                dataset_id = observation.get('dataset', {}).get('id')
                if self._valid_calculation(dataset_id, arguments, current):
                    current['evidence_ids'].append(dataset_id)
            if name in {'recommend_chart_images', 'render_chart_spec', 'render_histogram', 'prepare_histogram', 'show_chart'} and observation.get('cards'):
                current['chart'] = True
                dataset_id = arguments.get('dataset_id') or observation.get('loaded_dataset')
                valid = []
                for entry in observation['cards']:
                    try: card = self.artifacts[entry['id']]
                    except (KeyError, TypeError): continue
                    candidate_dataset = card.dataset_id if name == 'show_chart' else dataset_id
                    if self._valid_card(card, current, candidate_dataset): valid.append(card.id)
                if valid:
                    current['artifact_ids'] = valid
                    for tool in ('recommend_chart_images', 'render_chart_spec', 'render_histogram', 'prepare_histogram', 'show_chart'):
                        current['failed'].pop(tool, None)
        return current, calls

    @staticmethod
    def _signature(call):
        return json.dumps([call.get('name'), call.get('args', {})], sort_keys=True, default=str)

    def _source_matches(self, info, current):
        expected = current.get('required_sources') or []
        return not expected or self._source_key(info.source) in {self._source_key(s) for s in expected}

    @staticmethod
    def _has_scope(current):
        scope = current.get('scope', {})
        return bool(scope.get('conditions') or scope.get('any_conditions') or scope.get('unresolved'))

    def _scope_valid(self, executed, current, histogram_column=None):
        if not self._has_scope(current): return True
        if scope_matches(executed, current['scope'], histogram_column=histogram_column): return True
        current['scope_error'] = 'request_scope_unresolved' if current['scope'].get('unresolved') else 'request_scope_mismatch'
        self.diagnostics.emit('request_scope_rejected', request_id=current.get('request_id'),
            reason=current['scope_error'], columns=sorted({c['column'] for c in current['scope'].get('conditions', [])}),
            unresolved=current['scope'].get('unresolved', []))
        return False

    @staticmethod
    def _source_key(source):
        import sqlglot
        from utils.analysis_provenance import single_table, table_identity
        try:
            table = single_table(sqlglot.parse_one('SELECT * FROM ' + source, read='databricks'))
            return table_identity(table).casefold() if table is not None else source
        except (ValueError, TypeError, sqlglot.errors.SqlglotError):
            return source

    def _valid_card(self, card, current, dataset_id):
        if not card.image.startswith(b'\x89PNG\r\n\x1a\n'): return False
        if card.dataset_id != dataset_id: return False
        if current.get('kind') and card.kind != current['kind']: return False
        expected = current.get('required_columns') or current.get('columns', [])
        # Predicate columns restrict the rows; they need not be chart axes.
        if self._has_scope(current):
            filters = {c['column'] for c in (current['scope'].get('conditions', [])
                       + current['scope'].get('any_conditions', []))}
            expected = [column for column in expected if column not in filters]
        if expected and not set(expected).issubset(card.columns): return False
        if self.context:
            info = self.context.datasets.metadata.get(card.dataset_id)
            if info is None or not self._source_matches(info, current): return False
            if not self._fresh_for_request(info, current): return False
            value_column = card.columns[0] if card.kind == 'histogram' and len(card.columns) == 1 else None
            if not self._scope_valid(info, current, value_column): return False
            if info.coverage != 'complete' and not current.get('current_result_only'): return False
            if current.get('plan'):
                if self._source_key(info.source) != self._source_key(current['plan']['source']) or card.dataset_id != current.get('loaded_dataset'):
                    return False
            elif not current.get('current_result_only'):
                if info.coverage != 'complete': return False
                if info.grain == 'aggregate' and card.kind == 'histogram':
                    # A count distribution is not raw-predicate lineage. A direct
                    # whole-source count histogram can still prove its own scope.
                    import sqlglot
                    from sqlglot import exp
                    from utils.analysis_charts import validate_frequency_dataset
                    from utils.analysis_provenance import count_frequency_columns
                    try:
                        tree = sqlglot.parse_one(info.query, read='duckdb' if info.parent_id else 'databricks')
                        columns = count_frequency_columns(tree)
                        if not columns: return False
                        validate_frequency_dataset(self.context.datasets, info.id, *columns)
                        where = tree.args.get('where')
                        if where and not self._has_scope(current):
                            predicate = where.this
                            if not (isinstance(predicate, exp.Not) and isinstance(predicate.this, exp.Is)
                                    and isinstance(predicate.this.this, exp.Column) and predicate.this.this.name == columns[0]
                                    and isinstance(predicate.this.expression, exp.Null)): return False
                        if info.conditions and not self._has_scope(current): return False
                    except (ValueError, TypeError): return False
                elif (info.conditions and not self._has_scope(current)) or not info.predicate_known:
                    return False
        return True

    def _valid_calculation(self, dataset_id, arguments, current):
        if not self.context or dataset_id not in self.context.datasets.metadata: return False
        info = self.context.datasets.metadata[dataset_id]
        if not self._source_matches(info, current): return False
        if not self._fresh_for_request(info, current): return False
        if not self._scope_valid(info, current): return False
        if not measure_scope_matches(info.query, current.get('scope', {})): return False
        if arguments.get('current_result_only') and not current.get('current_result_only'): return False
        if info.coverage != 'complete' and not current.get('current_result_only'): return False
        if not info.query: return False
        try:
            import sqlglot
            from sqlglot import exp
            tree = sqlglot.parse_one(info.query, read='duckdb' if info.parent_id else 'databricks')
            operations = {node.sql_name() for node in tree.find_all(exp.AggFunc)}
            requested_operations = set(current.get('operations', []))
            if not (requested_operations - {'RATIO'}).issubset(operations): return False
            ratio = current.get('scope', {}).get('ratio') or {}
            if ('RATIO' in requested_operations and not tree.find(exp.Div)
                    and ratio.get('aggregation') != 'mean_zero_one'): return False
            columns = {c.name for c in tree.find_all(exp.Column)}
            condition_columns = {condition.column for condition in info.conditions}
            if not set(current.get('required_columns', [])).issubset(columns | set(info.columns) | condition_columns): return False
            if current.get('calculation') and not operations and not tree.find(exp.Div): return False
        except (ValueError, TypeError):
            return False
        self.diagnostics.emit('calculation_evidence', request_id=current.get('request_id'),
            dataset_id=dataset_id, source=info.source, coverage=info.coverage, grain=info.grain)
        return True

    def _complete(self, current):
        if current.get('data_load'): return bool(current.get('load_evidence_id'))
        if current.get('metadata_kind') and not current.get('metadata_evidence'): return False
        if current.get('profile_kind') and not current.get('profile_evidence'): return False
        if current.get('chart') and not current['artifact_ids']: return False
        if current.get('calculation') and not current['evidence_ids']: return False
        if current.get('chart') or current.get('calculation'): return True
        return not current['failed']

    def _answer(self, current):
        parts = []
        if current.get('data_load') and current.get('load_evidence_id') and self.context:
            info=self.context.datasets.metadata[current['load_evidence_id']]
            parts.append(f'승인한 조회로 {info.source} 데이터 {info.rows:,}행, {len(info.columns):,}열을 불러와 저장했습니다. '
                '이 결과는 구조와 예시 확인용 범위이며 전체 통계로 간주하지 않습니다.\n'
                +'컬럼: '+', '.join(info.columns))
        if current.get('metadata_evidence'):
            metadata = current['metadata_evidence']
            origin = ('승인 후 로딩된 실제 결과' if metadata.get('authority') == 'approved_select_star_result'
                      else '확인된 스키마 스냅샷')
            changed = ' 이전 스냅샷과 컬럼 구성이 달라 새 스키마를 사용했습니다.' if metadata.get('schema_changed') else ''
            kind = metadata.get('kind', 'columns')
            if kind == 'dtypes':
                parts.append(f"{origin} 기준으로 {metadata['table']}의 컬럼별 데이터 타입입니다.{changed}\n"
                             + '\n'.join(f"{column['name']}: {column['dtype']}" for column in metadata['schema']))
            elif kind in {'numeric_columns', 'categorical_columns'}:
                label = '수치형' if kind == 'numeric_columns' else '문자열/범주형'
                selected = metadata.get('selected_columns', [])
                parts.append(f"{origin} 기준으로 {metadata['table']}의 {label} 컬럼은 {len(selected)}개입니다.{changed}\n"
                             + (', '.join(selected) if selected else '해당 컬럼이 없습니다.'))
            else:
                parts.append(f"{origin} 기준으로 {metadata['table']}에는 컬럼이 {len(metadata['columns'])}개 있습니다.{changed}\n"
                             + ', '.join(metadata['columns']))
        if current.get('profile_evidence'):
            evidence = current['profile_evidence']
            profile = evidence['profile']
            kind = current.get('profile_kind')
            columns = profile.get('columns', [])
            parts.append(f"보유 데이터 프로파일 결과입니다. 출처: {profile.get('source')}\n분석 범위: {evidence.get('scope')}")
            if kind == 'missing':
                parts.append('\n'.join(
                    f"{column['name']}: 결측 {column['null_count']:,}건 ({column['null_ratio_pct']}%)"
                    for column in columns))
            elif kind == 'distinct':
                parts.append('\n'.join(
                    f"{column['name']}: 고유값 {column['distinct_count']:,}개"
                    for column in columns))
            else:
                lines = [f"행 {profile.get('rows', 0):,}개, 컬럼 {profile.get('column_count', 0):,}개"]
                for column in columns:
                    summary = column.get('numeric_summary')
                    if summary:
                        lines.append(
                            f"{column['name']}: 평균 {summary['mean']}, 중앙값 {summary['median']}, "
                            f"최솟값 {summary['min']}, 최댓값 {summary['max']}, 결측 {column['null_count']:,}건")
                    else:
                        lines.append(
                            f"{column['name']}: 고유값 {column['distinct_count']:,}개, 결측 {column['null_count']:,}건")
                parts.append('\n'.join(lines))
            if profile.get('column_page', {}).get('has_more'):
                parts.append('컬럼이 많아 이번 응답에는 일부 컬럼만 포함했습니다.')
        for card_id in current.get('artifact_ids', []):
            card = self.artifacts[card_id]
            parts.append(f'{card.title} 이미지를 생성했습니다.\n분석 범위: {card.scope}')
        if current.get('calculation') and current.get('evidence_ids'):
            info = self.context.datasets.metadata[current['evidence_ids'][-1]]
            frame = self.context.datasets.frames[info.id]
            scope = '요청 조건에 포함된 보유 데이터 전체' if info.coverage == 'complete' else '현재 보유한 일부 데이터'
            parts.append(f'보유 데이터로 계산한 결과입니다. 출처: {info.source}\n분석 범위: {scope}')
            requested_conditions=current.get('scope',{}).get('conditions',[])
            any_conditions=current.get('scope',{}).get('any_conditions',[])
            if requested_conditions or any_conditions:
                ops = {'eq':'=', 'ne':'≠', 'gt':'>', 'ge':'≥', 'lt':'<', 'le':'≤', 'in':'포함'}
                conjunction=' AND '.join(f"{c['column']} {ops[c['op']]} {c['value']}" for c in requested_conditions)
                disjunction=' OR '.join(f"{c['column']} {ops[c['op']]} {c['value']}" for c in any_conditions)
                rendered=' AND '.join(item for item in (conjunction, '('+disjunction+')' if disjunction else '') if item)
                parts.append('적용 조건: '+rendered)
            # Never publish unchecked numbers from the model as computed results.
            if frame.shape == (1, 1):
                labels = {'AVG':'평균', 'MEDIAN':'중앙값', 'SUM':'합계', 'COUNT':'건수', 'MIN':'최솟값', 'MAX':'최댓값', 'CORR':'피어슨 상관계수',
                          'RATIO':'비율(%)'}
                operations = current.get('operations', [])
                label = labels.get(operations[0], frame.columns[0]) if len(operations) == 1 else frame.columns[0]
                parts.append(f'{label}: {frame.iloc[0, 0]}')
            else:
                parts.append('```csv\n' + frame.head(15).to_csv(index=False).strip() + '\n```')
            if len(frame) > 15: parts.append(f'총 {len(frame)}행 중 앞 15행입니다. 전체 결과는 저장된 데이터에서 확인할 수 있습니다.')
        return '\n\n'.join(parts)

    def _limit_reason(self, current):
        if current['model_calls'] >= self.max_model_calls: return 'model_call_budget'
        if len(current['sent_calls']) >= self.max_tool_calls: return 'tool_call_budget'
        if current['model_seconds'] >= self.max_model_seconds: return 'model_time_budget'
        if any(count >= 2 for count in current['failed_signatures'].values()): return 'repeated_failed_tool'
        return None

    def _finish(self, current, last=None, reason=None):
        success = reason is None
        remote_block = current.get('remote_rejected') or any(o.get('status') == 'unavailable' for o in current['failed'].values())
        current['status'] = 'complete' if success else ('blocked' if remote_block else 'exhausted')
        current['stop_reason'] = reason
        text = self._answer(current) if success else remote_failure_message(current['failed'].values(), current.get('remote_rejected', False))
        if not success and not remote_block and reason not in {'missing_evidence', 'unresolved_failure'}:
            text = '반복 실행 한도에 도달해 분석을 중단했습니다. 검증된 완료 결과가 없으며 기존 데이터는 보존했습니다.'
        if not success and not remote_block and current.get('scope_error'):
            if current.get('scope', {}).get('unresolved'):
                current['status'] = 'blocked'
                current['stop_reason'] = 'request_scope_unresolved'
                text = '요청한 기간·조건을 저장된 컬럼 정보와 확정적으로 연결하지 못했습니다. 사용할 날짜 컬럼과 조건을 컬럼명·값으로 명시해주세요. 분석을 완료한 것으로 처리하지 않았으며 기존 데이터는 보존했습니다.'
            else:
                current['stop_reason'] = 'request_scope_mismatch'
                text = '생성된 분석의 기간·필터가 요청 조건과 일치하지 않아 결과를 채택하지 않았습니다. 조건을 유지한 복구가 실행 한도 안에 완료되지 않았습니다. 기존 데이터는 보존했습니다.'
        kwargs = {'analysis_status': 'answered' if success else current['status']}
        message = (last.model_copy(update={'content': text or last.content, 'tool_calls': [],
                   'additional_kwargs': {**last.additional_kwargs, **kwargs}}) if last else
                   AIMessage(content=text, additional_kwargs=kwargs))
        self.diagnostics.emit('completion_checked', request_id=current.get('request_id'),
            status=current['status'], reason=reason, attempts=current['attempts'],
            model_calls=current['model_calls'], tool_calls=len(current['sent_calls']),
            model_seconds=round(current['model_seconds'], 3),
            dataset_ids=current['evidence_ids'], artifact_ids=current['artifact_ids'])
        return {'recovery': current, 'messages': [message]}

    def _next_local(self, current, calls):
        if (self.context and current.get('profile_kind') and not current.get('profile_evidence')
                and not self._has_scope(current)):
            required = set(current.get('required_columns', []))
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and required.issubset(info.columns)
                and self._source_matches(info, current)
                and self._fresh_for_request(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known))]
            if len(candidates) == 1:
                arguments = {'dataset_id': candidates[0].id}
                if required:
                    arguments['columns'] = sorted(required)
                elif current.get('profile_kind') == 'distinct':
                    frame = self.context.datasets.frames[candidates[0].id]
                    categorical = [str(column) for column in frame.columns
                                   if _dtype_family(frame[column].dtype) == 'categorical']
                    if not categorical:
                        return None
                    arguments['columns'] = categorical
                if not any(c.get('name') == 'profile_dataset' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name':'profile_dataset', 'args':arguments}
        if (self.context and current.get('metadata_kind') in {'columns', 'dtypes', 'numeric_columns', 'categorical_columns'}
                and not current.get('metadata_evidence')):
            # A schema request has one safe local action when its source is
            # unambiguous. The inspection tool applies freshness/schema-drift
            # policy and can return needs_refresh; it never queries Databricks.
            known = []
            for item in self.context.reference_context:
                source = item.get('table', '')
                if source and source not in known:
                    known.append(source)
            for info in self.context.datasets.metadata.values():
                if info.source and info.source not in known:
                    known.append(info.source)
            requested = current.get('required_sources', [])
            if requested:
                wanted = {self._source_key(source) for source in requested}
                candidates = [source for source in known if self._source_key(source) in wanted]
            else:
                candidates = known
            if len(candidates) == 1:
                arguments = {'table': candidates[0]}
                from core.analysis_catalog import resolve_table_context
                inspection = resolve_table_context(
                    self.context.reference_context, self.context.datasets, candidates[0])
                if (inspection.get('status') == 'ready'
                        and not any(c.get('name') == 'inspect_table_context' and c.get('args') == arguments
                                    for c in calls.values())):
                    return {'name':'inspect_table_context', 'args':arguments}
        if current.get('plan') and current.get('loaded_dataset') and not current['artifact_ids']:
            arguments = {'dataset_id': current['loaded_dataset'], 'value_column': current['plan']['value_column'],
                         'weight_column': current['plan']['weight_column']}
            if not any(c.get('name') == 'render_histogram' and c.get('args') == arguments for c in calls.values()):
                return {'name': 'render_histogram', 'args': arguments}
        if (self.context and current.get('chart') and current.get('kind') in {'bar', 'line', 'scatter', 'boxplot'}
                and not current.get('fresh_source_required') and not current.get('artifact_ids')
                and not self._has_scope(current)):
            columns = current.get('required_columns', [])
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and set(columns).issubset(info.columns)
                and self._source_matches(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known
                        and self._fresh_for_request(info, current)))]
            if len(candidates) == 1:
                frame = self.context.datasets.frames[candidates[0].id]
                from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
                arguments = None
                if current['kind'] == 'boxplot' and len(columns) == 1 and is_numeric_dtype(frame[columns[0]]):
                    arguments = {'dataset_id':candidates[0].id,'kind':'boxplot','x':columns[0]}
                elif current['kind'] == 'scatter' and len(columns) == 2 and all(
                        is_numeric_dtype(frame[column]) for column in columns):
                    arguments = {'dataset_id':candidates[0].id,'kind':'scatter','x':columns[0],'y':columns[1]}
                elif current['kind'] == 'bar' and len(columns) == 1 and 1 <= frame[columns[0]].nunique() <= 50:
                    arguments = {'dataset_id':candidates[0].id,'kind':'bar','x':columns[0],
                                 'aggregation':'count','sort':'descending','top_n':50}
                elif current['kind'] == 'line' and len(columns) == 2:
                    time_columns = [column for column in columns if is_datetime64_any_dtype(frame[column])]
                    numeric_columns = [column for column in columns if is_numeric_dtype(frame[column])]
                    if len(time_columns) == 1 and len(numeric_columns) == 1 and frame[time_columns[0]].is_unique:
                        arguments = {'dataset_id':candidates[0].id,'kind':'line',
                                     'x':time_columns[0],'y':numeric_columns[0],'sort':'ascending'}
                if arguments and not any(c.get('name') == 'render_chart_spec' and c.get('args') == arguments
                                         for c in calls.values()):
                    return {'name':'render_chart_spec','args':arguments}
        scope = current.get('scope', {})
        if (self.context and current.get('chart') and current.get('kind') == 'histogram'
                and not current.get('fresh_source_required') and not current.get('chart_spec_requested')
                and not current.get('plan')
                and not current.get('artifact_ids') and not self._has_scope(current)
                and len(current.get('required_columns', [])) == 1):
            column=current['required_columns'][0]
            candidates=[info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and column in info.columns
                and self._source_matches(info,current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known
                        and self._fresh_for_request(info,current)))]
            if len(candidates) == 1:
                arguments={'source':candidates[0].source,'column':column,
                    'where_sql':'','current_result_only':bool(current.get('current_result_only'))}
                if not any(c.get('name') == 'prepare_histogram' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name':'prepare_histogram','args':arguments}
        if (self.context and current.get('calculation') and current.get('operations') == ['COUNT']
                and (scope.get('conditions') or scope.get('any_conditions') or current.get('whole_row_count'))
                and not scope.get('unresolved')
                and set(scope.get('columns', [])).issubset(
                    {item['column'] for item in (scope.get('conditions', [])+scope.get('any_conditions', []))})
                and not current.get('current_result_only') and not current.get('evidence_ids')):
            # A grounded filtered row count has one deterministic local plan.
            # This supplies a reliable fallback when a language model explains
            # the schema but fails to call the calculation tool. Stay
            # conservative: dispatch only when exactly one complete raw dataset
            # can satisfy every requested predicate.
            predicate_columns = {item['column'] for item in (
                scope.get('conditions', [])+scope.get('any_conditions', []))}
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.coverage == 'complete' and info.grain == 'raw' and info.predicate_known
                and predicate_columns.issubset(info.columns) and self._source_matches(info, current)
                and self._fresh_for_request(info, current)]
            # A model may first materialize the exact requested subset with
            # use_dataset. In that case both the source frame and its filtered
            # child are reusable candidates. Prefer the one whose persisted
            # predicates already prove the requested scope; otherwise the
            # apparent ambiguity needlessly sends the request back to the model.
            exact = [info for info in candidates if scope_matches(info, scope)]
            if len(exact) == 1:
                candidates = exact
            if len(candidates) == 1:
                any_conditions=scope.get('any_conditions', [])
                if any_conditions:
                    pieces=[]
                    if scope.get('conditions'): pieces.append(self._where_sql(scope['conditions']))
                    pieces.append('('+ ' OR '.join(self._where_sql([item]) for item in any_conditions)+')')
                    arguments={'dataset_id':candidates[0].id,
                        'query':'SELECT COUNT(*) AS count FROM data WHERE '+' AND '.join(pieces)}
                else:
                    arguments = {'dataset_id': candidates[0].id,
                        'query': 'SELECT COUNT(*) AS count FROM data',
                        'requested_conditions': deepcopy(scope.get('conditions', []))}
                if not any(c.get('name') == 'local_analysis_sql' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name': 'local_analysis_sql', 'args': arguments}
        if (self.context and current.get('calculation')
                and set(current.get('operations', [])) in ({'RATIO'}, {'COUNT', 'RATIO'})
                and not current.get('current_result_only') and not current.get('evidence_ids')):
            ratio = scope.get('ratio') or {}
            measure = scope.get('measure_conditions') or []
            if (ratio.get('column') and len(measure) == 1 and measure[0].get('op') == 'eq'
                    and measure[0].get('column') == ratio['column'] and not scope.get('unresolved')):
                population_mentions = set(scope.get('columns', [])) - {ratio['column']}
                population_conditions = {item['column'] for item in (
                    scope.get('conditions', [])+scope.get('any_conditions', []))}
                if not population_mentions.issubset(population_conditions):
                    return None
                needed = {item['column'] for item in (
                    scope.get('conditions', [])+scope.get('any_conditions', []))} | {ratio['column']}
                candidates = [info for info in self.context.datasets.metadata.values()
                    if info.coverage == 'complete' and info.grain == 'raw' and info.predicate_known
                    and needed.issubset(info.columns) and self._source_matches(info, current)
                    and self._fresh_for_request(info, current)]
                if len(candidates) == 1:
                    column = '"' + ratio['column'].replace('"', '""') + '"'
                    value = measure[0]['value']
                    if isinstance(value, bool): literal = 'TRUE' if value else 'FALSE'
                    elif isinstance(value, (int, float)): literal = repr(value)
                    else: literal = "'" + str(value).replace("'", "''") + "'"
                    prefix = 'COUNT(*) AS count, ' if 'COUNT' in current.get('operations', []) else ''
                    if ratio.get('aggregation') == 'mean_zero_one':
                        query = f'SELECT {prefix}100.0 * AVG({column}) AS percent FROM data'
                    else:
                        query = (f'SELECT {prefix}100.0 * SUM(CASE WHEN {column} = {literal} THEN 1 ELSE 0 END) '
                                 f'/ NULLIF(COUNT(*), 0) AS percent FROM data')
                    if scope.get('any_conditions'):
                        pieces=[]
                        if scope.get('conditions'):pieces.append(self._where_sql(scope['conditions']))
                        pieces.append('('+ ' OR '.join(self._where_sql([item])
                                      for item in scope['any_conditions'])+')')
                        query += ' WHERE '+' AND '.join(pieces)
                        arguments={'dataset_id':candidates[0].id,'query':query}
                    else:
                        arguments = {'dataset_id':candidates[0].id, 'query':query,
                                     'requested_conditions':deepcopy(scope.get('conditions', []))}
                    if not any(c.get('name') == 'local_analysis_sql' and c.get('args') == arguments
                               for c in calls.values()):
                        return {'name':'local_analysis_sql', 'args':arguments}
        if (self.context and current.get('calculation')
                and current.get('operations') == ['CORR'] and not self._has_scope(current)
                and not current.get('evidence_ids')):
            columns = current.get('required_columns', [])
            if len(columns) != 2:
                return None
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and (current.get('current_result_only') or info.predicate_known)
                and set(columns).issubset(info.columns) and self._source_matches(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and self._fresh_for_request(info, current)))]
            numeric_candidates = []
            from pandas.api.types import is_numeric_dtype
            for info in candidates:
                try:
                    frame = self.context.datasets.frames[info.id]
                    if all(is_numeric_dtype(frame[column]) for column in columns):
                        numeric_candidates.append(info)
                except (KeyError, OSError, ValueError, TypeError):
                    continue
            if len(numeric_candidates) == 1:
                quoted = ['"' + column.replace('"', '""') + '"' for column in columns]
                arguments = {'dataset_id':numeric_candidates[0].id,
                    'query':f'SELECT CORR({quoted[0]}, {quoted[1]}) AS correlation FROM data'}
                if current.get('current_result_only'):
                    arguments['current_result_only'] = True
                if not any(c.get('name') == 'local_analysis_sql' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name':'local_analysis_sql', 'args':arguments}
        aggregate_operations = current.get('operations', [])
        supported_aggregates = {'AVG', 'MEDIAN', 'SUM', 'MIN', 'MAX'}
        if (self.context and current.get('calculation') and len(aggregate_operations) > 1
                and set(aggregate_operations).issubset(supported_aggregates)
                and not self._has_scope(current) and not current.get('evidence_ids')):
            # Multiple descriptive aggregates over one numeric column have a
            # single deterministic local plan. This prevents a slow model turn
            # from dropping one requested statistic (for example average age
            # plus oldest age) when the complete frame is already available.
            columns = current.get('required_columns') or scope.get('columns', [])
            if len(columns) != 1:
                return None
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and (current.get('current_result_only') or info.predicate_known)
                and columns[0] in info.columns and self._source_matches(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and self._fresh_for_request(info, current)))]
            from pandas.api.types import is_numeric_dtype
            numeric_candidates = []
            for info in candidates:
                try:
                    if is_numeric_dtype(self.context.datasets.frames[info.id][columns[0]]):
                        numeric_candidates.append(info)
                except (KeyError, OSError, ValueError, TypeError):
                    continue
            if len(numeric_candidates) == 1:
                quoted = '"' + columns[0].replace('"', '""') + '"'
                aliases = {'AVG':'average', 'MEDIAN':'median', 'SUM':'sum',
                           'MIN':'minimum', 'MAX':'maximum'}
                projections = [f'{operation}({quoted}) AS {aliases[operation]}'
                               for operation in aggregate_operations]
                arguments = {'dataset_id':numeric_candidates[0].id,
                    'query':'SELECT ' + ', '.join(projections) + ' FROM data'}
                if current.get('current_result_only'):
                    arguments['current_result_only'] = True
                if not any(c.get('name') == 'local_analysis_sql' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name':'local_analysis_sql', 'args':arguments}
        return None

    @staticmethod
    def _where_sql(conditions):
        ops = {'eq':'=', 'ne':'<>', 'gt':'>', 'ge':'>=', 'lt':'<', 'le':'<='}
        parts = []
        for item in conditions:
            # This helper only builds DuckDB SQL for local_analysis_sql.
            column = '"' + item['column'].replace('"', '""') + '"'
            value = item['value']
            values = value if item['op'] == 'in' and isinstance(value, (list, tuple)) else [value]
            literals = []
            for entry in values:
                if entry is None: literals.append('NULL')
                elif isinstance(entry, bool): literals.append('TRUE' if entry else 'FALSE')
                elif isinstance(entry, (int, float)): literals.append(repr(entry))
                else: literals.append("'" + str(entry).replace("'", "''") + "'")
            if item['op'] == 'in': parts.append(column + ' IN (' + ', '.join(literals) + ')')
            elif value is None and item['op'] == 'eq': parts.append(column + ' IS NULL')
            elif value is None and item['op'] == 'ne': parts.append(column + ' IS NOT NULL')
            else: parts.append(column + ' ' + ops[item['op']] + ' ' + literals[0])
        return ' AND '.join(parts)

    def _cached_chart_call(self, current):
        """Return a local tool call that displays an already verified chart.

        Going through prepare_histogram emits the normal tool observation, so
        the repeated request visibly contains the image while avoiding a model
        call and any remote query.
        """
        if (current.get('fresh_source_required') or current.get('chart_spec_requested')
                or not current.get('chart') or not current.get('kind')
                or current.get('artifact_ids')):
            return None
        valid = []
        for card_id in self.artifacts:
            try:
                card = self.artifacts[card_id]
            except (KeyError, OSError, ValueError, TypeError):
                continue
            if card.kind != current['kind']:
                continue
            if self._valid_card(card, deepcopy(current), card.dataset_id):
                valid.append(card)
        if valid:
            card = valid[-1]
            if card.kind != 'histogram' or len(card.columns) != 1 or not self.context:
                return None
            info = self.context.datasets.metadata.get(card.dataset_id)
            if info is None:
                return None
            call = {'name':'show_chart', 'args':{'chart_id':card.id}}
            self.diagnostics.emit('cached_chart_reuse_planned',
                request_id=current.get('request_id'), chart_id=card.id,
                dataset_id=card.dataset_id, kind=card.kind)
            return call
        return None

    def _dispatch(self, current, call, last=None):
        call = {**call, 'id': str(uuid4()), 'type': 'tool_call'}
        current['sent_calls'].append(call['id'])
        self.diagnostics.emit('recovery_transition', request_id=current.get('request_id'), tool=call['name'])
        message = last.model_copy(update={'content': '', 'tool_calls': [call]}) if last else AIMessage(content='', tool_calls=[call])
        return {'recovery': current, 'messages': [message]}

    def before_step(self, state):
        current, calls = self._state(state)
        if (current.get('data_load') or current.get('plan') or current.get('chart') or current.get('calculation') or current.get('metadata_kind') or current.get('profile_kind')) and self._complete(current):
            return {**self._finish(current), 'jump_to': 'end'}
        reason = self._limit_reason(current)
        if reason: return {**self._finish(current, reason=reason), 'jump_to': 'end'}
        cached = self._cached_chart_call(current)
        if cached: return {**self._dispatch(current, cached), 'jump_to':'tools'}
        proposed = self._next_local(current, calls)
        if proposed: return {**self._dispatch(current, proposed), 'jump_to': 'tools'}
        current['model_started_at'] = time.time()
        scope_key = json.dumps(current.get('scope', {}), sort_keys=True, ensure_ascii=False)
        if self._has_scope(current) and current.get('scope_notified') != scope_key:
            current['scope_notified'] = scope_key
            return {'recovery':current, 'messages':[SystemMessage(content=self._scope_instruction(current),
                additional_kwargs={'lc_source':'recovery_scope'})]}
        return {'recovery': current}

    @staticmethod
    def _scope_instruction(current):
        return ('원래 사용자 요청과 저장된 데이터 문맥으로 확인한 분석 조건(JSON): '
            + json.dumps(current.get('scope', {}), ensure_ascii=False)
            + '\n이 조건과 이전 대화에서 유지된 조건을 모두 적용하세요. SQL이나 도구가 만든 다른 조건으로 원래 요청을 바꾸지 마세요. '
            'unresolved가 있으면 inspect_table_context로 관련 컬럼 정보를 확인하고, 확인할 수 없으면 조건을 명확히 요청하세요. '
            '필터 불일치 결과는 완료 증거가 아닙니다.')

    def _reject_scope_call(self, current, last):
        if current['attempts'] >= self.max_attempts:
            return self._finish(current, last, 'request_scope_mismatch')
        current['attempts'] += 1
        # Remove the entire unexecuted batch before HITL sees it. It creates
        # neither an approval card nor a fabricated tool result.
        return {'recovery':current, 'messages':[RemoveMessage(id=last.id),
            SystemMessage(content='도구 호출의 기간·필터가 요청 조건과 달라 실행 또는 원격 조회 승인 요청 전에 거절되었습니다. '
                + self._scope_instruction(current), additional_kwargs={'lc_source':'recovery_scope'})], 'jump_to':'model'}

    def _proposed_scope_valid(self, call, current):
        if not self._has_scope(current): return True
        arguments = call.get('args', {})
        if call.get('name') == 'local_analysis_sql':
            # Reject a wrong local population before its rows or aggregate are
            # shown to the model. Otherwise the incorrect value can anchor the
            # following repair turn even though it is never accepted as final
            # evidence. Explicit requested_conditions are the population
            # contract; when omitted, the SQL WHERE clause is the contract.
            requested = arguments.get('requested_conditions')
            executed = requested if requested is not None else arguments.get('query', '')
            return self._scope_valid(executed, current) and measure_scope_matches(
                arguments.get('query', ''), current.get('scope', {}))
        if call.get('name') == 'prepare_histogram':
            if current.get('fresh_source_required') and not arguments.get('fresh_source_required'):
                return False
            if arguments.get('current_result_only') and not current.get('current_result_only'):
                return False
            where = arguments.get('where_sql', '').strip()
            return self._scope_valid('SELECT * FROM data' + (' WHERE ' + where if where else ''), current)
        if call.get('name') == 'show_chart':
            try:
                card = self.artifacts[arguments.get('chart_id')]
            except (KeyError, OSError, ValueError, TypeError):
                return True
            return self._valid_card(card, current, card.dataset_id)
        if call.get('name') in {'recommend_chart_images', 'render_chart_spec', 'render_histogram'} and self.context:
            info = self.context.datasets.metadata.get(arguments.get('dataset_id'))
            if info is None: return True  # The tool adapter reports unknown IDs.
            column = arguments.get('value_column') or (
                arguments.get('x') if arguments.get('kind') == 'histogram' else None)
            return self._scope_valid(info, current, column)
        if call.get('name') != 'query_databricks': return True
        if current.get('current_result_only'):
            return False
        query = call.get('args', {}).get('query', '')
        column = None
        if current.get('kind') == 'histogram':
            import sqlglot
            from utils.analysis_provenance import count_frequency_columns
            try:
                columns = count_frequency_columns(sqlglot.parse_one(query, read='databricks'))
                column = columns[0] if columns else None
            except (TypeError, ValueError, sqlglot.errors.SqlglotError): pass
        return (self._scope_valid(query, current, column)
                and measure_scope_matches(query, current.get('scope', {})))

    @staticmethod
    def _fresh_for_request(info, current):
        if not current.get('fresh_source_required'):
            return True
        if not info.snapshot:
            return False
        from datetime import datetime
        try:
            observed = datetime.fromisoformat(str(info.snapshot).replace('Z', '+00:00')).timestamp()
            return observed + 0.001 >= float(current.get('request_started_at', float('inf')))
        except (TypeError, ValueError, OverflowError):
            return False

    @hook_config(can_jump_to=['model'])
    def after_model(self, state, runtime):
        messages = state.get('messages', [])
        if not messages or not isinstance(messages[-1], AIMessage): return None
        last = messages[-1]
        current, calls = self._state(state)
        current['model_calls'] += 1
        started = current.pop('model_started_at', None)
        if started: current['model_seconds'] += max(0, time.time() - started)
        for call in last.tool_calls:
            if call['name'] in {'recommend_chart_images', 'render_chart_spec', 'render_histogram', 'show_chart'}: current['chart'] = True
        remote_block = current.get('remote_rejected') or any(o.get('status') == 'unavailable' for o in current['failed'].values())
        if self._complete(current) and (current.get('plan') or not last.tool_calls):
            return self._finish(current, last)
        reason = self._limit_reason(current)
        if reason: return self._finish(current, last, reason)
        if last.tool_calls and remote_block and any(c['name'] == 'query_databricks' for c in last.tool_calls):
            return self._finish(current, last, 'remote_blocked')
        if any(not self._proposed_scope_valid(call, current) for call in last.tool_calls):
            return self._reject_scope_call(current, last)
        if current.get('plan') and not remote_block:
            plan = current['plan']
            if current.get('query_seen') != current.get('plan_call'):
                if not self._scope_valid(plan['query'], current, plan['value_column']):
                    current.pop('plan', None)
                    return self._reject_scope_call(current, last)
                return self._dispatch(current, {'name': 'query_databricks',
                    'args': {k: plan[k] for k in ('source', 'query', 'reason')}}, last)
            proposed = self._next_local(current, calls)
            if proposed: return self._dispatch(current, proposed, last)
        if last.tool_calls:
            if len(current['sent_calls']) + len(last.tool_calls) > self.max_tool_calls:
                return self._finish(current, last, 'tool_call_budget')
            current['sent_calls'].extend(c['id'] for c in last.tool_calls if c['id'] not in current['sent_calls'])
            return {'recovery': current}
        if remote_block or current['attempts'] >= self.max_attempts:
            return self._finish(current, last, 'missing_evidence')
        current['attempts'] += 1
        self.diagnostics.emit('recovery_replan', request_id=current.get('request_id'), attempts=current['attempts'],
            missing_chart=bool(current.get('chart') and not current['artifact_ids']),
            missing_calculation=bool(current.get('calculation') and not current['evidence_ids']))
        instruction = ('이전 응답은 완료 증거가 없어 채택되지 않았습니다. 원래 사용자 요청을 계속 수행하세요. '
            '수치/통계는 local_analysis_sql의 실제 계산 결과가 필요하고 결측·고유값·기초 통계는 profile_dataset의 구조화 결과가 필요합니다. 테이블 설명이나 미리보기는 계산 증거가 아닙니다. '
            '요청한 출처, 컬럼, 집계와 필터를 유지하세요. 지정 차트와 수정은 render_chart_spec을 사용하고, 원격 데이터가 필요한 히스토그램은 prepare_histogram(source, column, where_sql)을 사용하세요. '
            '이 도구는 먼저 재사용 가능한 보유 데이터를 찾고, 부족한 경우에만 승인형 로딩과 렌더링 계획을 만듭니다. '
            'query_databricks 호출이 승인 카드를 생성하며 실제 조회는 사용자 승인을 기다립니다. '
            '동일한 실패 호출을 반복하거나 증거 없이 완료했다고 말하지 마세요.')
        if self._has_scope(current): instruction += '\n' + self._scope_instruction(current)
        return {'recovery': current, 'messages': [RemoveMessage(id=last.id),
            SystemMessage(content=instruction, additional_kwargs={'lc_source': 'recovery', 'invalidated_message_id': last.id})], 'jump_to': 'model'}
