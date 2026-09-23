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


def _strip_outlier_method_scope(scope, column):
    """Do not mistake an outlier method token for a literal row filter."""
    if not column:
        return scope
    cleaned = deepcopy(scope)
    method_tokens = {'iqr', 'mad', 'zscore', 'sigma', '시그마'}
    for key in ('conditions', 'any_conditions'):
        cleaned[key] = [item for item in cleaned.get(key, []) if not (
            item.get('column') == column
            and isinstance(item.get('value'), str)
            and re.sub(r'[\s_-]+', '', item['value']).casefold() in method_tokens
        )]
    return cleaned


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
            elif re.search(r'선\s*(?:그래프|차트)|꺾은선|(?:누적|성장).{0,40}곡선|\bline(?:\s*chart)?\b|\bcurve\b', text, re.I): kind = 'line'
            else: kind = None
            chart = bool(kind or re.search(r'차트|시각화|그래프|\bchart|\bplot', text, re.I))
            count_rate_signal = bool(
                re.search(r'건수|고객\s*수|승객\s*수|접촉\s*수|행\s*수|\b(?:row\s*)?count\b', text, re.I)
                and re.search(r'비율|성공률|생존율|전환율|\b(?:ratio|rate|percentage|percent)\b', text, re.I))
            if chart and count_rate_signal and re.search(
                    r'이중\s*(?:Y\s*)?축|dual\s*(?:Y[- ]?)?axis|twinx', text, re.I):
                count_rate_layout = 'dual_axis'
            elif chart and count_rate_signal and re.search(
                    r'(?:2|두)\s*열.{0,12}(?:서브\s*플롯|패널)|split\s*panel', text, re.I):
                count_rate_layout = 'split_panel'
            else:
                count_rate_layout = None
            chart_cumulative = bool(chart and re.search(
                r'누적(?:합|곡선|성장)?|\bcumulative(?:\s+sum)?\b|\bcumsum\b', text, re.I))
            chart_spec_requested = bool(chart and (
                kind in {'bar', 'line', 'scatter', 'boxplot'} or
                chart_cumulative or re.search(
                    r'제목|축\s*라벨|정렬|상위\s*\d+|top\s*\d+|\bbins?\b|\d+\s*(?:개\s*)?구간|가로|세로', text, re.I)))
            time_series_context = bool(kind == 'line' and re.search(
                r'시계열|리샘플|재표본|시간\s*간격|날짜|일시|타임스탬프|'
                r'\b(?:time[ -]?series|resampl(?:e|ing)|datetime|timestamp)\b', text, re.I))
            if time_series_context and re.search(r'시간별|매\s*시간|hourly|\b1h\b', text, re.I):
                time_series_frequency = 'hour'
            elif time_series_context and re.search(r'일별|매일|daily|\b1d\b', text, re.I):
                time_series_frequency = 'day'
            elif time_series_context and re.search(r'주별|주간|weekly|\b1w\b', text, re.I):
                time_series_frequency = 'week'
            elif time_series_context and re.search(r'월별|매월|monthly|\b1m\b', text, re.I):
                time_series_frequency = 'month'
            else:
                time_series_frequency = None
            if time_series_frequency and re.search(
                    r'빈\s*(?:시간|날짜|구간).{0,12}(?:0|제로)|(?:gap|missing).{0,12}(?:zero|0)|fill.{0,8}zero',
                    text, re.I):
                time_series_gap_policy = 'zero'
            elif time_series_frequency and re.search(
                    r'빈\s*(?:시간|날짜|구간).{0,12}(?:NaN|결측)|(?:gap|missing).{0,12}(?:nan|null)',
                    text, re.I):
                time_series_gap_policy = 'nan'
            else:
                time_series_gap_policy = 'omit'
            timezone_match = re.search(
                r'(?:timezone|시간대)\s*(?:은|는|을|를|:|=)?\s*'
                r'(UTC|[A-Za-z_]+/[A-Za-z0-9_+\-]+(?:/[A-Za-z0-9_+\-]+)*)', text, re.I)
            time_series_timezone = timezone_match.group(1) if timezone_match else ''
            join_requested = bool(re.search(r'조인|병합|데이터\s*결합|\bjoin\b|\bmerge\b', text, re.I))
            if re.search(r'좌측|왼쪽|\bleft\s*(?:outer\s*)?join\b', text, re.I): join_how = 'left'
            elif re.search(r'우측|오른쪽|\bright\s*(?:outer\s*)?join\b', text, re.I): join_how = 'right'
            elif re.search(r'완전\s*외부|전체\s*외부|\b(?:full|outer)\s*(?:outer\s*)?join\b', text, re.I): join_how = 'outer'
            elif re.search(r'내부|\binner\s*join\b', text, re.I): join_how = 'inner'
            else: join_how = None
            if re.search(r'맨[\s-]*휘트니|mann[\s-]*whitney', text, re.I):
                statistical_kind = 'mann_whitney'
            elif re.search(r'카이\s*제곱|chi[\s-]*square', text, re.I):
                statistical_kind = 'chi_square'
            elif re.search(r'일원\s*분산|one[\s-]*way\s*anova|\banova\b', text, re.I):
                statistical_kind = 'one_way_anova'
            elif re.search(r'대응\s*(?:표본\s*)?t[\s-]*검정|paired\s*t[\s-]*test', text, re.I):
                statistical_kind = 'paired_t'
            elif re.search(r'독립\s*(?:표본\s*)?t[\s-]*검정|two[\s-]*sample\s*t[\s-]*test|independent\s*t[\s-]*test', text, re.I):
                statistical_kind = 'independent_t'
            elif re.search(r'(?:모평균|평균).{0,12}(?:신뢰\s*구간|confidence\s*interval)', text, re.I):
                statistical_kind = 'mean_ci'
            else:
                statistical_kind = None
            outlier_spec = None
            if not chart:
                if re.search(r'\bIQR\b|사분위\s*범위', text, re.I) and re.search(
                        r'이상치|극단치|상한|하한|fence|기준선', text, re.I):
                    match = re.search(r'(\d+(?:\.\d+)?)\s*\*?\s*IQR', text, re.I)
                    outlier_spec = {'method':'iqr', 'threshold':float(match.group(1)) if match else 1.5}
                elif re.search(r'Z[\s-]*Score|Z\s*점수|시그마|sigma', text, re.I) and re.search(
                        r'이상치|극단치|초과|범위|점검', text, re.I):
                    match = re.search(r'(?:Z[\s-]*Score|Z\s*점수).{0,12}?(\d+(?:\.\d+)?)|'
                                      r'(\d+(?:\.\d+)?)\s*[- ]?시그마', text, re.I)
                    value = next((group for group in match.groups() if group is not None), '3') if match else '3'
                    outlier_spec = {'method':'zscore', 'threshold':float(value)}
                elif re.search(r'\bMAD\b|중앙값\s*절대\s*편차', text, re.I) and re.search(
                        r'이상치|극단치|초과|범위|점검', text, re.I):
                    match = re.search(r'(?:MAD|수정\s*Z[\s-]*Score).{0,12}?(\d+(?:\.\d+)?)', text, re.I)
                    outlier_spec = {'method':'mad', 'threshold':float(match.group(1)) if match else 3.5}
                elif re.search(r'이상치|극단치', text, re.I):
                    match = re.search(r'(상위|하위|양쪽|상하위)\s*(\d+(?:\.\d+)?)\s*%', text, re.I)
                    if match and 0 < float(match.group(2)) < 50:
                        fraction = float(match.group(2)) / 100
                        outlier_spec = {'method':'quantile', 'threshold':1.5,
                            'lower_quantile':fraction, 'upper_quantile':1-fraction}
                if outlier_spec:
                    if re.search(r'범위|양쪽|상하위|상·?하한|하한.*상한|상한.*하한|절대값|\|Z\|', text, re.I):
                        outlier_spec['tail'] = 'both'
                    elif re.search(r'하한|미만|낮은|하위', text, re.I):
                        outlier_spec['tail'] = 'lower'
                    elif re.search(r'상한|초과|높은|고액|상위|최장', text, re.I):
                        outlier_spec['tail'] = 'upper'
                    else:
                        outlier_spec['tail'] = 'both'
            profile_kind = None
            if not chart and re.search(r'결측|누락|\b(?:null|missing|nan)\b', text, re.I):
                profile_kind = 'missing'
            elif not chart and re.search(r'고유값|유니크|\b(?:distinct|unique)\b', text, re.I):
                profile_kind = 'distinct'
            elif not chart and re.search(r'기초\s*통계|요약\s*통계|데이터\s*프로파일|\b(?:describe|profile)\b', text, re.I):
                profile_kind = 'summary'
            count_request = not chart and not profile_kind and bool(re.search(
                r'건수|인원\s*수|명수|빈도|결측|고유값|(?:사람|고객|승객|신청자|가입자|사용자|행)[\'\"]*(?:들)?(?:의)?\s*수',
                text))
            calculation = bool(re.search(
                r'평균|중앙값|합계|총합|최솟값|최댓값|최소값|최대값|표준편차|상관|비율|성공률|개수|몇\s*(?:명|개|건)|계산|통계|'
                r'[가-힣A-Za-z]+[율률]|\b(?:mean|average|avg|count|sum|median|std|correlation|ratio|rate|percentage|percent)\b', text, re.I)) or count_request
            if re.search(r'(?:이란|개념|정의|뜻).*(?:설명|알려)|(?:이란|란)\s*(?:무엇|뭐)', text):
                calculation = False
            names, sources, column_aliases = set(), set(), {}
            if self.context:
                from core.analysis_catalog import grounded_reference_columns
                for table in self.context.reference_context:
                    sources.add(table.get('table', ''))
                    for column in grounded_reference_columns(table, self.context.datasets):
                        name = column.get('name')
                        if not name:
                            continue
                        names.add(name)
                        column_aliases.setdefault(name, set()).update(
                            alias for alias in column.get('aliases', []) if isinstance(alias, str))
                for info in self.context.datasets.metadata.values():
                    names.update(info.columns)
                    sources.add(info.source)
            def mentioned(name):
                return bool(name and re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', text))
            def mentioned_alias(alias):
                if not alias:
                    return False
                return bool(re.search(
                    r'(?<![A-Za-z0-9_가-힣])' + re.escape(alias)
                    + r'(?=(?:은|는|이|가|을|를|의|과|와|별|간|에서|으로|에|도)?(?:[^가-힣]|$))', text))
            def mentioned_column(name):
                if mentioned(name) or any(mentioned_alias(alias) for alias in column_aliases.get(name, ())):
                    return True
                # A metadata alias such as "생존 여부" also grounds the
                # ordinary role noun "생존자". Keep this suffix rule tied to
                # TableContext metadata rather than any table-specific name.
                for alias in column_aliases.get(name, ()):
                    stem = re.sub(r'\s*(?:여부|유무|상태|율)$', '', alias).strip()
                    if len(stem) >= 2 and re.search(
                            r'(?<![A-Za-z0-9_])' + re.escape(stem)
                            + r'(?:자|여부|유무|상태|율)(?![A-Za-z0-9_])', text):
                        return True
                return False
            explicit_columns = [name for name in names if mentioned(name)]
            explicit_columns.sort(
                key=lambda name: (text.find(name) if text.find(name) >= 0 else len(text), name))
            mentioned_columns = [name for name in names if mentioned_column(name)]
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
            time_series_aggregation = None
            if time_series_frequency:
                operation_to_aggregation = {
                    'COUNT':'count', 'SUM':'sum', 'AVG':'mean', 'MEDIAN':'median',
                    'MIN':'min', 'MAX':'max'}
                declared = [operation_to_aggregation[operation] for operation in operations
                            if operation in operation_to_aggregation]
                if len(declared) == 1:
                    time_series_aggregation = declared[0]
                    calculation, operations = False, []
                else:
                    # A time bucket without one explicit aggregation is
                    # ambiguous. Leave it to the model rather than guessing.
                    time_series_frequency = None
            if chart_cumulative and set(operations) <= {'SUM'}:
                # The sum is the declared chart transform, not a separate
                # scalar result obligation.
                calculation, operations = False, []
            # A compound request first materializes a lineage-safe cohort, then
            # computes against that exact child dataset.  The first mentioned
            # column is the detector column because the prompt introduces the
            # outlier criterion before its follow-up metric.
            if outlier_spec and not mentioned_columns:
                outlier_spec = None
            outlier_column = mentioned_columns[0] if outlier_spec else None
            outlier_followup = bool(outlier_spec and (
                len(mentioned_columns) > 1 or 'RATIO' in operations))
            outlier_selection = (
                'inliers' if outlier_followup and re.search(
                    r'이상치.{0,12}(?:제외|제거)|(?:일반|정상)\s*(?:고객|승객|행)', text)
                else 'outliers')
            outlier_group_column = None
            outlier_metric_column = None
            outlier_metric_aggregation = None
            outlier_aggregate_mode = None
            top_match = re.search(r'(?:TOP|상위)\s*(\d+)', text, re.I)
            outlier_top_n = int(top_match.group(1)) if top_match else 0
            if outlier_followup:
                group_candidates = []
                top_group_candidates = []
                for name in mentioned_columns:
                    if name == outlier_column:
                        continue
                    terms = [name, *column_aliases.get(name, ())]
                    by_signal = any(re.search(
                        r'(?<![A-Za-z0-9_가-힣])' + re.escape(term)
                        + r'\s*(?:별|그룹별)(?![A-Za-z0-9_])', text, re.I) for term in terms)
                    top_signal = any(re.search(
                        r'(?:주요|빈도가\s*높은)\s*' + re.escape(term)
                        + r'|' + re.escape(term) + r'\s*(?:TOP|상위)\s*\d+', text, re.I)
                        for term in terms)
                    if by_signal or top_signal:
                        group_candidates.append(name)
                    if top_signal:
                        top_group_candidates.append(name)
                if len(group_candidates) == 1:
                    outlier_group_column = group_candidates[0]
                    aggregate_map = {'AVG':'mean', 'SUM':'sum', 'MEDIAN':'median',
                                     'MIN':'min', 'MAX':'max'}
                    declared = [aggregate_map[item] for item in operations if item in aggregate_map]
                    if len(top_group_candidates) == 1:
                        outlier_aggregate_mode = 'top_frequency'
                        metric_candidates = [column for column in mentioned_columns
                                             if column not in {outlier_column, outlier_group_column}]
                        if len(declared) == 1 and len(metric_candidates) == 1:
                            outlier_metric_aggregation = declared[0]
                            outlier_metric_column = metric_candidates[0]
                        elif declared:
                            outlier_aggregate_mode = None
                        if not outlier_top_n:
                            outlier_top_n = 3
                    elif re.search(r'별|그룹별', text):
                        metric_candidates = [column for column in mentioned_columns
                                             if column != outlier_group_column]
                        if len(metric_candidates) > 1 and outlier_column in metric_candidates:
                            non_detector = [column for column in metric_candidates
                                            if column != outlier_column]
                            if len(non_detector) == 1:
                                metric_candidates = non_detector
                        if len(declared) == 1 and len(metric_candidates) == 1:
                            outlier_aggregate_mode = (
                                'grouped_comparison' if re.search(
                                    r'(?:포함|제외).{0,12}(?:전|후)|(?:전|후).{0,12}(?:포함|제외)|비교',
                                    text, re.I)
                                else 'grouped_metric')
                            outlier_metric_aggregation = declared[0]
                            outlier_metric_column = metric_candidates[0]
                if outlier_top_n > 50:
                    outlier_aggregate_mode = None
            outlier_aggregate_requested = bool(outlier_aggregate_mode)
            if outlier_aggregate_requested:
                # aggregate_dataset provides the structured completion evidence
                # for this cohort request; do not also require arbitrary SQL.
                calculation = False
            if (join_requested and set(operations) <= {'COUNT'} and
                    re.search(r'조인\s*결과.*(?:행|열|컬럼)|(?:행|열|컬럼).*개수|결합된\s*컬럼', text, re.I)):
                # join_datasets returns these structural counts as grounded
                # evidence; a second SQL COUNT would add no information.
                calculation, operations = False, []
            if profile_kind:
                calculation, operations = False, []
            if statistical_kind:
                calculation, operations = False, []
            if outlier_spec and not outlier_followup:
                calculation, operations = False, []
            if count_rate_layout:
                # The chart tool returns the structured count, denominator,
                # numerator and percentage evidence for this compound request.
                calculation = False
            metadata_kind = None
            column_words = bool(re.search(r'컬럼|필드|\bcolumns?\b|\bfields?\b', text, re.I))
            if (not chart and not join_requested and not operations and column_words and
                    re.search(r'데이터\s*타입|자료형|\bdtypes?\b|\bdata\s*types?\b', text, re.I)):
                metadata_kind, calculation = 'dtypes', False
            elif (not chart and not join_requested and not operations and column_words and
                    re.search(r'수치형|숫자형|\bnumeric\b', text, re.I)):
                metadata_kind, calculation = 'numeric_columns', False
            elif (not chart and not join_requested and not operations and column_words and
                    re.search(r'문자열|범주형|카테고리형|\bcategorical\b|\bstring\b', text, re.I)):
                metadata_kind, calculation = 'categorical_columns', False
            elif (not chart and not join_requested and set(operations) <= {'COUNT'} and
                    column_words and
                    not re.search(r'고유|결측|중복|누락|빈도|\bnull\b|\bdistinct\b|\bmissing\b|\bunique\b', text, re.I) and
                    re.search(r'목록|개수|구조|이름|어떤|몇|전체|\blist\b|\bschema\b', text, re.I)):
                metadata_kind, calculation, operations = 'columns', False, []
            current = dict(request_id=human.id, attempts=0, chart=chart, kind=kind,
                join=join_requested, join_how=join_how,
                statistical_kind=statistical_kind,
                outlier_spec=outlier_spec,
                outlier_column=outlier_column,
                outlier_followup=outlier_followup,
                outlier_selection=outlier_selection,
                outlier_group_column=outlier_group_column,
                outlier_metric_column=outlier_metric_column,
                outlier_metric_aggregation=outlier_metric_aggregation,
                outlier_aggregate_mode=outlier_aggregate_mode,
                outlier_aggregate_requested=outlier_aggregate_requested,
                outlier_top_n=outlier_top_n,
                calculation=calculation, operations=operations, metadata_kind=metadata_kind,
                profile_kind=profile_kind, chart_spec_requested=chart_spec_requested,
                count_rate_layout=count_rate_layout,
                chart_cumulative=chart_cumulative,
                time_series_frequency=time_series_frequency,
                time_series_aggregation=time_series_aggregation,
                time_series_gap_policy=time_series_gap_policy,
                time_series_timezone=time_series_timezone,
                data_load=data_load,
                expected_load_source=human.additional_kwargs.get('source','') if data_load else '',
                expected_load_query=human.additional_kwargs.get('query','') if data_load else '',
                whole_row_count=bool(re.search(
                    r'전체\s*(?:행|레코드|데이터)(?:의)?\s*(?:수|개수)|총\s*(?:데이터\s*)?(?:행|레코드)\s*(?:수|개수)', text)),
                current_result_only=bool(current_loaded_reference or re.search(
                    r'(?:현재|보유|지금|이|그)\s*(?:로딩된\s*)?(?:[\d,]+\s*행\s*)?(?:결과|표본|샘플|데이터)|'
                    r'(?<!독립)(?<!대응)표본(?!\s*(?:평균|분산|크기|수))|샘플|일부\s*데이터', text)),
                fresh_source_required=fresh_source_required,
                request_started_at=time.time(),
                required_columns=mentioned_columns,
                explicit_columns=explicit_columns,
                required_sources=sorted(
                    (s for s in sources if mentioned(s) or mentioned(s.split('.')[-1])),
                    key=lambda source: min(
                        (position for position in (text.find(source), text.find(source.split('.')[-1])) if position >= 0),
                        default=len(text))),
                columns=[], failed={}, status='working')
            current['previous_scope'] = previous.get('scope', {})
            current['scope'] = resolve_request_scope(text, self.context, current['previous_scope'])
            # Prefer an explicitly named canonical grouping column over an
            # incidental alias match. For example, a short alias such as
            # "일" must not make ``job`` compete with an explicit ``day`` in
            # "일별(day)". This rule is derived from live schema names and is
            # independent of any particular table.
            if count_rate_layout:
                outcome = (current['scope'].get('ratio') or {}).get('column')
                explicit_groups = [column for column in explicit_columns if column != outcome]
                if outcome and len(explicit_groups) == 1:
                    current['required_columns'] = [explicit_groups[0], outcome]
            if outlier_spec:
                current['scope'] = _strip_outlier_method_scope(
                    current['scope'], outlier_column)
            if join_requested and len(current['required_sources']) == 2:
                # Two explicitly named sources are expected for a join and do
                # not represent the single-source ambiguity used by scalar and
                # chart requests.
                current['scope']['unresolved'] = [item for item in current['scope'].get('unresolved', [])
                                                  if item != 'ambiguous_source']
            if data_load:
                # The exact source and query are already bound to the approval
                # envelope.  Natural-language scope extraction from the reason
                # text would invent an analysis obligation.
                current.update(chart=False,kind=None,join=False,join_how=None,statistical_kind=None,outlier_spec=None,
                    outlier_column=None,outlier_followup=False,outlier_selection='outliers',
                    outlier_group_column=None,outlier_metric_column=None,
                    outlier_metric_aggregation=None,outlier_aggregate_mode=None,
                    outlier_aggregate_requested=False,outlier_top_n=0,
                    calculation=False,operations=[],metadata_kind=None,profile_kind=None,
                    chart_spec_requested=False,chart_cumulative=False,whole_row_count=False,current_result_only=False,fresh_source_required=False,
                    count_rate_layout=None,
                    time_series_frequency=None,time_series_aggregation=None,
                    time_series_gap_policy='omit',time_series_timezone='',
                    required_columns=[],explicit_columns=[])
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
                current['current_result_only'] = bool(
                    current.get('current_result_only') or previous.get('current_result_only'))
                if not operations and current['calculation']: current['operations'] = previous.get('operations', [])
                if current['calculation']:
                    # A follow-up such as "그중 segment A" mentions only a new
                    # predicate column. Keep the previous measure instead of
                    # accidentally aggregating the predicate itself.
                    predicate_columns = {item['column'] for item in (
                        current['scope'].get('conditions', [])
                        + current['scope'].get('any_conditions', []))}
                    measures = [column for column in current['required_columns']
                                if column not in predicate_columns]
                    if not measures:
                        previous_predicates = {item['column'] for item in (
                            previous.get('scope', {}).get('conditions', [])
                            + previous.get('scope', {}).get('any_conditions', []))}
                        measures = [column for column in previous.get('required_columns', [])
                                    if column not in previous_predicates]
                    current['required_columns'] = measures
                elif not current['required_columns']:
                    current['required_columns'] = previous.get('required_columns', [])
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
            current.update(data_load=True,chart=False,kind=None,join=False,join_how=None,statistical_kind=None,outlier_spec=None,
                outlier_column=None,outlier_followup=False,outlier_selection='outliers',
                outlier_group_column=None,outlier_metric_column=None,
                outlier_metric_aggregation=None,outlier_aggregate_mode=None,
                outlier_aggregate_requested=False,outlier_top_n=0,calculation=False,
                operations=[],metadata_kind=None,profile_kind=None,chart_spec_requested=False,chart_cumulative=False,whole_row_count=False,
                current_result_only=False,fresh_source_required=False)
            current['count_rate_layout'] = None
            current['scope']={'conditions':[],'any_conditions':[],
                'measure_conditions':[],'ratio':None,'unresolved':[],'columns':[]}
        if human and ('scope' not in current or current['scope'].get('unresolved')):
            # Metadata discovery may resolve an ambiguous column. Re-read only
            # the original human request, never model-generated SQL or summaries.
            current['scope'] = resolve_request_scope(str(human.content), self.context, current.get('previous_scope'))
            if current.get('outlier_spec'):
                current['scope'] = _strip_outlier_method_scope(
                    current['scope'], current.get('outlier_column'))
        for key, default in [('processed', []), ('sent_calls', []), ('failed_signatures', {}),
                ('evidence_ids', []), ('artifact_ids', []), ('failed', {}), ('attempts', 0),
                ('model_calls', 0), ('model_seconds', 0.0), ('columns', []),
                ('outlier_aggregate_evidence', {}), ('count_rate_evidence', None),
                ('count_rate_layout', None), ('count_rate_group_column', None),
                ('explicit_columns', [])]:
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
            failed = observation.get('status') in {'error', 'rejected', 'needs_data', 'needs_context', 'needs_refresh', 'unavailable', 'no_valid_chart'}
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
            if name == 'join_datasets' and observation.get('status') == 'ready':
                dataset_id = observation.get('dataset', {}).get('id')
                info = self.context.datasets.metadata.get(dataset_id) if self.context and dataset_id else None
                parent_ids = tuple(getattr(info, 'parent_ids', ())) if info is not None else ()
                expected_parents = (arguments.get('left_dataset_id'), arguments.get('right_dataset_id'))
                summary = observation.get('join_summary', {})
                if (info is not None and parent_ids == expected_parents
                        and info.rows == summary.get('actual_rows') == summary.get('expected_rows')
                        and set(current.get('required_columns', [])).issubset(info.columns)):
                    current['join_evidence'] = {
                        'dataset_id': dataset_id,
                        'parent_ids': list(parent_ids),
                        'summary': summary,
                        'scope': observation.get('scope'),
                    }
                    current['failed'].pop(name, None)
            if name == 'prepare_time_series' and observation.get('status') == 'ready':
                parent_id = arguments.get('dataset_id')
                child_id = observation.get('dataset', {}).get('id')
                parent = self.context.datasets.metadata.get(parent_id) if self.context else None
                child = self.context.datasets.metadata.get(child_id) if self.context else None
                result = observation.get('time_series_result', {})
                from utils.analysis_timeseries import dataset_digest
                valid = (
                    parent is not None and child is not None
                    and result.get('kind') == 'time_series_preparation'
                    and result.get('parent_dataset_id') == parent_id
                    and child.parent_id == parent_id and child.source == parent.source
                    and child.snapshot == parent.snapshot and child.grain == 'aggregate'
                    and result.get('time_column') == arguments.get('time_column')
                    and result.get('frequency') == arguments.get('frequency') == current.get('time_series_frequency')
                    and result.get('aggregation') == arguments.get('aggregation') == current.get('time_series_aggregation')
                    and result.get('gap_policy') == arguments.get('gap_policy') == current.get('time_series_gap_policy')
                    and result.get('output_rows') == child.rows
                    and result.get('data_sha256') == dataset_digest(self.context.datasets.frames[child_id])
                    and self._source_matches(parent, current)
                    and self._fresh_for_request(parent, current)
                    and (current.get('current_result_only')
                         or (parent.coverage == 'complete' and parent.predicate_known))
                    and not self._has_scope(current))
                if valid:
                    current['time_series_evidence'] = observation
                    current['time_series_dataset'] = child_id
                    current['failed'].pop(name, None)
            if name == 'statistical_test' and observation.get('status') == 'ready':
                dataset_id = arguments.get('dataset_id') or observation.get('dataset_id')
                info = self.context.datasets.metadata.get(dataset_id) if self.context and dataset_id else None
                result = observation.get('test_result', {})
                if (info is not None
                        and dataset_id == observation.get('dataset_id')
                        and result.get('kind') == arguments.get('test') == current.get('statistical_kind')
                        and set(current.get('required_columns', [])).issubset(result.get('columns', []))
                        and self._source_matches(info, current)
                        and self._fresh_for_request(info, current)
                        and (current.get('current_result_only')
                             or (info.coverage == 'complete' and info.predicate_known))
                        and self._statistical_scope_valid(info, arguments, current)):
                    current['statistical_evidence'] = observation
                    current['failed'].pop(name, None)
            if name in {'detect_outliers', 'select_outlier_rows'} and observation.get('status') == 'ready':
                parent_id = arguments.get('dataset_id')
                parent = self.context.datasets.metadata.get(parent_id) if self.context and parent_id else None
                result = observation.get('outlier_result', {})
                spec = current.get('outlier_spec') or {}
                common_valid = (
                    parent is not None
                    and result.get('kind') == 'outlier_detection'
                    and result.get('column') == arguments.get('column') == current.get('outlier_column')
                    and result.get('method') == arguments.get('method') == spec.get('method')
                    and result.get('tail') == arguments.get('tail') == spec.get('tail')
                    and (result.get('method') == 'quantile'
                         and arguments.get('lower_quantile') == spec.get('lower_quantile')
                         and arguments.get('upper_quantile') == spec.get('upper_quantile')
                         or result.get('method') != 'quantile'
                         and arguments.get('threshold') == spec.get('threshold'))
                    and self._source_matches(parent, current)
                    and self._fresh_for_request(parent, current)
                    and (current.get('current_result_only')
                         or (parent.coverage == 'complete' and parent.predicate_known)))
                if name == 'detect_outliers':
                    valid = (common_valid
                        and parent_id == observation.get('dataset_id')
                        and set(current.get('required_columns', [])) == {result.get('column')}
                        and not current.get('calculation')
                        and self._scope_valid(parent, current))
                else:
                    child_id = observation.get('dataset', {}).get('id')
                    child = self.context.datasets.metadata.get(child_id) if self.context and child_id else None
                    summary = observation.get('selection_summary', {})
                    expected_rows = (result.get('counts', {}).get('selected')
                        if arguments.get('selection', 'outliers') == 'outliers'
                        else result.get('sample', {}).get('valid_rows', 0)
                             - result.get('counts', {}).get('selected', 0))
                    digest = summary.get('data_sha256', '')
                    from utils.analysis_outliers import dataset_digest
                    actual_digest = (dataset_digest(self.context.datasets.frames[child_id])
                                     if child is not None else None)
                    valid = (common_valid and current.get('outlier_followup')
                        and child is not None and child.parent_id == parent_id
                        and child.source == parent.source and child.snapshot == parent.snapshot
                        and child.grain == 'raw' and not child.aggregation
                        and not child.predicate_known
                        and child.rows == summary.get('selected_rows') == expected_rows
                        and summary.get('parent_dataset_id') == parent_id
                        and summary.get('parent_rows') == parent.rows
                        and summary.get('selection') == arguments.get('selection') == current.get('outlier_selection')
                        and isinstance(digest, str) and len(digest) == 64
                        and digest == actual_digest
                        and bool(child.query) and not observation.get('preview'))
                if valid:
                    current['outlier_evidence'] = observation
                    if name == 'select_outlier_rows':
                        current['outlier_dataset'] = child_id
                    current['failed'].pop(name, None)
            if (name == 'aggregate_dataset' and observation.get('status') == 'ready'
                    and current.get('outlier_aggregate_requested')):
                parent_id = arguments.get('dataset_id')
                child_id = observation.get('dataset', {}).get('id')
                parent = self.context.datasets.metadata.get(parent_id) if self.context else None
                child = self.context.datasets.metadata.get(child_id) if self.context else None
                result = observation.get('aggregation_result', {})
                from utils.analysis_aggregate import dataset_digest
                actual_digest = (dataset_digest(self.context.datasets.frames[child_id])
                                 if child is not None else None)
                common_valid = (
                    parent_id == current.get('outlier_dataset')
                    and parent is not None and child is not None
                    and result.get('kind') == 'dataset_aggregation'
                    and result.get('parent_dataset_id') == parent_id
                    and child.parent_id == parent_id and child.source == parent.source
                    and child.snapshot == parent.snapshot and child.grain == 'aggregate'
                    and result.get('aggregation') == arguments.get('aggregation')
                    and result.get('value_column') == arguments.get('value_column', '')
                    and result.get('group_column') == arguments.get('group_column', '')
                    and result.get('top_n') == arguments.get('top_n', 0)
                    and result.get('output_rows') == child.rows
                    and result.get('data_sha256') == actual_digest
                    and isinstance(observation.get('preview'), list)
                    and not observation.get('rows'))
                evidence_key = None
                mode = current.get('outlier_aggregate_mode')
                if (common_valid and mode == 'top_frequency'
                        and arguments.get('aggregation') == 'count'
                        and arguments.get('group_column') == current.get('outlier_group_column')
                        and arguments.get('top_n') == current.get('outlier_top_n')):
                    evidence_key = 'grouped'
                elif (common_valid and mode == 'top_frequency'
                        and current.get('outlier_metric_aggregation')
                        and arguments.get('aggregation') == current.get('outlier_metric_aggregation')
                        and arguments.get('value_column') == current.get('outlier_metric_column')
                        and not arguments.get('group_column')):
                    evidence_key = 'overall'
                elif (common_valid and mode == 'grouped_metric'
                        and arguments.get('aggregation') == current.get('outlier_metric_aggregation')
                        and arguments.get('value_column') == current.get('outlier_metric_column')
                        and arguments.get('group_column') == current.get('outlier_group_column')
                        and not arguments.get('top_n', 0)):
                    evidence_key = 'grouped'
                if evidence_key:
                    current['outlier_aggregate_evidence'][evidence_key] = observation
                    current['failed'].pop(name, None)
            if (name == 'compare_group_aggregates' and observation.get('status') == 'ready'
                    and current.get('outlier_aggregate_mode') == 'grouped_comparison'):
                baseline_id = arguments.get('baseline_dataset_id')
                cohort_id = arguments.get('cohort_dataset_id')
                result_id = observation.get('dataset', {}).get('id')
                baseline = self.context.datasets.metadata.get(baseline_id) if self.context else None
                cohort = self.context.datasets.metadata.get(cohort_id) if self.context else None
                result_info = self.context.datasets.metadata.get(result_id) if self.context else None
                result = observation.get('comparison_result', {})
                from utils.analysis_compare import dataset_digest
                actual_digest = (dataset_digest(self.context.datasets.frames[result_id])
                                 if result_info is not None else None)
                valid = (
                    baseline is not None and cohort is not None and result_info is not None
                    and cohort_id == current.get('outlier_dataset')
                    and cohort.parent_id == baseline_id
                    and baseline.source == cohort.source == result_info.source
                    and baseline.snapshot == cohort.snapshot == result_info.snapshot
                    and tuple(result_info.parent_ids) == (baseline_id, cohort_id)
                    and result_info.grain == 'aggregate'
                    and result.get('kind') == 'group_aggregate_comparison'
                    and result.get('baseline_dataset_id') == baseline_id
                    and result.get('cohort_dataset_id') == cohort_id
                    and result.get('aggregation') == current.get('outlier_metric_aggregation')
                    and result.get('value_column') == current.get('outlier_metric_column')
                    and result.get('group_column') == current.get('outlier_group_column')
                    and result.get('output_rows') == result_info.rows
                    and result.get('data_sha256') == actual_digest
                    and isinstance(observation.get('preview'), list)
                    and not observation.get('rows'))
                if valid:
                    current['outlier_aggregate_evidence']['comparison'] = observation
                    current['failed'].pop(name, None)
            if name in {'local_analysis_sql', 'query_databricks'} and observation.get('status') == 'ready':
                dataset_id = observation.get('dataset', {}).get('id')
                if self._valid_calculation(dataset_id, arguments, current):
                    current['evidence_ids'].append(dataset_id)
            if name in {'recommend_chart_images', 'render_chart_spec', 'render_count_rate_chart', 'render_histogram', 'prepare_histogram', 'show_chart'} and observation.get('cards'):
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
                    if (name == 'render_count_rate_chart'
                            and self._count_rate_scope_valid(
                                self.context.datasets.metadata.get(dataset_id), arguments, current)
                            and observation.get('chart_spec', {}).get('layout') == current.get('count_rate_layout')
                            and isinstance(observation.get('render_summary', {}).get('points'), list)
                            and len(observation['render_summary']['points']) > 0
                            and len(observation['render_summary'].get('data_sha256', '')) == 64):
                        current['count_rate_evidence'] = observation
                    for tool in ('recommend_chart_images', 'render_chart_spec', 'render_count_rate_chart', 'render_histogram', 'prepare_histogram', 'show_chart'):
                        current['failed'].pop(tool, None)
        return current, calls

    @staticmethod
    def _signature(call):
        return json.dumps([call.get('name'), call.get('args', {})], sort_keys=True, default=str)

    def _source_matches(self, info, current):
        expected = current.get('required_sources') or []
        if not expected:
            return True
        expected_keys = {self._source_key(source) for source in expected}
        actual_keys = {self._source_key(info.source)}
        if self.context and (getattr(info, 'parent_ids', ()) or info.parent_id):
            pending = list(info.parent_ids) if getattr(info, 'parent_ids', ()) else [info.parent_id]
            visited = set()
            actual_keys = set()
            while pending:
                dataset_id = pending.pop()
                if dataset_id in visited:
                    continue
                visited.add(dataset_id)
                parent = self.context.datasets.metadata.get(dataset_id)
                if parent is None:
                    continue
                if getattr(parent, 'parent_ids', ()):
                    pending.extend(parent.parent_ids)
                elif parent.parent_id:
                    pending.append(parent.parent_id)
                else:
                    actual_keys.add(self._source_key(parent.source))
        return expected_keys.issubset(actual_keys)

    @staticmethod
    def _has_scope(current):
        scope = current.get('scope', {})
        return bool(scope.get('conditions') or scope.get('any_conditions') or scope.get('unresolved'))

    def _scope_valid(self, executed, current, histogram_column=None):
        if not self._has_scope(current): return True
        if scope_matches(executed, current['scope'], histogram_column=histogram_column): return True
        self._record_scope_error(current)
        return False

    def _record_scope_error(self, current):
        current['scope_error'] = 'request_scope_unresolved' if current['scope'].get('unresolved') else 'request_scope_mismatch'
        self.diagnostics.emit('request_scope_rejected', request_id=current.get('request_id'),
            reason=current['scope_error'], columns=sorted({c['column'] for c in current['scope'].get('conditions', [])}),
            unresolved=current['scope'].get('unresolved', []))

    def _statistical_scope_valid(self, info, arguments, current):
        """Allow an explicit list of all compared groups without filtering rows.

        In ``y='yes'`` versus ``y='no'``, the literals label the populations
        being compared. They do not narrow a complete raw dataset. The
        exception stays fail-closed: exactly one IN condition on the declared
        group column must equal every observed non-null level.
        """
        if not self._has_scope(current):
            return True
        valid = (arguments.get('test') in {'independent_t', 'mann_whitney'}
                 and self._all_group_levels_scope_valid(
                     info, arguments.get('group_column'), current))
        if valid:
            return True
        self._record_scope_error(current)
        return False

    def _all_group_levels_scope_valid(self, info, group_column, current):
        """Treat explicit values as labels only when they cover every group."""
        scope = current.get('scope', {})
        if (scope.get('unresolved') or scope.get('any_conditions')
                or scope.get('measure_conditions') or scope.get('ratio')
                or self.context is None or info.id not in self.context.datasets.frames
                or not group_column or group_column not in info.columns):
            return False
        conditions = scope.get('conditions', [])
        if (len(conditions) != 1 or conditions[0].get('column') != group_column
                or conditions[0].get('op') != 'in'):
            return False

        def canonical(value):
            if hasattr(value, 'item'):
                try:
                    value = value.item()
                except (TypeError, ValueError):
                    pass
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

        requested = {canonical(value) for value in conditions[0].get('value', [])}
        observed = {canonical(value) for value in
                    self.context.datasets.frames[info.id][group_column].dropna().unique().tolist()}
        return bool(requested) and requested == observed

    def _chart_scope_valid(self, info, arguments, current):
        if not self._has_scope(current):
            return True
        histogram_column = arguments.get('x') if arguments.get('kind') == 'histogram' else None
        if scope_matches(info, current['scope'], histogram_column=histogram_column):
            return True
        if (arguments.get('kind') == 'boxplot' and arguments.get('category')
                and self._all_group_levels_scope_valid(info, arguments['category'], current)):
            return True
        self._record_scope_error(current)
        return False

    def _count_rate_scope_valid(self, info, arguments, current):
        if info is None or not current.get('count_rate_layout'):
            return False
        scope = current.get('scope', {})
        ratio = scope.get('ratio') or {}
        measures = scope.get('measure_conditions') or []
        if (scope.get('unresolved') or scope.get('any_conditions')
                or len(measures) != 1 or measures[0].get('op') != 'eq'):
            return False
        outcome = arguments.get('outcome_column')
        group = arguments.get('group_column')
        required = set(current.get('required_columns', []))
        if (outcome != ratio.get('column') or measures[0].get('column') != outcome
                or arguments.get('success_value') != measures[0].get('value')
                or arguments.get('layout') != current.get('count_rate_layout')
                or not group or group == outcome
                or not {group, outcome}.issubset(info.columns)
                or not {group, outcome}.issubset(required)):
            return False
        return self._scope_valid(info, current)

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
        expected_kind = current.get('count_rate_layout') or current.get('kind')
        if expected_kind and card.kind != expected_kind: return False
        if current.get('count_rate_layout') and current.get('count_rate_group_column'):
            ratio = current.get('scope', {}).get('ratio') or {}
            expected = [current['count_rate_group_column'], ratio.get('column')]
            expected = [column for column in expected if column]
        else:
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
            if card.kind == 'boxplot' and len(card.columns) == 2:
                if not self._chart_scope_valid(info, {
                        'kind': 'boxplot', 'x': card.columns[0],
                        'category': card.columns[1]}, current):
                    return False
            elif not self._scope_valid(info, current, value_column):
                return False
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
        if arguments.get('current_result_only') and not current.get('current_result_only'):
            joined = current.get('join_evidence', {}).get('dataset_id')
            derived = current.get('outlier_dataset')
            if arguments.get('dataset_id') not in {joined, derived}:
                return False
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
            lineage_columns = set(info.columns)
            if self.context:
                pending = list(getattr(info, 'parent_ids', ()) or (() if not info.parent_id else (info.parent_id,)))
                visited = set()
                while pending:
                    parent_id = pending.pop()
                    if parent_id in visited:
                        continue
                    visited.add(parent_id)
                    parent = self.context.datasets.metadata.get(parent_id)
                    if parent is None:
                        continue
                    lineage_columns.update(parent.columns)
                    pending.extend(getattr(parent, 'parent_ids', ()) or
                                   (() if not parent.parent_id else (parent.parent_id,)))
            if not set(current.get('required_columns', [])).issubset(columns | lineage_columns | condition_columns): return False
            if current.get('calculation') and not operations and not tree.find(exp.Div): return False
        except (ValueError, TypeError):
            return False
        self.diagnostics.emit('calculation_evidence', request_id=current.get('request_id'),
            dataset_id=dataset_id, source=info.source, coverage=info.coverage, grain=info.grain)
        return True

    def _complete(self, current):
        if current.get('data_load'): return bool(current.get('load_evidence_id'))
        if current.get('count_rate_layout') and not current.get('count_rate_evidence'): return False
        if current.get('join') and not current.get('join_evidence'): return False
        if current.get('time_series_frequency') and not current.get('time_series_evidence'): return False
        if current.get('statistical_kind') and not current.get('statistical_evidence'): return False
        if current.get('outlier_spec') and not current.get('outlier_evidence'): return False
        if current.get('outlier_aggregate_requested'):
            evidence = current.get('outlier_aggregate_evidence', {})
            required_key = ('comparison' if current.get('outlier_aggregate_mode') == 'grouped_comparison'
                            else 'grouped')
            if required_key not in evidence: return False
            if (current.get('outlier_aggregate_mode') == 'top_frequency'
                    and current.get('outlier_metric_aggregation') and 'overall' not in evidence):
                return False
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
        if current.get('join_evidence') and self.context:
            evidence = current['join_evidence']
            summary = evidence['summary']
            info = self.context.datasets.metadata[evidence['dataset_id']]
            parts.append(
                f"보유 dataset 두 개를 {summary['how']} join해 {info.rows:,}행, {len(info.columns):,}열의 결과를 저장했습니다.\n"
                f"cardinality: {summary['relationship']}; key 일치 {summary['matched_distinct_keys']:,}개; "
                f"미일치 왼쪽 {summary['unmatched_left_rows']:,}행, 오른쪽 {summary['unmatched_right_rows']:,}행; "
                f"NULL key 왼쪽 {summary['left_null_key_rows']:,}행, 오른쪽 {summary['right_null_key_rows']:,}행.\n"
                f"분석 범위: {evidence.get('scope')}"
            )
        if current.get('time_series_evidence'):
            evidence = current['time_series_evidence']
            result = evidence['time_series_result']
            line = (
                f"{result['time_column']}을 {result['frequency']} 단위로 준비했습니다. "
                f"{result['aggregation']} 집계 {result['output_rows']:,}행, "
                f"timezone {result['timezone']}입니다.\n"
                f"사용 {result['complete_rows']:,}행, 제외 {result['dropped_rows']:,}행, "
                f"중복 시각 관측 {result['duplicate_time_rows']:,}행, "
                f"추가한 빈 구간 {result['gap_rows_added']:,}행(gap policy: {result['gap_policy']})."
            )
            if result.get('group_column'):
                line += f"\n{result['group_column']} 기준 {result['group_count']:,}개 series를 분리했습니다."
            line += f"\n기간: {result['start']} ~ {result['end']}.\n분석 범위: {evidence.get('scope')}"
            parts.append(line)
        if current.get('statistical_evidence'):
            evidence = current['statistical_evidence']
            result = evidence['test_result']
            sample = result['sample']
            line = (
                f"{result['method']} 결과입니다. 사용 {sample['complete_rows']:,}행, "
                f"결측 제외 {sample['dropped_rows']:,}행."
            )
            if result.get('statistic') is not None:
                line += f"\n통계량: {result['statistic']}; 자유도: {result.get('degrees_of_freedom')}; p-value: {result.get('p_value')}."
                line += (f" alpha={result['alpha']} 기준으로 귀무가설을 기각합니다."
                         if result.get('significant') else
                         f" alpha={result['alpha']} 기준으로 귀무가설을 기각할 근거가 부족합니다.")
            if result.get('estimate'):
                line += f"\n추정값({result['estimate']['name']}): {result['estimate']['value']}."
            if result.get('effect_size'):
                effect = result['effect_size']
                line += f"\n효과크기({effect['name']}): {effect.get('value')}."
            if result.get('confidence_intervals'):
                intervals = result['confidence_intervals']
                first = intervals[0]
                line += (f"\n{first.get('level', 1-result['alpha']):.1%} 신뢰구간"
                         f"({first['parameter']}): [{first['lower']}, {first['upper']}].")
                if len(intervals) > 1:
                    line += f" 추가 신뢰구간 {len(intervals)-1}개는 구조화 결과에 보존했습니다."
            if result.get('warnings'):
                line += "\n주의: " + " ".join(result['warnings'])
            if result.get('kind') == 'paired_t':
                line += "\n행 단위 쌍이 동일 관측 단위인지 데이터만으로 검증할 수 없습니다."
            else:
                line += "\n관측치 독립성은 데이터만으로 검증할 수 없습니다."
            line += f"\n분석 범위: {evidence.get('scope')}"
            parts.append(line)
        if current.get('outlier_evidence'):
            evidence = current['outlier_evidence']
            result = evidence['outlier_result']
            sample, counts = result['sample'], result['counts']
            thresholds, distribution = result['thresholds'], result['distribution']
            line = (
                f"{result['column']}에 {result['method']} {result['tail']} 기준을 적용했습니다. "
                f"유효값 {sample['valid_rows']:,}개, 결측 제외 {sample['missing_rows']:,}개.\n"
                f"하한 {thresholds['lower']}, 상한 {thresholds['upper']}; "
                f"선택된 이상치 {counts['selected']:,}개 ({counts['selected_percent']:.2f}%).\n"
                f"전체 유효값 범위: {distribution['minimum']} ~ {distribution['maximum']}; "
                f"하한 미만 {counts['lower']:,}개, 상한 초과 {counts['upper']:,}개."
            )
            if result.get('warnings'):
                line += "\n주의: " + " ".join(result['warnings'])
            line += f"\n분석 범위: {evidence.get('scope')}"
            parts.append(line)
        for evidence_key in ('overall', 'grouped'):
            evidence = current.get('outlier_aggregate_evidence', {}).get(evidence_key)
            if not evidence:
                continue
            result = evidence['aggregation_result']
            label = 'cohort 전체 집계' if evidence_key == 'overall' else 'cohort 그룹 집계'
            parts.append(
                f"{label}: {result['aggregation']}"
                + (f"({result['value_column']})" if result.get('value_column') else "(*)")
                + (f" by {result['group_column']}" if result.get('group_column') else "")
                + f" · 완전한 관측값 {result['complete_rows']:,}행 · 제외 {result['dropped_rows']:,}행\n"
                + '```csv\n'
                + self.context.datasets.frames[evidence['dataset']['id']].head(15).to_csv(index=False).strip()
                + '\n```\n분석 범위: ' + str(evidence.get('scope', ''))
            )
        comparison = current.get('outlier_aggregate_evidence', {}).get('comparison')
        if comparison:
            result = comparison['comparison_result']
            parts.append(
                f"원본 전체와 cohort의 그룹 집계 비교: {result['aggregation']}"
                + (f"({result['value_column']})" if result.get('value_column') else "(*)")
                + f" by {result['group_column']} · 기준 {result['baseline_complete_rows']:,}행 · "
                + f"cohort {result['cohort_complete_rows']:,}행\n"
                + '```csv\n'
                + self.context.datasets.frames[comparison['dataset']['id']].head(15).to_csv(index=False).strip()
                + '\n```\n분석 범위: ' + str(comparison.get('scope', ''))
            )
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
        if (self.context and current.get('time_series_frequency')
                and not current.get('time_series_evidence') and not self._has_scope(current)):
            required = list(current.get('required_columns', []))
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and not info.aggregation
                and set(required).issubset(info.columns)
                and self._source_matches(info, current)
                and self._fresh_for_request(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known))]
            plans = []
            from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
            import pandas as pd
            for info in candidates:
                frame = self.context.datasets.frames[info.id]
                time_columns = []
                for column in required:
                    series = frame[column]
                    if is_datetime64_any_dtype(series):
                        time_columns.append(column)
                    elif not is_numeric_dtype(series):
                        non_null = int(series.notna().sum())
                        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
                        if non_null >= 2 and int(parsed.notna().sum()) / non_null >= 0.95:
                            time_columns.append(column)
                if len(time_columns) != 1:
                    continue
                time_column = time_columns[0]
                remaining = [column for column in required if column != time_column]
                aggregation = current.get('time_series_aggregation')
                if aggregation == 'count':
                    value_column = ''
                    group_candidates = remaining
                else:
                    numeric_columns = [column for column in remaining
                                       if is_numeric_dtype(frame[column])]
                    if len(numeric_columns) != 1:
                        continue
                    value_column = numeric_columns[0]
                    group_candidates = [column for column in remaining if column != value_column]
                if len(group_candidates) > 1:
                    continue
                group_column = group_candidates[0] if group_candidates else ''
                if group_column and not 1 <= frame[group_column].nunique(dropna=True) <= 20:
                    continue
                plans.append((info, time_column, value_column, group_column))
            if len(plans) == 1:
                info, time_column, value_column, group_column = plans[0]
                arguments = {
                    'dataset_id':info.id,
                    'time_column':time_column,
                    'frequency':current['time_series_frequency'],
                    'aggregation':current['time_series_aggregation'],
                    'gap_policy':current.get('time_series_gap_policy', 'omit'),
                    'timezone':current.get('time_series_timezone', ''),
                    'max_output_rows':5000,
                }
                if value_column:
                    arguments['value_column'] = value_column
                if group_column:
                    arguments['group_column'] = group_column
                if not any(c.get('name') == 'prepare_time_series' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name':'prepare_time_series', 'args':arguments}
        if (self.context and current.get('time_series_evidence')
                and current.get('chart') and not current.get('artifact_ids')):
            evidence = current['time_series_evidence']['time_series_result']
            arguments = {
                'dataset_id':current['time_series_dataset'],
                'kind':'line',
                'x':evidence['time_column'],
                'y':evidence['value_column'],
                'aggregation':'none',
                'sort':'ascending',
            }
            if evidence.get('group_column'):
                arguments['category'] = evidence['group_column']
            if current.get('chart_cumulative'):
                arguments['cumulative'] = True
            if not any(c.get('name') == 'render_chart_spec' and c.get('args') == arguments
                       for c in calls.values()):
                return {'name':'render_chart_spec', 'args':arguments}
        if (self.context and current.get('outlier_spec')
                and not current.get('outlier_evidence') and not self._has_scope(current)):
            column = current.get('outlier_column')
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and not info.aggregation
                and column and column in info.columns
                and self._source_matches(info, current)
                and self._fresh_for_request(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known))]
            if len(candidates) == 1:
                info = candidates[0]
                frame = self.context.datasets.frames[info.id]
                from pandas.api.types import is_numeric_dtype
                if is_numeric_dtype(frame[column]):
                    spec = current['outlier_spec']
                    arguments = {'dataset_id':info.id, 'column':column,
                                 'method':spec['method'], 'tail':spec['tail'],
                                 'threshold':spec['threshold']}
                    if spec['method'] == 'quantile':
                        arguments.update(lower_quantile=spec['lower_quantile'],
                                         upper_quantile=spec['upper_quantile'])
                    tool_name = ('select_outlier_rows' if current.get('outlier_followup')
                                 else 'detect_outliers')
                    if tool_name == 'select_outlier_rows':
                        arguments['selection'] = current.get('outlier_selection', 'outliers')
                    if not any(c.get('name') == tool_name and c.get('args') == arguments
                               for c in calls.values()):
                        return {'name':tool_name, 'args':arguments}
        if (self.context and current.get('outlier_aggregate_requested')
                and current.get('outlier_dataset')):
            child_id = current['outlier_dataset']
            child = self.context.datasets.metadata.get(child_id)
            evidence = current.get('outlier_aggregate_evidence', {})
            if child is not None and child.grain == 'raw' and not child.aggregation:
                mode = current.get('outlier_aggregate_mode')
                if mode == 'grouped_comparison' and 'comparison' not in evidence:
                    baseline_id = child.parent_id
                    arguments = {
                        'baseline_dataset_id': baseline_id,
                        'cohort_dataset_id': child_id,
                        'aggregation': current['outlier_metric_aggregation'],
                        'value_column': current['outlier_metric_column'],
                        'group_column': current['outlier_group_column'],
                        'sort': 'group_ascending',
                    }
                    if (baseline_id and not any(
                            c.get('name') == 'compare_group_aggregates' and c.get('args') == arguments
                            for c in calls.values())):
                        return {'name':'compare_group_aggregates', 'args':arguments}
                if (mode == 'top_frequency' and current.get('outlier_metric_aggregation')
                        and 'overall' not in evidence):
                    arguments = {
                        'dataset_id': child_id,
                        'aggregation': current['outlier_metric_aggregation'],
                        'value_column': current['outlier_metric_column'],
                        'sort': 'descending',
                        'top_n': 0,
                    }
                    if not any(c.get('name') == 'aggregate_dataset' and c.get('args') == arguments
                               for c in calls.values()):
                        return {'name':'aggregate_dataset', 'args':arguments}
                if 'grouped' not in evidence:
                    if mode == 'top_frequency':
                        arguments = {
                            'dataset_id': child_id,
                            'aggregation': 'count',
                            'group_column': current['outlier_group_column'],
                            'sort': 'descending',
                            'top_n': current.get('outlier_top_n', 0),
                        }
                    elif mode == 'grouped_metric':
                        arguments = {
                            'dataset_id': child_id,
                            'aggregation': current['outlier_metric_aggregation'],
                            'value_column': current['outlier_metric_column'],
                            'group_column': current['outlier_group_column'],
                            'sort': 'descending',
                            'top_n': 0,
                        }
                    else:
                        arguments = None
                    if (arguments and not any(
                            c.get('name') == 'aggregate_dataset' and c.get('args') == arguments
                            for c in calls.values())):
                        return {'name':'aggregate_dataset', 'args':arguments}
        if (self.context and current.get('statistical_kind')
                and not current.get('statistical_evidence')):
            required = list(current.get('required_columns', []))
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and not info.aggregation
                and set(required).issubset(info.columns)
                and self._source_matches(info, current)
                and self._fresh_for_request(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known))]
            if len(candidates) == 1 and required:
                info = candidates[0]
                frame = self.context.datasets.frames[info.id]
                numeric = [column for column in required
                           if _dtype_family(frame[column].dtype) == 'numeric']
                low_cardinality = [column for column in required
                                   if frame[column].nunique(dropna=True) <= 20]
                kind = current['statistical_kind']
                arguments = None
                if kind == 'mean_ci' and len(required) == 1 and len(numeric) == 1:
                    arguments = {'dataset_id':info.id, 'test':kind, 'value_column':numeric[0]}
                elif kind == 'paired_t' and len(required) == 2 and len(numeric) == 2:
                    arguments = {'dataset_id':info.id, 'test':kind,
                                 'value_column':required[0], 'paired_column':required[1]}
                elif kind == 'chi_square' and len(required) == 2 and len(low_cardinality) == 2:
                    arguments = {'dataset_id':info.id, 'test':kind,
                                 'value_column':required[0], 'group_column':required[1]}
                elif kind in {'independent_t', 'one_way_anova', 'mann_whitney'} and len(required) == 2:
                    if len(numeric) == 1:
                        value_candidates = numeric
                        group_candidates = [column for column in required
                                            if column != numeric[0] and column in low_cardinality]
                    elif len(numeric) == 2:
                        cardinality = {column: frame[column].nunique(dropna=True) for column in numeric}
                        ordered = sorted(numeric, key=lambda column: cardinality[column])
                        group_candidates = [ordered[0]] if cardinality[ordered[0]] <= 20 and \
                            cardinality[ordered[0]] < cardinality[ordered[1]] else []
                        value_candidates = [ordered[1]] if group_candidates else []
                    else:
                        value_candidates, group_candidates = [], []
                    if len(value_candidates) == 1 and len(group_candidates) == 1:
                        arguments = {'dataset_id':info.id, 'test':kind,
                                     'value_column':value_candidates[0], 'group_column':group_candidates[0]}
                if (arguments and self._statistical_scope_valid(info, arguments, current)
                        and not any(c.get('name') == 'statistical_test' and c.get('args') == arguments
                                    for c in calls.values())):
                    return {'name':'statistical_test', 'args':arguments}
        if (self.context and current.get('join') and current.get('join_how')
                and not current.get('join_evidence') and not self._has_scope(current)):
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and not info.aggregation
                and not getattr(info, 'parent_ids', ())
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known
                        and self._fresh_for_request(info, current)))]
            requested_sources = current.get('required_sources', [])
            if requested_sources:
                ordered = []
                for source in requested_sources:
                    matches = [info for info in candidates
                               if self._source_key(info.source) == self._source_key(source)]
                    if len(matches) != 1:
                        ordered = []
                        break
                    ordered.append(matches[0])
                candidates = ordered
            if len(candidates) == 2 and candidates[0].id != candidates[1].id:
                shared = [column for column in current.get('required_columns', [])
                          if column in candidates[0].columns and column in candidates[1].columns]
                if 1 <= len(shared) <= 4:
                    arguments = {
                        'left_dataset_id': candidates[0].id,
                        'right_dataset_id': candidates[1].id,
                        'left_on': shared,
                        'right_on': shared,
                        'how': current['join_how'],
                    }
                    if not any(c.get('name') == 'join_datasets' and c.get('args') == arguments
                               for c in calls.values()):
                        return {'name':'join_datasets', 'args':arguments}
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
        if (self.context and current.get('count_rate_layout')
                and not current.get('count_rate_evidence')
                and not current.get('fresh_source_required')):
            scope = current.get('scope', {})
            ratio = scope.get('ratio') or {}
            measures = scope.get('measure_conditions') or []
            outcome = ratio.get('column')
            required = list(current.get('required_columns', []))
            groups = [column for column in required if column != outcome]
            explicit_groups = [column for column in current.get('explicit_columns', [])
                               if column != outcome and column in groups]
            if len(groups) != 1 and len(explicit_groups) == 1:
                groups = explicit_groups
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and not info.aggregation
                and outcome and len(groups) == 1
                and {groups[0], outcome}.issubset(info.columns)
                and self._source_matches(info, current)
                and self._fresh_for_request(info, current)
                and (current.get('current_result_only')
                    or (info.coverage == 'complete' and info.predicate_known))]
            if (len(candidates) == 1 and len(measures) == 1
                    and measures[0].get('column') == outcome
                    and measures[0].get('op') == 'eq'):
                info = candidates[0]
                frame = self.context.datasets.frames[info.id]
                group = groups[0]
                current['count_rate_group_column'] = group
                observed = {str(value).strip().casefold()
                            for value in frame[group].dropna().unique().tolist()}
                month_tokens = {
                    'jan','january','feb','february','mar','march','apr','april','may',
                    'jun','june','jul','july','aug','august','sep','sept','september',
                    'oct','october','nov','november','dec','december'}
                if observed and observed.issubset(month_tokens):
                    sorting = 'calendar_month'
                elif _dtype_family(frame[group].dtype) == 'numeric':
                    sorting = 'group_ascending'
                else:
                    sorting = 'count_descending'
                arguments = {
                    'dataset_id':info.id,
                    'group_column':group,
                    'outcome_column':outcome,
                    'success_value':measures[0].get('value'),
                    'layout':current['count_rate_layout'],
                    'sort':sorting,
                    'top_n':50,
                }
                if (self._count_rate_scope_valid(info, arguments, current)
                        and not any(c.get('name') == 'render_count_rate_chart'
                                    and c.get('args') == arguments for c in calls.values())):
                    return {'name':'render_count_rate_chart', 'args':arguments}
        if (self.context and current.get('chart') and not current.get('time_series_frequency')
                and current.get('kind') in {'bar', 'line', 'scatter', 'boxplot'}
                and not current.get('count_rate_layout')
                and not current.get('fresh_source_required') and not current.get('artifact_ids')):
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
                elif current['kind'] == 'boxplot' and len(columns) == 2:
                    numeric_columns = [column for column in columns if is_numeric_dtype(frame[column])]
                    if len(numeric_columns) == 1:
                        value_column = numeric_columns[0]
                        category_columns = [column for column in columns if column != value_column
                                            and frame[column].nunique(dropna=True) <= 20]
                    elif len(numeric_columns) == 2:
                        ordered = sorted(numeric_columns,
                            key=lambda column: frame[column].nunique(dropna=True))
                        category_columns = [ordered[0]] if (
                            frame[ordered[0]].nunique(dropna=True) <= 20
                            and frame[ordered[0]].nunique(dropna=True)
                                < frame[ordered[1]].nunique(dropna=True)) else []
                        value_column = ordered[1] if category_columns else None
                    else:
                        value_column, category_columns = None, []
                    if value_column and len(category_columns) == 1:
                        arguments = {'dataset_id':candidates[0].id,'kind':'boxplot',
                                     'x':value_column,'category':category_columns[0]}
                elif current['kind'] == 'scatter' and len(columns) == 2 and all(
                        is_numeric_dtype(frame[column]) for column in columns):
                    arguments = {'dataset_id':candidates[0].id,'kind':'scatter','x':columns[0],'y':columns[1]}
                elif current['kind'] == 'bar' and len(columns) == 1 and 1 <= frame[columns[0]].nunique() <= 50:
                    arguments = {'dataset_id':candidates[0].id,'kind':'bar','x':columns[0],
                                 'aggregation':'count','sort':'descending','top_n':50}
                elif current['kind'] == 'line' and len(columns) == 1:
                    column = columns[0]
                    unique = frame[column].nunique(dropna=True)
                    if is_numeric_dtype(frame[column]) and 2 <= unique <= 5_000:
                        arguments = {'dataset_id':candidates[0].id,'kind':'line','x':column,
                                     'aggregation':'count','sort':'ascending'}
                        if current.get('chart_cumulative'):
                            arguments['cumulative'] = True
                    elif current.get('chart_cumulative') and 2 <= unique <= 12:
                        month_tokens = {
                            'jan','january','feb','february','mar','march','apr','april','may',
                            'jun','june','jul','july','aug','august','sep','sept','september',
                            'oct','october','nov','november','dec','december'}
                        observed = {str(value).strip().casefold()
                                    for value in frame[column].dropna().unique().tolist()}
                        if observed and observed.issubset(month_tokens):
                            arguments = {'dataset_id':candidates[0].id,'kind':'line','x':column,
                                         'aggregation':'count','sort':'calendar_month',
                                         'cumulative':True}
                elif current['kind'] == 'line' and len(columns) == 2:
                    time_columns = [column for column in columns if is_datetime64_any_dtype(frame[column])]
                    numeric_columns = [column for column in columns if is_numeric_dtype(frame[column])]
                    if len(time_columns) == 1 and len(numeric_columns) == 1 and frame[time_columns[0]].is_unique:
                        arguments = {'dataset_id':candidates[0].id,'kind':'line',
                                     'x':time_columns[0],'y':numeric_columns[0],'sort':'ascending'}
                if (arguments and self._chart_scope_valid(candidates[0], arguments, current)
                        and not any(c.get('name') == 'render_chart_spec' and c.get('args') == arguments
                                    for c in calls.values())):
                    return {'name':'render_chart_spec','args':arguments}
        scope = current.get('scope', {})
        if (self.context and current.get('outlier_followup')
                and current.get('outlier_dataset') and current.get('calculation')
                and set(current.get('operations', [])) in ({'RATIO'}, {'COUNT', 'RATIO'})
                and not current.get('evidence_ids')):
            ratio = scope.get('ratio') or {}
            measure = scope.get('measure_conditions') or []
            child_id = current['outlier_dataset']
            child = self.context.datasets.metadata.get(child_id)
            if (child is not None and ratio.get('column') in child.columns
                    and len(measure) == 1 and measure[0].get('op') == 'eq'
                    and measure[0].get('column') == ratio.get('column')
                    and not scope.get('unresolved')):
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
                arguments = {'dataset_id':child_id, 'query':query, 'current_result_only':True}
                if not any(c.get('name') == 'local_analysis_sql' and c.get('args') == arguments
                           for c in calls.values()):
                    return {'name':'local_analysis_sql', 'args':arguments}
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
        deterministic_scalar = bool(current.get('current_result_only') or (
            len(aggregate_operations) > 1 and not self._has_scope(current)))
        if (self.context and current.get('calculation') and deterministic_scalar
                and len(aggregate_operations) >= 1
                and set(aggregate_operations).issubset(supported_aggregates)
                and not scope.get('unresolved') and not current.get('evidence_ids')):
            # One or more descriptive aggregates over one numeric measure have
            # a single deterministic local plan. Conditions are applied by the
            # bounded tool contract, so a loaded DataFrame never needs a model
            # merely to compute a basic filtered scalar.
            predicate_columns = {item['column'] for item in (
                scope.get('conditions', []) + scope.get('any_conditions', []))}
            columns = [column for column in current.get('required_columns', [])
                       if column not in predicate_columns]
            if not columns:
                columns = [column for column in scope.get('columns', [])
                           if column not in predicate_columns]
            if len(columns) != 1:
                return None
            needed_columns = predicate_columns | {columns[0]}
            candidates = [info for info in self.context.datasets.metadata.values()
                if info.grain == 'raw' and (current.get('current_result_only') or info.predicate_known)
                and needed_columns.issubset(info.columns) and self._source_matches(info, current)
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
                query = 'SELECT ' + ', '.join(projections) + ' FROM data'
                if scope.get('any_conditions'):
                    pieces = []
                    if scope.get('conditions'):
                        pieces.append(self._where_sql(scope['conditions']))
                    pieces.append('(' + ' OR '.join(
                        self._where_sql([item]) for item in scope['any_conditions']) + ')')
                    query += ' WHERE ' + ' AND '.join(pieces)
                    arguments = {'dataset_id':numeric_candidates[0].id, 'query':query}
                else:
                    arguments = {'dataset_id':numeric_candidates[0].id, 'query':query,
                                 'requested_conditions':deepcopy(scope.get('conditions', []))}
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
        if (current.get('data_load') or current.get('plan') or current.get('join')
                or current.get('time_series_frequency') or current.get('statistical_kind')
                or current.get('outlier_spec') or current.get('chart') or current.get('calculation')
                or current.get('metadata_kind') or current.get('profile_kind')) and self._complete(current):
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
        if call.get('name') in {'recommend_chart_images', 'render_chart_spec', 'render_count_rate_chart', 'render_histogram'} and self.context:
            info = self.context.datasets.metadata.get(arguments.get('dataset_id'))
            if info is None: return True  # The tool adapter reports unknown IDs.
            if call.get('name') == 'render_count_rate_chart':
                return self._count_rate_scope_valid(info, arguments, current)
            if call.get('name') == 'render_chart_spec':
                return self._chart_scope_valid(info, arguments, current)
            column = arguments.get('value_column') or (
                arguments.get('x') if arguments.get('kind') == 'histogram' else None)
            return self._scope_valid(info, current, column)
        if call.get('name') == 'statistical_test' and self.context:
            info = self.context.datasets.metadata.get(arguments.get('dataset_id'))
            return True if info is None else self._statistical_scope_valid(info, arguments, current)
        if call.get('name') == 'prepare_time_series' and self.context:
            info = self.context.datasets.metadata.get(arguments.get('dataset_id'))
            return (True if info is None else not self._has_scope(current)
                    and self._scope_valid(info, current))
        if call.get('name') == 'aggregate_dataset' and self.context:
            info = self.context.datasets.metadata.get(arguments.get('dataset_id'))
            if info is None:
                return True
            if current.get('outlier_aggregate_requested'):
                return info.id == current.get('outlier_dataset') and info.grain == 'raw'
            return self._scope_valid(info, current)
        if call.get('name') == 'compare_group_aggregates' and self.context:
            baseline = self.context.datasets.metadata.get(arguments.get('baseline_dataset_id'))
            cohort = self.context.datasets.metadata.get(arguments.get('cohort_dataset_id'))
            if baseline is None or cohort is None:
                return True
            return (current.get('outlier_aggregate_mode') == 'grouped_comparison'
                    and cohort.id == current.get('outlier_dataset')
                    and cohort.parent_id == baseline.id
                    and baseline.grain == cohort.grain == 'raw')
        if call.get('name') in {'detect_outliers', 'select_outlier_rows'} and self.context:
            info = self.context.datasets.metadata.get(arguments.get('dataset_id'))
            return True if info is None else self._scope_valid(info, current)
        if call.get('name') == 'join_datasets' and self.context:
            if current.get('scope', {}).get('any_conditions'):
                return False
            parents = [self.context.datasets.metadata.get(arguments.get(key))
                       for key in ('left_dataset_id', 'right_dataset_id')]
            if any(parent is None for parent in parents):
                return True
            for condition in current.get('scope', {}).get('conditions', []):
                matches = [parent for parent in parents if condition['column'] in parent.columns]
                if len(matches) != 1 or not scope_matches(matches[0], {
                        'conditions':[condition], 'any_conditions':[], 'unresolved':[], 'columns':[condition['column']]}):
                    return False
            return True
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

    def _fresh_for_request(self, info, current):
        if not current.get('fresh_source_required'):
            return True
        if self.context and (getattr(info, 'parent_ids', ()) or info.parent_id):
            parent_ids = info.parent_ids if getattr(info, 'parent_ids', ()) else (info.parent_id,)
            parents = [self.context.datasets.metadata.get(dataset_id) for dataset_id in parent_ids]
            return bool(parents) and all(parent is not None and self._fresh_for_request(parent, current)
                                         for parent in parents)
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
            if call['name'] in {'recommend_chart_images', 'render_chart_spec', 'render_count_rate_chart', 'render_histogram', 'show_chart'}: current['chart'] = True
            if call['name'] == 'join_datasets': current['join'] = True
            if call['name'] == 'statistical_test': current['statistical_kind'] = call.get('args', {}).get('test')
            if call['name'] == 'prepare_time_series' and not current.get('time_series_frequency'):
                current['time_series_frequency'] = call.get('args', {}).get('frequency')
                current['time_series_aggregation'] = call.get('args', {}).get('aggregation')
                current['time_series_gap_policy'] = call.get('args', {}).get('gap_policy', 'omit')
                current['time_series_timezone'] = call.get('args', {}).get('timezone', '')
            if (call['name'] in {'detect_outliers', 'select_outlier_rows'} and not current.get('outlier_spec')
                    and not current.get('calculation')
                    and len(current.get('required_columns', [])) == 1):
                current['outlier_spec'] = {key:call.get('args', {}).get(key)
                                           for key in ('method', 'tail', 'threshold',
                                                       'lower_quantile', 'upper_quantile')}
                current['outlier_column'] = call.get('args', {}).get('column')
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
            missing_calculation=bool(current.get('calculation') and not current['evidence_ids']),
            missing_join=bool(current.get('join') and not current.get('join_evidence')),
            missing_time_series=bool(current.get('time_series_frequency') and not current.get('time_series_evidence')),
            missing_statistical_test=bool(current.get('statistical_kind') and not current.get('statistical_evidence')),
            missing_outlier_detection=bool(current.get('outlier_spec') and not current.get('outlier_evidence')))
        instruction = ('이전 응답은 완료 증거가 없어 채택되지 않았습니다. 원래 사용자 요청을 계속 수행하세요. '
            '수치/통계는 local_analysis_sql의 실제 계산 결과가 필요하고 결측·고유값·기초 통계는 profile_dataset의 구조화 결과가 필요합니다. 테이블 설명이나 미리보기는 계산 증거가 아닙니다. '
            '두 로딩 dataset의 결합은 join_datasets로 cardinality와 lineage를 확인해야 합니다. many-to-many 차단을 우회하지 말고 먼저 한쪽 grain을 명확히 하세요. '
            '시간 재집계는 prepare_time_series로 datetime 파싱·timezone·중복 시각·gap·빈도·lineage를 확인한 뒤 파생 dataset을 render_chart_spec으로 그리세요. '
            '가설 검정과 평균 신뢰구간은 statistical_test의 구조화 결과가 완료 증거입니다. 표본 수·결측·가정·효과크기·신뢰구간을 확인하세요. '
            '이상치 기준과 건수는 detect_outliers의 구조화 결과가 완료 증거입니다. IQR·Z-score·MAD·분위수 기준, tail, 결측과 coverage를 확인하세요. '
            '요청한 출처, 컬럼, 집계와 필터를 유지하세요. 지정 차트와 수정은 render_chart_spec을 사용하고, 그룹별 전체 건수와 명시된 성공값 비율의 이중축·2열 패널은 render_count_rate_chart를 사용하세요. 원격 데이터가 필요한 히스토그램은 prepare_histogram(source, column, where_sql)을 사용하세요. '
            '이 도구는 먼저 재사용 가능한 보유 데이터를 찾고, 부족한 경우에만 승인형 로딩과 렌더링 계획을 만듭니다. '
            'query_databricks 호출이 승인 카드를 생성하며 실제 조회는 사용자 승인을 기다립니다. '
            '동일한 실패 호출을 반복하거나 증거 없이 완료했다고 말하지 마세요.')
        if self._has_scope(current): instruction += '\n' + self._scope_instruction(current)
        return {'recovery': current, 'messages': [RemoveMessage(id=last.id),
            SystemMessage(content=instruction, additional_kwargs={'lc_source': 'recovery', 'invalidated_message_id': last.id})], 'jump_to': 'model'}
