"""Local dataset and skill tools for the shared analysis loop."""
from dataclasses import asdict, replace
from sqlglot import exp, parse
from sqlglot.errors import SqlglotError
from core.analysis_catalog import compact_catalog, resolve_table_context

from core.analysis_tool_contract import (
    AnalysisToolContext,
    COMMON_TOOL_RESULT_SCHEMA,
    ToolDefinition,
    normalize_tool_result,
)
from utils.analysis_datasets import AnalysisNeed, Condition, DatasetStore, assess_reuse, filter_frame
from utils.analysis_skill_registry import AnalysisSkillRegistry
from utils.analysis_charts import (
    histogram_from_counts,
    recommend_charts,
    render_chart_spec as build_chart_from_spec,
    validate_frequency_dataset,
)
from utils.analysis_provenance import raw_conditions, query_conditions, query_coverage, single_table, table_identity
from core.analysis_sql import local_query, validate_query
from utils.analysis_profile import profile_dataset as build_dataset_profile
from utils.analysis_join import join_datasets as build_joined_dataset
from utils.analysis_statistics import statistical_test as run_statistical_test
from utils.analysis_outliers import (
    detect_outliers as run_outlier_detection,
    select_outlier_rows as run_outlier_selection,
)


def build_analysis_tools(context: AnalysisToolContext) -> list[ToolDefinition]:
    datasets = context.datasets
    registry = AnalysisSkillRegistry()

    def catalog():
        return compact_catalog({"datasets": [dict(display_name=f"결과 {index}", **asdict(info)) for index,info in enumerate(datasets.metadata.values(),1)],
                "skills": registry.list(), "available_tables": context.reference_context})


    def inspect_table_context(table):
        return resolve_table_context(context.reference_context, datasets, table)

    def inspect_dataset(dataset_id):
        info = datasets.metadata[dataset_id]
        frame = datasets.frames[dataset_id]
        return {"dataset": asdict(info), "dtypes": frame.dtypes.astype(str).to_dict(),
                "preview": frame.head(5).to_dict(orient="records")}

    def profile_dataset(dataset_id, columns=None, offset=0, limit=50):
        return build_dataset_profile(datasets, dataset_id, columns, offset, limit)

    def plan_source_discovery(catalog="", schema="", pattern="", limit=100):
        known_catalogs = {}
        for item in context.reference_context:
            parts = [part.strip().strip('`') for part in str(item.get('table', '')).split('.')]
            if len(parts) == 3 and parts[0]:
                known_catalogs.setdefault(parts[0].casefold(), parts[0])
        for info in datasets.metadata.values():
            parts = [part.strip().strip('`') for part in str(info.source).split('.')]
            if len(parts) == 3 and parts[0]:
                known_catalogs.setdefault(parts[0].casefold(), parts[0])
        requested = catalog.strip().strip('`')
        if requested:
            resolved = known_catalogs.get(requested.casefold())
            if resolved is None:
                return {
                    'status': 'needs_context',
                    'message': '현재 분석 문맥에서 확인된 catalog가 아닙니다. 임의 catalog를 조회하지 않습니다.',
                    'known_catalogs': sorted(known_catalogs.values()),
                }
        elif len(known_catalogs) == 1:
            resolved = next(iter(known_catalogs.values()))
        else:
            return {
                'status': 'needs_context',
                'message': '조회할 catalog를 하나로 특정해야 합니다.',
                'known_catalogs': sorted(known_catalogs.values()),
            }
        try:
            bounded_limit = int(limit)
        except (TypeError, ValueError):
            raise ValueError('limit은 정수여야 합니다.')
        if not 1 <= bounded_limit <= 200:
            raise ValueError('limit은 1~200이어야 합니다.')
        quoted_catalog = '`' + resolved.replace('`', '``') + '`'
        query = (
            'SELECT table_catalog, table_schema, table_name, table_type '
            f'FROM {quoted_catalog}.information_schema.tables'
        )
        predicates = []
        if schema.strip():
            predicates.append("table_schema = '" + schema.strip().replace("'", "''") + "'")
        if pattern.strip():
            predicates.append("instr(lower(table_name), lower('" + pattern.strip().replace("'", "''") + "')) > 0")
        if predicates:
            query += ' WHERE ' + ' AND '.join(predicates)
        query += f' ORDER BY table_catalog, table_schema, table_name LIMIT {bounded_limit}'
        validate_query(query)
        source = f'{resolved}.information_schema.tables'
        return {
            'status': 'planned',
            'discovery_plan': {
                'source': source,
                'query': query,
                'reason': '사용 가능한 테이블 목록을 information_schema에서 조회합니다.',
            },
            'scope': f'{resolved} catalog의 테이블 메타데이터를 최대 {bounded_limit}건 조회하는 계획입니다. 아직 실행되지 않았습니다.',
            'user_action': '이 계획을 query_databricks로 전달하면 사용자 승인 후에만 실행됩니다.',
        }

    def use_dataset(dataset_id, columns, conditions=None, current_result_only=False):
        info = datasets.metadata[dataset_id]
        need = AnalysisNeed(info.source, tuple(columns),
            conditions=tuple(Condition(**c) for c in (conditions or [])),
            current_result_only=current_result_only)
        decision = assess_reuse(info, need)
        if decision.action == "query_source":
            return {"status": "needs_data", **asdict(decision)}
        result = datasets.derive(dataset_id, need)
        return {"status": "ready", "dataset": asdict(result), **asdict(decision)}

    def join_datasets(left_dataset_id, right_dataset_id, left_on, right_on, how):
        return build_joined_dataset(
            datasets,
            left_dataset_id,
            right_dataset_id,
            left_on=left_on,
            right_on=right_on,
            how=how,
            max_rows=context.max_join_rows,
            max_expansion_ratio=context.max_join_expansion_ratio,
        )

    def statistical_test(dataset_id, test, value_column, group_column="",
                         paired_column="", alpha=0.05):
        return run_statistical_test(
            datasets,
            dataset_id,
            test=test,
            value_column=value_column,
            group_column=group_column,
            paired_column=paired_column,
            alpha=alpha,
        )

    def detect_outliers(dataset_id, column, method, tail="both", threshold=1.5,
                        lower_quantile=0.01, upper_quantile=0.99):
        return run_outlier_detection(
            datasets,
            dataset_id,
            column=column,
            method=method,
            tail=tail,
            threshold=threshold,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def select_outlier_rows(dataset_id, column, method, selection="outliers",
                            tail="both", threshold=1.5, lower_quantile=0.01,
                            upper_quantile=0.99):
        return run_outlier_selection(
            datasets,
            dataset_id,
            column=column,
            method=method,
            selection=selection,
            tail=tail,
            threshold=threshold,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def propose_query(source, query, reason):
        validate_query(query)
        return context.propose_query(source=source, query=query, reason=reason)

    def analyze_local(dataset_id, query, current_result_only=False, requested_conditions=None):
        info = datasets.metadata[dataset_id]
        corrections = []
        parsed = parse(query, read='duckdb')
        if (len(parsed) == 1 and isinstance(parsed[0], (exp.AggFunc, exp.Alias))
                and parsed[0].find(exp.AggFunc) and not parsed[0].find(exp.Select)
                and all(c.name in info.columns and not c.table for c in parsed[0].find_all(exp.Column))):
            # A standalone aggregate expression has a unique local SELECT form.
            query = exp.select(parsed[0]).from_('data').sql(dialect='duckdb')
            corrections.append('added_local_select_from')
        tree = validate_query(query, dialect='duckdb')
        table = single_table(tree)
        if table is not None and table.name != 'data':
            expected = source_key(info.source)
            actual = table_identity(table).casefold()
            # The selected dataset ID already binds one local frame. Repair only
            # its exact persisted source name or that source's final table name.
            expected_name = expected.rsplit('.', 1)[-1]
            if actual == expected or (not table.db and not table.catalog and table.name.casefold() == expected_name):
                original_name = table.name
                has_alias = bool(table.alias)
                table.set('this', exp.to_identifier('data'))
                table.set('db', None)
                table.set('catalog', None)
                if not has_alias:
                    for column in tree.find_all(exp.Column):
                        if column.table.casefold() == original_name.casefold():
                            column.set('table', exp.to_identifier('data'))
                query = tree.sql(dialect='duckdb')
                corrections.append('bound_source_table_to_local_data')
        columns = list(tree.find_all(exp.Column))
        if (isinstance(tree, exp.Select) and len(list(tree.find_all(exp.Select))) == 1
                and not list(tree.find_all(exp.Table)) and not tree.args.get('with_')
                and (columns or tree.find(exp.AggFunc))
                and all(not c.table and c.name in info.columns for c in columns)):
            # One bound DataFrame and known column references make this omitted
            # FROM unambiguous. Do not guess a column, filter, or remote source.
            tree = tree.from_('data')
            query = tree.sql(dialect='duckdb')
            corrections.append('added_local_data_from')
        sql_conditions = query_conditions(tree)
        requested = tuple(Condition(**item) for item in requested_conditions) if requested_conditions is not None else None
        # An omitted scope means the SQL's WHERE scope (or the whole source).
        # Never silently inherit a cached subset when the caller asks for all rows.
        scope = requested if requested is not None else (sql_conditions or ())
        need = AnalysisNeed(info.source, info.columns, conditions=scope,
            grain=info.grain if current_result_only else 'raw',
            aggregation=info.aggregation if current_result_only else '',
            current_result_only=current_result_only)
        decision = assess_reuse(info, need)
        if decision.action == 'query_source':
            return {'status': 'needs_data', **asdict(decision),
                    'message': decision.reason + ' 요청 범위를 requested_conditions로 명시하거나, 사용자가 현재 결과 자체를 분석할 때만 current_result_only를 사용하세요.'}
        input_frame = datasets.frames[dataset_id]
        if requested is not None:
            residual = tuple(c for c in requested if c not in info.conditions)
            input_frame = filter_frame(input_frame, residual)
        frame, truncated, tree = local_query(input_frame, query)
        aggregated = bool(tree.args.get("group") or tree.find(exp.AggFunc))
        safe_conditions = raw_conditions(tree)
        conditions = tuple(dict.fromkeys((*info.conditions, *(requested or ()), *(sql_conditions or ()))))
        coverage = query_coverage(tree, truncated=truncated)
        if coverage == 'complete':
            coverage = info.coverage
        result = datasets.register(frame, source=info.source,
            coverage=coverage,
            predicate_known=info.predicate_known and safe_conditions is not None,
            conditions=conditions, grain="aggregate" if aggregated else info.grain,
            aggregation=tree.sql() if aggregated else info.aggregation,
            parent_id=dataset_id, query=query, snapshot=info.snapshot)
        return {"status": "ready", "dataset": asdict(result),
                "preview": frame.head(15).to_dict(orient="records"),
                "applied_corrections": corrections}

    def chart_options(dataset_id, columns=None):
        previews = recommend_charts(datasets, dataset_id, columns)
        for card in previews:
            context.artifacts[card.id] = card
        return {"status": "ready" if previews else "no_valid_chart", "cards": [
            {"id": c.id, "dataset_id": c.dataset_id, "title": c.title,
             "reason": c.reason, "kind": c.kind, "columns": c.columns, "scope": c.scope}
            for c in previews]}

    def render_chart_spec(dataset_id, kind, x, y="", category="", aggregation="none",
                          sort="none", top_n=50, bins=20, title="", x_label="",
                          y_label="", orientation="vertical"):
        card, summary, spec = build_chart_from_spec(
            datasets, dataset_id, kind=kind, x=x, y=y, category=category,
            aggregation=aggregation, sort=sort, top_n=top_n, bins=bins,
            title=title, x_label=x_label, y_label=y_label, orientation=orientation)
        context.artifacts[card.id] = card
        return {"status": "ready", "cards": [card_entry(card)],
                "chart_spec": spec, "render_summary": summary}

    def card_entry(card):
        return {'id':card.id,'dataset_id':card.dataset_id,'title':card.title,
                'reason':card.reason,'kind':card.kind,'columns':card.columns,'scope':card.scope}

    def source_key(source):
        try:
            table = single_table(validate_query(f'SELECT * FROM {source}'))
            return table_identity(table).casefold() if table is not None else ''
        except (ValueError, TypeError, SqlglotError):
            return ''

    def normalized_query(tree):
        tree = tree.copy()
        for identifier in tree.find_all(exp.Identifier):
            identifier.set('quoted', False)
            identifier.set('this', identifier.name.casefold())
        return tree.sql(dialect='databricks')

    def prepare_histogram(source, column, where_sql="", fresh_source_required=False,
                          current_result_only=False):
        identity = source_key(source)
        matches=[t for t in context.reference_context if source_key(t.get('table','')) == identity and identity]
        known = any(source_key(info.source) == identity and column in info.columns
                    for info in datasets.metadata.values())
        if not identity or (not known and (len(matches)!=1 or column not in [c['name'] for c in matches[0].get('columns',[])])):
            return {'status':'needs_context','message':'정확한 테이블과 수치 컬럼을 inspect_table_context에서 확인하세요.'}
        # SQL is a read-only plan, never an execution or an approval.
        quoted='`'+column.replace('`','``')+'`'
        table='.'.join('`'+part.replace('`','``')+'`' for part in source.split('.'))
        query=f'SELECT {quoted}, COUNT(*) AS `__frequency` FROM {table}'
        query+=f' WHERE {quoted} IS NOT NULL'+(' AND ('+where_sql+')' if where_sql.strip() else '')
        query+=f' GROUP BY {quoted}'
        tree=validate_query(query)
        if len(list(tree.find_all(exp.Select)))!=1 or len(list(tree.find_all(exp.Table)))!=1:
            raise ValueError('필터에는 다른 조회나 테이블을 포함할 수 없습니다.')
        plan = {'source':source,'query':query,
            'reason':f'{column} 히스토그램에 필요한 값별 빈도를 조회합니다. 원본 행 전체는 가져오지 않습니다.',
            'value_column':column,'weight_column':'__frequency'}
        scope_tree = validate_query(f'SELECT * FROM {table}'+(' WHERE '+where_sql if where_sql.strip() else ''))
        conditions = query_conditions(scope_tree)

        def ready(info):
            validate_frequency_dataset(datasets, info.id, column, '__frequency')
            card = next((card for card in context.artifacts.values()
                         if card.dataset_id == info.id and card.kind == 'histogram'
                         and card.columns == (column,) and card.image.startswith(b'\x89PNG\r\n\x1a\n')), None)
            if card is None:
                card = histogram_from_counts(datasets, info.id, column, '__frequency')
                context.artifacts[card.id] = card
            return {'status':'ready','histogram_plan':plan,'loaded_dataset':info.id,
                    'cards':[card_entry(card)],'reused':True}

        if current_result_only and not fresh_source_required:
            # The user explicitly chose the loaded result as the population.
            # Render those rows directly and keep their partial/unknown scope
            # visible instead of converting the request into a source query.
            for info in reversed(list(datasets.metadata.values())):
                if source_key(info.source) != identity or column not in info.columns:
                    continue
                need=AnalysisNeed(info.source,(column,),conditions=conditions or (),
                    current_result_only=True)
                if assess_reuse(info,need).action == 'query_source':
                    continue
                previews=recommend_charts(datasets,info.id,[column])
                card=next((item for item in previews if item.kind == 'histogram'
                    and item.columns == (column,)),None)
                if card is None:
                    return {'status':'no_valid_chart',
                        'message':'현재 결과에는 히스토그램을 만들 수 있는 유효한 수치가 부족합니다.'}
                context.artifacts[card.id]=card
                return {'status':'ready','histogram_plan':plan,'loaded_dataset':info.id,
                    'cards':[card_entry(card)],'reused':True,'current_result_only':True}
            return {'status':'needs_data',
                'message':'현재 보유 결과에서 요청한 컬럼을 찾지 못했습니다. 원격 조회는 실행하지 않았습니다.'}

        # First reuse exact, validated count results, including a previous local
        # realization of this plan. Source filters applied before local SQL must
        # also fit the requested population; SQL text alone is insufficient.
        for info in ([] if fresh_source_required else reversed(list(datasets.metadata.values()))):
            if source_key(info.source) != identity or info.coverage != 'complete':
                continue
            try:
                candidate = validate_query(info.query, dialect='duckdb' if info.parent_id else 'databricks')
                if info.parent_id:
                    parent = datasets.metadata.get(info.parent_id)
                    if parent is None or not parent.predicate_known:
                        continue
                    scope_info = replace(info, grain='raw', aggregation='', predicate_known=True)
                    if assess_reuse(scope_info, AnalysisNeed(info.source, (column,), conditions=conditions or ())).action == 'query_source':
                        continue
                    local_table = single_table(candidate)
                    if local_table is None or local_table.name != 'data':
                        continue
                    local_table.replace(single_table(tree).copy())
                if normalized_query(candidate) == normalized_query(tree):
                    return ready(info)
            except (ValueError, KeyError, SqlglotError):
                continue

        # Complete compatible raw rows can fulfill the plan locally. The SQL
        # itself preserves even predicates outside the simple implication subset.
        for info in ([] if fresh_source_required else reversed(list(datasets.metadata.values()))):
            if source_key(info.source) != identity:
                continue
            need = AnalysisNeed(info.source, (column,), conditions=conditions or ())
            if assess_reuse(info, need).action == 'query_source':
                continue
            local_tree = tree.copy()
            single_table(local_tree).replace(exp.Table(this=exp.to_identifier('data')))
            result = analyze_local(info.id, local_tree.sql(dialect='duckdb'),
                requested_conditions=[asdict(c) for c in conditions or ()])
            if result['status'] == 'ready' and result['dataset']['coverage'] == 'complete':
                return ready(datasets.metadata[result['dataset']['id']])
        return {'status':'planned','histogram_plan':plan}

    def render_histogram(dataset_id, value_column, weight_column):
        card = histogram_from_counts(datasets, dataset_id, value_column, weight_column)
        context.artifacts[card.id] = card
        return {'status':'ready', 'cards':[card_entry(card)]}

    def show_chart(chart_id):
        card = context.artifacts[chart_id]
        # Persistent lookup verifies the image and its bound dataset without
        # creating another chart or running a calculation.
        datasets.metadata[card.dataset_id]
        return {'status':'ready', 'cards':[card_entry(card)], 'reused':True}

    def tool(name, description, properties, required, run):
        def normalized(*args, **kwargs):
            return normalize_tool_result(run(*args, **kwargs))
        return ToolDefinition(name, description, {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}, normalized,
            COMMON_TOOL_RESULT_SCHEMA)

    string = {"type": "string"}
    return [
        tool("list_analysis_context", "현재 보유 데이터와 읽을 수 있는 분석 스킬 목록. 원격 조회 없음.", {}, [], catalog),
        tool("read_analysis_skill", "등록된 분석 스킬을 필요할 때 읽습니다.", {"name": string}, ["name"], registry.read),
        tool("inspect_table_context", "테이블의 저장된 스키마 스냅샷과 승인 후 로딩된 실제 스키마를 비교합니다. stale 또는 needs_refresh이면 그 컬럼으로 새 SQL을 만들지 말고 반환된 SELECT * LIMIT 0 조회를 사용자에게 승인 요청하세요. 이 도구 자체는 원격 조회하지 않습니다. table은 available_tables의 정확한 테이블명입니다.",
             {"table": string}, ["table"], inspect_table_context),
        tool("inspect_dataset", "로딩된 datasets의 id만 사용하세요. 테이블명은 허용되지 않습니다. 결과 ID의 출처, 컬럼, 관측 단위와 최대 5개 미리보기 행을 확인합니다.",
             {"dataset_id": string}, ["dataset_id"], inspect_dataset),
        tool("profile_dataset", "로딩된 dataset의 행·컬럼 수, 결측치, 고유값 수, 수치 요약과 제한된 범주 빈도를 구조화해 계산합니다. 원격 조회가 없고 원시 행을 반환하지 않습니다. coverage와 grain을 함께 해석하세요.",
             {"dataset_id": string,
              "columns": {"type": "array", "items": string},
              "offset": {"type": "integer", "minimum": 0},
              "limit": {"type": "integer", "minimum": 1, "maximum": 64}},
             ["dataset_id"], profile_dataset),
        tool("plan_source_discovery", "현재 문맥에서 확인된 Databricks catalog의 테이블 목록을 찾는 읽기 전용 information_schema SQL을 만듭니다. 실행하지 않으며, 반환된 discovery_plan을 query_databricks로 전달할 때 다시 사용자 승인을 받아야 합니다.",
             {"catalog": string, "schema": string, "pattern": string,
              "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
             [], plan_source_discovery),
        tool("use_dataset", "원본 범위에 대한 raw 데이터 충분성을 검사하고 가능하면 로컬 필터링합니다. conditions는 AND입니다. OR를 AND로 바꾸지 마세요. current_result_only는 사용자가 현재 결과 자체만 분석할 때 사용합니다.",
             {"dataset_id": string, "columns": {"type": "array", "items": string},
              "conditions": {"type": "array", "items": {"type": "object",
                  "properties": {"column": string, "op": {"type": "string", "enum": ["eq", "ne", "gt", "ge", "lt", "le", "in"]},
                                 "value": {}}, "required": ["column", "op", "value"], "additionalProperties": False}},
              "current_result_only": {"type": "boolean"}}, ["dataset_id", "columns"], use_dataset),
        tool("join_datasets", "로딩된 두 raw 또는 명시적 aggregate dataset을 1~4개 key로 조인합니다. 실행 전에 key 자료형, NULL, 중복도, cardinality, 예상 행 수와 확장률을 검사합니다. many-to-many 또는 운영 한도 초과는 실행하지 않으며, 성공 결과는 두 부모 dataset ID와 snapshot lineage를 보존합니다. 조인 후 계산에는 반환된 dataset_id와 current_result_only=true를 사용하세요.",
             {"left_dataset_id": string, "right_dataset_id": string,
              "left_on": {"type":"array","items":string,"minItems":1,"maxItems":4,"uniqueItems":True},
              "right_on": {"type":"array","items":string,"minItems":1,"maxItems":4,"uniqueItems":True},
              "how": {"type":"string","enum":["inner","left","right","outer"]}},
             ["left_dataset_id","right_dataset_id","left_on","right_on","how"], join_datasets),
        tool("statistical_test", "로딩된 raw dataset에서 선언된 통계 검정을 실행하고 표본 수, 결측 제외, 가정 진단, 통계량, 자유도, p-value, 효과크기와 적용 가능한 신뢰구간을 구조화해 반환합니다. independent_t, paired_t, chi_square, one_way_anova, mann_whitney, mean_ci만 지원하며 임의 Python은 실행하지 않습니다.",
             {"dataset_id": string,
              "test": {"type":"string","enum":["independent_t","paired_t","chi_square","one_way_anova","mann_whitney","mean_ci"]},
              "value_column": string,
              "group_column": string,
              "paired_column": string,
              "alpha": {"type":"number","minimum":0.001,"maximum":0.2}},
             ["dataset_id","test","value_column"], statistical_test),
        tool("detect_outliers", "로딩된 raw dataset의 수치 컬럼에서 IQR, Z-score, MAD 또는 분위수 기준 이상치를 탐지합니다. 원시 행을 반환하거나 새 dataset을 만들지 않고 기준선, 표본·결측, 상·하한 이상치 수와 비율, 범위를 구조화해 반환합니다.",
             {"dataset_id": string,
              "column": string,
              "method": {"type":"string","enum":["iqr","zscore","mad","quantile"]},
              "tail": {"type":"string","enum":["both","upper","lower"]},
              "threshold": {"type":"number","minimum":0.1,"maximum":10},
              "lower_quantile": {"type":"number","minimum":0,"maximum":1},
              "upper_quantile": {"type":"number","minimum":0,"maximum":1}},
             ["dataset_id","column","method"], detect_outliers),
        tool("select_outlier_rows", "로딩된 raw dataset에서 IQR, Z-score, MAD 또는 분위수 기준 이상치/정상치 cohort를 파생 dataset으로 저장합니다. 원시 행은 반환하지 않고 부모 dataset ID, snapshot, 선택 행 수, predicate와 데이터 digest를 반환합니다. 후속 계산에는 반환된 dataset_id와 current_result_only=true를 사용하세요.",
             {"dataset_id": string,
              "column": string,
              "method": {"type":"string","enum":["iqr","zscore","mad","quantile"]},
              "selection": {"type":"string","enum":["outliers","inliers"]},
              "tail": {"type":"string","enum":["both","upper","lower"]},
              "threshold": {"type":"number","minimum":0.1,"maximum":10},
              "lower_quantile": {"type":"number","minimum":0,"maximum":1},
              "upper_quantile": {"type":"number","minimum":0,"maximum":1}},
             ["dataset_id","column","method"], select_outlier_rows),
        tool("propose_databricks_query", "Databricks 조회를 사용자에게 제안합니다. 이 도구는 실행하지 않습니다. 정확한 SQL과 이유를 표시하고 승인 대기합니다.",
             {"source": string, "query": string, "reason": string}, ["source", "query", "reason"], propose_query),
        tool("recommend_chart_images", "현재 데이터의 통계로 실제 이미지 후보를 만듭니다. 원격 조회 없음. 반환된 카드 ID의 이미지는 UI가 표시합니다.",
             {"dataset_id": string, "columns": {"type": "array", "items": string}},
             ["dataset_id"], chart_options),
        tool("render_chart_spec", "사용자가 지정한 차트 종류와 축·집계·정렬·상위 N개·bin·제목·축 라벨을 제한된 schema로 실제 PNG에 렌더링합니다. 임의 코드나 파일·URL은 받지 않습니다. 로딩된 dataset 범위를 넓히지 않으며 반환 scope와 sample 여부를 설명하세요.",
             {"dataset_id": string,
              "kind": {"type":"string","enum":["histogram","bar","line","scatter","boxplot"]},
              "x": string, "y": string, "category": string,
              "aggregation": {"type":"string","enum":["none","count","sum","mean","median","min","max"]},
              "sort": {"type":"string","enum":["none","ascending","descending"]},
              "top_n": {"type":"integer","minimum":1,"maximum":50},
              "bins": {"type":"integer","minimum":2,"maximum":100},
              "title": {"type":"string","maxLength":120},
              "x_label": {"type":"string","maxLength":120},
              "y_label": {"type":"string","maxLength":120},
              "orientation": {"type":"string","enum":["vertical","horizontal"]}},
             ["dataset_id","kind","x"], render_chart_spec),
        tool("prepare_histogram", "보유한 완전한 빈도·원본 데이터와 이미지를 먼저 재사용하여 히스토그램을 만듭니다. 데이터가 부족할 때만 승인형 조회 계획을 반환합니다. source는 정확한 테이블명, column은 수치 컬럼, where_sql은 유지해야 할 사용자 필터 SQL(없으면 빈 문자열)입니다. 사용자가 최신/현재 원본을 명시하면 fresh_source_required=true로 지정해 캐시를 재사용하지 않습니다. 직접 원격 조회하지 않습니다.",
             {"source":string,"column":string,"where_sql":string,
              "fresh_source_required":{"type":"boolean"}},["source","column"],prepare_histogram),
        tool("render_histogram", "완전한 값별 빈도 집계의 히스토그램을 생성합니다. value_column은 실제 수치값, weight_column은 해당 값의 COUNT(*) 빈도입니다. 원본 행이나 빈도 아닌 집계값을 넣지 마세요.",
             {"dataset_id":string,"value_column":string,"weight_column":string},
             ["dataset_id","value_column","weight_column"],render_histogram),
        tool("show_chart", "저장된 검증 완료 차트 이미지를 다시 표시합니다. 새 계산이나 원격 조회를 수행하지 않습니다.",
             {"chart_id":string}, ["chart_id"], show_chart),
        tool("local_analysis_sql", "보유 dataset을 data라는 로컬 테이블로 SELECT 계산합니다. requested_conditions는 요청 모집단의 AND 조건이며 SQL 전에 실제 적용합니다. 생략하면 SQL WHERE의 범위(WHERE가 없으면 전체 원본)를 요구합니다. 이전 필터 결과를 이어서 분석할 때도 요청 조건을 명시하세요. 범위를 넓힐 수 없는 캐시는 거절합니다. current_result_only는 사용자가 현재 결과 자체만 분석할 때만 true로 지정합니다.",
             {"dataset_id": string, "query": string, "current_result_only": {"type": "boolean"},
              "requested_conditions": {"type":"array","items":{"type":"object",
                  "properties":{"column":string,"op":{"type":"string","enum":["eq","ne","gt","ge","lt","le","in"]},"value":{}},
                  "required":["column","op","value"],"additionalProperties":False}}}, ["dataset_id", "query"], analyze_local),
    ]


def build_runtime_tools(session, datasets: DatasetStore) -> list[ToolDefinition]:
    """Compatibility bridge for the current runtime and existing callers."""
    def propose(**arguments):
        request = session.approvals.propose(**arguments, goal=session.last_goal)
        return {"status": "awaiting_approval", "request": asdict(request)}

    context = AnalysisToolContext(datasets, session.artifacts,
                                  session.reference_context, propose)
    tools = build_analysis_tools(context)
    session.context_provider = next(t.run for t in tools if t.name == "list_analysis_context")
    return tools
