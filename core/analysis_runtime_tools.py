"""Local dataset and skill tools for the shared analysis loop."""
from dataclasses import asdict
from duckdb import BinderException
from sqlglot import exp, parse
from sqlglot.errors import SqlglotError
from core.analysis_catalog import compact_catalog, resolve_table_context

from core.analysis_tool_contract import (
    AnalysisToolContext,
    COMMON_TOOL_RESULT_SCHEMA,
    ToolDefinition,
    normalize_tool_result,
)
from utils.analysis_datasets import AnalysisNeed, Condition, DatasetStore, assess_reuse, filter_frame, full_read_preflight, project_dataset, select_reusable_dataset
from utils.analysis_skill_registry import AnalysisSkillRegistry
from utils.analysis_charts import (
    histogram_from_counts,
    recommend_charts,
    render_count_rate_chart as build_count_rate_chart,
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
    winsorize_numeric_summary as run_winsorization,
)
from utils.analysis_timeseries import prepare_time_series as build_time_series
from utils.analysis_aggregate import aggregate_dataset as build_aggregate_dataset
from utils.analysis_compare import compare_group_aggregates as build_group_comparison
from utils.analysis_pivot import pivot_dataset as build_pivot_dataset
from utils.analysis_group_summary import summarize_groups as build_group_summary


def build_analysis_tools(context: AnalysisToolContext) -> list[ToolDefinition]:
    datasets = context.datasets
    registry = AnalysisSkillRegistry()

    def catalog():
        return compact_catalog({"datasets": [dict(display_name=f"결과 {index}", **asdict(info)) for index,info in enumerate(datasets.metadata.values(),1)],
                "skills": registry.list(), "available_tables": context.reference_context})


    def inspect_table_context(table):
        return resolve_table_context(context.reference_context, datasets, table)

    def resolve_analysis_intent(dataset_id):
        if context.semantic_resolver is None:
            return {'status':'needs_context', 'message':'의미 해석 서비스가 연결되지 않았습니다.'}
        return context.semantic_resolver.resolve(dataset_id)

    def inspect_dataset(dataset_id):
        if hasattr(datasets, 'inspect'):
            return datasets.inspect(dataset_id)
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
            grain=info.grain if current_result_only else 'raw',
            aggregation=info.aggregation if current_result_only else '',
            current_result_only=current_result_only)
        selection = select_reusable_dataset(datasets.metadata, dataset_id, need)
        decision = selection.decision
        if decision.action == "query_source":
            return {"status": "needs_data", "requested_dataset_id": dataset_id,
                    "selection_origin": selection.origin, **asdict(decision)}
        if decision.action == "derive_local":
            rejected = full_read_preflight(datasets, [selection.dataset_id])
            if rejected:
                return {**rejected, "requested_dataset_id": dataset_id,
                        "selected_dataset_id": selection.dataset_id,
                        "selection_origin": selection.origin}
        result = datasets.derive(selection.dataset_id, need)
        return {"status": "ready", "dataset": asdict(result),
                "requested_dataset_id": dataset_id,
                "selected_dataset_id": selection.dataset_id,
                "selection_origin": selection.origin, **asdict(decision)}

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

    def winsorize_numeric(dataset_id, column, lower_quantile=0.01,
                          upper_quantile=0.99):
        return run_winsorization(
            datasets,
            dataset_id,
            column=column,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    def prepare_time_series(dataset_id, time_column, value_column="", group_column="",
                            frequency="day", aggregation="count", gap_policy="omit",
                            timezone="", max_output_rows=5_000):
        return build_time_series(
            datasets,
            dataset_id,
            time_column=time_column,
            value_column=value_column,
            group_column=group_column,
            frequency=frequency,
            aggregation=aggregation,
            gap_policy=gap_policy,
            timezone=timezone,
            max_output_rows=max_output_rows,
        )

    def aggregate_dataset(dataset_id, aggregation, value_column="", group_column="",
                          sort="descending", top_n=0, max_groups=1_000,
                          max_output_rows=1_000):
        return build_aggregate_dataset(
            datasets,
            dataset_id,
            aggregation=aggregation,
            value_column=value_column,
            group_column=group_column,
            sort=sort,
            top_n=top_n,
            max_groups=max_groups,
            max_output_rows=max_output_rows,
        )

    def pivot_dataset(dataset_id, index_columns, column_columns, aggregation,
                      value_column="", success_value=None, conditions=None,
                      derived_bins=None,
                      margins=False, margins_name="전체", sort="ascending",
                      max_axis_values=100, max_output_cells=1_000):
        return build_pivot_dataset(
            datasets,
            dataset_id,
            index_columns=index_columns,
            column_columns=column_columns,
            aggregation=aggregation,
            value_column=value_column,
            success_value=success_value,
            conditions=conditions,
            derived_bins=derived_bins,
            margins=margins,
            margins_name=margins_name,
            sort=sort,
            max_axis_values=max_axis_values,
            max_output_cells=max_output_cells,
        )

    def summarize_groups(dataset_id, group_columns, metrics, conditions=None,
                         sort="group_ascending", max_groups=1_000,
                         max_output_rows=1_000):
        return build_group_summary(
            datasets,
            dataset_id,
            group_columns=group_columns,
            metrics=metrics,
            conditions=conditions,
            sort=sort,
            max_groups=max_groups,
            max_output_rows=max_output_rows,
        )

    def compare_group_aggregates(baseline_dataset_id, cohort_dataset_id, aggregation,
                                 group_column, value_column="", sort="group_ascending",
                                 max_groups=1_000, max_output_rows=1_000):
        return build_group_comparison(
            datasets,
            baseline_dataset_id,
            cohort_dataset_id,
            aggregation=aggregation,
            group_column=group_column,
            value_column=value_column,
            sort=sort,
            max_groups=max_groups,
            max_output_rows=max_output_rows,
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
        # ORDER BY may refer to a SELECT alias rather than an input column.
        # Projection expressions still retain their actual column references.
        output_aliases = {item.alias for item in tree.expressions
                          if isinstance(item, exp.Alias)}
        sql_column_names = {column.name for column in columns
                            if column.name not in output_aliases
                            or column.find_ancestor(exp.Alias) is not None}
        # An omitted scope means the SQL's WHERE scope (or the whole source).
        # Never silently inherit a cached subset when the caller asks for all rows.
        scope = requested if requested is not None else (sql_conditions or ())
        need = AnalysisNeed(info.source,
            tuple(dict.fromkeys((*sql_column_names, *(c.column for c in scope)))),
            conditions=scope,
            grain=info.grain if current_result_only else 'raw',
            aggregation=info.aggregation if current_result_only else '',
            current_result_only=current_result_only)
        selection = select_reusable_dataset(datasets.metadata, dataset_id, need)
        decision = selection.decision
        if decision.action == 'query_source':
            known_columns = {name for candidate in datasets.metadata.values()
                             if candidate.source == info.source for name in candidate.columns}
            unknown_sql_columns = sql_column_names - known_columns
            if unknown_sql_columns:
                # A model-invented column is a correctable SQL error, not
                # evidence that a costly remote reload is required.
                raise BinderException('Unknown local columns: ' + ', '.join(sorted(unknown_sql_columns)))
            return {'status': 'needs_data', **asdict(decision),
                    'requested_dataset_id': dataset_id,
                    'selection_origin': selection.origin,
                    'message': decision.reason + ' 요청 범위를 requested_conditions로 명시하거나, 사용자가 현재 결과 자체를 분석할 때만 current_result_only를 사용하세요.'}
        selected_info = datasets.metadata[selection.dataset_id]
        # A single, explicit-column SELECT can scan only its referenced
        # Parquet columns. Keep wildcard/CTE/compound queries on the general
        # path because their output may depend on the complete schema.
        projectable = (isinstance(tree, exp.Select)
            and len(list(tree.find_all(exp.Select))) == 1
            and not tree.args.get('with_') and not tree.args.get('joins')
            and not any(column.name in output_aliases and column.name in selected_info.columns
                        and column.find_ancestor(exp.Alias) is None for column in columns)
            and not any(isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)
                        for node in tree.walk()))
        projected_columns = tuple(dict.fromkeys((
            *sql_column_names, *(condition.column for condition in (requested or ())))))
        if projectable and set(projected_columns).issubset(selected_info.columns):
            input_frame = project_dataset(datasets, selection.dataset_id, projected_columns)
            if not projected_columns:
                # DuckDB cannot register a zero-column DataFrame. A private
                # row marker preserves COUNT(*) cardinality without decoding
                # the source's actual columns.
                input_frame["__telly_row_marker__"] = range(selected_info.rows)
        else:
            rejected = full_read_preflight(datasets, [selection.dataset_id])
            if rejected:
                return {**rejected, 'requested_dataset_id': dataset_id,
                        'selected_dataset_id': selection.dataset_id,
                        'selection_origin': selection.origin}
            input_frame = datasets.frames[selection.dataset_id]
        if requested is not None:
            residual = tuple(c for c in requested if c not in selected_info.conditions)
            input_frame = filter_frame(input_frame, residual)
        frame, truncated, tree = local_query(input_frame, query)
        aggregated = bool(tree.args.get("group") or tree.find(exp.AggFunc))
        safe_conditions = raw_conditions(tree)
        conditions = tuple(dict.fromkeys((*selected_info.conditions, *(requested or ()), *(sql_conditions or ()))))
        coverage = query_coverage(tree, truncated=truncated)
        if coverage == 'complete':
            coverage = selected_info.coverage
        result = datasets.register(frame, source=selected_info.source,
            coverage=coverage,
            predicate_known=selected_info.predicate_known and safe_conditions is not None,
            conditions=conditions, grain="aggregate" if aggregated else selected_info.grain,
            aggregation=tree.sql() if aggregated else selected_info.aggregation,
            parent_id=selection.dataset_id, query=query, snapshot=selected_info.snapshot)
        return {"status": "ready", "dataset": asdict(result),
                "preview": frame.head(15).to_dict(orient="records"),
                "applied_corrections": corrections,
                "requested_dataset_id": dataset_id,
                "selected_dataset_id": selection.dataset_id,
                "selection_origin": selection.origin}

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
                          y_label="", orientation="vertical", cumulative=False):
        card, summary, spec = build_chart_from_spec(
            datasets, dataset_id, kind=kind, x=x, y=y, category=category,
            aggregation=aggregation, sort=sort, top_n=top_n, bins=bins,
            title=title, x_label=x_label, y_label=y_label, orientation=orientation,
            cumulative=cumulative)
        context.artifacts[card.id] = card
        return {"status": "ready", "cards": [card_entry(card)],
                "chart_spec": spec, "render_summary": summary}

    def render_count_rate_chart(dataset_id, group_column, outcome_column, success_value,
                                layout="dual_axis", sort="none", top_n=50, title="",
                                x_label="", count_label="", rate_label=""):
        card, summary, spec = build_count_rate_chart(
            datasets, dataset_id, group_column=group_column,
            outcome_column=outcome_column, success_value=success_value,
            layout=layout, sort=sort, top_n=top_n, title=title,
            x_label=x_label, count_label=count_label, rate_label=rate_label)
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
                          current_result_only=False, dataset_id=""):
        identity = source_key(source)
        matches=[t for t in context.reference_context if source_key(t.get('table','')) == identity and identity]
        metadata = datasets.metadata
        matching = [info for info in metadata.values()
                    if source_key(info.source) == identity and column in info.columns]
        known = bool(matching)
        if not identity or (not known and (len(matches)!=1 or column not in [c['name'] for c in matches[0].get('columns',[])])):
            return {'status':'needs_context','message':'정확한 테이블과 수치 컬럼을 inspect_table_context에서 확인하세요.'}

        def lineage_root(info):
            seen = set()
            while info.id not in seen:
                seen.add(info.id)
                parents = info.parent_ids or ((info.parent_id,) if info.parent_id else ())
                if not parents:
                    return info.id
                if len(parents) != 1 or parents[0] not in metadata:
                    return None
                info = metadata[parents[0]]
            return None

        selected_id = dataset_id
        active = metadata.get(context.selected_dataset_id)
        if (not selected_id and not fresh_source_required and active is not None
                and source_key(active.source) == identity):
            selected_id = active.id
        if selected_id:
            selected = metadata.get(selected_id)
            if selected is None or source_key(selected.source) != identity:
                return {'status':'needs_context', 'message':'선택한 dataset ID가 이 출처의 보유 데이터와 일치하지 않습니다.'}
        elif matching and not fresh_source_required:
            roots = {lineage_root(info) for info in matching}
            if None in roots or len(roots) != 1 or (current_result_only and len(matching) != 1):
                return {'status':'needs_context',
                        'message':'같은 출처에 여러 원본 또는 결과가 있습니다. 분석할 dataset ID를 명시해주세요.',
                        'candidate_dataset_ids':[info.id for info in matching]}
            root_id = next(iter(roots))
            selected_id = (root_id if (column in metadata[root_id].columns
                                       and source_key(metadata[root_id].source) == identity) else
                           matching[0].id if len(matching) == 1 else '')
            if not selected_id:
                return {'status':'needs_context',
                        'message':'차트에 사용할 분기를 확정할 수 없습니다. dataset ID를 명시해주세요.',
                        'candidate_dataset_ids':[info.id for info in matching]}

        raw_id = ''
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
        if selected_id and not fresh_source_required and not current_result_only:
            selected = metadata[selected_id]
            decision = select_reusable_dataset(metadata, selected_id,
                AnalysisNeed(selected.source, (column,), conditions=conditions or ()))
            if decision.decision.action != 'query_source':
                raw_id = decision.dataset_id

        def in_selected_branch(info):
            if not selected_id:
                return True  # No local assets; only a remote plan can result.
            return (info.id == selected_id or
                    bool(raw_id and info.parent_id == raw_id))

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
            for info in ([metadata[selected_id]] if selected_id else []):
                if column not in info.columns:
                    continue
                need=AnalysisNeed(info.source,(column,),conditions=conditions or (),
                    grain=info.grain, aggregation=info.aggregation,
                    current_result_only=True)
                if assess_reuse(info,need).action == 'query_source':
                    continue
                chart_info = info
                if where_sql.strip():
                    local_where = scope_tree.args['where'].this.sql(dialect='duckdb')
                    filtered = analyze_local(info.id,
                        'SELECT * FROM data WHERE ' + local_where,
                        current_result_only=True)
                    if filtered['status'] != 'ready':
                        return filtered
                    chart_info = datasets.metadata[filtered['dataset']['id']]
                previews=recommend_charts(datasets,chart_info.id,[column])
                card=next((item for item in previews if item.kind == 'histogram'
                    and item.columns == (column,)),None)
                if card is None:
                    return {'status':'no_valid_chart',
                        'loaded_dataset':chart_info.id,
                        'message':'현재 결과에는 히스토그램을 만들 수 있는 유효한 수치가 부족합니다.'}
                context.artifacts[card.id]=card
                return {'status':'ready','histogram_plan':plan,'loaded_dataset':chart_info.id,
                    'cards':[card_entry(card)],'reused':True,'current_result_only':True}
            return {'status':'needs_data',
                'message':'현재 보유 결과에서 요청한 컬럼을 찾지 못했습니다. 원격 조회는 실행하지 않았습니다.'}

        # First reuse exact, validated count results, including a previous local
        # realization of this plan. Source filters applied before local SQL must
        # also fit the requested population; SQL text alone is insufficient.
        for info in ([] if fresh_source_required else reversed(list(metadata.values()))):
            if (source_key(info.source) != identity or info.coverage != 'complete'
                    or not in_selected_branch(info)):
                continue
            try:
                candidate = validate_query(info.query, dialect='duckdb' if info.parent_id else 'databricks')
                if info.parent_id:
                    parent = metadata.get(info.parent_id)
                    if parent is None or not parent.predicate_known:
                        continue
                    if assess_reuse(parent, AnalysisNeed(parent.source, (column,),
                            conditions=conditions or ())).action == 'query_source':
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
        for info in ([] if fresh_source_required else reversed(list(metadata.values()))):
            if source_key(info.source) != identity or info.id != raw_id:
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
        tool("resolve_analysis_intent", "분석 대상이 불명확할 때 저장된 외부 컬럼 설명으로 의미를 독립적으로 확인합니다. 모델 호출을 사용하며 실제 데이터 조회/계산은 하지 않습니다. 정의가 없거나 충돌하면 확인 질문이 필요합니다.",
             {"dataset_id": string}, ["dataset_id"], resolve_analysis_intent),
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
        tool("use_dataset", "원본 범위에 대한 raw 데이터 충분성을 검사하고 현재 결과가 부족하면 보존된 부모·같은 버전의 registry 데이터를 찾아 로컬 필터링합니다. needs_data는 이 탐색 후에만 반환합니다. conditions는 AND입니다. OR를 AND로 바꾸지 마세요. current_result_only는 사용자가 현재 결과 자체만 분석할 때 사용합니다.",
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
        tool("winsorize_numeric", "로딩된 raw dataset의 수치 컬럼을 지정한 하·상위 분위수 경계로 clip하고 원본/보정 평균·최솟값·최댓값과 실제 clip 건수를 구조화해 비교합니다. 원본 dataset을 변경하거나 행을 반환하지 않습니다.",
             {"dataset_id": string,
              "column": string,
              "lower_quantile": {"type":"number","exclusiveMinimum":0,"maximum":0.25},
              "upper_quantile": {"type":"number","minimum":0.75,"exclusiveMaximum":1}},
             ["dataset_id","column"], winsorize_numeric),
        tool("prepare_time_series", "로딩된 raw dataset의 실제 시간 컬럼을 검증하고 hour/day/week/month 단위로 제한된 재집계를 수행합니다. timezone, 파싱 실패, 중복 시각, 빈 구간, group cardinality와 출력 행 한도를 검사하며 부모 lineage를 가진 aggregate dataset을 저장합니다. 숫자 epoch 단위나 DST 충돌을 추측하지 않습니다.",
             {"dataset_id": string,
              "time_column": string,
              "value_column": string,
              "group_column": string,
              "frequency": {"type":"string","enum":["hour","day","week","month"]},
              "aggregation": {"type":"string","enum":["count","sum","mean","median","min","max"]},
              "gap_policy": {"type":"string","enum":["omit","zero","nan"]},
              "timezone": {"type":"string","maxLength":64},
              "max_output_rows": {"type":"integer","minimum":2,"maximum":5000}},
             ["dataset_id","time_column","frequency","aggregation"], prepare_time_series),
        tool("aggregate_dataset", "로딩된 raw dataset을 count, mean, sum, median, min, max 중 하나로 제한 집계합니다. 선택적 단일 group 컬럼, 정렬, TOP-N과 출력 한도를 검증하고 부모 lineage와 digest가 있는 aggregate dataset을 저장합니다. 이상치 cohort 후속 그룹 분석에는 select_outlier_rows가 반환한 dataset_id를 사용하세요.",
             {"dataset_id": string,
              "aggregation": {"type":"string","enum":["count","mean","sum","median","min","max"]},
              "value_column": string,
              "group_column": string,
              "sort": {"type":"string","enum":["ascending","descending"]},
              "top_n": {"type":"integer","minimum":0,"maximum":50},
              "max_groups": {"type":"integer","minimum":1,"maximum":1000},
              "max_output_rows": {"type":"integer","minimum":1,"maximum":1000}},
             ["dataset_id","aggregation"], aggregate_dataset),
        tool("pivot_dataset", "로딩된 raw dataset에서 실행 시점 schema의 행 축 1~3개와 열 축 1~2개를 사용해 bounded 피벗 표를 만듭니다. 단일 또는 제한된 복수 집계, 명시적 숫자 구간 축, 조건·총계·월 정렬·출력 셀 한도를 검증합니다. success_rate는 success_value를 반드시 명시합니다.",
             {"dataset_id":string,
              "index_columns":{"type":"array","items":string,"minItems":1,"maxItems":3,"uniqueItems":True},
              "column_columns":{"type":"array","items":string,"minItems":1,"maxItems":2,"uniqueItems":True},
              "aggregation":{"oneOf":[
                  {"type":"string","enum":["count","mean","sum","median","min","max","success_rate","overall_percent"]},
                  {"type":"array","items":{"type":"string","enum":["count","mean","sum","median","min","max"]},"minItems":2,"maxItems":6,"uniqueItems":True}]},
              "value_column":string,
              "success_value":{},
              "conditions":{"type":"array","items":{"type":"object",
                  "properties":{"column":string,"op":{"type":"string","enum":["eq","ne","gt","ge","lt","le","in"]},"value":{}},
                  "required":["column","op","value"],"additionalProperties":False}},
              "derived_bins":{"type":"array","maxItems":2,"items":{"type":"object",
                  "properties":{"source_column":string,"output_column":string,
                      "cut_points":{"type":"array","items":{"type":"number"},"minItems":1,"maxItems":20,"uniqueItems":True},
                      "labels":{"type":"array","items":string,"minItems":2,"maxItems":21,"uniqueItems":True}},
                  "required":["source_column","output_column","cut_points","labels"],"additionalProperties":False}},
              "margins":{"type":"boolean"},
              "margins_name":{"type":"string","minLength":1,"maxLength":40},
              "sort":{"type":"string","enum":["ascending","descending","calendar_month"]},
              "max_axis_values":{"type":"integer","minimum":1,"maximum":100},
              "max_output_cells":{"type":"integer","minimum":1,"maximum":10000}},
             ["dataset_id","index_columns","column_columns","aggregation"], pivot_dataset),
        tool("summarize_groups", "로딩된 raw dataset을 실행 시점 schema의 그룹 컬럼 1~3개로 나누고 최대 8개의 선언형 지표를 한 표에 계산합니다. count·수치 집계와 명시적 조건의 conditional_count·conditional_percent·conditional_mean만 지원합니다. 전체 필터와 조건부 지표의 분모를 분리하고 그룹·출력 한도, lineage와 digest를 검증합니다.",
             {"dataset_id":string,
              "group_columns":{"type":"array","items":string,"minItems":1,"maxItems":3,"uniqueItems":True},
              "metrics":{"type":"array","minItems":1,"maxItems":8,"items":{"type":"object",
                  "properties":{"name":{"type":"string","minLength":1,"maxLength":80},
                      "aggregation":{"type":"string","enum":["count","mean","sum","median","min","max","conditional_count","conditional_percent","conditional_mean"]},
                      "value_column":string,
                      "condition":{"type":"object","properties":{"column":string,
                          "op":{"type":"string","enum":["eq","ne","gt","ge","lt","le","in"]},"value":{}},
                          "required":["column","op","value"],"additionalProperties":False},
                      "empty_value":{"type":"number"}},
                  "required":["name","aggregation"],"additionalProperties":False}},
              "conditions":{"type":"array","items":{"type":"object",
                  "properties":{"column":string,"op":{"type":"string","enum":["eq","ne","gt","ge","lt","le","in"]},"value":{}},
                  "required":["column","op","value"],"additionalProperties":False}},
              "sort":{"type":"string","enum":["group_ascending","group_descending","none"]},
              "max_groups":{"type":"integer","minimum":1,"maximum":1000},
              "max_output_rows":{"type":"integer","minimum":1,"maximum":1000}},
             ["dataset_id","group_columns","metrics"], summarize_groups),
        tool("compare_group_aggregates", "같은 source·snapshot에서 기준 raw dataset과 그 lineage 후손 cohort의 동일한 그룹 집계를 비교합니다. 그룹별 기준값, cohort값, 차이와 변화율을 계산하고 두 부모 ID와 digest를 보존합니다. 임의의 서로 무관한 dataset은 비교하지 않습니다.",
             {"baseline_dataset_id": string,
              "cohort_dataset_id": string,
              "aggregation": {"type":"string","enum":["count","mean","sum","median","min","max"]},
              "group_column": string,
              "value_column": string,
              "sort": {"type":"string","enum":["group_ascending","difference_descending","none"]},
              "max_groups": {"type":"integer","minimum":1,"maximum":1000},
              "max_output_rows": {"type":"integer","minimum":1,"maximum":1000}},
             ["baseline_dataset_id","cohort_dataset_id","aggregation","group_column"],
             compare_group_aggregates),
        tool("propose_databricks_query", "Databricks 조회를 사용자에게 제안합니다. 이 도구는 실행하지 않습니다. 정확한 SQL과 이유를 표시하고 승인 대기합니다.",
             {"source": string, "query": string, "reason": string}, ["source", "query", "reason"], propose_query),
        tool("recommend_chart_images", "현재 데이터의 통계로 실제 이미지 후보를 만듭니다. 원격 조회 없음. 반환된 카드 ID의 이미지는 UI가 표시합니다.",
             {"dataset_id": string, "columns": {"type": "array", "items": string}},
             ["dataset_id"], chart_options),
        tool("render_chart_spec", "사용자가 지정한 차트 종류와 축·집계·정렬·상위 N개·bin·제목·축 라벨을 제한된 schema로 실제 PNG에 렌더링합니다. 선 그래프는 숫자 축 빈도와 영문 월의 calendar_month 정렬·누적 변환을 지원합니다. 임의 코드나 파일·URL은 받지 않습니다. 로딩된 dataset 범위를 넓히지 않으며 반환 scope와 sample 여부를 설명하세요.",
             {"dataset_id": string,
              "kind": {"type":"string","enum":["histogram","bar","line","scatter","boxplot"]},
              "x": string, "y": string, "category": string,
              "aggregation": {"type":"string","enum":["none","count","sum","mean","median","min","max"]},
              "sort": {"type":"string","enum":["none","ascending","descending","calendar_month"]},
              "top_n": {"type":"integer","minimum":1,"maximum":50},
              "bins": {"type":"integer","minimum":2,"maximum":100},
              "title": {"type":"string","maxLength":120},
              "x_label": {"type":"string","maxLength":120},
              "y_label": {"type":"string","maxLength":120},
              "orientation": {"type":"string","enum":["vertical","horizontal"]},
              "cumulative": {"type":"boolean"}},
             ["dataset_id","kind","x"], render_chart_spec),
        tool("render_count_rate_chart", "로딩된 raw dataset에서 그룹별 전체 행 수와 명시한 outcome 성공값의 비율을 계산해 이중 Y축 또는 2열 패널 PNG로 렌더링합니다. 성공률 분모는 outcome 비결측 행이며 points와 digest를 반환합니다. 성공값·분모·컬럼을 추측하지 않습니다.",
             {"dataset_id": string,
              "group_column": string,
              "outcome_column": string,
              "success_value": {},
              "layout": {"type":"string","enum":["dual_axis","split_panel"]},
              "sort": {"type":"string","enum":["none","group_ascending","count_descending","calendar_month"]},
              "top_n": {"type":"integer","minimum":1,"maximum":50},
              "title": {"type":"string","maxLength":120},
              "x_label": {"type":"string","maxLength":120},
              "count_label": {"type":"string","maxLength":120},
              "rate_label": {"type":"string","maxLength":120}},
             ["dataset_id","group_column","outcome_column","success_value"],
             render_count_rate_chart),
        tool("prepare_histogram", "보유한 완전한 빈도·원본 데이터와 이미지를 먼저 재사용하여 히스토그램을 만듭니다. 여러 원본·버전·분기가 있으면 dataset_id로 사용자가 선택한 데이터의 계보를 지정하세요. 명시하지 않아 모호하면 다른 결과를 임의 선택하지 않습니다. source는 정확한 테이블명, column은 수치 컬럼, where_sql은 유지할 사용자 필터 SQL입니다. 최신 데이터 요청에만 fresh_source_required=true를 사용합니다. 직접 원격 조회하지 않습니다.",
             {"source":string,"column":string,"where_sql":string,"dataset_id":string,
              "current_result_only":{"type":"boolean"},
              "fresh_source_required":{"type":"boolean"}},["source","column"],prepare_histogram),
        tool("render_histogram", "완전한 값별 빈도 집계의 히스토그램을 생성합니다. value_column은 실제 수치값, weight_column은 해당 값의 COUNT(*) 빈도입니다. 원본 행이나 빈도 아닌 집계값을 넣지 마세요.",
             {"dataset_id":string,"value_column":string,"weight_column":string},
             ["dataset_id","value_column","weight_column"],render_histogram),
        tool("show_chart", "저장된 검증 완료 차트 이미지를 다시 표시합니다. 새 계산이나 원격 조회를 수행하지 않습니다.",
             {"chart_id":string}, ["chart_id"], show_chart),
        tool("local_analysis_sql", "보유 dataset을 data라는 로컬 테이블로 SELECT 계산합니다. 현재 결과가 부족하면 보존된 부모·같은 버전의 registry 원본을 먼저 탐색합니다. requested_conditions는 요청 모집단의 AND 조건이며 SQL 전에 실제 적용합니다. 생략하면 SQL WHERE의 범위(WHERE가 없으면 전체 원본)를 요구합니다. 이전 필터 결과를 이어서 분석할 때도 요청 조건을 명시하세요. current_result_only는 사용자가 현재 결과 자체만 분석할 때만 true로 지정합니다.",
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
