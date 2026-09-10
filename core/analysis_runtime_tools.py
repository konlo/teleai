"""Local dataset and skill tools for the shared analysis loop."""
from dataclasses import asdict
from core.analysis_catalog import compact_catalog

from core.analysis_tool_contract import AnalysisToolContext, ToolDefinition
from utils.analysis_datasets import AnalysisNeed, Condition, DatasetStore, assess_reuse
from utils.analysis_skill_registry import AnalysisSkillRegistry
from utils.analysis_charts import recommend_charts, histogram_from_counts
from core.analysis_sql import local_query, validate_query


def build_analysis_tools(context: AnalysisToolContext) -> list[ToolDefinition]:
    datasets = context.datasets
    registry = AnalysisSkillRegistry()

    def catalog():
        return compact_catalog({"datasets": [dict(display_name=f"결과 {index}", **asdict(info)) for index,info in enumerate(datasets.metadata.values(),1)],
                "skills": registry.list(), "available_tables": context.reference_context})


    def inspect_table_context(table):
        matches = [item for item in context.reference_context
                   if item.get('table') == table or item.get('table', '').split('.')[-1] == table]
        if len(matches) != 1:
            return {'status': 'needs_context', 'message': '저장된 테이블 정보가 없거나 이름이 모호합니다. 목록의 정확한 테이블명을 확인하세요.'}
        return {'status': 'ready', 'table_context': matches[0],
                'scope': '저장된 테이블 설명입니다. 현재 접근 권한이나 실제 로딩된 데이터를 뜻하지 않습니다.'}

    def inspect_dataset(dataset_id):
        info = datasets.metadata[dataset_id]
        frame = datasets.frames[dataset_id]
        return {"dataset": asdict(info), "dtypes": frame.dtypes.astype(str).to_dict(),
                "preview": frame.head(5).to_dict(orient="records")}

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

    def propose_query(source, query, reason):
        validate_query(query)
        return context.propose_query(source=source, query=query, reason=reason)

    def analyze_local(dataset_id, query, current_result_only=False):
        info = datasets.metadata[dataset_id]
        if not current_result_only and (info.coverage != 'complete' or not info.predicate_known):
            return {'status': 'needs_data', 'message': '현재 데이터의 원본 범위 완전성을 확인할 수 없습니다. 사용자가 현재 결과만 분석하려는 경우에만 current_result_only를 사용하고, 원본 통계에는 승인형 조회를 제안하세요.'}
        frame, truncated, tree = local_query(datasets.frames[dataset_id], query)
        from sqlglot import exp
        aggregated = bool(tree.args.get("group") or tree.find(exp.AggFunc))
        result = datasets.register(frame, source=info.source,
            coverage="truncated" if truncated else info.coverage,
            # SQL transformations require richer lineage before population reuse.
            predicate_known=False, grain="aggregate" if aggregated else info.grain,
            aggregation=tree.sql() if aggregated else info.aggregation,
            parent_id=dataset_id, query=query, snapshot=info.snapshot)
        return {"status": "ready", "dataset": asdict(result),
                "preview": frame.head(15).to_dict(orient="records")}

    def chart_options(dataset_id, columns=None):
        previews = recommend_charts(datasets, dataset_id, columns)
        for card in previews:
            context.artifacts[card.id] = card
        return {"status": "ready" if previews else "no_valid_chart", "cards": [
            {"id": c.id, "dataset_id": c.dataset_id, "title": c.title,
             "reason": c.reason, "kind": c.kind, "columns": c.columns, "scope": c.scope}
            for c in previews]}

    def prepare_histogram(source, column, where_sql=""):
        matches=[t for t in context.reference_context if t.get('table')==source]
        if len(matches)!=1 or column not in [c['name'] for c in matches[0].get('columns',[])]:
            return {'status':'needs_context','message':'정확한 테이블과 수치 컬럼을 inspect_table_context에서 확인하세요.'}
        # SQL is a read-only plan, never an execution or an approval.
        quoted='`'+column.replace('`','``')+'`'
        table='.'.join('`'+part.replace('`','``')+'`' for part in source.split('.'))
        query=f'SELECT {quoted}, COUNT(*) AS `__frequency` FROM {table}'
        query+=f' WHERE {quoted} IS NOT NULL'+(' AND ('+where_sql+')' if where_sql.strip() else '')
        query+=f' GROUP BY {quoted}'
        tree=validate_query(query)
        from sqlglot import exp
        if len(list(tree.find_all(exp.Select)))!=1 or len(list(tree.find_all(exp.Table)))!=1:
            raise ValueError('필터에는 다른 조회나 테이블을 포함할 수 없습니다.')
        return {'status':'planned','histogram_plan':{'source':source,'query':query,
            'reason':f'{column} 히스토그램에 필요한 값별 빈도를 조회합니다. 원본 행 전체는 가져오지 않습니다.',
            'value_column':column,'weight_column':'__frequency'}}

    def render_histogram(dataset_id, value_column, weight_column):
        card = histogram_from_counts(datasets, dataset_id, value_column, weight_column)
        context.artifacts[card.id] = card
        return {'status':'ready', 'cards':[{'id':card.id,'dataset_id':card.dataset_id,
            'title':card.title,'reason':card.reason,'kind':card.kind,'columns':card.columns,'scope':card.scope}]}

    def tool(name, description, properties, required, run):
        return ToolDefinition(name, description, {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}, run)

    string = {"type": "string"}
    return [
        tool("list_analysis_context", "현재 보유 데이터와 읽을 수 있는 분석 스킬 목록. 원격 조회 없음.", {}, [], catalog),
        tool("read_analysis_skill", "등록된 분석 스킬을 필요할 때 읽습니다.", {"name": string}, ["name"], registry.read),
        tool("inspect_table_context", "저장된 테이블의 컬럼과 설명을 확인합니다. 원격 조회나 DataFrame 로딩 없음. table은 available_tables의 테이블명입니다.",
             {"table": string}, ["table"], inspect_table_context),
        tool("inspect_dataset", "로딩된 datasets의 id만 사용하세요. 테이블명은 허용되지 않습니다. 결과 ID의 출처, 컬럼, 관측 단위와 최대 5개 미리보기 행을 확인합니다.",
             {"dataset_id": string}, ["dataset_id"], inspect_dataset),
        tool("use_dataset", "원본 범위에 대한 raw 데이터 충분성을 검사하고 가능하면 로컬 필터링합니다. conditions는 AND입니다. OR를 AND로 바꾸지 마세요. current_result_only는 사용자가 현재 결과 자체만 분석할 때 사용합니다.",
             {"dataset_id": string, "columns": {"type": "array", "items": string},
              "conditions": {"type": "array", "items": {"type": "object",
                  "properties": {"column": string, "op": {"type": "string", "enum": ["eq", "ne", "gt", "ge", "lt", "le", "in"]},
                                 "value": {}}, "required": ["column", "op", "value"], "additionalProperties": False}},
              "current_result_only": {"type": "boolean"}}, ["dataset_id", "columns"], use_dataset),
        tool("propose_databricks_query", "Databricks 조회를 사용자에게 제안합니다. 이 도구는 실행하지 않습니다. 정확한 SQL과 이유를 표시하고 승인 대기합니다.",
             {"source": string, "query": string, "reason": string}, ["source", "query", "reason"], propose_query),
        tool("recommend_chart_images", "현재 데이터의 통계로 실제 이미지 후보를 만듭니다. 원격 조회 없음. 반환된 카드 ID의 이미지는 UI가 표시합니다.",
             {"dataset_id": string, "columns": {"type": "array", "items": string}},
             ["dataset_id"], chart_options),
        tool("prepare_histogram", "로딩된 결과가 없는 히스토그램의 실행 계획을 만듭니다. source는 정확한 테이블명, column은 수치 컬럼, where_sql은 유지해야 할 사용자 필터 SQL(없으면 빈 문자열)입니다. 반환된 계획은 실행기가 승인 카드와 빈도 히스토그램 생성으로 연결합니다. 직접 원격 조회하지 않습니다.",
             {"source":string,"column":string,"where_sql":string},["source","column"],prepare_histogram),
        tool("render_histogram", "완전한 값별 빈도 집계의 히스토그램을 생성합니다. value_column은 실제 수치값, weight_column은 해당 값의 COUNT(*) 빈도입니다. 원본 행이나 빈도 아닌 집계값을 넣지 마세요.",
             {"dataset_id":string,"value_column":string,"weight_column":string},
             ["dataset_id","value_column","weight_column"],render_histogram),
        tool("local_analysis_sql", "보유 dataset을 data라는 로컬 테이블로 SELECT 계산합니다. 네트워크/파일 접근 없음. 현재 데이터 범위 안에서만 분석하며 결과는 새 ID로 반환합니다. current_result_only는 사용자가 현재 결과 자체만 분석할 때만 true로 지정합니다.",
             {"dataset_id": string, "query": string, "current_result_only": {"type": "boolean"}}, ["dataset_id", "query"], analyze_local),
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
