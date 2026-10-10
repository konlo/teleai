"""Bounded COUNT(*) tool plan from an LLM-authored, observed population."""
from sqlglot import exp
from core.analysis_catalog import resolve_table_context
from core.analysis_agent.predicate_sql import where_sql


def next_call(context,current,remote_available,remote_blocked):
    scope=current.get('scope') or {}
    sources=current.get('required_sources') or []
    if (not context or current.get('intent_origin')!='llm' or not current.get('whole_row_count')
            or current.get('evidence_ids') or current.get('current_result_only')
            or not remote_available or remote_blocked or len(sources)!=1
            or any(scope.get(k) for k in ('unresolved','join_edges','measure_conditions','ratio'))):return None
    observed=resolve_table_context(context.reference_context,context.datasets,sources[0])
    if observed.get('status')!='ready':return {'name':'inspect_table_context','args':{'table':sources[0]}}
    columns={c['name'] for c in observed['table_context']['columns']}
    conditions=scope.get('conditions',[]);alternatives=scope.get('any_conditions',[])
    if not {c['column'] for c in conditions+alternatives}.issubset(columns):return None
    parts=[]
    if conditions:parts.append('('+where_sql(conditions,context.sql_dialect)+')')
    if alternatives:parts.append('('+' OR '.join(where_sql([c],context.sql_dialect) for c in alternatives)+')')
    table=exp.to_table(sources[0],quoted=True).sql(dialect=context.sql_dialect)
    query='SELECT COUNT(*) AS count FROM '+table+(' WHERE '+' AND '.join(parts) if parts else '')
    return {'name':'query_databricks','args':{'source':sources[0],'query':query,
        'reason':'확인된 테이블과 요청 조건의 전체 행 수를 데이터베이스에서 계산합니다.'}}
