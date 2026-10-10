"""Compile verified scalar goals against discovered schemas, without prose routing."""
from sqlglot import exp
from core.analysis_catalog import resolve_table_context
from core.analysis_agent.predicate_sql import where_sql
from core.analysis_agent.dtypes import family

OPERATIONS={'AVG':'average','SUM':'sum','MIN':'minimum','MAX':'maximum'}

def next_call(context,current,remote_available,remote_blocked):
    scope=current.get('scope') or {}
    sources=current.get('required_sources') or []
    columns=current.get('required_columns') or []
    operations=current.get('operations') or []
    if (not context or current.get('intent_origin')!='llm' or not current.get('calculation')
            or current.get('evidence_ids') or current.get('current_result_only')
            or current.get('scalar_grouping') or current.get('group_columns')
            or not remote_available or remote_blocked or len(sources)!=1 or len(columns)!=1
            or not operations or not set(operations)<=set(OPERATIONS)
            or any(scope.get(k) for k in ('unresolved','join_edges','measure_conditions','ratio'))):
        return None
    observed=resolve_table_context(context.reference_context,context.datasets,sources[0])
    if observed.get('status')!='ready':return None
    known={c['name']:c.get('dtype') for c in observed['table_context']['columns']}
    predicates=scope.get('conditions',[])+scope.get('any_conditions',[])
    if (columns[0] not in known or family(known[columns[0]])!='numeric'
            or not {c['column'] for c in predicates}<=set(known)):
        return None
    quote=exp.column(columns[0],quoted=True).sql(dialect=context.sql_dialect)
    projections=[f'{op}({quote}) AS {OPERATIONS[op]}' for op in operations]
    clauses=[]
    if scope.get('conditions'):clauses.append('('+where_sql(scope['conditions'],context.sql_dialect)+')')
    if scope.get('any_conditions'):clauses.append('('+' OR '.join(where_sql([c],context.sql_dialect) for c in scope['any_conditions'])+')')
    table=exp.to_table(sources[0],quoted=True).sql(dialect=context.sql_dialect)
    query='SELECT '+', '.join(projections)+' FROM '+table+(' WHERE '+' AND '.join(clauses) if clauses else '')
    return {'name':'query_databricks','args':{'source':sources[0],'query':query,
        'reason':'확인된 스키마와 요청한 연산·조건으로 데이터베이스에서 집계합니다.'}}
