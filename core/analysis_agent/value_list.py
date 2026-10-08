"""Bounded distinct-value inspection; evidence, never model prose, completes it."""
import json
import re
import time

from sqlglot import exp
from core.analysis_catalog import _source_key, _quoted_table, resolve_table_context
from core.analysis_sql import validate_query
from core.analysis_tool_contract import json_tool_value
from core.analysis_agent.intent_scope import scope_matches
from utils.analysis_provenance import single_table, table_identity
from utils.analysis_datasets import Condition, filter_frame, preview_dataset

LIMIT = 100


def requested(text):
    if re.search(r'개수|몇\s*개|빈도|비율|평균|최대|최소|\b(?:count|how many|frequency|ratio|average|max|min)\b', text, re.I):
        return False
    return bool(re.search(r'어떤\s*(?:값|종류)|값들.{0,25}(?:되어|있|보여|알려)|'
        r'(?:고유값|유니크|값\s*목록|범주\s*목록).{0,20}(?:보여|알려|확인|출력)|'
        r'\b(?:distinct|unique|possible)\s+values\b|\bwhat\s+values\b', text, re.I))


def target(context, current):
    columns = current.get('required_columns', [])
    text = current.get('request_text', '')
    direct = [column for column in columns if re.search(re.escape(column)
        + r'[`\"]?(?:은|는|의|에)?\s*(?:어떤\s*)?(?:값|고유값|유니크)',text,re.I)]
    if len(direct)==1:column=direct[0]
    elif len(columns)==1:column=columns[0]
    else:return None
    known = {item['table'] for item in context.reference_context
        if any(c.get('name')==column for c in item.get('columns',[]))}
    known.update(info.source for info in context.datasets.metadata.values()
        if column in info.columns and ' | ' not in info.source)
    explicit = current.get('required_sources') or []
    if explicit:
        known = {source for source in known if _source_key(source) in {_source_key(s) for s in explicit}}
    selected = context.datasets.metadata.get(context.selected_dataset_id)
    if selected and selected.source in known and not explicit:return selected.source,column
    return (next(iter(known)),column) if len(known)==1 else None


def _receipt(context, info):
    reader=getattr(context,'remote_receipt_reader',None)
    if reader is None:return False
    record=reader(info.id)
    if not record:return False
    result=record.get('result') or {}; saved=result.get('dataset') or {}
    return bool(record.get('status')=='completed' and result.get('status')=='ready'
        and record.get('query')==info.query and saved.get('query')==info.query
        and saved.get('source')==info.source and saved.get('rows')==info.rows
        and list(saved.get('columns',[]))==list(info.columns) and saved.get('id')==info.id)


def retained(context, source, column, conditions, *, allowed_ids=None):
    """Reuse only an exact identity DISTINCT with a completed execution receipt."""
    scope={'conditions':conditions,'any_conditions':[],'unresolved':[]}
    for info in reversed(list(context.datasets.metadata.values())):
        if allowed_ids is not None and info.id not in allowed_ids:continue
        if (_source_key(info.source)!=_source_key(source) or list(info.columns)!=[column]
                or info.parent_id or info.parent_ids or not info.query):continue
        try:
            tree=validate_query(info.query,dialect=context.sql_dialect)
            table=single_table(tree)
            projection=tree.expressions[0] if len(tree.expressions)==1 else None
            if (table is None or _source_key(table_identity(table))!=_source_key(source)
                    or not tree.args.get('distinct') or not isinstance(projection,exp.Column)
                    or projection.name!=column or tree.args.get('group') or tree.args.get('having')
                    or tree.args.get('qualify') or tree.args.get('offset')
                    or not scope_matches(info.query,scope,dialect=context.sql_dialect)
                    or not _receipt(context,info)):continue
            limit=tree.args.get('limit')
            bound=int(limit.expression.this) if limit and limit.expression.is_int else None
            if limit and bound is None:continue
            # A query result can be completely fetched while LIMIT still cuts
            # off population values. Equality with the bound is never full-list proof.
            complete=((bound is None and info.coverage=='complete') or
                (info.coverage in {'complete','unknown'} and bound is not None and info.rows<bound))
            values=preview_dataset(context.datasets,info.id,limit=LIMIT+1)[column].tolist()
            return {'dataset_id':info.id,'source':source,'column':column,
                'values':_values(values[:LIMIT]),'has_more':len(values)>LIMIT or not complete,
                'scope':'모집단 조건의 고유값','conditions':conditions,'reused':True}
        except (ValueError,TypeError,KeyError,OSError):continue
    return None


def _values(values):
    return [json_tool_value(value[:256] if isinstance(value,str) else value) for value in values]


def complete_from_receipts(context, current):
    """Finish the declared obligation from this turn's verified DISTINCT receipt.

    Query completion is not itself value-list evidence. Recheck its projection,
    population, LIMIT and durable receipt before reading the bounded values.
    """
    if not current.get('value_list_requested') or current.get('value_list_evidence'):
        return None
    scope=current.get('scope') or {}
    if current.get('current_result_only') or any(scope.get(k) for k in
            ('unresolved','any_conditions','measure_conditions','join_edges','ratio')):
        return None
    wanted=target(context,current)
    ids={e['dataset_id'] for e in current.get('remote_query_evidence',{}).values()
         if e.get('dataset_id')}
    if not wanted or not ids:return None
    return retained(context,*wanted,scope.get('conditions',[]),allowed_ids=ids)


def prepare(context, source, column, conditions=None, current_result_only=False, fresh=False):
    conditions=conditions or []
    if not fresh and not current_result_only:
        result=retained(context,source,column,conditions)
        if result:return {'status':'ready','value_list':result}
    selected=context.datasets.metadata.get(context.selected_dataset_id)
    if (not fresh and selected and selected.grain=='raw' and selected.predicate_known
            and _source_key(selected.source)==_source_key(source)
            and (selected.coverage=='complete' or current_result_only)
            and column in selected.columns
            and scope_matches(selected.conditions,{'conditions':[],'unresolved':[],'any_conditions':[]})):
        cols=list(dict.fromkeys([column,*[c['column'] for c in conditions]]))
        if set(cols)<=set(selected.columns):
            values={};start=time.monotonic()
            batches=context.datasets.frames.batches(selected.id,cols,expected_rows=selected.rows,batch_size=1024)
            try:
                for batch in batches:
                    if batch.nbytes>8*1024*1024 or time.monotonic()-start>30:
                        raise MemoryError('값 목록의 배치/시간 한도를 초과했습니다.')
                    frame=filter_frame(batch.to_pandas(),tuple(Condition(**c) for c in conditions))
                    for value in frame[column].drop_duplicates().tolist():
                        normalized=json_tool_value(value)
                        key=json.dumps(normalized,sort_keys=True,ensure_ascii=False)
                        values.setdefault(key,normalized)
                        if len(values)>LIMIT:break
                    if len(values)>LIMIT:break
            finally:batches.close()
            return {'status':'ready','value_list':{'dataset_id':selected.id,'source':source,'column':column,
                'values':_values(list(values.values())[:LIMIT]),'has_more':len(values)>LIMIT,
                'scope':'선택한 결과 범위의 고유값' if current_result_only else '모집단 조건의 고유값',
                'conditions':conditions,'reused':True}}
    if current_result_only:
        return {'status':'needs_context','message':'선택한 결과 범위에서 확인 가능한 원본이 필요합니다.'}
    inspection=resolve_table_context(context.reference_context,context.datasets,source)
    if inspection['status']!='ready':return inspection
    columns={c['name'] for c in inspection['table_context'].get('columns',[])}
    from utils.analysis_latest_filters import validate
    conditions=validate(conditions,'before_selection' if conditions else '',columns)
    if column not in columns:return {'status':'needs_context','message':'실제 스키마에서 값 목록 컬럼을 확인해주세요.'}
    quote=lambda name:'`'+name.replace('`','``')+'`'
    # Build typed predicates through the AST; no text interpolation of values.
    nodes=[]
    for item in conditions:
        col=exp.column(item['column'],quoted=True)
        if item['op']=='in':node=exp.In(this=col,expressions=[exp.convert(v) for v in item['value']])
        else:node={'eq':exp.EQ,'ne':exp.NEQ,'gt':exp.GT,'ge':exp.GTE,'lt':exp.LT,'le':exp.LTE}[item['op']](this=col,expression=exp.convert(item['value']))
        nodes.append(node)
    query='SELECT DISTINCT '+quote(column)+' FROM '+_quoted_table(source)
    if nodes:query+=' WHERE '+exp.and_(*nodes).sql(dialect=context.sql_dialect)
    query+=' ORDER BY '+quote(column)+' LIMIT '+str(LIMIT+1)
    validate_query(query,dialect=context.sql_dialect)
    return {'status':'planned','value_list_plan':{'source':source,'query':query,
        'reason':'요청 조건의 고유값을 최대 101행 조회해 100개 표시와 초과 여부를 확인합니다.'}}


def render(runtime,current):
    from html import escape
    proof=current['value_list_evidence']
    def display(value):
        text='NULL' if value is None else str(value)
        text=escape(text).replace('\n',' ').replace('\r',' ')[:256]
        return re.sub(r'([\\`*_\[\]|])',r'\\\1',text)
    values=proof['values']
    text=f"{display(proof['source'])}의 **{display(proof['column'])} 값 목록**입니다.\n\n"
    text+= '\n'.join('- '+display(value) for value in values) if values else '해당 조건에 값이 없습니다.'
    text+='\n\n범위: '+proof['scope']+'.'
    if proof.get('conditions'):text+=' 요청한 필터 조건을 적용했습니다.'
    text+= (' 최대 100개를 표시하며 전체 목록이 아닐 수 있습니다.' if proof['has_more']
            else f' 고유값 {len(values)}개를 모두 표시했습니다.')
    text+=' 긴 문자열은 256자로 제한하여 표시합니다.'
    return text
