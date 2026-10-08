"""Bounded row display, separated from source loading and analytical selection."""
import re
from sqlglot import parse_one
from core.analysis_catalog import full_schema_source, resolve_table_context
from utils.analysis_provenance import table_identity
from utils.analysis_datasets import preview_dataset


def requested(text):
    if not isinstance(text,str) or re.search(r'그려|시각화|histogram|chart|plot|조인|join|평균|합계|고유|결측',text,re.I):
        return None
    unit=r'(?:rows?|records?|레코드|행|건)(?![A-Za-z_])'
    number=r'(?<![\d.])([+-]?[\d,]+)(?![\d.,])'
    # Count-unit and unit-count word orders both express a bounded preview.
    # ASCII boundaries admit Korean particles and "row10개" without matching
    # parts of column/table identifiers such as row_id or records_archive.
    match=re.search(number+r'\s*(?:개\s*)?'+unit,text,re.I)
    if not match:
        match=re.search(r'(?<![A-Za-z0-9_])'+unit+r'\s*(?:를|을|만)?\s*'+number+r'\s*(?:개|건|행)?',text,re.I)
    if not match or not re.search(r'보여|출력|추출|미리보기|표|(?<![A-Za-z0-9_])(?:table|preview|show|display|extract|sample|head)(?![A-Za-z0-9_])',text,re.I):
        return None
    limit=int(match[1].replace(',',''))
    return {'limit':limit,'question':('표 미리보기는 1~200행으로 요청해주세요.' if not 1<=limit<=200 else '')}


def key(source):
    return str(source).replace('`','').casefold()


def evidence(store,dataset_id,limit):
    info=store.metadata[dataset_id]
    frame=preview_dataset(store,dataset_id,limit=limit)
    if list(frame.columns)!=list(info.columns) or len(frame)!=min(info.rows,limit):
        raise ValueError('Stored preview shape does not match dataset metadata')
    return {'dataset_id':info.id,'source':info.source,'snapshot':info.snapshot,
            'columns':list(info.columns),'total_rows':info.rows,'rows':len(frame),'limit':limit}


def frame(store,proof):
    info=store.metadata[proof['dataset_id']]
    if any(proof.get(k)!=v for k,v in {'source':info.source,'snapshot':info.snapshot,
            'columns':list(info.columns),'total_rows':info.rows}.items()):
        raise ValueError('Table display references a different dataset')
    result=preview_dataset(store,info.id,limit=proof['limit'])
    if len(result)!=proof['rows'] or list(result.columns)!=proof['columns']:
        raise ValueError('Table display shape mismatch')
    return result


def prepare(context,source,limit=10,where_sql='',current_result_only=False,fresh=False):
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=200:
        raise ValueError('limit must be an integer between 1 and 200')
    store=context.datasets
    if current_result_only:
        selected=store.metadata.get(context.selected_dataset_id)
        if not selected or key(selected.source)!=key(source) or where_sql:
            return {'status':'needs_context','message':'현재 결과의 필터·출처를 먼저 확정해야 합니다.'}
        return {'status':'ready','table_preview_evidence':evidence(store,selected.id,limit)}
    inspection=resolve_table_context(context.reference_context,store,source)
    if inspection.get('status') not in {'ready','needs_refresh'}:
        return {'status':'needs_context','message':'미리볼 테이블의 실제 출처를 확인해주세요.'}
    source=inspection.get('table_context',{}).get('table') or source
    columns=[c['name'] for c in inspection.get('table_context',{}).get('columns',[]) if c.get('name')]
    if not fresh and not where_sql and inspection.get('status')=='ready':
        for info in reversed(list(store.metadata.values())):
            if (key(info.source)!=key(source) or info.grain!='raw' or info.parent_id
                    or tuple(info.columns)!=tuple(columns)
                    or info.rows<limit and info.coverage!='complete'):
                continue
            if info.query:
                tree=parse_one(info.query,read=context.sql_dialect)
                if (full_schema_source(info.query,dialect=context.sql_dialect) is None
                        or tree.args.get('where') or tree.args.get('order')):
                    continue
                bound=tree.args.get('limit')
                if bound and int(bound.expression.this)<limit:
                    continue
            elif info.coverage!='complete' or not info.predicate_known:
                continue
            return {'status':'ready','table_preview_evidence':evidence(store,info.id,limit)}
    query='SELECT * FROM '+'.'.join('`'+p.replace('`','``')+'`' for p in source.split('.'))
    if where_sql:query+=' WHERE ('+where_sql+')'
    query+=f' LIMIT {limit}'
    from core.analysis_load_plan import source_plan
    source_plan(source,query,dialect=context.sql_dialect)
    return {'status':'planned','row_preview_plan':{'source':source,'query':query,
        'reason':f'요청한 표 미리보기용으로 최대 {limit}행만 조회합니다. 기존 분석 기준은 유지합니다.'}}


def accept_remote(context,current,receipt):
    plan=current.get('row_preview_plan') or {}
    info=context.datasets.metadata.get(receipt.get('dataset_id'))
    limit=current['row_preview_spec']['limit']
    if (not info or info.query!=plan.get('query') or key(info.source)!=key(plan.get('source'))
            or info.grain!='raw' or info.rows>limit
            or full_schema_source(info.query,dialect=context.sql_dialect) is None):
        return None
    return evidence(context.datasets,info.id,limit)


def valid_plan(context,current,plan):
    try:
        query=plan['query']
        source=full_schema_source(query,dialect=context.sql_dialect)
        tree=parse_one(query,read=context.sql_dialect)
        bound=tree.args.get('limit')
        return bool(source is not None and key(table_identity(source))==key(plan['source'])
            and key(plan['source']) in {key(s) for s in current.get('required_sources',[])}
            and bound and int(bound.expression.this)==current['row_preview_spec']['limit']
            and not any(tree.args.get(k) for k in ('group','having','offset','distinct')))
    except (KeyError,ValueError,TypeError):
        return False


def render(runtime,current):
    proof=current['table_preview_evidence']
    frame(runtime.context.datasets,proof)
    return (f"{proof['source']}의 데이터 {proof['rows']}행, {len(proof['columns'])}열을 표로 표시했습니다. "
            '원본과 기존 분석 기준은 보존했습니다. 이 미리보기는 전체 분포 통계가 아닙니다.'
            + (' 이번 조회 결과는 0행입니다.' if not proof['rows'] else ''))
