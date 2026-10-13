"""Bounded whole-source numeric bins with independently checked SQL receipts."""
from io import BytesIO
from uuid import uuid4
import re

import numpy as np
from sqlglot import exp
from sqlglot.errors import SqlglotError
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter

from core.analysis_sql import validate_query
from core.analysis_catalog import resolve_table_context
from utils.analysis_provenance import table_identity
from utils.analysis_datasets import project_dataset
from utils.analysis_charts import ChartPreview, _apply_unicode_font

WEIGHT='__frequency'
FIELDS=('__hist_min','__hist_max','__hist_total','__hist_bins')


def quote(value):
    return chr(96)+value.replace(chr(96),chr(96)*2)+chr(96)


def query_for(source,column,bins=20,where_sql='',dialect='databricks'):
    if type(bins) is not int or not 2<=bins<=100:
        raise ValueError('bins must be an integer in 2..100')
    if not source or not column or column in (WEIGHT,*FIELDS):
        raise ValueError('Distinct observed source/value names are required')
    table='.'.join(quote(p) for p in source.split('.'))
    col=quote(column)
    predicate=f'{col} IS NOT NULL'+(' AND ('+where_sql+')' if where_sql.strip() else '')
    # One physical source query; at most bins output rows, regardless of the
    # source cardinality. Numeric conversion is explicit, never TRY_CAST/drop.
    query=f"""SELECT __bucket AS {col}, COUNT(*) AS {quote(WEIGHT)},
 MIN(__raw_lo) AS __hist_min, MAX(__raw_hi) AS __hist_max,
 MAX(__n) AS __hist_total, {bins} AS __hist_bins
FROM (
 SELECT CASE WHEN __hi = __lo THEN 0 ELSE
 LEAST({bins-1}, GREATEST(0, CAST(FLOOR((__v - __lo) / (__hi - __lo) * {bins}) AS BIGINT)))
 END AS __bucket, __raw_lo, __raw_hi, __n
 FROM (
  SELECT __v, __raw_lo, __raw_hi, __n,
   CASE WHEN __raw_lo = __raw_hi THEN __raw_lo - GREATEST(0.5, ABS(__raw_lo)*1e-12) ELSE __raw_lo END AS __lo,
   CASE WHEN __raw_lo = __raw_hi THEN __raw_hi + GREATEST(0.5, ABS(__raw_hi)*1e-12) ELSE __raw_hi END AS __hi
  FROM (
   SELECT CAST({col} AS DOUBLE) AS __v,
    MIN(CAST({col} AS DOUBLE)) OVER () AS __raw_lo,
    MAX(CAST({col} AS DOUBLE)) OVER () AS __raw_hi,
    COUNT(*) OVER () AS __n
   FROM {table} WHERE {predicate}
  ) AS __hist_values
 ) AS __hist_bounds
) AS __hist_rows GROUP BY __bucket"""
    return validate_query(query).sql(dialect=dialect)


def descriptor(query,dialect='databricks'):
    """Recognize only the exact supported template, not arbitrary bucket SQL."""
    try:
        tree=validate_query(query,dialect=dialect)
        tables=list(tree.find_all(exp.Table))
        if len(tables)!=1 or len(list(tree.find_all(exp.Select)))!=4:return None
        projections={item.alias:item for item in tree.expressions if isinstance(item,exp.Alias)}
        count=projections.get('__hist_bins')
        if count is None or not isinstance(count.this,exp.Literal) or count.this.is_string:return None
        bins=int(count.this.this)
        inner=tables[0].find_ancestor(exp.Select)
        value=inner.expressions[0].this
        if not isinstance(value,exp.Cast) or not isinstance(value.this,exp.Column):return None
        column=value.this.name;source=table_identity(tables[0])
        mandatory=validate_query('SELECT * FROM data WHERE '+quote(column)+' IS NOT NULL',dialect=dialect).args['where'].this
        predicate=inner.args['where'].this
        extra=''
        if predicate!=mandatory:
            if not isinstance(predicate,exp.And) or predicate.this!=mandatory:return None
            extra=predicate.expression
            if isinstance(extra,exp.Paren):extra=extra.this
            extra=extra.sql(dialect=dialect)
        expected=validate_query(query_for(source,column,bins,extra,dialect),dialect=dialect)
        if tree!=expected:return None
        return {'source':source,'column':column,'bins':bins,'where_sql':extra,
            'population_query':f'SELECT {quote(column)} FROM '+
                '.'.join(quote(p) for p in source.split('.'))+' WHERE '+predicate.sql(dialect=dialect)}
    except (ValueError,TypeError,KeyError,AttributeError,SqlglotError):
        return None


def plan_for(context,source,column,bins=20,where_sql=''):
    observed=resolve_table_context(context.reference_context,context.datasets,source)
    if observed.get('status')!='ready':
        raise ValueError('Fresh observed schema is required for source bins')
    types={c['name']:c.get('dtype','') for c in observed['table_context'].get('columns',[])}
    if column not in types or not re.search(r'int|float|double|decimal|numeric|real|number',str(types[column]),re.I):
        raise ValueError('An observed numeric source column is required')
    query=query_for(source,column,bins,where_sql,context.sql_dialect)
    if descriptor(query,context.sql_dialect) is None:
        raise ValueError('Unsupported source bin predicate')
    return {'source':source,'query':query,'value_column':column,'weight_column':WEIGHT,
        'bins':bins,'reason':'전체 유효 수치의 구간별 빈도를 DB에서 집계합니다. 원본 전체를 전송하지 않습니다.'}


def target(context,current):
    if (not context or current.get('intent_origin')!='llm' or not current.get('chart')
            or current.get('kind')!='histogram' or current.get('current_result_only')
            or current.get('categorical_distribution') or current.get('requested_join')
            or current.get('chart_group_spec') or current.get('calculation')
            or current.get('latest_per_key_spec') or current.get('custom_analysis_spec')
            or current.get('advanced_eda_spec') or current.get('chart_axis_spec')):return None
    scope=current.get('scope') or {}
    if any(scope.get(k) for k in ('unresolved','join_edges','ratio','measure_conditions')):return None
    sources=current.get('required_sources') or []
    columns=current.get('required_columns') or []
    if len(sources)!=1 or len(columns)!=1:return None
    if not current.get('fresh_source_required'):
        from utils.analysis_datasets import AnalysisNeed,Condition,assess_reuse
        if not scope.get('any_conditions'):
            try:
                need=AnalysisNeed(source=sources[0],columns=tuple(columns),
                    conditions=tuple(Condition(**c) for c in scope.get('conditions',[])))
                if any(assess_reuse(info,need).action!='query_source' for info in context.datasets.metadata.values()):
                    # Existing complete raw inputs use prepare_histogram's
                    # original-preserving local path rather than another load.
                    return None
            except (ValueError,TypeError):return None
    from core.analysis_agent.predicate_sql import population_sql
    args={'source':sources[0],'column':columns[0],'bins':current.get('histogram_bins') or 20,
        'where_sql':population_sql(scope,context.sql_dialect),
        'fresh':bool(current.get('fresh_source_required')),
        **{k:v for k,v in current.get('chart_presentation_spec',{}).items() if k in ('legend','palette')}}
    try:plan_for(context,**{k:args[k] for k in ('source','column','bins','where_sql')})
    except (ValueError,KeyError):return None
    return args


def same_predicate(left,right,dialect):
    def tree(predicate):
        if not predicate.strip():return None
        value=validate_query('SELECT * FROM data WHERE '+predicate,dialect=dialect).args['where'].this
        def normalize(node):
            if isinstance(node,exp.Paren):return node.this
            if isinstance(node,exp.Identifier):return exp.Identifier(this=node.name.casefold())
            return node
        return value.transform(normalize)
    try:return tree(left)==tree(right)
    except (ValueError,TypeError,AttributeError,SqlglotError):return False


def valid_call(recovery,current,args):
    wanted=target(recovery.context,current)
    if not wanted:return False
    if args.get('source')!=wanted['source'] or args.get('column')!=wanted['column']:return False
    if current.get('histogram_bins') and args.get('bins',20)!=current['histogram_bins']:return False
    if current.get('fresh_source_required') and not args.get('fresh'):return False
    if not same_predicate(args.get('where_sql',''),wanted['where_sql'],recovery.sql_dialect):return False
    spec=current.get('chart_presentation_spec') or {}
    if spec.get('stacked'):return False
    if any(args.get(k,d)!=spec.get(k,d) for k,d in (('legend',False),('palette','default'))):return False
    return recovery._scope_valid('SELECT * FROM data'+(' WHERE '+args['where_sql'] if args.get('where_sql') else ''),
        current,args['column'])


def values(store,dataset_id,column,dialect='databricks'):
    info=store.metadata[dataset_id]
    desc=descriptor(info.query,dialect)
    if (desc is None or desc['column']!=column or info.parent_id or info.parent_ids
            or info.coverage!='complete' or info.grain!='aggregate' or not info.aggregation):
        raise ValueError('Complete executed source bin SQL is required')
    frame=project_dataset(store,dataset_id,[column,WEIGHT,*FIELDS])
    if frame.empty or len(frame)>desc['bins']:raise ValueError('No complete numeric bin result')
    matrix=frame.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():raise ValueError('Finite numeric values are required')
    ids=matrix[:,0];weights=matrix[:,1]
    if (np.any(ids!=np.floor(ids)) or np.any(ids<0) or np.any(ids>=desc['bins'])
            or len(set(ids))!=len(ids) or np.any(weights<=0) or np.any(weights!=np.floor(weights))):
        raise ValueError('Unique bin indices and positive integer frequencies are required')
    stats=matrix[:,2:]
    if not np.all(stats==stats[0]) or stats[0,3]!=desc['bins']:
        raise ValueError('Bin statistics are inconsistent')
    lo,hi,total=stats[0,:3]
    if lo>hi or total>2**53 or total!=np.floor(total) or int(sum(map(int,weights)))!=int(total):
        raise ValueError('Full population frequency sum is inconsistent')
    if lo==hi:
        pad=max(.5,abs(lo)*1e-12);lo-=pad;hi+=pad
    edges=np.linspace(lo,hi,desc['bins']+1)
    if not np.all(np.diff(edges)>0):raise ValueError('Numeric precision does not support these bins')
    counts=np.zeros(desc['bins'],dtype=np.int64)
    for i,w in zip(ids,weights):counts[int(i)]=int(w)
    return desc,edges,counts,int(total)


def render(store,dataset_id,column,*,dialect='databricks',legend=False,palette='default'):
    if type(legend) is not bool or palette not in {'default','high_contrast'}:
        raise ValueError('Unsupported histogram presentation')
    desc,edges,counts,total=values(store,dataset_id,column,dialect)
    fig=Figure(figsize=(6,3.5));ax=fig.subplots()
    ax.bar(edges[:-1],counts,width=np.diff(edges),align='edge',edgecolor='white',
        color='#0072B2' if palette=='high_contrast' else '#3278b9',label=column)
    if legend:ax.legend()
    ax.set(xlabel=column,ylabel='Count')
    # Fonts available on the host may lack the Unicode minus glyph.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:format(value,'g')))
    spec={'mode':'source_numeric_bins','aggregation':'count','bins':desc['bins'],
        'bin_edges':edges.tolist(),'bin_counts':counts.tolist(),'total_count':total,
        'null_policy':'exclude','legend':legend,'legend_labels':[column] if legend else [],
        'stacked':False,'palette':palette,'numeric_representation':'double'}
    buffer=BytesIO();FigureCanvasAgg(fig);_apply_unicode_font(fig);fig.tight_layout()
    fig.savefig(buffer,format='png',dpi=110)
    return ChartPreview(str(uuid4()),dataset_id,column+' 분포',
        'DB에서 전체 유효 수치의 구간별 빈도를 계산했습니다. 수치 표현은 DOUBLE이며 NULL은 제외했습니다.',
        'histogram',(column,),f'전체 유효값 {total:,}개 · complete · DB bins {desc["bins"]} · NULL 제외',
        buffer.getvalue(),spec)


def valid_card(recovery,current,card):
    try:
        wanted=target(recovery.context,current)
        if not wanted or card.kind!='histogram' or card.columns!=(wanted['column'],):return False
        info=recovery.context.datasets.metadata[card.dataset_id]
        from core.analysis_agent.row_preview import key
        if key(info.source)!=key(wanted['source']):return False
        desc,edges,counts,total=values(recovery.context.datasets,card.dataset_id,wanted['column'],recovery.sql_dialect)
        spec=card.render_spec
        if (spec.get('bin_edges')!=edges.tolist() or spec.get('bin_counts')!=counts.tolist()
                or spec.get('total_count')!=total or spec.get('bins')!=desc['bins']):return False
        if current.get('histogram_bins') and spec['bins']!=current['histogram_bins']:return False
        if not same_predicate(desc['where_sql'],wanted['where_sql'],recovery.sql_dialect):return False
        receipt=recovery.context.remote_receipt_reader(card.dataset_id)
        if not receipt or receipt.get('status')!='completed':return False
        return recovery._scope_valid(desc['population_query'],current,wanted['column'],record_error=False)
    except (ValueError,TypeError,KeyError):return False


def execute(recovery,current,args,query_call,runtime):
    """Use the ordinary execution boundary; persist only verified complete bins."""
    if not valid_call(recovery,current,args):
        return {'status':'needs_context','error_code':'histogram_goal_mismatch',
                'message':'현재 확정 목표의 출처·수치 컬럼·조건·표현을 유지해야 합니다.'}
    context=recovery.context
    plan=plan_for(context,**{k:args[k] for k in ('source','column','bins','where_sql')})
    receipt=None;result=None
    if not args.get('fresh'):
        for info in reversed(list(context.datasets.metadata.values())):
            if info.source!=plan['source'] or info.query!=plan['query']:continue
            saved=context.remote_receipt_reader(info.id)
            if saved and saved.get('status')=='completed':
                try:values(context.datasets,info.id,args['column'],context.sql_dialect)
                except (ValueError,KeyError,TypeError):continue
                receipt=saved;result=saved['result'];break
    reused=result is not None
    if result is None:
        result=query_call(**{k:plan[k] for k in ('source','query','reason')},runtime=runtime)
        if result.get('status')!='ready':return {**result,'source_histogram_plan':plan}
        receipt=recovery.approval_ledger.get(runtime.tool_call_id)
    from core.analysis_agent.remote_completion import verified_receipt
    proof=verified_receipt(recovery.approval_ledger,context,
        {'id':receipt['id'],'args':{k:receipt[k] for k in ('source','query','reason')}},result)
    if not proof:return {'status':'unavailable','error_code':'remote_receipt_unverified'}
    try:
        card=render(context.datasets,proof['dataset_id'],args['column'],dialect=context.sql_dialect,
                    legend=args.get('legend',False),palette=args.get('palette','default'))
        if not valid_card(recovery,current,card):raise ValueError('Histogram result does not meet the goal')
        context.artifacts[card.id]=card
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'no_valid_chart','error_code':'source_bins_invalid',
                'message':str(exc),'source_histogram_plan':plan,'query_result':result,'receipt_id':receipt['id']}
    return {'status':'ready','source_histogram_plan':plan,'query_result':result,
        'receipt_id':receipt['id'],'reused':reused,'loaded_dataset':card.dataset_id,
        'dataset':result['dataset'],'cards':[{'id':card.id,'dataset_id':card.dataset_id,
            'title':card.title,'reason':card.reason,'kind':card.kind,'columns':card.columns,'scope':card.scope}]}
