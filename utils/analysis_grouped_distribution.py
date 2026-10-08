"""Complete two-axis COUNT distributions, rendered without expanding raw rows."""
from io import BytesIO
from uuid import uuid4
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from utils.analysis_datasets import project_dataset
from utils.analysis_provenance import single_table, query_coverage


def frequency_columns(tree):
    if (single_table(tree) is None or query_coverage(tree)!='complete' or
        any(tree.args.get(k) for k in ('distinct','having','qualify')) or tree.find(exp.Window)
        or len(tree.expressions)!=3):return None
    group=tree.args.get('group')
    if not group or len(group.expressions)!=2 or any(not isinstance(e,exp.Column) for e in group.expressions):return None
    if any(v for k,v in group.args.items() if k!='expressions'):return None
    axes=[];weight=None
    for item in tree.expressions:
        node=item.this if isinstance(item,exp.Alias) else item
        if isinstance(node,exp.Column) and not isinstance(item,exp.Alias):axes.append(node.name)
        elif (isinstance(item,exp.Alias) and isinstance(node,exp.Count)
              and isinstance(node.this,exp.Star) and not node.expressions):weight=item.alias
        else:return None
    if len(set(axes+[weight]))!=3 or set(axes)!={e.name for e in group.expressions}:return None
    return (*axes,weight)


def render(store,dataset_id,value,category,weight,bins=8):
    from utils.analysis_charts import ChartPreview,_apply_unicode_font
    info=store.metadata[dataset_id]
    tree=sqlglot.parse_one(info.query,read='duckdb' if info.parent_id else 'databricks')
    if info.grain!='aggregate' or info.coverage!='complete' or not info.aggregation or frequency_columns(tree)!=(value,category,weight):
        raise ValueError('수치·그룹별 완전한 COUNT(*) 실행 출처가 필요합니다.')
    if info.parent_id:
        parent=store.metadata[info.parent_id]
        if parent.grain!='raw' or parent.coverage!='complete' or not parent.predicate_known or parent.source!=info.source:
            raise ValueError('완전한 동일 출처 원본에서 집계해야 합니다.')
    if not 2<=int(bins)<=100:raise ValueError('bin은 2~100이어야 합니다.')
    frame=project_dataset(store,dataset_id,[value,category,weight])
    frame[value]=pd.to_numeric(frame[value],errors='raise')
    frame[weight]=pd.to_numeric(frame[weight],errors='raise')
    if frame.empty or not np.isfinite(frame[value]).all() or not np.isfinite(frame[weight]).all():raise ValueError('유효한 값과 빈도가 필요합니다.')
    if (frame[weight]<0).any() or (frame[weight]%1!=0).any() or frame.duplicated([value,category]).any():raise ValueError('그룹별 빈도는 고유한 값 쌍과 0 이상 정수여야 합니다.')
    labels=sorted(frame[category].dropna().unique(),key=str)
    if not 1<=len(labels)<=20 or len({str(x) for x in labels})!=len(labels):raise ValueError('범례 그룹은 구별 가능한 1~20개여야 합니다.')
    excluded=int(frame.loc[frame[category].isna(),weight].sum())
    frame=frame.dropna(subset=[category])
    if frame[weight].sum()<=0:raise ValueError('표시할 빈도가 없습니다.')
    edges=np.histogram_bin_edges(frame[value],bins=int(bins))
    fig=Figure(figsize=(6,3.5));ax=fig.subplots()
    arrays=[frame.loc[frame[category]==label,value] for label in labels]
    weights=[frame.loc[frame[category]==label,weight] for label in labels]
    colors=['#3278b9','#e58632','#38946c','#b94d66','#7956a1','#8c6d31',
        '#238b9e','#bc80bd','#969696','#80b1d3','#fdb462','#b3de69',
        '#fb8072','#bebada','#8dd3c7','#d9d9d9','#ccebc5','#ffed6f','#fccde5','#a6cee3'][:len(labels)]
    counts,_,_=ax.hist(arrays,weights=weights,bins=edges,histtype='bar',
        label=[str(x) for x in labels],color=colors,edgecolor='white')
    ax.legend(title=category,fontsize=8);ax.set(xlabel=value,ylabel='Count')
    spec={'kind':'histogram','x':value,'category':category,'bins':int(bins),'legend':True,
        'legend_labels':[str(x) for x in labels],'colors':colors,'bin_edges':edges.tolist(),
        'series_counts':np.asarray(counts).reshape(len(labels),-1).astype(int).tolist(),
        'group_totals':[int(w.sum()) for w in weights],'total_count':int(frame[weight].sum()),
        'excluded_null_group_count':excluded,'null_policy':'exclude'}
    buffer=BytesIO();FigureCanvasAgg(fig);_apply_unicode_font(fig);fig.tight_layout();fig.savefig(buffer,format='png',dpi=110)
    return ChartPreview(str(uuid4()),dataset_id,f'{category}별 {value} 분포',
        '공통 bin의 그룹별 COUNT(*)를 서로 다른 색의 막대로 표시하고 범례를 추가했습니다.',
        'histogram',(value,category),f'빈도 합계 {spec["total_count"]:,} · complete · 그룹 결측 제외 {excluded:,} · {info.query}',buffer.getvalue(),spec)


def prepare(context,source,column,category,where_sql,bins,normalized_query,analyze_local,card_entry,source_key,current_result_only=False,fresh_source_required=False):
    from core.analysis_sql import validate_query
    def quote(name):return '`'+name.replace('`','``')+'`'
    table='.'.join(quote(p) for p in source.split('.'))
    names=set()
    for item in context.reference_context:
        if source_key(item.get('table',''))==source_key(source):names.update(c['name'] for c in item.get('columns',[]))
    for info in context.datasets.metadata.values():
        if source_key(info.source)==source_key(source):names.update(info.columns)
    if column==category or not {column,category}.issubset(names):return {'status':'needs_context','message':'현재 schema에서 수치와 그룹 컬럼을 확인해주세요.'}
    query=f'SELECT {quote(column)}, {quote(category)}, COUNT(*) AS `__frequency` FROM {table} WHERE {quote(column)} IS NOT NULL'+(' AND ('+where_sql+')' if where_sql.strip() else '')+f' GROUP BY {quote(column)}, {quote(category)}'
    tree=validate_query(query,dialect=context.sql_dialect)
    if frequency_columns(tree)!=(column,category,'__frequency'):raise ValueError('허용된 단일 출처 집계만 계획할 수 있습니다.')
    plan={'source':source,'query':query,'reason':f'{column} 분포를 {category}별 색상·범례로 표시하기 위한 빈도 집계입니다. 원본 전체를 가져오지 않습니다.',
        'value_column':column,'category':category,'weight_column':'__frequency','bins':int(bins)}
    def ready(info):
        card=next((c for c in context.artifacts.values() if c.dataset_id==info.id and c.render_spec.get('category')==category and c.render_spec.get('bins')==int(bins) and c.render_spec.get('legend')),None)
        if card is None:
            card=render(context.datasets,info.id,column,category,'__frequency',bins);context.artifacts[card.id]=card
        return {'status':'ready','histogram_plan':plan,'loaded_dataset':info.id,'cards':[card_entry(card)],'reused':True}
    if not fresh_source_required:
        for info in reversed(list(context.datasets.metadata.values())):
            if source_key(info.source)!=source_key(source) or info.coverage!='complete':continue
            try:
                parsed=validate_query(info.query,dialect='duckdb' if info.parent_id else context.sql_dialect)
                if info.parent_id:
                    if single_table(parsed) is None:continue
                    single_table(parsed).replace(single_table(tree).copy())
                if normalized_query(parsed)==normalized_query(tree):return ready(info)
            except (ValueError,TypeError,sqlglot.errors.SqlglotError):continue
        def compatible_raw(info):
            if (source_key(info.source)!=source_key(source) or info.grain!='raw'
                    or info.coverage!='complete' or not info.predicate_known or info.conditions
                    or not {column,category}.issubset(info.columns)):return False
            if not info.query:return True
            try:return not validate_query(info.query,dialect=context.sql_dialect).args.get('limit')
            except (ValueError,TypeError,sqlglot.errors.SqlglotError):return False
        raw=[i for i in context.datasets.metadata.values() if compatible_raw(i)]
        if current_result_only:raw=[i for i in raw if i.id==context.selected_dataset_id]
        if len(raw)==1:
            local=tree.copy();single_table(local).replace(exp.Table(this=exp.to_identifier('data')))
            result=analyze_local(raw[0].id,local.sql(dialect='duckdb'))
            if result['status']=='ready':return ready(context.datasets.metadata[result['dataset']['id']])
    if current_result_only:return {'status':'needs_data','message':'현재 결과에는 그룹별 빈도가 없습니다. 원본 또는 그룹별 빈도 결과를 선택해주세요.'}
    return {'status':'planned','histogram_plan':plan}
