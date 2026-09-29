"""Latest record selection with explicit ordering and immutable distribution output."""
from dataclasses import asdict
import pandas as pd
from utils.analysis_datasets import project_dataset
from utils.analysis_charts import render_chart_spec


def latest_distribution(context, dataset_id, key_columns, order_column, value_column,
                        categorical=True, tie_break_columns=None, *, bins=20, null_policy='reject'):
    if null_policy not in {'reject','drop_before_selection'}:
        raise ValueError('지원되는 결측 정책은 reject 또는 drop_before_selection입니다.')
    store=context.datasets
    info=store.metadata[dataset_id]
    tie_break_columns=tie_break_columns or []
    order=[order_column,*tie_break_columns]
    columns=list(dict.fromkeys([*key_columns,*order,value_column]))
    if not key_columns or len(set(key_columns))!=len(key_columns) or len(set(order))!=len(order):
        raise ValueError('키와 정렬 컬럼은 중복 없이 지정해야 합니다.')
    if set(key_columns)&set(order) or value_column in key_columns or not set(columns)<=set(info.columns):
        raise ValueError('키·정렬·분포 대상 컬럼을 현재 스키마에서 구분해야 합니다.')
    if info.grain!='raw' or info.coverage!='complete' or not info.predicate_known:
        return {'status':'needs_context','error_code':'latest_scope_incomplete',
                'message':'완전한 행 데이터가 필요합니다. 표본이나 집계 결과로 전체 키별 최신 행을 결정할 수 없습니다.'}
    execution_mode = 'pandas_projection'
    excluded_rows=0
    if hasattr(store.frames, 'batches') and info.rows > 20_000:
        from utils.analysis_latest_sql import select_latest
        selected, issue, excluded_rows = select_latest(store, info, columns, key_columns, order, null_policy=null_policy)
        if issue:
            return issue
        execution_mode = 'bounded_local_sql'
    else:
        frame=project_dataset(store,dataset_id,columns).reset_index(drop=True)
        if frame.empty:
            return {'status':'needs_context','message':'조건에 해당하는 행이 없습니다. 최신행 분포를 그릴 데이터가 없습니다.'}
        import numpy as np
        if any(np.isinf(frame[c].dropna()).any() for c in columns if pd.api.types.is_float_dtype(frame[c])):
            return {'status':'needs_context','error_code':'latest_nonfinite_policy',
                'message':'키·정렬·분포 컬럼에 무한대가 있습니다. 결측 행 제외와 다른 처리 기준이 필요합니다.'}
        missing=frame[columns].isna().any(axis=1)
        excluded_rows=int(missing.sum())
        if excluded_rows and null_policy=='reject':
            return {'status':'needs_context','error_code':'latest_null_policy',
                    'message':'키·정렬·분포 컬럼에 결측값이 있습니다. 어떤 결측 행을 제외하거나 별도 범주로 처리할지 알려주세요.'}
        frame=frame.loc[~missing].copy()
        if frame.empty:
            return {'status':'needs_context','error_code':'latest_empty','message':'결측 행을 제외한 뒤 분석할 행이 없습니다.'}
        sort_frame=frame.copy()
        for column in order:
            if not (pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_datetime64_any_dtype(frame[column])):
                try:sort_frame[column]=pd.to_datetime(frame[column],errors='raise',utc=True)
                except (ValueError,TypeError):
                    return {'status':'needs_context','error_code':'latest_order_type',
                            'message':'최신 순서를 나타내는 시간 또는 수치 컬럼을 선택해주세요.'}
        ranked=sort_frame.sort_values(order,ascending=False,kind='stable')
        winners=ranked.drop_duplicates(key_columns,keep='first')
        # Only ties at the maximum matter; older duplicate timestamps are harmless.
        top=ranked.merge(winners[[*key_columns,*order]],on=[*key_columns,*order],how='inner')
        if top.duplicated(key_columns,keep=False).any():
            return {'status':'needs_context','error_code':'latest_order_tie',
                    'message':'같은 키에 최신 정렬값이 동일한 행이 여러 개입니다. 동률일 때 사용할 추가 정렬 컬럼을 알려주세요.'}
        selected=frame.loc[winners.index].copy()
    if not selected[key_columns].drop_duplicates().shape[0]==len(selected):
        raise ValueError('최신 행의 키 유일성을 확인하지 못했습니다.')
    if categorical and selected[value_column].nunique()>50:
        return {'status':'needs_context','error_code':'latest_category_limit',
                'message':'범주가 50개를 넘습니다. 표시할 상위 범주 수나 묶는 기준을 알려주세요.'}
    contract={'kind':'latest_per_key','key_columns':list(key_columns),
        'order_columns':order,'descending':True,'null_policy':null_policy,'tie_policy':'reject',
        'input_dataset_id':dataset_id,'input_snapshot':info.snapshot}
    chosen=store.register(selected,source=info.source,parent_id=info.id,snapshot=info.snapshot,
        coverage='complete',predicate_known=True,conditions=info.conditions,row_selection=contract)
    count_column='__key_count'
    while count_column in columns:count_column+='_'  # Avoid schema-specific reserved names.
    counted=selected.groupby(value_column,dropna=False).size().reset_index(name=count_column)
    if int(counted[count_column].sum())!=len(selected):
        raise ValueError('분포 합계와 선택한 키 수가 일치하지 않습니다.')
    distribution=store.register(counted,source=info.source,parent_id=chosen.id,snapshot=info.snapshot,
        coverage='complete',predicate_known=True,conditions=info.conditions,grain='aggregate',
        aggregation='count_by_value_after_latest_per_key')
    if categorical:
        card,summary,spec=render_chart_spec(store,distribution.id,kind='bar',x=value_column,
            y=count_column,aggregation='none',title=f'{value_column} · 키별 최신 행 분포',y_label='Unique keys')
    else:
        card,summary,spec=render_chart_spec(store,chosen.id,kind='histogram',x=value_column,bins=bins)
    context.artifacts[card.id]=card
    return {'status':'ready','dataset':asdict(chosen),'distribution':asdict(distribution),
        'latest_selection':contract,'selected_keys':len(selected),'input_rows':info.rows,'excluded_rows':excluded_rows,
        'count_column':count_column,'value_column':value_column,'execution_mode':execution_mode,
        'counts':counted.head(50).to_dict('records'),'counts_truncated':len(counted)>50,
        'chart_spec':spec,'render_summary':summary,
        'cards':[{'id':card.id,'kind':card.kind,'title':card.title,'dataset_id':card.dataset_id}]}
