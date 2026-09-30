"""Schema-grounded remote latest distribution plan and bounded result validation."""
from dataclasses import asdict
import math
import pandas as pd
from core.analysis_catalog import resolve_table_context, _quoted_table, _source_key
from core.analysis_sql import validate_query
from utils.analysis_charts import render_chart_spec
from utils.analysis_datasets import project_dataset

FIELDS = ['__kind', '__value', '__frequency', '__input_rows', '__null_rows',
          '__tied_keys', '__selected_keys', '__categories', '__nonfinite_rows']


def _quote(name):
    return '`' + name.replace('`', '``') + '`'


def plan(context, source, key_columns, order_column, value_column, *, categorical=True, bins=20, tie_break_columns=None, null_policy='reject', conditions=None, filter_stage=''):
    if null_policy not in {'reject','drop_before_selection'}:
        raise ValueError('지원하지 않는 최신행 결측 정책입니다.')
    if type(categorical) is not bool or type(bins) is not int or not 2 <= bins <= 100:
        raise ValueError('categorical은 bool, bins는 2~100 정수여야 합니다.')
    observed = resolve_table_context(context.reference_context, context.datasets, source)
    schema = observed.get('table_context') or {}
    if observed.get('status') != 'ready' or schema.get('freshness') not in {'fresh', 'current_loaded_schema'}:
        return {'status': 'needs_refresh', 'refresh_query': 'SELECT * FROM '+_quoted_table(source)+' LIMIT 0',
                'message': '현재 테이블 스키마 확인이 필요합니다.'}
    types = {c['name']: c.get('dtype', '').lower() for c in schema.get('columns', [])}
    from utils.analysis_latest_filters import validate, validate_types, sql as filter_sql, lineage
    conditions=validate(conditions,filter_stage,types)
    validate_types(conditions,types)
    order = [order_column, *(tie_break_columns or [])]
    columns = [*key_columns, *order, value_column]
    if (not 1 <= len(key_columns) <= 4 or len(set(columns)) != len(columns) or not set(columns) <= types.keys()):
        return {'status': 'needs_context', 'message': '현재 스키마에서 서로 다른 키·정렬·분포 컬럼을 확인해주세요.'}
    import re
    if any(not re.search(r'int|long|short|byte|float|double|decimal|numeric|timestamp|datetime|date', types[c]) for c in order):
        return {'status': 'needs_context', 'message': '최신 순서를 확인할 수 있는 수치 또는 시간 자료형 컬럼이 필요합니다.'}
    if any(re.search(r'array|struct|map|binary', types[c]) for c in columns):
        return {'status': 'needs_context', 'message': '키·정렬·분포 컬럼은 단일 값 자료형이어야 합니다.'}
    if not categorical and not re.search(r'int|long|short|byte|float|double|decimal|numeric', types[value_column]):
        return {'status':'needs_context','message':'수치 히스토그램은 실제 수치 자료형 컬럼이 필요합니다.'}
    keys = ['_k'+str(i) for i in range(len(key_columns))]
    order_aliases = ['_order'] + ['_tie'+str(i) for i in range(len(order)-1)]
    aliases = [*keys, *order_aliases, '_value']
    role_aliases=list(aliases)
    extras=[c['column'] for c in conditions if c['column'] not in columns]
    extras=list(dict.fromkeys(extras))
    columns.extend(extras)
    aliases.extend('_filter'+str(i) for i in range(len(extras)))
    predicate=filter_sql(conditions,dialect='databricks',aliases=dict(zip(columns,aliases)),types=types)
    projection = ', '.join(_quote(c)+' AS '+a for c, a in zip(columns, aliases))
    nulls = ' OR '.join(a+' IS NULL' for a in role_aliases)
    infinite = []
    for c, a in zip(columns, aliases):
        if a in role_aliases and re.search(r'float|double', types[c]):
            nulls += ' OR isnan('+a+')'
            infinite.append('ABS('+a+") = CAST('Infinity' AS DOUBLE)")
    infinite_test=' OR '.join(infinite) or 'FALSE'
    group = ', '.join(keys)
    ordering = ', '.join(a+' DESC' for a in order_aliases)
    valid_rows = (' WHERE NOT ('+nulls+')') if null_policy=='drop_before_selection' else ''
    before=predicate if filter_stage=='before_selection' else 'TRUE'
    after=predicate if filter_stage=='after_selection' else 'TRUE'
    query = f'''WITH original AS (SELECT {projection} FROM {_quoted_table(source)}),
base AS (SELECT * FROM original WHERE {before}),
quality AS (SELECT (SELECT COUNT(*) FROM original) AS n, COALESCE(SUM(CASE WHEN {nulls} THEN 1 ELSE 0 END), 0) AS missing,
COALESCE(SUM(CASE WHEN {infinite_test} THEN 1 ELSE 0 END),0) AS nonfinite FROM base),
ranked AS (SELECT *, DENSE_RANK() OVER (PARTITION BY {group} ORDER BY {ordering}) AS _rank FROM base{valid_rows}),
winners AS (SELECT * FROM ranked WHERE _rank = 1),
latest AS (SELECT * FROM winners WHERE {after}),
key_counts AS (SELECT {group}, COUNT(*) AS n FROM winners GROUP BY {group}),
frequencies AS (SELECT CAST(_value AS STRING) AS v, COUNT(*) AS n FROM latest GROUP BY _value)
SELECT 0 AS __kind, CAST(NULL AS STRING) AS __value, 0 AS __frequency,
n AS __input_rows, missing AS __null_rows,
(SELECT COUNT(*) FROM key_counts WHERE n > 1) AS __tied_keys,
(SELECT COUNT(*) FROM latest) AS __selected_keys,
(SELECT COUNT(*) FROM frequencies) AS __categories, nonfinite AS __nonfinite_rows FROM quality
UNION ALL
SELECT 1, v, n, 0, 0, 0, 0, 0, 0 FROM frequencies
ORDER BY __kind, __value LIMIT 52'''
    if not categorical:
        # Min/max and counts use the same statement/snapshot as latest selection.
        # Empty bins are generated explicitly; the maximum belongs to the last bin.
        prefix = query[:query.index('frequencies AS')]
        query = prefix + f"""bounds AS (SELECT MIN(CAST(_value AS DOUBLE)) AS lo, MAX(CAST(_value AS DOUBLE)) AS hi FROM latest),
ranges AS (SELECT CASE WHEN lo = hi THEN lo - 0.5 ELSE lo END AS lo,
CASE WHEN lo = hi THEN hi + 0.5 ELSE hi END AS hi FROM bounds),
bucketed AS (SELECT CASE WHEN CAST(_value AS DOUBLE) >= hi THEN {bins-1}
ELSE GREATEST(0, LEAST({bins-1}, CAST(FLOOR((CAST(_value AS DOUBLE)-lo)/NULLIF(hi-lo,0)*{bins}) AS INT))) END AS bucket
FROM latest CROSS JOIN ranges),
frequencies AS (SELECT bucket, COUNT(*) AS n FROM bucketed GROUP BY bucket),
bin_ids AS (SELECT EXPLODE(SEQUENCE(0,{bins-1})) AS bucket)
SELECT 0 AS __kind, CAST(NULL AS STRING) AS __value, 0 AS __frequency,
n AS __input_rows, missing AS __null_rows,
(SELECT COUNT(*) FROM key_counts WHERE n > 1) AS __tied_keys,
(SELECT COUNT(*) FROM latest) AS __selected_keys,
{bins} AS __categories, nonfinite AS __nonfinite_rows, (SELECT lo FROM ranges) AS __lower, (SELECT hi FROM ranges) AS __upper FROM quality
UNION ALL
SELECT 1, CAST(b.bucket AS STRING), COALESCE(f.n,0), 0,0,0,0,0,0,
r.lo+(r.hi-r.lo)*b.bucket/{bins}, r.lo+(r.hi-r.lo)*(b.bucket+1)/{bins}
FROM bin_ids b CROSS JOIN ranges r LEFT JOIN frequencies f ON b.bucket=f.bucket
ORDER BY __kind, __lower LIMIT {bins+1}"""
    validate_query(query)
    return {'status': 'planned', 'remote_latest_plan': {'source': source, 'query': query,
        'reason': '키별 최신행과 결측·동률을 동일 조회에서 검증하고 분포 빈도만 가져옵니다.',
        'key_columns': key_columns, 'order_column': order_column, 'value_column': value_column,
        'categorical':categorical, 'bins':bins, 'tie_break_columns':order[1:], 'null_policy':null_policy,
        **lineage(conditions,filter_stage), 'schema_fingerprint': schema.get('schema_fingerprint')}}


def prepare(context, source, key_columns, order_column, value_column, result_dataset_id='', *, categorical=True, bins=20, tie_break_columns=None, null_policy='reject', conditions=None, filter_stage=''):
    outcome = plan(context, source, key_columns, order_column, value_column, categorical=categorical, bins=bins, tie_break_columns=tie_break_columns, null_policy=null_policy, conditions=conditions, filter_stage=filter_stage)
    if outcome['status'] != 'planned' or not result_dataset_id:
        return outcome
    from utils.analysis_latest_filters import lineage
    spec = outcome['remote_latest_plan']
    store = context.datasets
    fields = FIELDS if categorical else FIELDS + ['__lower','__upper']
    limit = 52 if categorical else bins+1
    info = store.metadata.get(result_dataset_id)
    if (info is None or info.query != spec['query'] or _source_key(info.source) != _source_key(source)
            or info.coverage == 'truncated' or list(info.columns) != fields or not 1 <= info.rows <= limit):
        return {'status': 'needs_context', 'message': '정확한 최신행 조회의 완전한 결과를 확인하지 못했습니다.'}
    frame = project_dataset(store, info.id, fields)
    if len(frame) != info.rows:
        raise ValueError('저장된 조회 행 수가 일치하지 않습니다.')
    def integer(value):
        if isinstance(value, bool) or pd.isna(value) or not math.isfinite(float(value)):
            raise ValueError('조회 통계는 유한한 정수여야 합니다.')
        result = int(value)
        if result < 0 or value != result:
            raise ValueError('조회 통계는 음수가 아닌 정수여야 합니다.')
        return result
    numeric = [c for c in FIELDS if c != '__value']
    for column in numeric:
        frame[column] = frame[column].map(integer)
    header = frame[frame.__kind == 0]
    data = frame[frame.__kind == 1]
    if len(header) != 1 or len(header)+len(data) != len(frame):
        raise ValueError('최신행 조회의 검증 행이 올바르지 않습니다.')
    h = header.iloc[0]
    if h.__nonfinite_rows:
        return {'status':'needs_context','error_code':'latest_nonfinite_policy',
            'message':'키·정렬·분포 컬럼에 무한대가 있습니다. 결측 행 제외 정책으로 무한대를 임의로 제외하지 않았습니다. 별도 처리 기준이 필요합니다.'}
    if h.__null_rows and null_policy=='reject':
        return {'status': 'needs_context', 'error_code': 'latest_null_policy',
                'message': '키·정렬·분포 컬럼에 결측값이 있습니다. 결측값 처리 기준이 필요합니다.'}
    if h.__tied_keys:
        return {'status': 'needs_context', 'error_code': 'latest_order_tie',
                'message': '같은 키의 최신 정렬값이 동률입니다. 추가 정렬 기준이 필요합니다.'}
    if categorical and h.__categories > 50:
        return {'status': 'needs_context', 'error_code': 'latest_category_limit',
                'message': '범주가 50개를 넘습니다. 상위 범주 또는 묶는 기준을 알려주세요. 일부 결과를 전체 분포로 표시하지 않았습니다.'}
    if h.__null_rows > h.__input_rows:
        raise ValueError('결측 행 수가 전체 행 수보다 큽니다.')
    if h.__input_rows == h.__null_rows:
        return {'status': 'needs_context', 'message': '조회 대상에 행이 없습니다.'}
    if h.__selected_keys==0 and data.empty:
        return {'status':'needs_context','error_code':'latest_empty','message':'조건을 적용한 최신행이 0개입니다. 원본은 보존했습니다.'}
    if (h.__frequency != 0 or not pd.isna(h.__value) or h.__selected_keys > h.__input_rows-h.__null_rows
            or h.__selected_keys < 1 or h.__categories != len(data)
            or data.__value.isna().any() or data.__value.duplicated().any()
            or (data.__frequency < (1 if categorical else 0)).any() or int(data.__frequency.sum()) != h.__selected_keys
            or (data[FIELDS[3:]] != 0).any().any()):
        raise ValueError('원격 최신행 분포의 합계·검증 통계가 일치하지 않습니다.')
    if not categorical:
        return _histogram_result(context, info, spec, h, data, bins)
    count_column = '__key_count' if value_column != '__key_count' else '__key_count_'
    counts = data[['__value', '__frequency']].rename(columns={'__value': value_column, '__frequency': count_column})
    selection = {'kind': 'remote_latest_per_key_distribution', 'key_columns': key_columns,
        'order_columns': [order_column, *(tie_break_columns or [])], 'descending': True, 'null_policy': null_policy, 'tie_policy': 'reject',
        'input_result_id': info.id, 'input_snapshot': info.snapshot,**lineage(spec.get('conditions'),spec.get('filter_stage'))}
    distribution = store.register(counts, source=info.source, parent_id=info.id, snapshot=info.snapshot,
        coverage='complete', predicate_known=True, grain='aggregate',
        aggregation='remote_count_by_value_after_latest_per_key', row_selection=selection)
    card, summary, chart_spec = render_chart_spec(store, distribution.id, kind='bar',
        x=value_column, y=count_column, aggregation='none', title=value_column+' · 키별 최신행 분포', y_label='고유 키 수')
    context.artifacts[card.id] = card
    return {'status': 'ready', 'remote_latest_plan': spec, 'distribution': asdict(distribution),
        'input_result_id': info.id, 'input_rows': int(h.__input_rows), 'selected_keys': int(h.__selected_keys),'excluded_rows':int(h.__null_rows),
        'count_column': count_column, 'counts': counts.to_dict('records'), 'execution_mode': 'remote_sql',
        'chart_spec': chart_spec, 'render_summary': summary,
        'cards': [{'id': card.id, 'kind': card.kind, 'title': card.title, 'dataset_id': card.dataset_id}]}


def _histogram_result(context, info, spec, header, data, bins):
    """Validate actual edges/counts before rendering pre-binned bars, never bin centers."""
    from utils.analysis_latest_filters import lineage
    import numpy as np
    from utils.analysis_charts import ChartPreview, _apply_unicode_font
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from io import BytesIO
    from uuid import uuid4
    from hashlib import sha256
    import json
    values = data[['__lower','__upper']].to_numpy(dtype=float)
    lo, hi = float(header.__lower), float(header.__upper)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError('유한한 구간 경계가 필요합니다.')
    expected = np.linspace(lo, hi, bins+1)
    if (len(data) != bins or not np.isfinite(values).all() or not np.isfinite(expected).all()
            or not np.all(np.diff(expected) > 0)
            or list(data.__value.astype(str)) != [str(i) for i in range(bins)]
            or not np.allclose(values[:,0], expected[:-1], rtol=1e-12, atol=1e-12)
            or not np.allclose(values[:,1], expected[1:], rtol=1e-12, atol=1e-12)):
        raise ValueError('원격 히스토그램 구간 경계가 유한하고 연속적인 동일 폭 구간이어야 합니다.')
    selection = {'kind':'remote_latest_per_key_distribution','key_columns':spec['key_columns'],
        'order_columns':[spec['order_column'], *spec.get('tie_break_columns',[])],'descending':True,'null_policy':spec.get('null_policy','reject'),'tie_policy':'reject',
        'input_result_id':info.id,'input_snapshot':info.snapshot,**lineage(spec.get('conditions'),spec.get('filter_stage'))}
    counts = data[['__lower','__upper','__frequency']].copy().reset_index(drop=True)
    distribution = context.datasets.register(counts, source=info.source,parent_id=info.id,snapshot=info.snapshot,
        coverage='complete',predicate_known=True,grain='aggregate',aggregation='remote_histogram_after_latest_per_key',row_selection=selection)
    heights = counts.__frequency.to_numpy(dtype='int64')
    fig = Figure(figsize=(6,3.5)); ax = fig.subplots()
    ax.bar(expected[:-1], heights, width=np.diff(expected), align='edge', color='#3278b9', edgecolor='white')
    title = spec['value_column']+' · 키별 최신행 분포'
    ax.set(xlabel=spec['value_column'],ylabel='고유 키 수',title=title)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    render_spec = {'kind':'histogram','bins':bins,'edges':expected.tolist(),'counts':heights.tolist(),
        'aggregation':'prebinned','closed':'left; final bin includes right edge'}
    render_spec['data_digest'] = sha256(json.dumps(render_spec,sort_keys=True).encode()).hexdigest()
    buffer=BytesIO(); FigureCanvasAgg(fig); _apply_unicode_font(fig); fig.tight_layout(); fig.savefig(buffer,format='png',dpi=110)
    card=ChartPreview(str(uuid4()),distribution.id,title,'DB에서 계산한 구간별 정확한 빈도입니다.',
        'histogram',(spec['value_column'],),f"전체 {int(header.__selected_keys):,}개 키 · {bins}개 구간",buffer.getvalue(),render_spec)
    context.artifacts[card.id]=card
    return {'status':'ready','remote_latest_plan':spec,'distribution':asdict(distribution),
        'input_result_id':info.id,'input_rows':int(header.__input_rows),'selected_keys':int(header.__selected_keys),'excluded_rows':int(header.__null_rows),
        'count_column':'__frequency','counts':counts.to_dict('records'),'execution_mode':'remote_sql',
        'chart_spec':render_spec,'render_summary':{'bins':bins,'total':int(heights.sum())},
        'cards':[{'id':card.id,'kind':card.kind,'title':card.title,'dataset_id':card.dataset_id}]}
