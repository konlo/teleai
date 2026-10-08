"""Exact whole-source scatter coordinates, with bounded transfer and receipts.

Duplicate rows at the same coordinates are represented by their frequency.
This is not sampling or binning. No source frame or selected dataset is replaced.
"""
import re
from dataclasses import asdict
from io import BytesIO
from uuid import uuid4

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LogNorm

from core.analysis_catalog import resolve_table_context
from core.analysis_agent.chart_binding import axis_roles
from core.analysis_agent.row_preview import key
from utils.analysis_charts import ChartPreview, _apply_unicode_font
from utils.analysis_datasets import project_dataset, stored_dataset_digest

WEIGHT = '__telly_coordinate_frequency'


def coordinate_query(source, x, y):
    def quote(name): return '`' + name.replace('`', '``') + '`'
    return (f'SELECT {quote(x)}, {quote(y)}, COUNT(*) AS {quote(WEIGHT)} '
            f'FROM {".".join(quote(p) for p in source.split("."))} '
            f'WHERE {quote(x)} IS NOT NULL AND {quote(y)} IS NOT NULL '
            f'GROUP BY {quote(x)}, {quote(y)}')


def fetch_limit(query, dialect, max_rows, max_coordinate_rows=None):
    """Separate coordinate units from raw-row policy, only for the exact plan.

    This limit is opt-in at adapter construction. Ordinary SQL, raw SELECTs,
    other aggregates and modified coordinate queries keep the generic cap.
    """
    if max_coordinate_rows is None:return max_rows
    from sqlglot import exp, parse_one
    from utils.analysis_provenance import table_identity
    try:
        tree=parse_one(query,read=dialect)
        selections=tree.expressions
        tables=list(tree.find_all(exp.Table))
        if (len(selections)!=3 or len(tables)!=1 or not all(
                isinstance(c,exp.Column) and not c.table for c in selections[:2])):
            return max_rows
        x,y=(c.name for c in selections[:2])
        if x==y or WEIGHT in (x,y):return max_rows
        expected=parse_one(coordinate_query(table_identity(tables[0]),x,y),read=dialect)
        if tree==expected:return max_coordinate_rows
    except (ValueError,TypeError,AttributeError):pass
    return max_rows


def target(context, current):
    scope = current.get('scope') or {}
    text = current.get('request_text', '')
    if (not context or current.get('kind') != 'scatter' or not current.get('chart')
            or current.get('current_result_only') or current.get('calculation')
            or current.get('chart_group_spec') or current.get('requested_join')
            or any(scope.get(k) for k in ('conditions', 'any_conditions', 'unresolved',
                                         'ratio', 'measure_conditions', 'join_edges'))
            or (current.get('intent_origin')!='llm' and not re.search(r'전체|모든|모집단|\b(?:all|whole|entire|population)\b', text, re.I))):
        return None
    sources = current.get('required_sources') or []
    columns = list(dict.fromkeys(current.get('required_columns') or []))
    if len(sources) != 1 or len(columns) != 2:
        return None
    # This controller owns the default exact scatter, not custom styling or
    # regression overlays. Those requests retain their model/tool obligations.
    if current.get('intent_origin')!='llm' and re.search(r'제목|축\s*라벨|추세선|회귀선|로그\s*축|색상|마커|'
                 r'\b(?:title|xlabel|ylabel|legend|colou?r|marker|alpha|logarithmic|regression)\b',text,re.I):
        return None
    axes = current.get('chart_axes') or ({} if current.get('intent_origin')=='llm' else axis_roles(text, columns))
    if not axes:
        if current.get('intent_origin')=='llm':return None
        columns.sort(key=lambda c: text.casefold().find(c.casefold()))
        axes = dict(zip(('x', 'y'), columns))
    if set(axes.values()) != set(columns):
        return None
    return {'source': sources[0], **axes}


def plan_for(context, source, x, y):
    observed = resolve_table_context(context.reference_context, context.datasets, source)
    if observed.get('status') != 'ready':
        raise ValueError('Fresh source schema is required')
    table = observed['table_context']
    types = {c['name']: c.get('dtype', '') for c in table.get('columns', [])}
    if x == y or WEIGHT in (x, y) or not all(c in types and re.search(
            r'int|float|double|decimal|numeric|real|number', str(types[c]), re.I) for c in (x, y)):
        raise ValueError('Two observed numeric axes are required')
    source = table['table']
    sql = coordinate_query(source,x,y)
    from core.analysis_load_plan import source_plan
    source_plan(source, sql, dialect=context.sql_dialect)
    return {'source': source, 'query': sql,
            'reason': '전체 원본의 두 수치 축을 정확한 좌표별 빈도로 집계합니다. 표본 없이 중복 좌표의 전송만 줄이며 원본은 보존합니다.'}


def proof(context, dataset_id, source, x, y):
    plan = plan_for(context, source, x, y)
    info = context.datasets.metadata[dataset_id]
    receipt = context.remote_receipt_reader(dataset_id) if context.remote_receipt_reader else None
    if (not receipt or receipt.get('status') != 'completed' or info.query != plan['query']
            or receipt.get('query') != plan['query']
            or key(receipt.get('source')) != key(plan['source'])
            or key(info.source) != key(plan['source']) or info.coverage != 'complete'
            or info.rows>context.max_scatter_coordinates
            or info.grain != 'aggregate' or list(info.columns) != [x, y, WEIGHT]):
        raise ValueError('A complete exact-coordinate query receipt is required')
    published=(receipt.get('result') or {}).get('dataset') or {}
    if any(published.get(k)!=getattr(info,k) for k in ('id','snapshot','query','rows','coverage','grain')):
        raise ValueError('Coordinate metadata differs from its execution receipt')
    data = project_dataset(context.datasets, dataset_id, [x, y, WEIGHT])
    if len(data) != info.rows or not len(data) or data.duplicated([x, y]).any():
        raise ValueError('Coordinates must be nonempty and unique')
    numeric = data.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Coordinates and frequencies must be finite')
    counts = numeric[WEIGHT]
    if not (counts.gt(0) & counts.eq(np.floor(counts))).all():
        raise ValueError('Coordinate frequencies must be positive integers')
    total = sum(int(v) for v in counts)
    return {'mode': 'exact_coordinate_frequency', 'source': info.source,
            'query': info.query, 'snapshot': info.snapshot, 'x': x, 'y': y,
            'coordinate_count': info.rows, 'drawable_rows': total,
            'weight_column': WEIGHT, 'data_sha256': stored_dataset_digest(context.datasets, info.id)}


def prepare(context, source, x, y, fresh=False, dataset_id=''):
    plan = plan_for(context, source, x, y)
    candidates = ([dataset_id] if dataset_id else [] if fresh else
                  [i.id for i in reversed(list(context.datasets.metadata.values()))
                   if i.query == plan['query'] and i.coverage == 'complete'])
    for identity in candidates:
        try:
            checked = proof(context, identity, source, x, y)
        except ValueError:
            if dataset_id:
                return {'status': 'unavailable', 'error_code': 'whole_scatter_incomplete',
                        'message': '전체 산점도에 필요한 좌표 결과가 잘렸거나 비어 있거나 검증되지 않았습니다. '
                                   '표본을 전체로 표시하지 않습니다. 원본은 보존했습니다.'}
            continue
        from utils.analysis_image_validation import validate_chart_image
        for card_id in reversed(list(context.artifacts)):
            retained=context.artifacts[card_id]
            if (retained.dataset_id==identity and retained.kind=='scatter'
                    and retained.columns==(x,y) and retained.render_spec==checked):
                try:validate_chart_image(retained.image)
                except ValueError:continue
                return {'status':'ready','cards':[asdict(retained)|{'image':None}],
                        'loaded_dataset':identity,'reused_chart':True}
        data = project_dataset(context.datasets, identity, [x, y, WEIGHT])
        fig = Figure(figsize=(8, 5)); FigureCanvasAgg(fig)
        ax = fig.subplots()
        weights = data[WEIGHT].to_numpy(dtype=float)
        dots = ax.scatter(data[x], data[y], c=weights, s=9, alpha=.75,
                          norm=LogNorm(vmin=1, vmax=max(2, weights.max())), cmap='viridis')
        fig.colorbar(dots, ax=ax, label='동일 좌표의 원본 행 수')
        title = f'{x}와 {y} 관계'
        ax.set(xlabel=x, ylabel=y, title=title)
        _apply_unicode_font(fig); fig.tight_layout()
        output = BytesIO(); fig.savefig(output, format='png', dpi=110)
        card = ChartPreview(str(uuid4()), identity, title,
            '전체 유효 좌표를 표시하고 중복 빈도를 색상으로 표현합니다.', 'scatter', (x, y),
            f"전체 유효 원본 {checked['drawable_rows']:,}행 · 고유 좌표 {checked['coordinate_count']:,}개 · "
            '동일 좌표는 빈도로 압축 · 표본/구간 집계 없음 · 두 축의 NULL 제외 · complete',
            output.getvalue(), checked)
        validate_chart_image(card.image)
        context.artifacts[card.id] = card
        return {'status': 'ready', 'cards': [asdict(card) | {'image': None}],
                'loaded_dataset': identity}
    return {'status': 'planned', 'source_scatter_plan': plan}


def valid_card(context, current, card):
    wanted = target(context, current)
    if not wanted or card.kind != 'scatter' or card.columns != (wanted['x'], wanted['y']):
        return False
    try:
        return card.render_spec == proof(context, card.dataset_id, **wanted)
    except (ValueError, KeyError, TypeError):
        return False


def next_call(context, current, remote_available, remote_blocked):
    wanted = target(context, current)
    if not wanted or current.get('artifact_ids') or current.get('source_scatter_error'):
        return None
    # A complete retained raw snapshot is preferable to any new remote read.
    if not current.get('fresh_source_required') and not current.get('source_scatter_plan'):
        if any(i.grain=='raw' and i.coverage=='complete' and i.predicate_known
               and not i.conditions and key(i.source)==key(wanted['source'])
               and {wanted['x'],wanted['y']}.issubset(i.columns)
               for i in context.datasets.metadata.values()):
            return None
    try: plan = plan_for(context, **wanted)
    except ValueError: return None
    if current.get('source_scatter_plan') == plan:
        for receipt in current.get('remote_query_evidence', {}).values():
            info = context.datasets.metadata.get(receipt['dataset_id'])
            if info and info.query == plan['query']:
                return {'name': 'prepare_source_scatter', 'args': {**wanted,
                    'fresh': bool(current.get('fresh_source_required')), 'dataset_id': info.id}}
        return {'name': 'query_databricks', 'args': plan} if remote_available and not remote_blocked else None
    return {'name': 'prepare_source_scatter', 'args': {**wanted,
            'fresh': bool(current.get('fresh_source_required'))}}


def valid_call(context, current, call):
    wanted = target(context, current)
    if not wanted: return False
    args = call.get('args', {})
    try:
        if call['name'] == 'query_databricks':
            return args == current.get('source_scatter_plan') == plan_for(context, **wanted)
        if call['name'] == 'prepare_source_scatter':
            return (all(args.get(k) == v for k, v in wanted.items())
                    and bool(args.get('fresh')) == bool(current.get('fresh_source_required'))
                    and (not args.get('dataset_id') or args['dataset_id'] in
                         {r['dataset_id'] for r in current.get('remote_query_evidence', {}).values()}))
    except (ValueError, KeyError, TypeError): pass
    return False
