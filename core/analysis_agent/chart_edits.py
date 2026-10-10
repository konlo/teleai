"""Compile presentation edits from a verified chart, without reading prose."""
from copy import deepcopy


def resolve(goal, current, context):
    if len(goal['tasks'])!=1 or goal['tasks'][0]['capability']!='chart_adjust':return goal
    options=goal['tasks'][0]['options']
    if 'y_max' in options:return goal
    prior=current.get('confirmed_analysis') or {}
    ids=prior.get('artifact_ids') or []
    if not ids:raise ValueError('presentation change requires a verified previous chart')
    card=context.artifacts[ids[-1]]
    if card.kind not in {'histogram','bar'}:
        raise ValueError('legend/stack/palette edit requires a verified frequency chart')
    if 'bins' in options and card.kind!='histogram':
        raise ValueError('bin adjustment requires a verified numeric histogram')
    sources=prior.get('required_sources') or []
    if goal['sources'] and goal['sources']!=sources:
        raise ValueError('presentation edit cannot replace its verified source')
    if goal['fresh_source_required']:
        raise ValueError('presentation edit cannot refresh or replace its verified population')
    scope=prior.get('scope') or {}
    for key in ('conditions','any_conditions','measure_conditions','ratio'):
        if goal[key] and goal[key]!=scope.get(key):
            raise ValueError('presentation edit cannot change population '+key)
    spec=card.render_spec
    x=spec.get('x') or (card.columns[0] if card.columns else None)
    if not x:raise ValueError('verified chart has no declared distribution subject')
    chart={'kind':card.kind,'axes':{'x':x},
           **{k:spec.get(k,default) for k,default in
              (('legend',False),('stacked',False),('palette','default'))},**options}
    if spec.get('category'):
        chart.update(kind='histogram',category=spec['category'])
    if 'bins' not in options:
        if spec.get('bins') is not None and chart['kind']=='histogram':chart['bins']=spec['bins']
        elif spec.get('bin_edges') and chart['kind']=='histogram':chart['bins']=len(spec['bin_edges'])-1
    return {**deepcopy(goal),'sources':deepcopy(sources),'columns':list(card.columns),
            'current_result_only':prior.get('current_result_only',False),
            **{k:deepcopy(scope.get(k,[] if k!='ratio' else None))
               for k in ('conditions','any_conditions','measure_conditions','ratio')},
            'tasks':[{'capability':'chart','options':chart}]}


def rebin_call(context,current,valid_card):
    """Render from the prior verified frequency asset; do not query raw rows."""
    chart_id=current.get('rebin_chart_id')
    if (not chart_id or current.get('artifact_ids') or not context or not current.get('chart')
            or current.get('kind')!='histogram' or current.get('histogram_bins') is None):return None
    card=context.artifacts[chart_id]
    if card.kind!='histogram' or len(card.columns)!=1:return None
    observed=deepcopy(current);observed['histogram_bins']=card.render_spec.get('bins')
    if not valid_card(card,observed,card.dataset_id):return None
    from sqlglot import parse_one
    from utils.analysis_provenance import count_frequency_columns
    info=context.datasets.metadata[card.dataset_id]
    columns=count_frequency_columns(parse_one(info.query,read='duckdb' if info.parent_id else context.sql_dialect))
    if not columns:return None
    return {'name':'render_histogram','args':{'dataset_id':card.dataset_id,
        'value_column':columns[0],'weight_column':columns[1],
        'bins':current['histogram_bins'],**current.get('chart_presentation_spec',{})}}
