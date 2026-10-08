"""A display-only follow-up binds to the last verified frequency chart."""
from copy import deepcopy
import re
import sqlglot


def bind(text, current, context, artifacts):
    match=re.fullmatch(r'\s*(?:y\s*축|y[- ]?axis)\s*(?:을|를)?\s*([\d,]+(?:\.\d+)?)\s*(?:으로|로|to)?\s*(?:해\s*줘|해\s*주세요|바꿔\s*줘|설정해\s*줘)?\s*[.!]?\s*',text,re.I)
    if not match or context is None:return
    previous=current.get('confirmed_analysis') or {}
    keys=previous.get('artifact_ids',[])
    if not keys:return
    try:
        card=artifacts[keys[-1]]
        from utils.analysis_charts import validate_frequency_dataset
        from utils.analysis_provenance import count_frequency_columns
        info=context.datasets.metadata[card.dataset_id]
        columns=count_frequency_columns(sqlglot.parse_one(info.query,read='duckdb' if info.parent_id else context.sql_dialect))
        if not columns or card.kind not in {'bar','histogram'} or card.columns!=(columns[0],):return
        if card.render_spec.get('category'):return
        validate_frequency_dataset(context.datasets,card.dataset_id,*columns)
        maximum=float(match[1].replace(',',''))
        if maximum<=0:return
        scope=deepcopy(previous.get('scope') or {})
        if scope.get('sources')!=[info.source] or scope.get('unresolved'):return
    except (KeyError,ValueError,TypeError,OSError,sqlglot.errors.ParseError):return
    current.update(chart=True,kind=card.kind,calculation=False,operations=[],metadata_kind=None,
        required_sources=[info.source],required_columns=[columns[0]],scope=scope,
        categorical_distribution=card.kind=='bar',chart_spec_requested=True,
        chart_axis_spec={'dataset_id':card.dataset_id,'value_column':columns[0],
            'weight_column':columns[1],'categorical':card.kind=='bar','y_max':maximum})


def next_call(current):
    if current.get('chart_axis_spec') and not current.get('artifact_ids'):
        return {'name':'render_histogram','args':dict(current['chart_axis_spec'])}


def bind_goal(current,context,maximum):
    """Bind a model-authored display change to proven frequency data, without prose."""
    previous=current.get('confirmed_analysis') or {}
    ids=previous.get('artifact_ids') or []
    if not ids:raise ValueError('y_max requires a verified previous frequency chart')
    card=context.artifacts[ids[-1]]
    # The previously accepted card's dataset is persisted in its display proof.
    proof=previous.get('chart_axis_spec') or previous.get('chart_display_evidence')
    dataset_id=proof.get('dataset_id') if proof else card.dataset_id
    if card.kind not in {'bar','histogram'} or card.render_spec.get('category'):
        raise ValueError('y_max requires a simple frequency chart')
    if not dataset_id:
        raise ValueError('y_max requires a verified frequency dataset; clarify the chart to change')
    info=context.datasets.metadata[dataset_id]
    from utils.analysis_provenance import count_frequency_columns
    from utils.analysis_charts import validate_frequency_dataset
    columns=count_frequency_columns(sqlglot.parse_one(info.query,read='duckdb' if info.parent_id else context.sql_dialect))
    if not columns or info.source not in current['required_sources']:raise ValueError('y_max scope mismatch')
    validate_frequency_dataset(context.datasets,dataset_id,*columns)
    current['chart_axis_spec']={'dataset_id':dataset_id,'value_column':columns[0],
        'weight_column':columns[1],'categorical':current.get('kind')=='bar','y_max':float(maximum)}
