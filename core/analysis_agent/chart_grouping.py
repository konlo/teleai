"""Bind histogram color roles to actual schema and the last delivered chart."""
from copy import deepcopy
import re
import sqlglot
from sqlglot import exp
from utils.analysis_provenance import query_conditions


def requested(text):
    return bool(re.search(r'(?<![A-Za-z])legend(?![A-Za-z])|범례|색(?:상)?\s*(?:구분|구별|을|으로|좀)|\b(?:hue|color|colour)\b',text,re.I))


def bind(text, current, context, artifacts, messages):
    if not requested(text) or context is None:return None
    if current.get('kind') and current['kind']!='histogram':return None
    prior=None
    for message in reversed(messages):
        for key in reversed(message.additional_kwargs.get('analysis_artifact_ids',[])):
            try:card=artifacts[key]
            except (KeyError,ValueError,OSError):continue
            if card.dataset_id==context.selected_dataset_id:
                if card.kind!='histogram':return None
                prior=card;break
        if prior:break
    if prior is None:
        return {'question':'색상을 구분할 histogram의 수치 컬럼과 그룹 컬럼을 지정해주세요.'}
    info=context.datasets.metadata[prior.dataset_id]
    if current.get('required_sources') and info.source not in current['required_sources']:
        return {'question':'현재 차트와 요청한 출처가 다릅니다. 대상 차트를 먼저 선택해주세요.'}
    value=prior.columns[0]
    names=set(info.columns)
    for table in context.reference_context:
        if table.get('table')==info.source:names.update(c['name'] for c in table.get('columns',[]))
    scope=deepcopy(current.get('scope',{}))
    # The rendered COUNT lineage is independent evidence of the population.
    # Drop only the renderer's null exclusion before extracting predicates.
    try:
        tree=sqlglot.parse_one(info.query,read=context.sql_dialect)
        terms=[]
        def flatten(node):
            if isinstance(node,exp.Paren):return flatten(node.this)
            if isinstance(node,exp.And):return flatten(node.this)+flatten(node.expression)
            return [node]
        for term in flatten(tree.args['where'].this) if tree.args.get('where') else []:
            if (isinstance(term,exp.Not) and isinstance(term.this,exp.Is)
                    and isinstance(term.this.this,exp.Column) and term.this.this.name==value
                    and isinstance(term.this.expression,exp.Null)):continue
            terms.append(term)
        tree.set('where',exp.Where(this=exp.and_(*terms)) if terms else None)
        conditions=query_conditions(tree)
    except (ValueError,KeyError,TypeError,sqlglot.errors.ParseError):conditions=None
    if conditions is None:
        return {'question':'기존 차트의 분석 조건을 확인할 수 없습니다. 대상 조건을 명시해주세요.'}
    from dataclasses import asdict
    inherited=[asdict(c) for c in conditions]
    explicit=[name for name in names if name!=value and re.search(
        r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',text,re.I)]
    candidates=explicit or [c['column'] for c in inherited if c['op']=='in' and len(c['value'])>1]
    candidates=list(dict.fromkeys(candidates))
    if len(candidates)!=1:
        return {'question':'범례와 색상으로 구분할 그룹 컬럼을 하나 지정해주세요.'}
    category=candidates[0]
    # A style-only follow-up keeps the exact delivered population. A newly
    # stated numeric interval/filter remains the parser's responsibility.
    from core.analysis_agent.numeric_scope import intervals
    if not list(intervals(text)) and not re.search(r'조건|이상|이하|초과|미만|where|filter',text,re.I):
        scope={'conditions':inherited,'any_conditions':[],'measure_conditions':[],
            'ratio':None,'unresolved':[],'columns':list(dict.fromkeys(c['column'] for c in inherited)),
            'sources':[info.source]}
    from core.analysis_agent.eda_contract import histogram_bins
    current.update(chart=True,kind='histogram',calculation=False,operations=[],operation_pending=False,
        required_sources=[info.source],required_columns=[value,category],scope=scope,
        chart_spec_requested=True,chart_group_spec={'source':info.source,'value_column':value,
        'category':category,'bins':histogram_bins(text) or current.get('histogram_bins') or prior.render_spec.get('bins',8)},
        histogram_bins=None,categorical_distribution=False)
    return current['chart_group_spec']
