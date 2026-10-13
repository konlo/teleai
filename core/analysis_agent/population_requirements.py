"""Validate the independent reader's quoted restrictions, before SQL planning.

The model interprets prose; this module checks its declared evidence. It never
routes natural language with keywords or supplies dataset-specific identifiers.
"""
from datetime import datetime
from core.analysis_agent.relative_period import SPEC

SCHEMA={'type':'object','additionalProperties':False,
    'required':['clauses','question'],'properties':{
        'question':{'type':'string','maxLength':500},
        'clauses':{'type':'array','maxItems':16,'items':{
            'type':'object','additionalProperties':False,
            'required':['quote','kind','columns','status'],
            'properties':{'quote':{'type':'string','minLength':1,'maxLength':384},
                'kind':{'type':'string','enum':['predicate','absolute_time','relative_time','source_scope']},
                'columns':{'type':'array','maxItems':8,'items':{'type':'string'}},
                'status':{'type':'string','enum':['resolved','unresolved']},
                'relative_period':SPEC,
                'scope_basis':{'const':'unfiltered_source'}}}}}}


def model_schema():
    """The production reader must declare relative periods, not date placeholders."""
    from copy import deepcopy
    schema=deepcopy(SCHEMA);item=schema['properties']['clauses']['items']
    relative=deepcopy(item)
    relative['required'].append('relative_period')
    relative['properties']['kind']={'const':'relative_time'}
    relative['properties']['status']={'const':'unresolved'}
    other=deepcopy(item)
    other['properties']['kind']={'type':'string','enum':['predicate','absolute_time']}
    other['properties'].pop('scope_basis')
    other['properties'].pop('relative_period')
    relative['properties'].pop('scope_basis')
    scope=deepcopy(item)
    scope['required'].append('scope_basis')
    scope['properties'].pop('relative_period')
    scope['properties'].update(kind={'const':'source_scope'},columns={'const':[]},
                               status={'const':'resolved'})
    schema['properties']['clauses']['items']={'anyOf':[relative,other,scope]}
    return schema

INSTRUCTIONS='''Read EVERY restriction in CURRENT_USER_REQUEST independently of any
earlier task/filter-change decision. restriction_review.clauses records each
CURRENT row restriction with its exact short quote, kind, actual schema columns,
and resolved/unresolved status. Negated actions, axis settings, output columns
and preview row counts are not row restrictions. No new restrictions => clauses=[].
An explicit request for the WHOLE UNFILTERED SOURCE is a source_scope clause:
kind=source_scope, scope_basis=unfiltered_source, columns=[], status=resolved.
It has no column predicate. Use change=clear and empty conditions/any_conditions
to remove filters; never modify with an invented predicate for an output column.
"All matching rows" WITH conditions retains those conditions and is NOT unfiltered_source.
Output/chart/measure columns are not restriction clauses unless the user actually
constrains their values. A histogram axis alone must never be predicate/unresolved.
Resolved predicates must occur in your resulting population; do not omit clauses
because the earlier selector said filters were unchanged. Explicit removal of a
filter is reflected in removed_columns or change=clear, not a new predicate.
Absolute time needs BOTH explicitly supplied date boundaries. For relative time,
include relative_period={anchor:source_max|today|unspecified,amount:integer,
unit:year|month|day,column:actual schema name or ""}. An answer specifying the
last date in the DATA establishes source_max. Do not ask for literal dates then:
the executor will find the real MAX. Preserve the original duration and task.
Select the observed datetime column from schema when unambiguous; strings with
date semantics will be validated by a database probe, never assumed to be valid.
Mark relative_time unresolved until a probe resolves it; do not invent literal
boundaries. Only genuinely unknown/ambiguous anchors or columns need a question.
For a relative_time clause, do NOT emit a predicate for its date column in
conditions/any_conditions: relative_period is the executable intent, and the
database observation supplies its literal bounds. Never emit between with
textual date placeholders, reversed dates or a year number in place of dates.
question is a concise Korean clarification for unresolved clauses, else "".
Unresolved clauses use columns=[] if not established and do not invent predicates.
Return restriction_review alongside the population delta. Never output SQL.'''


def question(review, request, resolution_status=None):
    """Render an actionable question from the typed missing information."""
    kinds={c['kind'] for c in review['clauses'] if c['status']=='unresolved'}
    if 'relative_time' in kinds:
        specs=[c.get('relative_period',{}) for c in review['clauses'] if c['kind']=='relative_time']
        if specs and all(s.get('anchor')=='source_max' for s in specs):
            if not all(s.get('column') for s in specs):
                return '데이터의 마지막 날짜를 기준으로 적용하겠습니다. 기간에 사용할 날짜 컬럼을 확인해주세요.'
            messages={
                'unavailable':'기준일 조회가 완료되지 않아 기간을 확정하지 못했습니다. 연결 진단을 확인해주세요.',
                'unknown_submission':'이전 기준일 조회의 제출 상태가 불명확하여 중복 조회하지 않습니다. DB 실행 상태를 확인해야 합니다.',
                'empty_time_column':'선택한 날짜 컬럼에 값이 없어 기준일을 계산할 수 없습니다.',
                'mixed_date_format':'날짜 문자열의 형식이 섞여 있습니다. 같은 시간 기준의 날짜 타입으로 변환이 필요합니다.',
                'invalid_date_format':'날짜 컬럼에 해석할 수 없는 값 또는 시간대가 있습니다. 날짜 형식 확인이 필요합니다.',
                'unsupported_period':'요청한 기간이 날짜 계산에서 지원하는 범위를 벗어났습니다.',
                'needs_approval':'현재 수동 승인 설정으로 기준일 조회가 실행되지 않았습니다.',
                'invalid_receipt':'기준일 조회 결과의 출처와 완전성을 검증하지 못했습니다.'}
            if resolution_status in messages:return '마지막 날짜 기준은 확인했습니다. '+messages[resolution_status]
            return '마지막 날짜 기준은 확인했습니다. 날짜 값의 형식 또는 기준일 조회 결과를 확인해야 합니다.'
        if specs and all(s.get('anchor')=='today' for s in specs):
            return '오늘 기준은 확인했습니다. 기간에 사용할 날짜 컬럼의 시간대와 날짜 범위를 확인해야 합니다.'
        return '요청한 기간을 오늘 기준으로 적용할까요, 데이터의 마지막 날짜 기준으로 적용할까요?'
    if 'absolute_time' in kinds:
        return '기간을 적용할 날짜 컬럼과 시작·종료 날짜를 확인해주세요.'
    text=review['question'].strip()
    if text==request.strip() or any(text==c['quote'] for c in review['clauses']):
        return '요청한 조건에 사용할 컬럼과 정확한 조건 값을 확인해주세요.'
    return text


def validate(review, request, result, raw, schema):
    from jsonschema import Draft202012Validator,ValidationError
    try:Draft202012Validator(SCHEMA).validate(review)
    except ValidationError as exc:
        # jsonschema's full exception contains the instance (user quotes).
        raise ValueError('Invalid restriction review contract: '+str(exc.validator)) from None
    names={c['name'] for t in schema for c in t.get('columns',[])}
    predicates=result['conditions']+result['any_conditions']
    if (any(c['kind']=='source_scope' for c in review['clauses'])
            and any(c['kind']!='source_scope' for c in review['clauses'])):
        raise ValueError('Unfiltered source scope cannot include column restrictions')
    removed=set(raw.get('removed_columns',[]))
    for clause in review['clauses']:
        if clause['quote'] not in request:
            raise ValueError('Restriction evidence must quote the CURRENT request exactly')
        if clause['columns'] and (not names or not set(clause['columns']).issubset(names)):
            raise ValueError('Restriction columns must come from the observed schema')
        spec=clause.get('relative_period')
        if clause['kind']=='source_scope':
            if (clause.get('scope_basis')!='unfiltered_source' or clause['columns']
                    or clause['status']!='resolved' or spec):
                raise ValueError('Source scope requires resolved unfiltered_source and no columns')
            if predicates or raw.get('change')!='clear':
                raise ValueError('Unfiltered source scope must clear all population predicates')
            continue
        if 'scope_basis' in clause:
            raise ValueError('Column restrictions cannot declare source scope')
        if spec and spec['column'] and clause['columns']!=[spec['column']]:
            raise ValueError('Relative period column must match the observed restriction column')
        if clause['status']=='unresolved':continue
        if clause['kind']=='relative_time':
            raise ValueError('Relative time needs resolved reference/boundaries; clarify rather than omit the period')
        if not clause['columns']:
            raise ValueError('Resolved restrictions need observed columns')
        for column in clause['columns']:
            bound=[p for p in predicates if p['column']==column]
            if not bound and column not in removed and raw.get('change')!='clear':
                raise ValueError('Quoted restriction has no matching population predicate. '
                    'Output/axis columns are not row restrictions; remove an output-only clause. '
                    'For an explicit whole unfiltered source use source_scope with change=clear; '
                    'for a real value restriction supply its predicate.')
            if clause['kind']=='absolute_time':
                lower=[];upper=[]
                for term in bound:
                    values=term['value'] if term['op']=='between' else [term['value']]
                    if term['op'] not in {'between','ge','gt','le','lt'}:continue
                    for value in values:
                        if not isinstance(value,str) or value not in request:
                            raise ValueError('Absolute time boundaries must be explicit CURRENT date literals')
                        datetime.fromisoformat(value.replace('Z','+00:00'))
                    if term['op'] in {'between','ge','gt'}:lower.append(values[0])
                    if term['op'] in {'between','le','lt'}:upper.append(values[-1])
                if not lower or not upper:
                    raise ValueError('Absolute time restriction needs both boundaries')
    pending=[c for c in review['clauses'] if c['status']=='unresolved']
    missing=any(c['kind']!='relative_time' or not c.get('relative_period',{}).get('column')
        or c.get('relative_period',{}).get('anchor')!='source_max' for c in pending)
    # An established source-MAX intent awaits a tool observation, not a user
    # answer. Do not force the model to fabricate a redundant question.
    if (missing and not review['question'].strip()) or (not pending and review['question'].strip()):
        raise ValueError('Unresolved restrictions require a clarification question; resolved reviews do not')
    return review


def canonical_delta(raw, review, request, schema):
    """Compile an explicit model-authored whole-source scope into a clear delta.

    No prose matching or missing restriction removal: mixed/invalid clauses or
    declared predicates retain the original rejection path.
    """
    if (not isinstance(review,dict) or not review.get('clauses')
            or raw.get('change') not in {'keep','modify'}
            or any(raw.get(k) for k in ('conditions','any_conditions','included_values'))
            or any(not isinstance(c,dict) or c.get('kind')!='source_scope'
                   for c in review['clauses'])):
        return raw
    proposed={**raw,'change':'clear','evidence_quote':request}
    try:validate(review,request,proposed,proposed,schema)
    except (ValueError,TypeError,KeyError):return raw
    return proposed
