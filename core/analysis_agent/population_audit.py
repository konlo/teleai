"""Independent LLM reading of population changes, without the proposed filters."""
import json
from types import SimpleNamespace
from langchain_core.messages import HumanMessage, SystemMessage
from core.analysis_agent.json_contract import invoke_role
from core.analysis_agent.goal_contract import response_schema, predicates

CAPABILITIES={'chart','row_preview','value_list','profile','calculation','row_count',
              'statistics','outliers','time_series','group_summary','pivot','compare'}


def applies(goal, current=None):
    if not any(t['capability'] in CAPABILITIES for t in goal['tasks']):return False
    if any(t['capability']=='row_preview' for t in goal['tasks']):
        from core.analysis_agent.intent_memory import prior_intent
        prior=prior_intent(current or {},goal['current_result_only'])
        # Source-reference labels describe how a name was resolved, not
        # permission to erase its population. The canonical identity owns scope.
        previous=prior.get('scope',{}) if goal['sources']==prior.get('required_sources') else {}
        # A new unfiltered preview has an output limit, no inherited population
        # to reconcile. The normal goal review still verifies its obligations.
        return any(goal[k] or previous.get(k) for k in ('conditions','any_conditions'))
    return True


def null_declared(population):
    return any(c.get('op') in {'is_null','not_null'} or c.get('value') is None
        or (isinstance(c.get('value'),list) and None in c['value'])
        for key in ('conditions','any_conditions') for c in population.get(key,[]))


def model_for(model, scope=None, request=None, goal=None):
    from core.analysis_agent.model_roles import json_role
    properties=response_schema()['properties']
    keys=['change','evidence_quote','included_values','conditions','any_conditions','removed_columns']
    schema={
        'type':'object','additionalProperties':False,'required':keys,
        'properties':{'change':{'type':'string','enum':['keep','modify','clear','new_source']},
            'evidence_quote':{'type':'string'},
            'included_values':{'type':'array','maxItems':32,'items':{'type':'object',
                'additionalProperties':False,'required':['column','values','quote'],
                'properties':{'column':{'type':'string'},'values':{'type':'array','minItems':1,
                    'items':{'type':['string','number','boolean']}},'quote':{'type':'string'}}}},
            **{k:properties[k] for k in ('conditions','any_conditions')},
            'removed_columns':{'type':'array','items':{'type':'string'},'maxItems':32}}}
    if goal is not None and not (null_declared(goal) or null_declared(scope or {})):
        # The audit cannot invent a new NULL role absent from both the
        # independently reviewed goal and the previously confirmed population.
        for key in ('conditions','any_conditions'):
            variants=schema['properties'][key]['items']['anyOf']
            schema['properties'][key]['items']['anyOf']=[v for v in variants
                if not set(v['properties']['op'].get('enum',[])) & {'is_null','not_null'}]
            for variant in schema['properties'][key]['items']['anyOf']:
                value=variant['properties']['value']
                if isinstance(value.get('type'),list):value['type']=[t for t in value['type'] if t!='null']
                if value.get('type')=='array' and isinstance(value.get('items',{}).get('type'),list):
                    value['items']['type']=[t for t in value['items']['type'] if t!='null']
    membership=[]
    if scope is not None:
        membership=[c['column'] for c in scope.get('conditions',[]) if c['op'] in {'eq','in'}]
        if membership:
            schema['properties']['included_values']['items']['properties']['column']={
                'type':'string','enum':list(dict.fromkeys(membership))}
        else:schema['properties']['included_values']={'const':[]}
    if request is not None:
        schema['properties']['evidence_quote']={'type':'string','enum':['',request]}
        if membership:
            schema['properties']['included_values']['items']['properties']['quote']={'const':request}
    return json_role(model,schema,1024)


def reconcile(result, scope, request, same_source):
    """Validate a model-authored scope delta before accepting full predicates."""
    if set(result)-{'removed_columns','included_values'}!={'conditions','any_conditions','change','evidence_quote'}:
        raise ValueError('invalid population audit fields')
    removed=result.get('removed_columns',[])
    if not isinstance(removed,list) or any(not isinstance(c,str) or not c for c in removed):
        raise ValueError('invalid removed population columns')
    for key in ('conditions','any_conditions'):predicates(result[key])
    # Opposing overlapping numeric bounds in an OR admit essentially every
    # value. Never silently execute that common range-interpretation mistake.
    for lower in result['any_conditions']:
        for upper in result['any_conditions']:
            if (lower['column']==upper['column'] and lower['op'] in {'gt','ge'}
                    and upper['op'] in {'lt','le'}
                    and type(lower['value']) in (int,float) and type(upper['value']) in (int,float)
                    and lower['value']<=upper['value']):
                raise ValueError('Overlapping numeric bounds in any_conditions (OR) remove the range. '
                                 'Re-read CURRENT request; an inclusive lower AND upper range belongs '
                                 'in conditions with between [lower,upper], never any_conditions.')
    change=result['change'];quote=result['evidence_quote']
    if change not in {'keep','modify','clear','new_source'} or not isinstance(quote,str):
        raise ValueError('invalid scope delta')
    from copy import deepcopy
    result=deepcopy(result)
    additions=result.get('included_values',[])
    if not isinstance(additions,list):raise ValueError('invalid included category values')
    combined={}
    for addition in additions:
        if (not isinstance(addition,dict) or set(addition)!={'column','values','quote'}
                or not isinstance(addition['column'],str) or not isinstance(addition['values'],list)
                or not addition['values'] or any(type(v) not in (str,int,float,bool) for v in addition['values'])
                or not isinstance(addition['quote'],str) or not addition['quote'].strip()
                or addition['quote'] not in request or result['any_conditions'] or scope.get('any_conditions')):
            raise ValueError('included values need an exact CURRENT quote and an unambiguous AND population')
        column=addition['column']
        prior=[c for c in scope.get('conditions',[]) if c['column']==column]
        if len(prior)!=1 or prior[0]['op'] not in {'eq','in'}:
            raise ValueError('included values need a previous equality or membership for '+column)
        values=combined.get(column,([prior[0]['value']] if prior[0]['op']=='eq' else prior[0]['value']))+addition['values']
        unique=[]
        for value in values:
            if value not in unique:unique.append(value)
        combined[column]=unique
        result['conditions']=[c for c in result['conditions'] if c['column']!=column]+[
            {'column':column,'op':'in','value':unique}]
        result['change']='modify'
        result['evidence_quote']=addition['quote']
    from core.analysis_agent.population_equivalence import equivalent
    if not removed and equivalent(result,scope):
        # An imperfect delta label cannot invalidate an unchanged population.
        # Evidence is required for actual changes, not mere chart requests.
        return {k:scope.get(k,[]) for k in ('conditions','any_conditions')}
    if same_source and change=='new_source':
        raise ValueError('same source cannot reset population as a new source')
    if change in {'modify','clear'} and (not quote.strip() or quote not in request):
        raise ValueError('population change requires an exact quote from CURRENT request')
    if change=='keep':
        return {k:scope.get(k,[]) for k in ('conditions','any_conditions')}
    if change=='clear' and any(result[k] for k in ('conditions','any_conditions')):
        raise ValueError('clear population must have empty conditions')
    if change=='modify' and same_source:
        # A delta replaces only its declared dimensions. Omission is never
        # permission to discard another dimension of a large population.
        affected={c['column'] for key in ('conditions','any_conditions') for c in result[key]}|set(removed)
        prior_or=scope.get('any_conditions',[])
        if prior_or and not result['any_conditions'] and affected.intersection(c['column'] for c in prior_or):
            if not set(c['column'] for c in prior_or).issubset(removed):
                raise ValueError('cannot partially rewrite a previous OR population')
        keep_or=prior_or if not result['any_conditions'] and not set(removed).intersection(c['column'] for c in prior_or) else []
        return {'conditions':[c for c in scope.get('conditions',[]) if c['column'] not in affected]+result['conditions'],
                'any_conditions':result['any_conditions'] or keep_or}
    return {k:result[k] for k in ('conditions','any_conditions')}


def audit(interpreter,current,data,goal,selection=None):
    """Return full resulting filters; never accept an existing plan as evidence."""
    from core.analysis_agent.intent_memory import prior_intent
    prior=prior_intent(current,goal['current_result_only'])
    same_source=goal['sources']==prior.get('required_sources')
    scope=prior.get('scope',{}) if same_source else {}
    if (same_source and selection and selection.get('changes_filters') is False
            and selection.get('population_basis')=='source_population'):
        # A separate language role owns permission to change restrictions. A
        # later predicate generator cannot expand an unchanged-population task.
        # Legacy contracts without this decision still take the normal audit.
        interpreter.diagnostics.emit('goal_population_contract_applied',request_id=current['request_id'],
            basis='unchanged_filters',same_source=True,
            changed=any(goal[k]!=scope.get(k,[]) for k in ('conditions','any_conditions')))
        return {**goal,**{k:scope.get(k,[]) for k in ('conditions','any_conditions')}}
    population_model=model_for(interpreter.population_model,scope,current['request_text'],goal) or interpreter.population_model
    request=HumanMessage(content=json.dumps({
        'CURRENT_USER_REQUEST':current['request_text'],
        'previous_requested_population':{k:scope.get(k,[]) for k in ('conditions','any_conditions')},
        'requested_previous_subject':data.get('requested_previous_subject'),
        'previous_population_authority':prior.get('status'),
        'schema':data.get('tables',[]),
    },ensure_ascii=False))
    system=SystemMessage(content='''population_audit_v1: Independently interpret EVERY restriction in CURRENT_USER_REQUEST. Return JSON only.
conditions are AND; any_conditions are OR. Numeric inclusive ranges use between [lower,upper] in conditions, never OR.
change=keep only when no row restrictions change. change=modify for new or replaced restrictions even when there are no previous filters.
Return ONLY the added/replaced conditions. Unmentioned previous dimensions are automatically kept.
Multiple alternative category VALUES for ONE column use IN with ALL those values. Never keep only the first value.
If the current request says to include another value alongside a previous category, put ONLY the additional value in included_values [{column:the previous filter column, values:[new value], quote:CURRENT_USER_REQUEST}].
For an explicitly named complete category set use conditions column IN [all requested values], included_values=[].
change=clear ONLY if CURRENT explicitly removes ALL filters. change=new_source only for an actually different source, not naming the same table.
removed_columns lists explicitly removed dimensions, otherwise []. Styling, bins, axes and row preview limit are NOT predicates.
For modify/clear evidence_quote copies the entire CURRENT_USER_REQUEST exactly. Otherwise evidence_quote="".
Do not invent NULL predicates or restrictions. Keep explicitly requested or previously confirmed NULL predicates only. Supplied schema maps business terms to real column names. Return no explanation.''')
    req=SimpleNamespace(state={'recovery':current},system_message=system,messages=[request],tools=[])
    def override(**changes):
        revised=SimpleNamespace(**{**vars(req),**changes});revised.override=override;return revised
    req.override=override
    def invoke():return invoke_role(population_model,req,interpreter.budget)
    for attempt in range(2):
        response=(interpreter.model_recovery.auxiliary_call(current,invoke)
                  if interpreter.model_recovery else invoke())
        raw=json.loads(response.content)
        try:
            if null_declared(raw) and not (null_declared(goal) or null_declared(scope)):
                raise ValueError('Unrequested NULL predicate: absent from reviewed request goal and confirmed prior population. Re-read the original request; do not invent a missing-value filter for an ordinary aggregate.')
            result=reconcile(raw,scope,current['request_text'],same_source)
            break
        except ValueError as exc:
            interpreter.diagnostics.emit('goal_population_delta_rejected',request_id=current['request_id'],
                attempt=attempt+1,detail=str(exc),change=raw.get('change'),
                has_current_quote=isinstance(raw.get('evidence_quote'),str) and bool(raw['evidence_quote'].strip())
                    and raw['evidence_quote'] in current['request_text'])
            if attempt:
                from core.analysis_agent.population_equivalence import equivalent
                if same_source and equivalent(goal,scope):
                    # Neither the reviewed goal nor a grounded audit authorized
                    # a new population. Retain its verified restrictions rather
                    # than retry unrelated goal generation or silently widen it.
                    result={k:scope.get(k,[]) for k in ('conditions','any_conditions')}
                    interpreter.diagnostics.emit('goal_population_audit_conflict',request_id=current['request_id'],
                        resolution='retain_unchanged_reviewed_population')
                    break
                raise
            req.messages.append(HumanMessage(content='Repair ONLY the population delta: '+str(exc)+
                '. Re-read EVERY current restriction. Preserve unmentioned prior dimensions. '
                'For a modified population evidence_quote copies the full CURRENT_USER_REQUEST exactly.'))
    grounded={c['column'] for key in result for c in goal[key]}
    grounded.update(c['column'] for key in result for c in scope.get(key,[]))
    from core.analysis_agent.source_references import identifier_mentioned
    for key in result:
        for condition in result[key]:
            column=condition['column']
            if column not in grounded and not identifier_mentioned(column,current['request_text']):
                interpreter.diagnostics.emit('goal_population_audit_conflict',request_id=current['request_id'],
                    resolution='ungrounded_new_filter_column',column=column)
                raise ValueError('Population audit introduced ungrounded filter column '+column+
                    '. Output row limits are not column filters. Rebuild conditions from the original request.')
    interpreter.diagnostics.emit('goal_population_audited',request_id=current['request_id'],
        changed=any(result[k]!=goal[k] for k in result),condition_count=len(result['conditions']),
        alternative_count=len(result['any_conditions']))
    return {**goal,**result}
