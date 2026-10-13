"""Two model readings of an immutable, task-scoped intent; no prose routing."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
from core.analysis_agent.goal_contract import response_schema, validate_goal, canonical_sources, compile_goal
from core.analysis_agent.json_contract import invoke_role
from core.analysis_agent.model_roles import json_role
from core.analysis_agent.intent_schema import PROTOCOL,OUTLINE_PROTOCOL,schema,outline_schema,review_schema
from core.analysis_agent.population_requirements import validate as validate_review


def intent_role(model,contract,tokens):
    role=json_role(model,contract,tokens)
    from langchain_ollama import ChatOllama
    # Semantic interpretation needs the configured model's reasoning. Only
    # the final JSON is admitted; private reasoning is never persisted here.
    return role.model_copy(update={'reasoning':True}) if isinstance(role,ChatOllama) else role


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def validate(plan, current, data, context, outline=None):
    """Structural and evidence admission, with no natural-language classification."""
    from jsonschema import Draft202012Validator, ValidationError
    try:Draft202012Validator(schema(outline,pending_population=bool(data.get('unparsed_user_intent')))).validate(plan)
    except ValidationError as error:raise ValueError('intent plan schema: '+str(error.validator)) from None
    value=deepcopy(plan);mode=value['mode'];steps=value['steps'];request=current['request_text']
    if not value['objective'].strip():raise ValueError('plan objective is required')
    if mode=='execute' and (not steps or value['question']):raise ValueError('execute needs steps and no question')
    if mode!='execute' and steps:raise ValueError('explain/clarify cannot execute')
    if mode=='clarify' and not value['question'].strip():raise ValueError('clarify needs a question')
    if mode=='explain' and value['question']:raise ValueError('explain cannot ask a question')
    from core.analysis_agent.output_memory import available
    references=available(context,current.get('output_references',[]))
    prior=data.get('requested_previous_analysis') or data.get('verified_previous') or {}
    previous_sources=canonical_sources(prior.get('required_sources',[]),context)
    ids=set()
    for index,step in enumerate(steps):
        if step['id']!='s'+str(index+1):raise ValueError('step IDs must be ordered s1,s2,...')
        if not set(step['depends_on'])<=ids:raise ValueError('dependency must reference an earlier step')
        ids.add(step['id'])
        quote=step['request_quote']
        if not quote.strip() or quote not in request:raise ValueError('task needs its exact CURRENT request quote')
        goal=deepcopy(step['goal'])
        if len(goal['tasks'])!=1:raise ValueError('each independently scoped step needs exactly one output')
        if (goal['source_reference']=='selected_dataset' and not data.get('selected_dataset')
                and goal['tasks'][0]['capability']!='table_list'):
            raise ValueError('selected_dataset is unavailable. Use previous_analysis with its observed physical sources when CURRENT refers to the confirmed previous subject, or explicit for a table named CURRENT. Never invent a selected dataset.')
        decision=step['population']['basis']
        basis={'population_basis':decision,'result_reference_quote':'' if decision=='source_population' else quote,
               'current_result_only':decision in {'displayed_result','selected_original'}}
        cap=goal['tasks'][0]['capability']
        inventory=cap=='table_list'
        sources=canonical_sources(goal['sources'],context)
        from core.analysis_agent.source_references import source_mentions,qualified_mentions
        candidates=SimpleNamespace(reference_context=list(context.reference_context)+[{'table':s} for s in sources],datasets=context.datasets)
        quoted_sources={m['source'] for m in source_mentions(quote,candidates)}
        for mention in qualified_mentions(quote):
            parts=mention['source'].split('.')
            column_reference=(parts[-1] in goal['columns'] and any(
                s.split('.')[-len(parts[:-1]):]==parts[:-1] for s in sources))
            if not column_reference:quoted_sources.update(canonical_sources([mention['source']],context))
        if not inventory and quoted_sources and (not sources or not set(sources)<=quoted_sources):
            raise ValueError('source conflicts with the physical namespace named for this task')
        if goal['source_reference']=='explicit' and not inventory:
            unquoted=not sources or not set(sources)<=quoted_sources
            if unquoted:
                # Preserve the physical source chosen by the model. A mistaken
                # provenance label is not a different table/population. Never
                # fill an empty source or override a newly named namespace.
                named=data.get('literal_source_mentions') or data.get('literal_qualified_mentions')
                observed={t.get('table') for t in context.reference_context}
                observed.update(d.source for d in context.datasets.metadata.values())
                if named or not sources or not set(sources)<=observed:
                    raise ValueError('source must match CURRENT identity or an observed catalog candidate; never replace a named namespace')
                if sources==previous_sources:
                    goal['source_reference']='previous_analysis'
                    step['source_binding_origin']='verified_previous_identity'
                else:step['source_binding_origin']='observed_catalog_candidate'

        goal['current_result_only']=basis['current_result_only']
        reference_id=step['population'].get('output_reference_id','')
        if decision!='displayed_result':reference_id=''
        step['population']['output_reference_id']=reference_id
        if basis['population_basis']=='displayed_result':
            ref=next((r for r in references if r['reference_id']==reference_id),None)
            # An earlier step's output can be bound after its verified completion.
            if ref is None and reference_id not in step['depends_on']:
                raise ValueError('display requires an eligible receipt or dependency step')
            if ref:step['output_reference']=ref
        elif reference_id:raise ValueError('non-displayed population cannot bind a display receipt')
        if basis['population_basis']=='unavailable_result':
            raise ValueError('unavailable display must ask a clarification, not execute')
        if basis['population_basis']=='selected_original' and not data.get('selected_dataset'):
            raise ValueError('selected original is not available')
        delta={'removed_columns':[],'included_values':[],**step['population_delta']}
        delta['evidence_quote']=quote if delta['change'] in {'modify','clear'} else ''
        review=deepcopy(step['restriction_review'])
        pending=[c for c in review['clauses'] if c['status']=='unresolved']
        review['question']='요청한 조건에 사용할 컬럼과 정확한 조건 값을 확인해주세요.' if pending else ''
        from core.analysis_agent.population_requirements import canonical_delta
        delta=canonical_delta(delta,review,quote,data.get('tables',[]))
        validate_review(review,quote,delta,delta,data.get('tables',[]))
        same=sources==previous_sources
        from core.analysis_agent.population_audit import reconcile, verify_filter_grounding
        previous_scope=prior.get('scope') or {} if same else {}
        step['goal']=goal
        from core.analysis_agent.pending_population import recover
        previous_scope=recover(step,data,previous_scope,sources,previous_sources,context)
        resolved=reconcile(delta,previous_scope,quote,same)
        if basis['population_basis']=='unfiltered_source' and (delta['change']!='clear' or any(resolved.values())):
            raise ValueError('unfiltered population must explicitly clear predicates')
        if inventory and any(v and v not in request for v in goal['tasks'][0]['options'].values()):
            raise ValueError('inventory namespaces require literal CURRENT identifiers')
        if inventory and (sources or any(resolved.values()) or goal['columns']):
            raise ValueError('inventory must have its own namespace scope, no table population')
        # Every non-inherited predicate dimension needs a quoted restriction.
        declared={col for clause in review['clauses'] for col in clause['columns']}
        prior_terms=previous_scope.get('conditions',[])+previous_scope.get('any_conditions',[])
        for term in resolved['conditions']+resolved['any_conditions']:
            if term not in prior_terms and term['column'] not in declared:
                raise ValueError('new predicate lacks quoted restriction evidence')
        temporal={c.get('relative_period',{}).get('column') for c in review['clauses'] if c['kind']=='relative_time'}
        for key in resolved:
            if any(c['column'] in temporal for c in resolved[key]):
                raise ValueError('relative dates must come from observation, not model predicates')
        goal=validate_goal({'objective':quote,'mode':'execute','question':'',
            'measure_conditions':[],'ratio':None,**goal,**resolved})
        goal['sources']=sources
        step.update(goal=goal,population_delta=delta,population_decision=basis,restriction_review=review)
    return value


def interpret(interpreter,current,messages,remote_available):
    from core.analysis_agent.clarification_memory import resume
    resume(interpreter,current,messages)
    data=interpreter.payload(current,messages)
    from core.analysis_agent.custom_input_context import names
    wanted=[m['source'] for m in data.get('literal_source_mentions',[])] or (data.get('verified_previous') or {}).get('required_sources',[])
    observed=names(interpreter.context,data,wanted)
    if observed and len(observed)<=128:data['observed_input_column_names']=observed
    # Enrich only matching source schemas; metadata remains outside static instructions.
    for table in data['tables']:
        info=next((t for t in interpreter.context.reference_context if t.get('table')==table['table']),None)
        if info:
            table['columns']=[{k:c[k] for k in ('name','dtype','database_dtype') if k in c} for c in (info.get('columns') or [])[:128]]
    outline_role=intent_role(interpreter.model,outline_schema(),1024)
    outline_request=SimpleNamespace(state={'recovery':current},system_message=SystemMessage(content=OUTLINE_PROTOCOL),
        messages=[HumanMessage(content=json.dumps({'CURRENT':data},ensure_ascii=False,default=str))],tools=[])
    def override_outline(**changes):
        result=SimpleNamespace(**{**vars(outline_request),**changes});result.override=override_outline;return result
    outline_request.override=override_outline
    def read_outline():return invoke_role(outline_role,outline_request,interpreter.budget)
    from jsonschema import Draft202012Validator,ValidationError
    for outline_attempt in range(2):
        reply=interpreter.model_recovery.auxiliary_call(current,read_outline) if interpreter.model_recovery else read_outline()
        try:
            outline=json.loads(reply.content)
            interpreter.diagnostics.emit('intent_outline_observed',request_id=current['request_id'],
                mode=outline.get('mode') if isinstance(outline,dict) else None,
                step_shapes=[{'grouped':s.get('grouped'),'scope_change':s.get('scope_change')}
                    for s in outline.get('steps',[]) if isinstance(s,dict)] if isinstance(outline,dict) else [])
            try:Draft202012Validator(outline_schema()).validate(outline)
            except ValidationError as error:
                raise ValueError('Invalid intent outline: '+str(error.validator)+' at '+ '/'.join(map(str,error.absolute_path))) from None
            if outline['mode']=='execute' and not outline['steps']:raise ValueError('Executable outline needs outputs')
            if outline['mode']!='execute' and outline['steps']:raise ValueError('Non-executable outline cannot have outputs')
            if outline['mode']!='clarify':outline['question']=''
            ids=set()
            for index,choice in enumerate(outline['steps']):
                if choice['id']!='s'+str(index+1) or not set(choice['depends_on'])<=ids:
                    raise ValueError('Outline dependency/ID invalid')
                if not choice['request_quote'].strip() or choice['request_quote'] not in current['request_text']:
                    raise ValueError('Outline output must quote CURRENT')
                ids.add(choice['id'])
            break
        except (ValueError,TypeError,KeyError) as error:
            interpreter.diagnostics.emit('intent_outline_rejected',request_id=current['request_id'],
                attempt=outline_attempt+1,detail=str(error)[:300])
            if outline_attempt==1:raise
            outline_request.messages=[HumanMessage(content=json.dumps({'CURRENT':data,
                'validation_error':str(error)[:400]},ensure_ascii=False,default=str))]
    interpreter.diagnostics.emit('intent_outline_selected',request_id=current['request_id'],
        mode=outline['mode'],capabilities=[s['capability'] for s in outline['steps']])
    def review_role():
        return intent_role(interpreter.model,review_schema(outline,
            allow_selected=bool(data.get('selected_dataset')),
            pending_population=bool(data.get('unparsed_user_intent'))),4096)
    role=review_role()
    payload={'CURRENT':{**data,'task_options':{k:v for k,v in data['task_options'].items()
        if k in {s['capability'] for s in outline['steps']}}},'phase':'REVIEW','selected_outputs':outline}
    raw=None
    for attempt in range(3):
        req=SimpleNamespace(state={'recovery':current},system_message=SystemMessage(content=PROTOCOL),
            messages=[HumanMessage(content=json.dumps(payload,ensure_ascii=False,separators=(',',':'),default=str))],tools=[])
        def override(**changes):
            result=SimpleNamespace(**{**vars(req),**changes});result.override=override;return result
        req.override=override
        def call():return invoke_role(role,req,interpreter.budget)
        reply=(interpreter.model_recovery.auxiliary_call(current,call) if interpreter.model_recovery else call())
        try:
            raw=json.loads(reply.content)
            if isinstance(raw,dict) and raw.get('mode')=='revise_outline':
                Draft202012Validator(review_schema(outline,
                    allow_selected=bool(data.get('selected_dataset')),
                    pending_population=bool(data.get('unparsed_user_intent')))).validate(raw)
                revised=raw['outline'];seen=set()
                for index,choice in enumerate(revised['steps']):
                    if (choice['id']!='s'+str(index+1) or not set(choice['depends_on'])<=seen
                        or not choice['request_quote'].strip() or choice['request_quote'] not in current['request_text']):
                        raise ValueError('Revised outline must retain ordered dependencies and exact CURRENT output quotes')
                    seen.add(choice['id'])
                if revised==outline or attempt==2:raise ValueError('Outline revision did not converge within review budget')
                outline=revised;role=review_role()
                payload.update(selected_outputs=outline,phase='REVIEW',
                    CURRENT={**data,'task_options':{k:v for k,v in data['task_options'].items()
                        if k in {s['capability'] for s in outline['steps']}}})
                interpreter.diagnostics.emit('intent_outline_revised',request_id=current['request_id'],
                    capabilities=[s['capability'] for s in outline['steps']],attempt=attempt+1)
                continue
            interpreter.diagnostics.emit('intent_plan_observed',request_id=current['request_id'],
                phase=payload['phase'],mode=raw.get('mode') if isinstance(raw,dict) else None,
                restriction_shapes=[[{k:c.get(k) for k in ('kind','columns','status','scope_basis','relative_period')} for c in s.get('restriction_review',{}).get('clauses',[]) if isinstance(c,dict)] for s in raw.get('steps',[]) if isinstance(s,dict)] if isinstance(raw,dict) else [],
                population_shapes=[{k:s.get('population',{}).get(k) for k in ('basis','changes_filters')} for s in raw.get('steps',[]) if isinstance(s,dict)] if isinstance(raw,dict) else [],
                step_count=len(raw.get('steps',[])) if isinstance(raw,dict) and isinstance(raw.get('steps'),list) else None,
                question_chars=len(raw.get('question','')) if isinstance(raw,dict) and isinstance(raw.get('question'),str) else None)
            plan=validate(raw,current,data,interpreter.context,outline)
        except (ValueError,TypeError,KeyError) as error:
            interpreter.diagnostics.emit('intent_plan_rejected',request_id=current['request_id'],
                phase=payload['phase'],attempt=attempt+1,detail=str(error)[:240])
            if attempt==2:return interpreter.blocked(current,error,'intent_plan_validation')
            payload.update(validation_error=str(error)[:400],phase='REVIEW',proposed_plan=raw)
            continue
        from core.analysis_agent.task_execution import start
        interpreter.diagnostics.emit('intent_plan_accepted',request_id=current['request_id'],
            steps=len(plan['steps']),interpretation_calls=outline_attempt+attempt+2,
            plan_sha256=digest(plan),scope_mode='per_task',reviewed=True)
        if interpreter.model_recovery:interpreter.model_recovery.ledger.sync(current)
        return start(current,plan,interpreter.context,remote_available)
    raise AssertionError('bounded plan loop did not return')
