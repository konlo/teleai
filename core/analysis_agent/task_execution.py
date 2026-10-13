"""Checkpointed sequential execution of independently scoped model tasks.

Task transitions retain the single user-turn budget and completed receipts. They
never dispatch a tool: execution still returns to the model/action cycle.
"""
from copy import deepcopy
from langchain_core.messages import HumanMessage
from core.analysis_agent.goal_contract import compile_goal, pending_state

# Only execution-accounting/observation state crosses task boundaries. Intent,
# options, candidates, failed local tools and completion evidence do not.
ACCOUNTING=('request_started_at','model_calls','model_seconds','sent_calls','processed',
    'intent_revisions','intent_replan_observation',
    'model_attempt_budget','model_attempts','action_cycle_history','action_validation_observations',
    'observation_query_ids','observation_query_evidence','temporal_observations',
    'accounted_aux_calls','accounted_aux_seconds','accounted_model_failures',
    'accounted_model_failure_seconds','model_retries','_new_model_failure_seconds')


def start(current,plan,context,remote_available):
    if plan['mode']!='execute':
        goal={'objective':plan['objective'],'mode':plan['mode'],'source_reference':'explicit',
            'sources':[],'columns':[],'conditions':[],'any_conditions':[],'measure_conditions':[],
            'ratio':None,'current_result_only':False,'fresh_source_required':False,
            'question':plan['question'],'tasks':[]}
        return compile_goal(current,goal,context,remote_available)
    from core.analysis_agent.intent_plan import digest
    current=deepcopy(current)
    current['intent_plan']={'version':2,'request_id':current['request_id'],
        'request_text':current['request_text'],'sha256':digest(plan),
        'spec':deepcopy(plan),'active_step_id':None,'completed':[]}
    return activate(current,plan['steps'][0],context,remote_available)


def activate(previous,step,context,remote_available):
    plan=deepcopy(previous['intent_plan'])
    if any(s['id']==step['id'] for s in plan['completed']):raise ValueError('Verified task cannot execute again')
    done={s['id'] for s in plan['completed']}
    if not set(step['depends_on'])<=done:raise ValueError('Task dependencies are not verified')
    # The original user request ID remains the graph message/budget identity.
    human=HumanMessage(content=plan['request_text'],id=previous['request_id'])
    base=pending_state(human,{},context)
    base.update({key:deepcopy(previous[key]) for key in ACCOUNTING if key in previous})
    base['confirmed_analysis']=deepcopy(previous.get('confirmed_analysis') or {})
    base['output_references']=deepcopy(previous.get('output_references',[]))
    base['intent_plan']=plan
    base['intent_plan']['active_step_id']=step['id']
    base['active_task_id']=step['id']
    base['active_task_quote']=step['request_quote']
    # The accepted goal has canonical sources and reconciled final population.
    goal=deepcopy(step['goal'])
    if goal['sources']:
        base['requested_subject']={'sources':goal['sources'],'request_id':base['request_id'],
            'status':'model_interpreted_not_execution_evidence'}
    if step.get('output_reference'):base['requested_output_reference']=deepcopy(step['output_reference'])
    ref_id=step['population']['output_reference_id']
    if ref_id in step['depends_on']:
        receipt=next(s for s in plan['completed'] if s['id']==ref_id)
        reference=receipt.get('output_reference')
        if reference is None:raise ValueError('Dependency has no eligible displayed-result receipt')
        base['requested_output_reference']=deepcopy(reference)
    # Calendar anchors are supplied by existing guarded source observation,
    # never guessed by either model reading and never taken from a sample.
    from core.analysis_agent.relative_period import resolve
    review=deepcopy(step['restriction_review'])
    for clause in review['clauses']:
        if clause['kind']=='relative_time' and clause['status']=='unresolved':
            terms=None if goal['current_result_only'] else resolve(context,base,goal['sources'],clause)
            if terms:
                affected={t['column'] for t in terms}
                if any(t['column'] in affected for t in goal['any_conditions']):
                    raise ValueError('Relative period cannot replace an OR population')
                goal['conditions']=[t for t in goal['conditions'] if t['column'] not in affected]+terms
                clause['status']='resolved'
    if any(c['status']=='unresolved' for c in review['clauses']):
        from core.analysis_agent.population_requirements import question
        from core.analysis_agent.clarification_memory import remember
        question_text=question(review,base['request_text'],base.get('temporal_resolution_status'))
        remember(base,goal,question_text)
        goal={**goal,'mode':'clarify','tasks':[],'question':question_text}
    base=compile_goal(base,goal,context,remote_available)
    from core.analysis_agent.output_memory import bind
    bind(context,base)
    base['population_requirement_review']=review
    # Task evidence is bound to the compiled (including observed dates) goal.
    from core.analysis_agent.intent_plan import digest
    base['active_task_goal_sha256']=digest(base['goal'])
    return base


def advance(recovery,current,text):
    """Record a verified receipt, then select the next *declared* ready task."""
    if not current.get('intent_plan'):return None
    plan=deepcopy(current['intent_plan']);key=plan['active_step_id']
    if key in {s['id'] for s in plan['completed']}:return None
    from core.analysis_agent.completion import active_contracts
    receipts={c.name:deepcopy(current.get(c.evidence_key)) for c in active_contracts(current)}
    from core.analysis_agent.intent_plan import digest
    if digest(current['goal'])!=current['active_task_goal_sha256']:
        raise ValueError('Active task goal changed after admission')
    proof=current.get('table_preview_evidence') or current.get('chart_display_evidence')
    output_reference=None
    if proof:
        output_reference={'reference_id':current['request_id']+':'+key,
            'capability':current['goal']['tasks'][0]['capability'],
            'required_sources':deepcopy(current['required_sources']),
            'required_columns':deepcopy(current['required_columns']),
            'display_evidence':deepcopy(proof),'scope':deepcopy(current['scope'])}
    record={'id':key,'status':'verified','goal':deepcopy(current['goal']),
        'scope':deepcopy(current['scope']),'receipts':receipts,'text':text,
        'evidence_ids':deepcopy(current.get('evidence_ids',[])),
        'artifact_ids':deepcopy(current.get('artifact_ids',[])),
        'output_reference':output_reference}
    plan['completed'].append(record)
    current['intent_plan']=plan
    if output_reference:
        current['output_references']=(current.get('output_references',[])+[output_reference])[-8:]
    done={s['id'] for s in plan['completed']}
    persisted=recovery.context.runtime_services.get('plan',lambda:{})()
    if persisted.get('status')=='ready':
        edges={t['id'].split(':')[-1]:[k.split(':')[-1] for k in t['depends_on']]
            for t in persisted['plan']['tasks']}
        for step in plan['spec']['steps']:
            if step['id'] in edges:step['depends_on']=edges[step['id']]
    current['intent_plan']=plan
    remaining=[s for s in plan['spec']['steps'] if s['id'] not in done]
    recovery.diagnostics.emit('intent_task_verified',request_id=current['request_id'],task_id=key,
        remaining=len(remaining),evidence_ids=record['evidence_ids'],artifact_ids=record['artifact_ids'])
    if not remaining:return None
    ready=next((s for s in remaining if set(s['depends_on'])<=done),None)
    if ready is None:raise ValueError('No ready pending task; dependency contract violated')
    next_state=activate(current,ready,recovery.context,recovery.remote_available)
    recovery.context.current_analysis=deepcopy(next_state)
    return {'recovery':next_state,'jump_to':'model'}


def final_text(current,text):
    plan=current.get('intent_plan') or {}
    if len(plan.get('spec',{}).get('steps',[]))<=1:return text
    completed=plan.get('completed',[])
    if len(completed)!=len(plan['spec']['steps']):raise ValueError('Cannot complete with unfinished tasks')
    current['artifact_ids']=list(dict.fromkeys(i for s in completed for i in s['artifact_ids']))
    current['evidence_ids']=list(dict.fromkeys(i for s in completed for i in s['evidence_ids']))
    return '\n\n'.join(f"작업 {s['id'][1:]} · {s['goal']['objective']}\n\n{s['text']}" for s in completed)
