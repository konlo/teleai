"""Bind a clarification answer to its unfinished request using a semantic role."""
from copy import deepcopy
import json
from types import SimpleNamespace
from langchain_core.messages import HumanMessage, SystemMessage


def carry(previous):
    if previous.get('status')=='complete':return None
    saved=previous.get('pending_clarification')
    if saved:return deepcopy(saved)
    # Older checkpoints have the question but did not retain the executable
    # intent. The role may recover it from actual bounded user turns, never
    # from an assistant's invented explanation.
    if previous.get('status')=='needs_context' and previous.get('goal_question'):
        return {'question':previous['goal_question'],'legacy':True}


def remember(current, goal, question):
    current['pending_clarification']={'request_text':current['request_text'],
        'request_id':current['request_id'],'sources':goal['sources'],
        'goal':deepcopy(goal),'question':question}


def resume(interpreter,current,messages):
    pending=current.get('pending_clarification')
    if not pending or interpreter.selection_model is None:return
    prior=[m for m in messages if m.type=='human' and m.id!=current['request_id']
           and isinstance(m.content,str)][-4:]
    choices={m.id:m.content for m in prior if m.id} if pending.get('legacy') else {}
    if pending.get('request_id'):choices[pending['request_id']]=pending['request_text']
    if not choices:return
    from core.analysis_agent.model_roles import json_role
    from core.analysis_agent.json_contract import invoke_role
    schema={'type':'object','additionalProperties':False,'required':['action','request_id','answer_quote'],
        'properties':{'action':{'type':'string','enum':['answer','new_request']},
            'request_id':{'type':'string','enum':['',*choices]},'answer_quote':{'type':'string'}}}
    model=json_role(interpreter.selection_model,schema,512) or interpreter.selection_model
    req=SimpleNamespace(state={'recovery':current},tools=[],messages=[HumanMessage(content=json.dumps({
        'CURRENT_ANSWER':current['request_text'],'pending':pending,
        'actual_prior_user_requests':[{'id':k,'text':v} for k,v in choices.items()]},ensure_ascii=False))],
        system_message=SystemMessage(content='clarification_answer_v1: Decide whether CURRENT_ANSWER answers the pending question or is a new independent request. '
        'For an answer select the ORIGINAL unfinished analytical request ID (not a previous short clarification answer), '
        'and copy CURRENT_ANSWER exactly into answer_quote. The original task, source, columns and duration remain active. '
        'A new task, explicit change of table or unrelated topic is new_request, request_id="", answer_quote="". '
        'Use actual user turns only; pending intent is NOT execution evidence. Return JSON only.'))
    def override(**changes):
        revised=SimpleNamespace(**{**vars(req),**changes});revised.override=override;return revised
    req.override=override
    def invoke():return invoke_role(model,req,interpreter.budget)
    reply=(interpreter.model_recovery.auxiliary_call(current,invoke) if interpreter.model_recovery else invoke())
    value=json.loads(reply.content)
    from jsonschema import validate
    validate(value,schema)
    if value['action']=='new_request':
        if value['request_id'] or value['answer_quote']:raise ValueError('New request cannot inherit a clarification')
        current.pop('pending_clarification',None);return
    if value['request_id'] not in choices or value['answer_quote']!=current['request_text']:
        raise ValueError('Clarification must bind an actual request and exact current answer')
    original=choices[value['request_id']]
    current['user_answer_text']=current['request_text']
    current['request_text']=original+'\n추가 확인 답변: '+current['user_answer_text']
    if pending.get('sources'):
        current['requested_subject']={'sources':pending['sources'],'status':'pending_clarification_not_execution'}
    current['pending_clarification']={**pending,'legacy':False,
        'request_id':value['request_id'],'request_text':current['request_text'],
        'sources':(current.get('requested_subject') or {}).get('sources') or
                   (current.get('confirmed_analysis') or {}).get('required_sources',[])}
    interpreter.diagnostics.emit('goal_clarification_answer_bound',request_id=current['request_id'],
        original_request_id=value['request_id'],legacy=bool(pending.get('legacy')))
