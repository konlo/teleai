"""A model-selected semantic replan, preserving budgets and verified assets."""
from copy import deepcopy
import json
from langchain_core.messages import HumanMessage,ToolMessage
from langgraph.types import Command


def valid(current,ledger):
    from core.analysis_agent.recovery import remote_blocked
    return bool(ledger and current.get('intent_origin')=='llm' and current.get('goal',{}).get('mode')=='execute'
        and not current.get('goal_pending') and current.get('status')!='complete'
        and current.get('intent_revisions',0)<1
        and not (current.get('intent_plan') or {}).get('completed')
        and not current.get('artifact_ids') and not current.get('evidence_ids')
        and not remote_blocked(current) and not ledger.uncertain())


def execute(recovery,current,reason,call_id):
    if not valid(current,recovery.approval_ledger) or not reason.strip() or len(reason)>400:
        return {'status':'unavailable','error_code':'intent_replan_unavailable',
            'message':'이 요청의 검증 결과·미확정 조회·재계획 한도는 보존해야 합니다.'}
    from core.analysis_agent.goal_contract import pending_state
    from core.analysis_agent.intent_memory import KEYS
    from core.analysis_agent.task_execution import ACCOUNTING
    base=pending_state(HumanMessage(content=current['request_text'],id=current['request_id']),{},recovery.context)
    for key in ACCOUNTING+('confirmed_analysis','output_references','schema_subject',
            'remote_query_ids','remote_query_evidence','requested_subject','unparsed_user_intent'):
        if key in current:base[key]=deepcopy(current[key])
    base['requested_analysis']={**{k:deepcopy(current[k]) for k in KEYS if k in current},
        'status':'interpreted_not_completed; NOT result evidence'}
    base['intent_revisions']=current.get('intent_revisions',0)+1
    base['intent_replan_observation']={'reason':reason,'previous_goal':deepcopy(current['goal']),
        'authority':'model goal is fallible; reread original HUMAN request, never invent outputs'}
    recovery.diagnostics.emit('intent_replan_requested',request_id=current['request_id'],
        revision=base['intent_revisions'],reason_chars=len(reason))
    return Command(update={'recovery':base,'messages':[ToolMessage(tool_call_id=call_id,
        name='review_analysis_intent',content=json.dumps({'status':'replanning',
            'message':'Original human request will be reinterpreted; budgets, receipts and assets remain intact.'}))]})
