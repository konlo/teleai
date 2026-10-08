"""End abandoned work without executing pending calls or changing assets."""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, ToolMessage


def cancellation_update(values, reason):
    current = deepcopy(values.get('recovery') or {})
    current.update(status='cancelled', stop_reason=reason)
    messages = values.get('messages') or []
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    unresolved = {call['id']: call for m in messages if isinstance(m, AIMessage)
                  for call in m.tool_calls if call['id'] not in answered}
    additions = [ToolMessage(tool_call_id=key, name=call['name'], status='error',
        content=json.dumps({'status':'cancelled', 'reason':reason, 'executed':False}))
        for key, call in unresolved.items()]
    text = '미완료 요청을 종료했습니다. 기존 데이터와 결과는 보존했으며 새 요청을 입력할 수 있습니다.'
    additions.append(AIMessage(content=text, additional_kwargs={
        'analysis_status':'cancelled', 'analysis_complete':False,
        'analysis_artifact_ids':[], 'analysis_request_id':current.get('request_id')}))
    return {'recovery':current, 'messages':additions, 'queued_user_request':None}
