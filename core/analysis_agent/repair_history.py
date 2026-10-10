"""Project superseded failed calls without modifying the durable graph history."""
from hashlib import sha256
import json
from langchain_core.messages import AIMessage,ToolMessage,SystemMessage

def compact_failed_calls(messages):
    failures={}
    for index,message in enumerate(messages):
        if not isinstance(message,ToolMessage):continue
        try:value=json.loads(message.content)
        except (ValueError,TypeError):continue
        if isinstance(value,dict) and value.get('status') in {'error','rejected','unavailable'}:
            failures[message.tool_call_id]=(index,value)
    eligible=[]
    for index,message in enumerate(messages):
        if (isinstance(message,AIMessage) and message.tool_calls
                and all(c['id'] in failures and c['name'] not in
                    {'query_databricks','cancel_database_query','inspect_execution_status'} for c in message.tool_calls)):
            eligible.append((index,message))
    if len(eligible)<2:return messages
    removed=set();brief=[]
    for index,message in eligible[:-1]:
        removed.add(index)
        for call in message.tool_calls:
            result_index,value=failures[call['id']];removed.add(result_index)
            brief.append({'tool':call['name'],'arguments_sha256':sha256(
                json.dumps(call['args'],sort_keys=True,default=str).encode()).hexdigest(),
                'error_code':value.get('error_code'),'error_type':value.get('error_type')})
            # Repair guidance for a superseded failure is already represented
            # by the bounded failure record; keep all other system contracts.
            after=result_index+1
            if (after<len(messages) and isinstance(messages[after],SystemMessage)
                    and str(messages[after].content).startswith('도구 실패를 관찰했습니다.')):
                removed.add(after)
    note=SystemMessage(content='이전 실패 호출의 모델 입력만 압축했습니다. 같은 실패 인자를 반복하지 마세요. '
        '최신 실패의 코드·인자·오류는 그대로 유지합니다. 완료 근거가 아닙니다. '+
        json.dumps(brief,ensure_ascii=False))
    insertion=min(removed)
    return [item for index,message in enumerate(messages)
            for item in ([note] if index==insertion else [])+([] if index in removed else [message])]
