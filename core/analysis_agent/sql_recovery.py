"""Reconcile a legacy SQL rejection using an exact durable tool observation.

No SQL is replayed. A lost connection or missing observation stays uncertain.
"""
import json
from langchain_core.messages import AIMessage, ToolMessage

REJECTED_ERRNOS=frozenset({1064,1149,1054,1146,1305})

def reconcile(runtime):
    if runtime.sql_dialect!='mysql' or runtime.ledger is None:return 0
    try:
        with runtime._exclusive():
            return _reconcile_locked(runtime)
    except RuntimeError as error:
        if str(error)=='현재 대화가 실행 중입니다.':return 0
        raise


def _reconcile_locked(runtime):
    messages=runtime.agent.get_state(runtime.config).values.get('messages',[])
    calls={call['id']:call for m in messages if isinstance(m,AIMessage) for call in m.tool_calls}
    repaired=0
    for message in messages:
        if not isinstance(message,ToolMessage) or message.name!='query_databricks':continue
        try:
            proof=json.loads(message.content)
            call=calls[message.tool_call_id]
            if (proof.get('status')!='unavailable' or proof.get('error_type')!='ProgrammingError'
                    or proof.get('database_errno') not in REJECTED_ERRNOS):continue
            saved=runtime.ledger.get(message.tool_call_id)
            envelope={k:saved[k] for k in ('source','query','reason','connection')}
            if any(call['args'].get(k)!=envelope[k] for k in ('source','query','reason')):continue
            if runtime.ledger.confirm_sql_rejection(message.tool_call_id,envelope,proof['database_errno']):
                runtime.diagnostics.emit('legacy_sql_rejection_reconciled',tool_call_id=message.tool_call_id,
                    database_errno=proof['database_errno'],remote_reexecuted=False)
                repaired+=1
        except (ValueError,KeyError,TypeError):continue
    return repaired
