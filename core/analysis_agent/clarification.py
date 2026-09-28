"""Recognize narrow, schema-grounded answers to a pending analysis question."""
import re

from core.analysis_agent.intent_scope import resolve_request_scope


def continue_analysis(text, previous, context):
    """Return the original request plus an explicit binding, or no continuation.

    This is not an LLM inference or an approval. Only a canonical column name
    (for a numeric measure), or one explicit equality (for a qualified count),
    answers this question. Fresh analytical requests remain separate turns.
    The normal planner revalidates source, types, population and permissions.
    """
    pending = previous.get('pending_clarification') or {}
    if not pending or not context or previous.get('stop_reason') != 'analysis_target_unresolved':
        return None
    if pending.get('selected_dataset_id') != context.selected_dataset_id:
        return None
    columns = {name for info in context.datasets.metadata.values() for name in info.columns}
    reply = text.strip()
    operation = set(previous.get('operations', []))
    if reply.strip('`') in columns and operation and operation <= {'AVG', 'SUM', 'MIN', 'MAX', 'MEDIAN'}:
        return pending['request_text'] + '\n분석 대상 컬럼: ' + reply
    # A count needs the value, not merely a plausible column. Restrict syntax
    # so a separate question cannot accidentally inherit the previous task.
    literal = r'''(?:'(?:[^']|'')*'|"(?:[^"]|"")*"|[+-]?\d+(?:\.\d+)?|true|false)'''
    if operation != {'COUNT'} or not re.fullmatch(r'`?[^\s`=]+`?\s*=\s*' + literal, reply, re.I):
        return None
    scope = resolve_request_scope(reply, context)
    conditions = scope.get('conditions', [])
    if (scope.get('unresolved') or scope.get('any_conditions')
            or len(conditions) != 1 or conditions[0]['op'] != 'eq'
            or conditions[0]['column'] not in columns):
        return None
    return pending['request_text'] + '\n확인한 조건: ' + reply
