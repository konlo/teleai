"""Schema meaning questions stay distinct from exhaustive column listings."""
import re
import json


def type_question(text):
    """Recognize table-schema types without requiring the word 'column'."""
    edge=lambda word:r'(?<![A-Za-z0-9_])(?:'+word+r')(?![A-Za-z0-9_])'
    return bool(re.search(r'자료형|(?:데이터\s*)?타입|'+edge(r'dtypes?|data\s*types?'),text,re.I)
        and re.search(r'컬럼|필드|항목|테이블|스키마|'+edge(r'columns?|fields?|tables?|schema'),text,re.I))


def observed_subject(observation, arguments, context):
    """Bind a conversational table only to an executed schema observation."""
    from core.analysis_catalog import _source_key
    table=(observation.get('table_context') or {}).get('table')
    if (observation.get('status')!='ready' or not table or not context
            or _source_key(table)!=_source_key(arguments.get('table',''))):return None
    known={_source_key(t.get('table','')) for t in context.reference_context}
    known.update(_source_key(d.source) for d in context.datasets.metadata.values())
    if _source_key(table) not in known:return None
    return {'table':table,'authority':observation.get('authority'),
        'schema_fingerprint':(observation.get('table_context') or {}).get('schema_fingerprint'),
        'selection_at_observation':context.selected_dataset_id}


def prior_subject(current,messages,context):
    """Keep the schema topic separate from selected analytical result IDs.

    For old checkpoints reconstruct only from matched tool calls/results, never
    from an assistant's claim. Explicit selection or a completed new analysis
    supersedes the preceding metadata topic.
    """
    if not context:return None
    if (current.get('selection_at_confirmation') is not None
            and current['selection_at_confirmation']!=context.selected_dataset_id):return None
    subject=current.get('schema_subject')
    if subject and subject.get('selection_at_observation')!=context.selected_dataset_id:return None
    if current.get('status')=='complete' and any(current.get(k) for k in
            ('chart','calculation','data_load','row_preview_spec','value_list_requested')):return None
    if subject:return subject
    from langchain_core.messages import AIMessage,ToolMessage
    calls={c['id']:c for m in messages if isinstance(m,AIMessage) for c in m.tool_calls}
    for m in reversed(messages):
        if not isinstance(m,ToolMessage) or m.name!='inspect_table_context':continue
        call=calls.get(m.tool_call_id)
        if not call or call.get('name')!='inspect_table_context':continue
        try:
            data=json.loads(m.content)
            if not isinstance(data,dict):continue
            subject=observed_subject(data,call.get('args') or {},context)
            if subject:return subject
        except (ValueError,TypeError):continue
    return None


def semantic_columns(text):
    return bool(re.search(r'컬럼|필드|항목|\bcolumns?\b|\bfields?\b',text,re.I)
        and re.search(r'의미|나타내|표현|represent|시간|날짜|\b(?:date|time|temporal)\b|'
            r'관련(?:된)?\s*(?:컬럼|필드|항목)|(?:컬럼|필드|항목).{0,20}관련|related.{0,10}(?:columns?|fields?)',text,re.I)
        and not re.search(r'전체\s*(?:목록|리스트)|모든\s*(?:컬럼|필드)|all\s+columns',text,re.I))


def reference(text):
    return bool(re.search(r'여기|현재|이\s*테이블|이\s*데이터|이\s*자료|\bthis\s+(?:table|dataset)\b',text,re.I))


def corrective_goal(text,messages,human_id):
    if not re.search(r'찾아\s*줘야|테이블에서\s*(?:찾아|확인)|대신.{0,15}테이블|wrong\s+table',text,re.I):return text
    from langchain_core.messages import HumanMessage
    previous=next((m for m in reversed(messages) if isinstance(m,HumanMessage) and m.id!=human_id
        and not m.additional_kwargs.get('lc_source') and not m.additional_kwargs.get('approval_decision')),None)
    if previous is not None and semantic_columns(str(previous.content)):
        return str(previous.content)+'\n사용자의 대상 정정: '+text
    return text
