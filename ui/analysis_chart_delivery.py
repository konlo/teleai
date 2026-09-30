"""Deliver stored chart assets from tool observations and verified final replies."""
import json
from langchain_core.messages import AIMessage, ToolMessage
from utils.analysis_image_validation import validate_chart_image


def chart_references(message):
    if isinstance(message, AIMessage):
        refs = message.additional_kwargs.get('analysis_artifact_ids', [])
        return list(dict.fromkeys(item for item in refs if isinstance(item, str))) if isinstance(refs,list) else []
    if isinstance(message, ToolMessage):
        try:data=json.loads(message.content)
        except (ValueError,TypeError):return []
        refs=data.get('cards', []) if isinstance(data,dict) else []
        if not isinstance(refs,list):return []
        return list(dict.fromkeys(item['id'] for item in refs
            if isinstance(item,dict) and isinstance(item.get('id'),str)))
    return []


def displayable_chart(artifacts, chart_id):
    card=artifacts[chart_id]
    validate_chart_image(card.image)
    return card


def chart_delivery_plan(messages):
    """Only show accepted artifacts when a turn has a verified final reply.

    Legacy history without attachment metadata keeps the tool-card fallback.
    A rejected or failed chart must not leak through an intermediate tool card.
    """
    plan={}
    group=[]
    def flush():
        allowed=next((set(chart_references(m)) for m in reversed(group)
            if isinstance(m,AIMessage) and 'analysis_artifact_ids' in m.additional_kwargs),None)
        for message in group:
            plan[message.id]=[key for key in chart_references(message)
                              if allowed is None or key in allowed]
    for message in messages:
        if message.type=='human':
            flush();group=[]
        group.append(message)
    flush()
    return plan
