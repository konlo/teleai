"""Ground source identities in user text, without classifying tasks or conditions."""
from copy import deepcopy
import re


def identifier_mentioned(identifier, text):
    """Literal identifier grounding only; never identifies an action or filter."""
    return bool(re.search(r'(?<![\w$])'+re.escape(identifier)+r'(?![A-Za-z0-9_$])',text,re.I))


def source_mentions(text, context):
    """Return unambiguous catalog identities with literal mention evidence.

    This is identifier validation, not an intent router. The model still decides
    whether a mention requests a preview, explanation, exclusion, or a chart.
    """
    known={t['table'] for t in context.reference_context if t.get('table')}
    known.update(d.source for d in context.datasets.metadata.values())
    found={}
    for source in known:
        parts=source.replace('`','').split('.')
        for n in range(len(parts),0,-1):
            alias='.'.join(parts[-n:])
            matches=[s for s in known if s.replace('`','').casefold().split('.')[-n:]==alias.casefold().split('.')]
            if len(matches)!=1:continue
            match=re.search(r'(?<![\w$])'+re.escape(alias)+r'(?![A-Za-z0-9_$])',text,re.I)
            if match:
                found[source]={'source':source,'quote':match.group()}
                break
    return sorted(found.values(),key=lambda item:item['source'])


def requested_subject(current, messages, context):
    """User-named subject after the last completed analysis, never result proof."""
    saved=deepcopy(current.get('requested_subject') or {})
    confirmed_id=(current.get('confirmed_analysis') or {}).get('request_id')
    human=[m for m in messages if m.type=='human' and isinstance(m.content,str)]
    if confirmed_id:
        index=next((i for i,m in enumerate(human) if m.id==confirmed_id),None)
        if index is not None:human=human[index+1:]
    for message in human[-16:]:
        if message.id==current['request_id']:continue
        mentions=source_mentions(message.content,context)
        if mentions:
            saved={'sources':[m['source'] for m in mentions], 'request_id':message.id,
                   'mentions':mentions,'status':'user_named_not_execution_evidence'}
    return saved
