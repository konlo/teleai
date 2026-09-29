"""Bind synchronous remote results to durable execution receipts before answering."""
import re


def catalog_read_requested(current):
    sources = current.get('required_sources') or []
    text = current.get('request_text', '')
    return bool(sources) and all(
        len(parts := str(source).replace('`', '').lower().split('.')) == 3
        and parts[-2] == 'information_schema' for source in sources
    ) and bool(re.search(r'목록|리스트|\blist\b', text, re.I)) and not re.search(
        r'뜻|의미|정의|설명|\b(?:meaning|definition|explain)\b', text, re.I)


def verified_receipt(ledger, context, call, observation):
    if ledger is None or context is None or observation.get('status') != 'ready':
        return None
    try:
        receipt = ledger.get(call['id'])
        arguments = call.get('args', {})
        if receipt['status'] != 'completed' or any(
                receipt.get(key) != arguments.get(key) for key in ('source', 'query', 'reason')):
            return None
        saved = receipt.get('result') or {}
        dataset = saved.get('dataset') or {}
        observed = observation.get('dataset') or {}
        info = context.datasets.metadata.get(dataset.get('id'))
        if (saved.get('status') != 'ready' or info is None
                or any(observed.get(key) != dataset.get(key) for key in ('id', 'source', 'rows', 'columns'))
                or info.query != arguments.get('query') or info.source != dataset.get('source')
                or info.rows != dataset.get('rows') or list(info.columns) != dataset.get('columns')):
            return None
        return {'dataset_id': info.id, 'source': info.source, 'query': info.query,
                'rows': info.rows, 'columns': list(info.columns), 'coverage': info.coverage}
    except (KeyError, TypeError, ValueError):
        return None


def remote_queries_ready(current):
    calls = current.get('remote_query_ids') or []
    return bool(calls) and all(key in current.get('remote_query_evidence', {}) for key in calls)


def deferred_execution_claim(content):
    """Defense for unclassified requests; execution receipts remain the primary gate."""
    if not isinstance(content, str):
        return False
    # An explanation that explicitly denies the capability is not a promise.
    sentences = re.split(r'(?<=[.!?。])\s+|\n', content)
    content = ' '.join(sentence for sentence in sentences if not re.search(
        r'(?:기능|자동\s*알림).{0,20}(?:없습니다|지원하지\s*않|제공하지\s*않)|'
        r'(?:안내|알려).{0,12}(?:드릴|줄)\s*수\s*없', sentence))
    # No result-delivery worker exists. Do not accept an assistant's invented
    # future callback as a final answer, including when it never called a tool.
    return bool(re.search(
        r'(?:조회\s*)?결과(?:가|는)?\s*(?:도착|반환|준비|나오|오면).{0,100}'
        r'(?:알려|안내|보여|제공|전달)|'
        r'(?:조회|분석|작업)(?:가|이)?\s*완료되면.{0,80}(?:알려|안내|보여|전달)|'
        r'(?:I(?:\s+will|[’\']ll)\s+(?:notify|let\s+you\s+know)|'
        r'(?:when|once)\s+(?:the\s+)?results?\s+(?:arrive|are\s+ready|come\s+back))',
        content, re.I | re.S))
