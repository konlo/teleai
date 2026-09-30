"""Bind synchronous remote results to durable execution receipts before answering."""
import re


UNVERIFIED_DELIVERY_NOTICE = (
    '확인되지 않은 결과 대기 안내를 차단했습니다. '
    '백그라운드 조회나 자동 완료 알림은 예약되지 않았습니다. '
    '실제 결과 또는 실행 상태를 확인해야 합니다. 기존 데이터는 보존했습니다.')


def table_list_requested(text):
    """Recognize a catalog table inventory, excluding table-column requests."""
    if not isinstance(text, str):
        return False
    if re.search(r'컬럼|필드|항목|\b(?:columns?|fields?|dtypes?|data\s+types?)\b', text, re.I):
        return False
    return bool(
        re.search(r'(?:테이블|\btables?\b).{0,45}(?:목록|리스트|\blist\b|어떤|보여|있(?:어|나요|지))', text, re.I)
        or re.search(r'(?:목록|리스트|\blist\b|어떤).{0,45}(?:테이블|\btables?\b)', text, re.I)
    )


def catalog_read_requested(current):
    sources = current.get('required_sources') or []
    text = current.get('request_text', '')
    return bool(sources) and all(
        len(parts := str(source).replace('`', '').lower().split('.')) == 3
        and parts[-2] == 'information_schema' for source in sources
    ) and table_list_requested(text) and not re.search(
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
    if isinstance(content, list):
        # Providers may return visible text as blocks instead of a plain string.
        # Reasoning/image/tool blocks are not user-visible answer text.
        content = ''.join(block if isinstance(block, str) else block.get('text', '')
            for block in content if isinstance(block, str) or
            (isinstance(block, dict) and block.get('type') in {'text', 'output_text'}
             and isinstance(block.get('text'), str)))
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
