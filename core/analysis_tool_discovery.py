"""Bounded discovery over the actual registered tools, without executing them."""
from hashlib import sha256
import json
import re


def discover_tools(definitions, skills, query, *, limit=3, allowed_names=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 400:
        raise ValueError('검색어는 1~400자여야 합니다.')
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError('도구 정의는 한 번에 최대 3개만 읽을 수 있습니다.')
    tokens = set(re.findall(r'[a-z0-9_]+|[가-힣]+', query.casefold()))
    ranked = []
    for definition in definitions:
        if definition.name == 'search_analysis_tools':
            continue
        schema = definition.schema()
        name = definition.name
        if name == 'propose_databricks_query' and allowed_names is not None:
            name = 'query_databricks'
            schema['name'] = name
        if allowed_names is not None and name not in allowed_names:
            continue
        searchable = json.dumps(schema, ensure_ascii=False).casefold()
        score = sum(8 if token == name else 3 if token in name else 1
                    for token in tokens if token in searchable)
        if score:
            ranked.append((score, name, schema))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    matches = []
    for _, name, schema in ranked[:limit]:
        encoded = json.dumps(schema, sort_keys=True, ensure_ascii=False)
        matches.append({**schema, 'version':sha256(encoded.encode()).hexdigest(),
            'permission':('exact_sql_approval_required' if name in
                          {'query_databricks', 'propose_databricks_query'} else 'local_tool'),
            'related_skills':[{'name':s['name'], 'description':s['description']}
                for s in skills if any(token in (s['name']+' '+s['description']).casefold()
                                       for token in tokens)][:3]})
    result = {'status':'ready', 'matches':matches,
              'total_matches':len(ranked), 'more_available':len(ranked)>limit,
              'message':'등록 도구의 실제 계약입니다. 검색은 실행·승인·완료의 근거가 아닙니다.'}
    # Do not silently truncate a JSON schema and make it look executable.
    while len(json.dumps(result, ensure_ascii=False)) > 18000 and matches:
        matches.pop()
        result['more_available'] = True
    return result
