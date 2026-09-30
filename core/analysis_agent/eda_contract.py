"""Explicit EDA options grounded in the request, independent of model output."""
import re


def histogram_bins(text):
    patterns = (
        r'구간\s*(?:수|개수)\s*(?:는|를|을|만|:|=)?\s*(\d+)',
        r'(\d+)\s*(?:개(?:의)?\s*)?(?:구간|bins?\b)',
        r'\bbins?\s*(?:count\s*)?(?:=|:|of)?\s*(\d+)',
    )
    values = {int(m.group(1)) for pattern in patterns for m in re.finditer(pattern, text, re.I)}
    return next(iter(values)) if len(values) == 1 else None


def profile_request_text(text):
    # Remove only an explicit quoted-token conversion clause, not requests
    # for missing counts/ratios alongside a calculation.
    return re.sub(r'''["'`][^"'`\n]{1,64}["'`]\s*(?:문자열)?(?:만|을|를)?\s*
        (?:결측값?|NULL|NaN)(?:으로|로)?\s*(?:처리|변환|간주)(?:해줘|해|하세요|한다)?''',
        '', text, flags=re.I | re.X)


def group_columns(text, columns):
    result = []
    for column in columns:
        name = r'[`"\']?' + re.escape(column) + r'[`"\']?(?![A-Za-z0-9_])'
        if (re.search(r'(?<![A-Za-z0-9_])' + name + r'\s*(?:별|마다|기준으로)', text, re.I)
                or re.search(r'\b(?:by|(?:in|for)\s+each)\s+' + name, text, re.I)):
            result.append(column)
    return result


def group_sort(text, metrics, group, aliases=None):
    """Bind an explicit sort direction to one requested metric or the group key."""
    direction=re.search(r'내림차순|오름차순|\bdescending\b|\bascending\b',text,re.I)
    if not direction:return {'sort':'group_ascending','sort_by':''}
    descending=bool(re.search(r'내림차순|descending',direction[0],re.I))
    preceding=text[max(0,direction.start()-45):direction.start()]
    patterns={'mean':r'평균|\b(?:mean|average|avg)\b','median':r'중앙값|\bmedian\b',
        'count':r'건수|인원수|고객\s*수|승객\s*수|\bcount\b','sum':r'합계|총합|\bsum\b',
        'min':r'최소|최솟|\bmin\b','max':r'최대|최댓|\bmax\b'}
    aliases=aliases or {}
    suffix=r'(?:을|를|의)?\s*(?:기준(?:으로)?|순(?:으로)?)?\s*$'
    terms=[group,*aliases.get(group,())]
    if any(re.search(r'(?<![A-Za-z0-9_])[`"\']?'+re.escape(term)+r'[`"\']?(?:\))?'+suffix,preceding,re.I) for term in terms):
        return {'sort':'group_descending' if descending else 'group_ascending','sort_by':''}
    candidates=[]
    for metric in metrics:
        operation=patterns.get(metric['aggregation'],r'(?!)')
        column=metric.get('value_column','')
        terms=[column,*aliases.get(column,())] if column else []
        column_pattern='(?:'+'|'.join(re.escape(term) for term in terms if term)+')' if column else r'(?!)'
        target=(r'(?:'+operation+r')\s*(?:'+column_pattern+r')?(?:\('+column_pattern+r'\))?'
            +r'|'+column_pattern+r'(?:\))?\s*(?:'+operation+r')')
        if re.search('(?:'+target+')'+suffix,preceding,re.I):candidates.append(metric['name'])
    if len(candidates)==1:
        return {'sort':'metric_descending' if descending else 'metric_ascending','sort_by':candidates[0]}
    return None
