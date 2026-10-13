"""Bounded population contract shapes, without quotes, values or model thoughts."""
from hashlib import sha256


def summary(raw, review, request, schema):
    names={c['name'] for t in schema for c in t.get('columns',[])}
    def choice(value, allowed):
        return value if isinstance(value,str) and value in allowed else '<invalid>'
    def column(value):
        return value if isinstance(value,str) and value in names else '<unobserved>'
    def terms(key):
        value=raw.get(key,[]) if isinstance(raw,dict) else []
        return [{'column':column(p.get('column')),'op':choice(p.get('op'),
                 {'eq','ne','gt','ge','lt','le','between','in','not_in','is_null','not_null'})}
                for p in value[:32] if isinstance(p,dict)] if isinstance(value,list) else []
    clauses=review.get('clauses',[]) if isinstance(review,dict) else []
    result=[]
    for c in clauses[:16] if isinstance(clauses,list) else []:
        if not isinstance(c,dict):continue
        quote=c.get('quote')
        result.append({'kind':choice(c.get('kind'),
            {'predicate','absolute_time','relative_time','source_scope'}),
            'status':choice(c.get('status'),{'resolved','unresolved'}),
            'columns':[column(n) for n in c.get('columns',[])[:8]] if isinstance(c.get('columns'),list) else [],
            'quote_in_request':isinstance(quote,str) and bool(quote) and quote in request,
            'quote_sha256':sha256(quote.encode()).hexdigest() if isinstance(quote,str) else None})
    return {'change':choice(raw.get('change') if isinstance(raw,dict) else None,
            {'keep','modify','clear','new_source'}),
            'predicate_shapes':terms('conditions'),'alternative_shapes':terms('any_conditions'),
            'restriction_shapes':result,'has_question':bool(review.get('question')) if isinstance(review,dict) else False}
