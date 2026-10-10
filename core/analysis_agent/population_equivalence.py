"""Conservative equivalence of validated predicates, without reading prose.

Only conjunction ordering, BETWEEN expansion and set/singleton IN are normalized.
Different operators, NULL membership and alternative groups remain distinct.
"""
import json


def normalized(items):
    terms=[]
    for item in items:
        c,op,v=item['column'],item['op'],item['value']
        if op=='between':
            terms.extend([{'column':c,'op':'ge','value':v[0]},
                          {'column':c,'op':'le','value':v[1]}])
        elif op in {'in','not_in'} and None not in v:
            values=sorted(set(json.dumps(x,sort_keys=True,ensure_ascii=False) for x in v))
            if len(values)==1:
                terms.append({'column':c,'op':'eq' if op=='in' else 'ne','value':json.loads(values[0])})
            else:terms.append({'column':c,'op':op,'value':[json.loads(x) for x in values]})
        else:terms.append(item)
    return sorted(set(json.dumps(t,sort_keys=True,ensure_ascii=False) for t in terms))


def equivalent(left,right):
    return all(normalized(left.get(k,[]))==normalized(right.get(k,[]))
               for k in ('conditions','any_conditions'))
