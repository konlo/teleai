"""Provider predicates: an operation determines the shape of its literal."""
from copy import deepcopy


def predicate_shapes(candidate):
    """Debug structure without recording predicate values or unbounded output."""
    result=[]
    if not isinstance(candidate,dict):return result
    for key in ('conditions','any_conditions','measure_conditions'):
        clauses=candidate.get(key)
        if not isinstance(clauses,list):continue
        for p in clauses[:32]:
            if not isinstance(p,dict):continue
            value=p.get('value')
            result.append({'column':p.get('column'),'op':p.get('op'),
                'value_type':type(value).__name__,
                'item_types':[type(v).__name__ for v in value[:8]] if isinstance(value,list) else []})
    return result


def condition_schema(*,allow_between=True):
    scalar={'type':['string','number','boolean','null']}
    ordered={'type':['string','number']}
    variants=[]
    for operations,value in [
        (['eq','ne'],scalar),
        (['gt','ge','lt','le'],{'type':['string','number','boolean']}),
        (['in','not_in'],{'type':'array','maxItems':100,'items':scalar}),
        (['is_null','not_null'],{'const':None}),
    ]:
        variants.append({'type':'object','additionalProperties':False,
            'required':['column','op','value'], 'properties':{
                'column':{'type':'string'},'op':{'type':'string','enum':operations},
                'value':deepcopy(value)}})
    if allow_between:
        variants.append({'type':'object','additionalProperties':False,
            'required':['column','op','value'],'properties':{
                'column':{'type':'string'},'op':{'const':'between'},
                'value':{'type':'array','minItems':2,'maxItems':2,'items':ordered}}})
    return {'anyOf':variants}
