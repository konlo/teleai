"""Serialize validated population contracts; never interpret user prose."""
from sqlglot import exp


def where_sql(conditions, dialect):
    terms=[]
    operations={'eq':exp.EQ,'ne':exp.NEQ,'gt':exp.GT,'ge':exp.GTE,'lt':exp.LT,'le':exp.LTE}
    for item in conditions:
        column=exp.Column(this=exp.Identifier(this=item['column'],quoted=True))
        op=item['op'];value=item['value']
        if op in {'is_null','not_null'} or (op in {'eq','ne'} and value is None):
            term=exp.Is(this=column,expression=exp.Null())
            if op in {'not_null','ne'}:term=exp.Not(this=term)
        elif op in {'in','not_in'}:
            term=exp.In(this=column,expressions=[exp.convert(v) for v in value])
            if op=='not_in':term=exp.Not(this=term)
        elif op=='between':
            term=exp.Between(this=column,low=exp.convert(value[0]),high=exp.convert(value[1]))
        elif op in operations:
            term=operations[op](this=column,expression=exp.convert(value))
        else:raise ValueError('unsupported predicate operation')
        terms.append(term)
    return exp.and_(*terms).sql(dialect=dialect) if terms else ''


def population_sql(scope, dialect):
    """Serialize the complete AND population plus its one OR group."""
    parts=[]
    if scope.get('conditions'):
        conditions=[]
        for term in scope['conditions']:
            if term['op']=='between':
                conditions.extend({'column':term['column'],'op':op,'value':value}
                                  for op,value in zip(('ge','le'),term['value']))
            else:conditions.append(term)
        parts.append('('+where_sql(conditions,dialect)+')')
    if scope.get('any_conditions'):
        parts.append('('+' OR '.join(where_sql([term],dialect)
                                    for term in scope['any_conditions'])+')')
    return ' AND '.join(parts)
