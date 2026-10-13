"""Resolve model-declared calendar periods from a verified source MAX receipt."""
from datetime import datetime
import calendar

SPEC={'type':'object','additionalProperties':False,
    'required':['anchor','amount','unit','column'], 'properties':{
        'anchor':{'type':'string','enum':['unspecified','source_max','today']},
        'amount':{'type':'integer','minimum':1,'maximum':10000},
        'unit':{'type':'string','enum':['year','month','day']},'column':{'type':'string'}}}


def resolve(context,current,sources,clause):
    spec=clause.get('relative_period')
    if not spec or spec['anchor']=='unspecified':return None
    if len(sources)!=1 or not spec['column'] or clause['columns']!=[spec['column']]:return None
    if spec['anchor']!='source_max':return None
    service=(getattr(context,'runtime_services',None) or {}).get('resolve_source_period')
    if not service:return None
    proof=service(current,sources[0],spec)
    if proof.get('status')!='ready':
        current['temporal_resolution_status']=proof.get('status','unavailable')
        return None
    current['temporal_resolution']=proof
    return proof['conditions']


def boundaries(value,amount,unit):
    """Calendar year/month subtraction includes leap-day clipping, not 365 days."""
    end=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if unit=='day':
        from datetime import timedelta
        start=end-timedelta(days=amount)
    else:
        months=end.year*12+end.month-1-amount*(12 if unit=='year' else 1)
        year,month=divmod(months,12);month+=1
        start=end.replace(year=year,month=month,day=min(end.day,calendar.monthrange(year,month)[1]))
    raw=str(value)
    lower=start.date().isoformat() if len(raw)==10 else start.isoformat(sep='T' if 'T' in raw else ' ')
    if raw.endswith('Z'):lower=lower.replace('+00:00','Z')
    return lower,raw


def probe_query(source,column,dtype,dialect):
    """String dates must be homogeneous ISO UTC before lexical comparisons."""
    q=lambda name:'`'+name.replace('`','``')+'`'
    table='.'.join(q(p) for p in source.split('.'));col=q(column)
    from core.analysis_agent.dtypes import family
    kind=family(dtype)
    if kind not in {'temporal','categorical'}:raise ValueError('Period column needs an observed date or string type')
    temporal=kind=='temporal'
    if temporal:invalid='0';formats='1'
    else:
        # Only sortable ISO dates / UTC timestamps are accepted. Mixed offsets
        # and arbitrary strings require a typed transformation, not guessing.
        pattern='^[0-9]{4}-[0-9]{2}-[0-9]{2}([ T][0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?([+]00:00|Z)?)?$'
        regex='RLIKE' if dialect=='databricks' else 'REGEXP'
        cast=(f'TRY_CAST({col} AS TIMESTAMP)' if dialect=='databricks'
              else f'CAST({col} AS DATETIME)')
        invalid=f'SUM(CASE WHEN {col} IS NOT NULL AND (NOT ({col} {regex} \'{pattern}\') OR {cast} IS NULL) THEN 1 ELSE 0 END)'
        formats=f"COUNT(DISTINCT REGEXP_REPLACE({col}, '[0-9]', '0'))"
    return (f'SELECT MAX({col}) AS `__period_max`, COUNT({col}) AS `__period_count`, '
            f'{invalid} AS `__period_invalid`, {formats} AS `__period_formats` FROM {table}')
