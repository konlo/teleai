"""Explicit filter stage shared by local and remote latest-record selection."""
import math
import re
from sqlglot import exp
from utils.analysis_datasets import Condition

STAGES={'before_selection','after_selection'}

def validate(conditions, stage, columns):
    conditions=conditions or []
    if not isinstance(conditions,list) or len(conditions)>16:
        raise ValueError('최신행 필터는 최대 16개 조건 목록이어야 합니다.')
    if (conditions and stage not in STAGES) or (not conditions and stage):
        raise ValueError('필터 조건과 선택 전/후 적용 단계를 함께 지정해주세요.')
    output=[]
    for item in conditions:
        condition=Condition(**item)
        if condition.column not in columns:
            raise ValueError('필터 컬럼을 현재 스키마에서 확인해주세요.')
        values=list(condition.value) if condition.op=='in' else [condition.value]
        if not values or len(values)>100 or any(
            type(v) not in (str,int,float,bool) or
            (isinstance(v,float) and not math.isfinite(v)) for v in values):
            raise ValueError('필터에는 유한한 단일 값 또는 최대 100개 값 목록이 필요합니다.')
        output.append({'column':condition.column,'op':condition.op,
            'value':values if condition.op=='in' else condition.value})
    return output

def sql(conditions, *, dialect='duckdb', aliases=None, types=None):
    nodes=[]
    for item in conditions:
        column=exp.column((aliases or {}).get(item['column'],item['column']),quoted=True)
        if item['op']=='in':
            node=exp.In(this=column,expressions=[exp.convert(v) for v in item['value']])
        else:
            cls={'eq':exp.EQ,'ne':exp.NEQ,'gt':exp.GT,'ge':exp.GTE,'lt':exp.LT,'le':exp.LTE}[item['op']]
            node=cls(this=column,expression=exp.convert(item['value']))
        # Match pandas filter_frame's SQL-style null exclusion, including !=.
        valid=exp.Not(this=exp.Is(this=column.copy(),expression=exp.Null()))
        if re.search(r'float|double',str((types or {}).get(item['column'],'')).lower()):
            valid=exp.and_(valid,exp.Not(this=exp.Anonymous(this='isnan',expressions=[column.copy()])))
        nodes.append(exp.and_(valid,node))
    return exp.and_(*nodes).sql(dialect=dialect) if nodes else 'TRUE'

def validate_types(conditions,types):
    """Do not let SQL implicitly coerce literals that pandas rejects."""
    for item in conditions:
        dtype=str(types.get(item['column'],'')).lower()
        values=item['value'] if item['op']=='in' else [item['value']]
        if re.search(r'string|str|object|varchar|char',dtype):
            valid=all(isinstance(v,str) for v in values)
        elif re.search(r'bool',dtype):
            valid=all(type(v) is bool for v in values)
        elif re.search(r'int|long|short|byte|float|double|decimal|numeric',dtype):
            valid=all(type(v) in (int,float) for v in values)
        elif re.search(r'date|time',dtype):
            import pandas as pd
            try:
                valid=all(isinstance(v,str) and not pd.isna(pd.Timestamp(v)) for v in values)
            except (ValueError,TypeError):valid=False
        else:valid=False
        if not valid:raise ValueError('필터 비교값의 자료형이 관측된 컬럼 자료형과 다릅니다.')

def lineage(conditions,stage):
    return {'conditions':conditions,'filter_stage':stage} if conditions else {}

def matches(args,spec):
    return ((args.get('conditions') or [])==(spec.get('conditions') or [])
        and (args.get('filter_stage') or '')==(spec.get('filter_stage') or ''))
