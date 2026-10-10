"""Declarative postconditions agreed before generated Python is executed."""
from hashlib import sha256
import json
from jsonschema import Draft202012Validator

TEXT={'type':'string','minLength':1,'maxLength':128}
NAMES={'type':'array','items':TEXT,'minItems':1,'maxItems':8,'uniqueItems':True}
AGGREGATIONS=['mean','sum','min','max','median','std','count']
def step(op,properties,required):
    return {'type':'object','additionalProperties':False,
            'properties':{'op':{'const':op},**properties},'required':['op',*required]}
SCHEMA={'type':'object','additionalProperties':False,'required':['steps','output_columns'],
    'properties':{'output_columns':NAMES,'steps':{'type':'array','minItems':1,'maxItems':8,'items':{'anyOf':[
        step('sort',{'columns':NAMES,'ascending':{'type':'boolean'}},['columns','ascending']),
        step('rolling',{'column':TEXT,'output_column':TEXT,'window':{'type':'integer','minimum':1,'maximum':10000},
             'min_periods':{'type':'integer','minimum':1,'maximum':10000},
             'partial_window_quote':{'type':'string','minLength':1,'maxLength':256},
             'aggregation':{'enum':AGGREGATIONS}},['column','output_column','window','min_periods','aggregation']),
        step('diff',{'column':TEXT,'output_column':TEXT,'periods':{'type':'integer','minimum':1,'maximum':10000}},
             ['column','output_column','periods']),
        step('aggregate',{'column':TEXT,'output_column':TEXT,'aggregation':{'enum':AGGREGATIONS}},
             ['column','output_column','aggregation'])]}}}}

def validate(contract,input_columns):
    error=next(Draft202012Validator(SCHEMA).iter_errors(contract),None)
    if error:raise ValueError('Invalid result contract field '+'.'.join(map(str,error.absolute_path))+' ('+error.validator+')')
    known=set(input_columns)
    for item in contract['steps']:
        needed=set(item['columns']) if item['op']=='sort' else {item['column']}
        if not needed<=known:raise ValueError('Result contract references unavailable input/derived columns')
        if item['op']=='rolling' and item['min_periods']>item['window']:
            raise ValueError('Rolling min_periods must not exceed window')
        if item['op']=='aggregate':known={item['output_column']}
        elif item['op']!='sort':known.add(item['output_column'])
    if not set(contract['output_columns'])<=known:raise ValueError('Result contract output columns are not defined')
    return contract

def digest(contract):
    return sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate_intent(contract,request):
    """Partial windows change the calculation and need current user evidence."""
    for item in contract['steps']:
        if item['op']=='rolling' and item['min_periods']<item['window']:
            quote=item.get('partial_window_quote','')
            if not quote or quote not in request:
                raise ValueError('Use min_periods=window for complete rolling windows by default. Partial windows require partial_window_quote copied from an explicit CURRENT request for partial windows; do not invent that requirement.')

def verified(current):
    contract=(current.get('custom_analysis_spec') or {}).get('result_contract')
    receipt=current.get('custom_analysis_evidence') or {}
    return bool(contract and receipt.get('semantic_verified') is True
                and receipt.get('result_contract_hash')==digest(contract))
