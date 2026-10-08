"""Single source for model output grammar and executable task parameters.

Capability branches prevent a provider's JSON grammar from accepting arbitrary
keys which the executor will reject later. This contains no table knowledge.
"""
from copy import deepcopy

TEXT = {'type': 'string'}
NUMBER = {'type': 'number'}
BOOL = {'type': 'boolean'}
STRINGS = {'type': 'array', 'items': TEXT, 'maxItems': 32}

def enum(*values):
    return {'type': 'string', 'enum': list(values)}

def obj(properties, required=()):
    return {'type': 'object', 'additionalProperties': False,
            'properties': properties, 'required': list(required)}

OPTION_SCHEMAS = {
    'metadata': obj({'kind': enum('columns','dtypes','numeric_columns','categorical_columns')}, ['kind']),
    'table_list': obj({'catalog': TEXT, 'schema': TEXT}),
    'row_preview': obj({'limit': {'type':'integer','minimum':1,'maximum':200}}, ['limit']),
    'row_count': obj({}), 'value_list': obj({}), 'data_load': obj({}),
    'calculation': obj({'operations': {'type':'array','items':enum('AVG','SUM','MIN','MAX','MEDIAN','COUNT','CORR','RATIO'),
                                      'minItems':1,'maxItems':8},
                        'grouped':BOOL,'group_columns':STRINGS}, ['operations']),
    'profile': obj({'kind': enum('missing','distinct','summary')}, ['kind']),
    'chart': obj({'kind': enum('histogram','bar','scatter','line','boxplot','recommend'),
                  'axes':obj({'x':TEXT,'y':TEXT}),
                  'bins':{'type':'integer','minimum':2,'maximum':100},
                  'category':TEXT,'categorical':BOOL}, ['kind']),
    'chart_adjust':obj({'y_max':{'type':'number','exclusiveMinimum':0}},['y_max']),
    'join': obj({'how':enum('inner','left','right','outer')}, ['how']),
    'statistics': obj({'test':enum('mann_whitney','chi_square','one_way_anova','paired_t','independent_t','mean_ci')}, ['test']),
    'time_series': obj({'frequency':enum('hour','day','week','month'),
                        'aggregation':enum('mean','sum','count','median','min','max'),
                        'gap_policy':enum('omit','zero','nan'),'timezone':TEXT}, ['frequency','aggregation']),
    'winsorization': obj({'column':TEXT,'lower_quantile':NUMBER,'upper_quantile':NUMBER},
                          ['column','lower_quantile','upper_quantile']),
    'outliers': obj({'column':TEXT,'method':enum('iqr','zscore','mad'),'threshold':NUMBER,
                     'tail':enum('both','upper','lower'),'selection':enum('outliers','inliers')},
                    ['column','method','threshold']),
    'pivot': obj({'aggregation':enum('count','mean','sum','median','min','max','success_rate','overall_percent'),
                  'value_column':TEXT,'success_value':{'type':['string','number','boolean','null']},
                  'index_columns':STRINGS,'column_columns':STRINGS,'margins':BOOL,
                  'sort':enum('ascending','descending','none')}, ['aggregation','index_columns','column_columns']),
    'group_summary': obj({'columns':STRINGS,'metrics':{'type':'array','minItems':1,'maxItems':8,
        'items':obj({'name':TEXT,'aggregation':TEXT,'value_column':TEXT,'condition':{'type':'object'},
                     'empty_value':{'type':['string','number','boolean','null']}}, ['name','aggregation'])}}, ['columns','metrics']),
    'latest_per_key': obj({'key_columns':STRINGS,'order_column':TEXT,'value_column':TEXT,'categorical':BOOL,
                           'bins':{'type':'integer','minimum':2,'maximum':100},'tie_break_columns':STRINGS,
                           'null_policy':enum('reject','drop_before_selection'),
                           'filter_stage':enum('','before_selection','after_selection')},
                          ['key_columns','order_column','value_column','categorical']),
}
OPTIONS = {name:set(schema['properties']) for name,schema in OPTION_SCHEMAS.items()}

def task_schema():
    branches=[]
    for name,schema in OPTION_SCHEMAS.items():
        options=deepcopy(schema)
        if name=='chart':
            variants=[]
            for kind in schema['properties']['kind']['enum']:
                variant=deepcopy(schema)
                variant['properties']['kind']={'const':kind}
                if kind in {'histogram','bar'}:
                    variant['properties']['axes']=obj({'x':TEXT})
                if kind not in {'histogram','recommend'}:
                    variant['properties'].pop('category')
                if kind=='bar':
                    variant['properties'].pop('bins')
                variants.append(variant)
            options={'anyOf':variants}
        branches.append(obj({'capability':{'const':name},'options':options},['capability','options']))
    return {'anyOf':branches}

def option_protocol():
    """Exact names and enums, shared with the grammar rather than hand copies."""
    return {name:{key:(field.get('enum') or field.get('type'))
                  for key,field in spec['properties'].items()}
            for name,spec in OPTION_SCHEMAS.items()}

def compact_option_protocol():
    """Names/enums only; the provider grammar already enforces field types."""
    return {name:','.join(key+('='+('|'.join(map(str,field))) if isinstance(field,list) else '')
                         for key,field in options.items()) for name,options in option_protocol().items()}
