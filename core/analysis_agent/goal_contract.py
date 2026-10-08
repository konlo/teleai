"""Model-authored goals, compiled into execution contracts without reading prose."""
from copy import deepcopy
import json
import time
from core.analysis_agent.goal_schema import OPTIONS, task_schema

GOAL_KEYS={'objective','mode','sources','columns','conditions','any_conditions',
           'measure_conditions','ratio','current_result_only','fresh_source_required','question','tasks'}
OPERATIONS={'AVG','SUM','MIN','MAX','MEDIAN','COUNT','CORR','RATIO'}


def response_schema():
    """Provider JSON grammar; the same strict contract is checked after inference."""
    strings={'type':'array','items':{'type':'string'}}
    condition={'type':'object','additionalProperties':False,'required':['column','op','value'],
        'properties':{'column':{'type':'string'},'op':{'type':'string','enum':
            ['eq','ne','gt','ge','lt','le','between','in','not_in','is_null','not_null']},
            'value':{'type':['string','number','boolean','null','array'],'items':{'type':['string','number','boolean','null']}}}}
    alternative=deepcopy(condition)
    alternative['properties']['op']['enum'].remove('between')
    return {'type':'object','additionalProperties':False,'required':sorted(GOAL_KEYS|{'source_reference'}),
        'properties':{'objective':{'type':'string'},'mode':{'type':'string','enum':['execute','explain','clarify']},
            'source_reference':{'type':'string','enum':['explicit','previous_analysis','selected_dataset']},
            'sources':strings,'columns':strings,
            **{k:{'type':'array','items':alternative if k=='any_conditions' else condition}
               for k in ('conditions','any_conditions','measure_conditions')},
            'ratio':{'type':['object','null']},'current_result_only':{'type':'boolean'},
            'fresh_source_required':{'type':'boolean'},'question':{'type':'string'},
            'tasks':{'type':'array','maxItems':8,'items':task_schema()}}}


def string_list(value, name, limit=32):
    if not isinstance(value,list) or len(value)>limit or any(not isinstance(v,str) or not v or len(v)>512 for v in value):
        raise ValueError(name+' must be a bounded list of strings')
    return list(dict.fromkeys(value))


def predicates(value):
    if not isinstance(value,list) or len(value)>32:raise ValueError('conditions must be a list')
    for item in value:
        if (not isinstance(item,dict) or set(item)!={'column','op','value'}
                or not isinstance(item['column'],str) or not item['column']
                or item['op'] not in {'eq','ne','gt','ge','lt','le','between','in','not_in','is_null','not_null'}):
            raise ValueError('invalid condition structure')
        literal=item['value']
        values=literal if isinstance(literal,list) else [literal]
        if len(values)>100 or any(v is not None and not isinstance(v,(str,int,float,bool)) for v in values):
            raise ValueError('condition values must be bounded literals')
        if item['op'] in {'in','not_in'} and not isinstance(literal,list):raise ValueError('IN needs a list')
        if item['op']=='between':
            if (not isinstance(literal,list) or len(literal)!=2 or
                    any(type(v) not in (int,float,str) for v in literal) or
                    type(literal[0]) is not type(literal[1]) or literal[0]>literal[1]):
                raise ValueError('between needs ordered [lower, upper] boundaries')
    return deepcopy(value)


def validate_goal(value):
    if not isinstance(value,dict) or set(value)-{'source_reference'}!=GOAL_KEYS:raise ValueError('goal_schema_v1 fields are required exactly')
    if value.get('source_reference','explicit') not in {'explicit','previous_analysis','selected_dataset'}:raise ValueError('invalid source_reference')
    if len(json.dumps(value,ensure_ascii=False))>20000:raise ValueError('goal is too large')
    if not isinstance(value['objective'],str) or not value['objective'].strip():raise ValueError('objective is required')
    if value['mode'] not in {'execute','explain','clarify'}:raise ValueError('invalid goal mode')
    string_list(value['sources'],'sources',8);string_list(value['columns'],'columns')
    for key in ('conditions','any_conditions','measure_conditions'):predicates(value[key])
    if any(c['op']=='between' for c in value['any_conditions']):
        raise ValueError('between is one AND range in conditions; a nested range in OR needs a separate supported goal')
    for lower in value['any_conditions']:
        for upper in value['any_conditions']:
            if (lower['column']==upper['column'] and lower['op']=='ge' and upper['op']=='le'
                    and type(lower['value']) in (int,float) and type(upper['value']) in (int,float)
                    and lower['value']<=upper['value']):
                raise ValueError('OR lower/upper boundaries cover all non-null values, not a range. Recheck the original AND/OR; use between in conditions for a range.')
    for key in ('current_result_only','fresh_source_required'):
        if type(value[key]) is not bool:raise ValueError(key+' must be boolean')
    if value['current_result_only'] and value['fresh_source_required']:raise ValueError('current result cannot be refreshed')
    if not isinstance(value['question'],str):raise ValueError('question must be a string')
    if value['ratio'] is not None and not isinstance(value['ratio'],dict):raise ValueError('ratio must be an object or null')
    if value['measure_conditions'] and value['ratio'] is None:
        raise ValueError('measure_conditions are only for an explicit ratio; do not invent NULL filters for SUM/AVG')
    if value['ratio'] is not None:
        ratio=value['ratio']
        if (set(ratio)-{'column','aggregation'} or not isinstance(ratio.get('column'),str)
                or not ratio['column'] or not value['measure_conditions']):raise ValueError('ratio requires explicit numerator and column')
    tasks=value['tasks']
    if not isinstance(tasks,list) or len(tasks)>8:raise ValueError('tasks must be a bounded list')
    if value['mode']=='execute' and not tasks:raise ValueError('execution requires result obligations')
    if value['mode']=='clarify' and (tasks or not value['question'].strip()):raise ValueError('clarification needs only a question')
    if value['mode']=='explain' and tasks:raise ValueError('explanation cannot execute tasks. Showing schema, names, types or stored data requires mode execute. A prohibition on charts only removes chart tasks; preserve the positive metadata/data request.')
    capabilities={t.get('capability') for t in tasks if isinstance(t,dict)}
    if (value['mode']=='execute' and capabilities-{'table_list','chart_adjust'} and not value['sources']
            and value.get('source_reference','explicit')=='explicit'):
        raise ValueError('execution needs an explicit source from verified context, or a clarification')
    if 'chart_adjust' in capabilities and len(capabilities)>1:
        raise ValueError('display adjustment binds one verified previous chart; use a separate task for a new analysis')
    if 'table_list' in capabilities and len(capabilities)>1:
        raise ValueError('inventory and table analysis require separate source scopes; clarify or split the goal')
    seen=set()
    for task in tasks:
        if not isinstance(task,dict) or set(task)!={'capability','options'}:raise ValueError('invalid task')
        cap=task['capability'];options=task['options']
        if cap not in OPTIONS:raise ValueError('unknown capability; use one of '+', '.join(sorted(OPTIONS)))
        if not isinstance(options,dict):raise ValueError(cap+'.options must be an object')
        extra=set(options)-OPTIONS[cap]
        if extra:raise ValueError(cap+'.options has unsupported keys '+', '.join(sorted(extra))+
                                 '; allowed keys: '+', '.join(sorted(OPTIONS[cap])))
        if cap in seen:raise ValueError('duplicate capability; use options for multiple measures. metadata dtypes includes both column names and types: use one metadata task with kind dtypes for that combined request.')
        seen.add(cap)
        from core.analysis_agent.goal_options import validate_options
        validate_options(cap,options,value)
        if cap=='row_preview':
            n=options.get('limit')
            if type(n) is not int or not 1<=n<=200:raise ValueError('preview limit must be 1..200')
        if cap=='calculation':
            ops=string_list(options.get('operations'),'operations',8)
            if not ops or set(ops)-OPERATIONS:raise ValueError('invalid operations')
            if not value['columns']:raise ValueError('calculation needs explicit measures; whole table COUNT uses row_count')
        if cap=='metadata' and options.get('kind') not in {'columns','dtypes','numeric_columns','categorical_columns'}:raise ValueError('invalid metadata kind')
        if cap=='profile' and options.get('kind') not in {'missing','distinct','summary'}:raise ValueError('invalid profile kind')
        if cap=='chart':
            if options.get('kind') not in {'histogram','bar','scatter','line','boxplot','recommend'}:raise ValueError('invalid chart kind')
            if options.get('category') and options['kind'] not in {'histogram','recommend'}:
                raise ValueError('category groups a numeric histogram only; frequency bar uses one x column and implicit count Y. Filter columns do not define chart groups.')
            if 'bins' in options and options['kind']=='bar':raise ValueError('frequency bar has categorical labels, not numeric bins')
            if 'bins' in options and (type(options['bins']) is not int or not 2<=options['bins']<=100):raise ValueError('invalid bins')
            axes=options.get('axes',{})
            if options.get('kind')=='bar' and axes.get('y'):
                raise ValueError('This bar task is a frequency chart: x only, with implicit count Y. For a numeric distribution by category use histogram with category, or recommend; do not invent an aggregation.')
            if not isinstance(axes,dict) or set(axes)-{'x','y'} or any(v!='' and v not in value['columns'] for v in axes.values()):
                raise ValueError('invalid chart axes '+repr(axes)+' for goal columns '+repr(value['columns'])+
                                 '; histogram frequency Y-axis is implicit, omit y. Axes must reference goal.columns.')
        if cap=='join' and options.get('how') not in {'inner','left','right','outer'}:raise ValueError('invalid join mode')
    return deepcopy(value)


def pending_state(human, previous, context):
    from core.analysis_agent.conversation_context import prior_analysis
    from core.analysis_agent.schema_questions import prior_subject
    prior=prior_analysis(previous,context)
    current={
        'request_id':human.id,'request_text':str(human.content),'status':'working',
        'intent_origin':'llm','goal_pending':True,'request_started_at':time.time(),
        'required_sources':[],'required_columns':[],'explicit_columns':[],
        'scope':{'sources':[],'columns':[],'conditions':[],'any_conditions':[],
                 'measure_conditions':[],'ratio':None,'join_edges':[],'unresolved':[]},
        'previous_scope':deepcopy(prior.get('scope',{})),
        'confirmed_analysis':deepcopy(prior),
        'chart':False,'kind':None,'calculation':False,'operations':[],
        'metadata_kind':None,'catalog_discovery_source':'','catalog_discovery_schema':'',
        'profile_kind':None,'preview_limit':0,'data_load':False,'whole_row_count':False,
        'current_result_only':False,'fresh_source_required':False,'requested_result_rows':None,
        'requested_join':False,'join':False,'join_how':None,'statistical_kind':None,
        'latest_per_key_spec':None,'row_preview_spec':None,'chart_spec_requested':False,
        'value_list_requested':False,'operation_pending':False,'schema_semantic':False,
        'outlier_spec':None,'outlier_column':None,'time_series_frequency':None,
        'histogram_bins':None,'chart_cumulative':False,
        'processed':[],'sent_calls':[],'evidence_ids':[],'artifact_ids':[],
        'failed':{},'failed_signatures':{},'attempts':0,'model_calls':0,'model_seconds':0.,
        'remote_query_ids':[],'remote_query_evidence':{},'columns':[],
    }
    subject=prior_subject(previous,[],context)
    if subject:current['schema_subject']=subject
    if previous.get('status')!='complete' and previous.get('requested_subject'):
        current['requested_subject']=deepcopy(previous['requested_subject'])
    if previous.get('request_id')==human.id:
        # Reinterpreting an in-flight legacy checkpoint cannot reset budgets.
        for key in ('request_started_at','model_calls','model_seconds','attempts','sent_calls'):
            if key in previous:current[key]=deepcopy(previous[key])
    return current


def canonical_sources(sources, context):
    known={t['table'] for t in context.reference_context if t.get('table')}
    known.update(d.source for d in context.datasets.metadata.values())
    resolved=[]
    for source in sources:
        parts=source.replace('`','').casefold().split('.')
        matches=[s for s in known if s.replace('`','').casefold().split('.')[-len(parts):]==parts]
        if len(matches)>1:raise ValueError('source name is ambiguous: '+source)
        resolved.append(matches[0] if matches else source.replace('`',''))
    return resolved


def compile_goal(current, goal, context, remote_available=False):
    """Translate an accepted goal; no natural-language recognition is permitted."""
    goal=validate_goal(goal);current=deepcopy(current)
    sources=canonical_sources(goal['sources'],context)
    reference=goal.get('source_reference','explicit')
    # Inventory uses its observed namespace, never a selected/previous dataset.
    # A generic source-reference label cannot prevent safe catalog discovery.
    inventory=any(t['capability']=='table_list' for t in goal['tasks'])
    if goal['mode']=='execute' and reference!='explicit' and not inventory:
        if reference=='previous_analysis':
            prior=current.get('confirmed_analysis') or {}
            requested=current.get('requested_subject') or {}
            if requested.get('sources'):
                bound=canonical_sources(requested['sources'],context)
                known={t['table'] for t in context.reference_context if t.get('table')}
                if any(s not in known for s in bound):raise ValueError('requested subject needs observed catalog identity')
                if goal['current_result_only']:raise ValueError('a requested but unfinished subject is not a stored result; inspect/query the source')
            else:
                if prior.get('status')!='complete':raise ValueError('previous_analysis needs a verified completed subject')
                bound=prior.get('required_sources') or []
        else:
            info=context.datasets.metadata.get(context.selected_dataset_id)
            bound=[info.source] if info else []
        if not bound:raise ValueError(reference+' has no verified source')
        if sources and sources!=bound:raise ValueError(reference+' source mismatch; verified subject is '+repr(bound))
        sources=deepcopy(bound)
    columns=[] if goal['columns']==['*'] and any(t['capability']=='row_preview' for t in goal['tasks']) else goal['columns']
    scope={k:deepcopy(goal[k]) for k in ('conditions','any_conditions','measure_conditions','ratio')}
    for key in ('conditions','measure_conditions'):
        scope[key]=[expanded for c in scope[key] for expanded in
                    ([{'column':c['column'],'op':'ge','value':c['value'][0]},
                      {'column':c['column'],'op':'le','value':c['value'][1]}]
                     if c['op']=='between' else [c])]
    scope.update(sources=sources,columns=list(dict.fromkeys(c['column'] for k in
        ('conditions','any_conditions','measure_conditions') for c in scope[k])),unresolved=[],join_edges=[])
    current.update(goal=goal,goal_pending=False,goal_version=1,
        required_sources=sources,required_columns=columns,explicit_columns=columns,
        scope=scope,current_result_only=goal['current_result_only'],fresh_source_required=goal['fresh_source_required'],
        explanation_only=goal['mode']=='explain',goal_question=goal['question'] if goal['mode']=='clarify' else '')
    for task in goal['tasks']:
        cap=task['capability'];o=task['options']
        if cap=='row_preview':current['row_preview_spec']={'limit':o['limit'],'question':''}
        elif cap=='metadata':current['metadata_kind']=o['kind']
        elif cap=='table_list':
            catalog=o.get('catalog') or getattr(context,'source_namespace','')
            if not catalog:raise ValueError('table inventory needs an observed namespace')
            current.update(catalog_discovery_source='information_schema.tables' if context.sql_dialect=='mysql' else catalog+'.information_schema.tables',
                catalog_discovery_schema=o.get('schema',''),required_sources=['information_schema.tables' if context.sql_dialect=='mysql' else catalog+'.information_schema.tables'])
        elif cap=='row_count':current.update(whole_row_count=True,calculation=True,operations=['COUNT'])
        elif cap=='calculation':current.update(calculation=True,operations=o['operations'],
            named_measure_columns=[c for c in columns if c not in o.get('group_columns',[])],
            scalar_group_columns=o.get('group_columns',[]),
            scalar_grouping=bool(o.get('group_columns')) or o.get('grouped',False))
        elif cap=='profile':current['profile_kind']=o['kind']
        elif cap=='value_list':current['value_list_requested']=True
        elif cap=='chart':
            plotted=[o.get('axes',{}).get(k) for k in ('x','y') if o.get('axes',{}).get(k)] or columns
            if len(goal['tasks'])==1:current['required_columns']=plotted
            kind=o['kind'];current.update(chart=True,kind=None if kind=='recommend' else kind,
                chart_axes={k:v for k,v in o.get('axes',{}).items() if v},histogram_bins=o.get('bins'),
                chart_spec_requested=kind in {'bar','scatter','line','boxplot'},categorical_distribution=o.get('categorical',kind=='bar'))
            if len(sources)==len(plotted)==1 and kind in {'histogram','recommend'} and not o.get('category'):
                from core.analysis_catalog import resolve_table_context
                from core.analysis_agent.dtypes import family
                observed=resolve_table_context(context.reference_context,context.datasets,sources[0])
                types=[c.get('database_dtype') or c.get('dtype') for c in
                       observed.get('table_context',{}).get('columns',[]) if c.get('name')==plotted[0]]
                if observed.get('status')=='ready' and len(types)==1:
                    dtype=family(types[0])
                    if dtype in {'numeric','categorical'}:
                        current.update(kind='histogram' if dtype=='numeric' else 'bar',
                                       categorical_distribution=dtype=='categorical',chart_spec_requested=False)
            if o.get('category'):
                if len(sources)!=1 or not columns:raise ValueError('grouped chart needs a source and measure')
                from core.analysis_catalog import resolve_table_context
                from core.analysis_agent.dtypes import family
                observed=resolve_table_context(context.reference_context,context.datasets,sources[0])
                types=[c.get('database_dtype') or c.get('dtype') for c in
                       observed.get('table_context',{}).get('columns',[]) if c.get('name')==plotted[0]]
                if observed.get('status')=='ready' and len(types)==1 and family(types[0])!='numeric':
                    raise ValueError('Grouped histogram requires a numeric x measure; observed '+plotted[0]+' is '+str(types[0])+'. Use a frequency bar for a categorical subject; do not turn filter columns into chart groups.')
                current.update(chart_group_spec={'source':sources[0],'value_column':plotted[0],'category':o['category'],'bins':o.get('bins',8)},chart_spec_requested=True,histogram_bins=None)
            elif kind=='bar' and len(sources)==len(plotted)==1 and not o.get('axes',{}).get('y'):
                current.update(chart_spec_requested=False,categorical_distribution=True)
            if current['current_result_only']:
                prior=current['confirmed_analysis'];proof=prior.get('table_preview_evidence') or prior.get('chart_display_evidence')
                if proof and set(columns).issubset(proof.get('columns',[])) and proof.get('source') in sources:
                    current.update(display_dataset_id=proof['dataset_id'],chart_display_evidence=deepcopy(proof),requested_result_rows=proof['rows'])
        elif cap=='chart_adjust':
            from core.analysis_agent.chart_axis import bind_goal
            prior=current.get('confirmed_analysis') or {}
            bound_sources=prior.get('required_sources') or []
            if sources and sources!=bound_sources:raise ValueError('chart_adjust must reference the previously verified chart source '+repr(bound_sources))
            if goal['fresh_source_required']:raise ValueError('chart_adjust cannot refresh its population')
            for key in ('conditions','any_conditions','measure_conditions','ratio'):
                if goal[key] and goal[key]!=(prior.get('scope') or {}).get(key):raise ValueError('chart_adjust cannot change the verified population '+key)
            current.update(chart=True,kind=prior.get('kind'),scope=deepcopy(prior.get('scope',scope)),
                required_sources=deepcopy(bound_sources),
                required_columns=deepcopy(prior.get('required_columns',columns)),
                chart_axes=deepcopy(prior.get('chart_axes',{})),
                categorical_distribution=prior.get('kind')=='bar',chart_spec_requested=True)
            bind_goal(current,context,o['y_max'])
        elif cap=='join':current.update(join=True,requested_join=True,join_how=o['how'])
        elif cap=='statistics':current['statistical_kind']=o.get('test')
        elif cap=='time_series':current.update(time_series_frequency=o.get('frequency'),time_series_aggregation=o.get('aggregation'),time_series_gap_policy=o.get('gap_policy','omit'),time_series_timezone=o.get('timezone',''))
        elif cap=='winsorization':current.update(winsor_column=o.get('column'),winsor_spec={k:o[k] for k in ('lower_quantile','upper_quantile') if k in o})
        elif cap=='outliers':current.update(outlier_column=o.get('column'),outlier_spec={k:o[k] for k in ('method','threshold','tail') if k in o},outlier_selection=o.get('selection','outliers'))
        elif cap=='pivot':current.update(pivot_requested=True,pivot_aggregation=o.get('aggregation'),pivot_value_column=o.get('value_column'),pivot_success_value=o.get('success_value'),pivot_index_columns=o.get('index_columns',[]),pivot_column_columns=o.get('column_columns',[]),pivot_conditions=scope['conditions'],pivot_margins=o.get('margins',False),pivot_sort=o.get('sort','ascending'))
        elif cap=='group_summary':current.update(group_summary_requested=True,group_summary_columns=o.get('columns',[]),group_summary_metrics=o.get('metrics',[]),group_summary_conditions=scope['conditions'])
        elif cap=='latest_per_key':
            if len(sources)!=1:raise ValueError('latest selection needs one source')
            spec={**o,'source':sources[0],'conditions':scope['conditions'],'filter_scope_bound':True,
                'remote':remote_available and not current['current_result_only'],'context_dataset_id':context.selected_dataset_id}
            if not spec['remote']:spec['dataset_id']=context.selected_dataset_id
            current.update(latest_per_key_spec=spec,chart=True,kind='bar' if o.get('categorical',True) else 'histogram',required_columns=[o.get('value_column')],chart_spec_requested=True)
        elif cap=='data_load':current['data_load']=True
    if current.get('row_preview_spec'):
        if len(sources)!=1:raise ValueError('preview needs one exact source or clarification')
        if current['current_result_only']:
            selected=context.datasets.metadata.get(context.selected_dataset_id)
            if not selected or selected.source not in sources:
                raise ValueError('current_result_only cannot use an unrelated selected dataset; a named table preview needs false')
    # Compilation cannot publish, choose an active dataset, or execute SQL.
    return current
