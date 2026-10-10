"""Validate structured task parameters, never infer semantics from user prose."""
import math

RESULT_TOOLS={
    'custom_analysis_spec':{'execute_analysis_python'},
    'advanced_eda_spec':{'render_advanced_eda'},
    'export_spec':{'export_analysis_result'},
    'chart':{'render_chart_spec','render_histogram','recommend_chart_images','prepare_histogram','show_chart','prepare_source_scatter','render_count_rate_chart'},
    'statistical_kind':{'statistical_test'},'winsor_spec':{'winsorize_numeric'},
    'outlier_spec':{'detect_outliers','select_outlier_rows'},'time_series_frequency':{'prepare_time_series'},
    'pivot_requested':{'pivot_dataset'},'group_summary_requested':{'summarize_groups'},
    'join':{'join_datasets'},'row_preview_spec':{'prepare_row_preview'},
    'value_list_requested':{'inspect_value_list'},
}


def allowed_result_call(current,name):
    """A planner/tool cannot add an output obligation rejected by the goal."""
    return all(current.get(flag) for flag,names in RESULT_TOOLS.items() if name in names)


def validate_options(cap, options, goal):
    def enum(key, choices, required=True):
        if (required or key in options) and options.get(key) not in choices:
            raise ValueError('invalid '+cap+'.'+key)
    def column(key):
        if not isinstance(options.get(key),str) or not options[key]:
            raise ValueError(cap+'.'+key+' requires an explicit column')
    def columns(key):
        v=options.get(key)
        if not isinstance(v,list) or not v or len(v)>32 or any(not isinstance(c,str) or not c for c in v):
            raise ValueError(cap+'.'+key+' requires explicit columns')
    for key in ('grouped','categorical','margins','legend','stacked'):
        if key in options and type(options[key]) is not bool:raise ValueError(key+' must be boolean')
    if 'bins' in options and (type(options['bins']) is not int or not 2<=options['bins']<=100):raise ValueError('invalid bins')
    if cap=='table_list':
        for key in options:
            if not isinstance(options[key],str):raise ValueError('namespace must be text')
    if cap=='custom_analysis' and (not isinstance(options.get('description'),str) or not options['description'].strip()):
        raise ValueError('Custom analysis requires an explicit description')
    if cap=='custom_analysis':
        from core.analysis_agent.python_result_contract import validate
        validate(options.get('result_contract'),goal['columns'])
    if cap=='advanced_eda':
        enum('kind',{'correlation_heatmap','scatter_matrix','distribution_panels'})
        if not 2<=len(goal['columns'])<=6:raise ValueError('Advanced EDA requires 2..6 explicit columns')
    if cap=='export':enum('format',{'csv','parquet','png'})
    if cap=='calculation' and (options.get('grouped') or options.get('group_columns')):
        columns('group_columns')
        if not set(options['group_columns']).issubset(goal['columns']):
            raise ValueError('group columns '+repr(options['group_columns'])+' must be explicit goal columns '+repr(goal['columns'])+
                             '; ungrouped totals must omit group_columns or use []')
    elif cap=='calculation' and 'group_columns' in options and options['group_columns']!=[]:
        raise ValueError('group_columns must be a list')
    if cap in {'chart','chart_adjust'} and 'palette' in options:
        enum('palette',{'default','high_contrast'})
    if cap=='chart_adjust':
        if not options:raise ValueError('chart adjustment needs a requested display property')
        if 'bins' in options and (type(options['bins']) is not int or not 2<=options['bins']<=100):
            raise ValueError('bins must be an integer in 2..100')
        if 'y_max' in options:
            if len(options)!=1:raise ValueError('axis limit and presentation edits require separate tasks')
            v=options['y_max']
            if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('y_max must be positive finite')
    if cap=='statistics':
        enum('test',{'mann_whitney','chi_square','one_way_anova','paired_t','independent_t','mean_ci'})
        if not goal['columns']:raise ValueError('statistical columns required')
    if cap=='time_series':
        enum('frequency',{'hour','day','week','month'});enum('aggregation',{'count','sum','mean','median','min','max'})
        enum('gap_policy',{'omit','zero','nan'},False)
        if len(goal['columns'])!=2:raise ValueError('time series needs time and measure columns')
    if cap=='winsorization':
        column('column')
        low,high=options.get('lower_quantile'),options.get('upper_quantile')
        if type(low) not in (int,float) or type(high) not in (int,float) or not 0<low<=.25<.75<=high<1:
            raise ValueError('winsorization requires ordered quantiles')
    if cap=='outliers':
        column('column');enum('method',{'iqr','zscore','mad'});enum('tail',{'both','upper','lower'},False)
        enum('selection',{'outliers','inliers'},False)
        v=options.get('threshold')
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('outlier threshold required')
    if cap=='pivot':
        enum('aggregation',{'count','mean','sum','median','min','max','success_rate','overall_percent'})
        columns('index_columns');columns('column_columns')
        if options['aggregation'] not in {'count','overall_percent'}:column('value_column')
        if options['aggregation']=='success_rate' and 'success_value' not in options:raise ValueError('success value required')
        enum('sort',{'ascending','descending','none'},False)
    if cap=='group_summary':
        columns('columns')
        from utils.analysis_group_summary import AGGREGATIONS
        metrics=options.get('metrics')
        if not isinstance(metrics,list) or not 1<=len(metrics)<=8:raise ValueError('group metrics required')
        for metric in metrics:
            if (not isinstance(metric,dict) or set(metric)-{'name','aggregation','value_column','condition','empty_value'}
                    or not isinstance(metric.get('name'),str) or not metric['name']
                    or metric.get('aggregation') not in AGGREGATIONS):raise ValueError('invalid group metric')
    if cap=='latest_per_key':
        columns('key_columns');column('order_column');column('value_column')
        if type(options.get('categorical')) is not bool:raise ValueError('latest distribution type required')
        enum('null_policy',{'reject','drop_before_selection'},False)
        stage=options.get('filter_stage','')
        if (goal['conditions'] and stage not in {'before_selection','after_selection'}) or (not goal['conditions'] and stage):
            raise ValueError('latest filter requires an explicit before/after stage')
        if goal['any_conditions'] or goal['measure_conditions']:raise ValueError('latest task cannot simplify OR/measure conditions')
        if 'tie_break_columns' in options:
            v=options['tie_break_columns']
            if not isinstance(v,list) or any(not isinstance(c,str) or not c for c in v):raise ValueError('invalid tie breakers')
