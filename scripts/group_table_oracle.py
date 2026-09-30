"""Independent grouped result grading; no agent implementation import for expected values."""
import json
import numpy as np
import pandas as pd
from sqlglot import exp, parse_one


def result_columns(info, grading):
    """Bind metric meaning from recorded operations, never from result numbers."""
    operations={}
    if info.query.startswith('GROUP SUMMARY '):
        spec=json.loads(info.query[len('GROUP SUMMARY '):])
        if spec['group_columns'] != grading['groups']:raise ValueError('Wrong grouping')
        groups=list(spec['group_columns'])
        for metric in spec['metrics']:
            signature=(metric['aggregation'],metric.get('value_column',''))
            if signature in operations:raise ValueError('Duplicate metric')
            operations[signature]=metric['name']
    else:
        tree=parse_one(info.query,read='duckdb')
        # This grader only supports plain grouped reductions; complex SQL needs its own oracle.
        if not isinstance(tree,exp.Select) or tree.args.get('with_') or len(list(tree.find_all(exp.Select)))!=1:
            raise ValueError('Unsupported grouped SQL shape')
        group=tree.args.get('group')
        if not group or [c.name for c in group.expressions if isinstance(c,exp.Column)]!=grading['groups']:
            raise ValueError('Wrong grouping')
        groups=[]
        names={exp.Avg:'mean',exp.Median:'median',exp.Min:'min',exp.Max:'max',exp.Sum:'sum',exp.Count:'count'}
        for projection in tree.expressions:
            expression=projection.this if isinstance(projection,exp.Alias) else projection
            alias=projection.alias_or_name
            if isinstance(expression,exp.Column):groups.append(alias);continue
            if isinstance(expression,exp.Round):expression=expression.this
            kind=names.get(type(expression))
            if not kind or isinstance(expression.this,exp.Distinct):raise ValueError('Unsupported metric expression')
            column=expression.this.name if isinstance(expression.this,exp.Column) else ''
            if kind=='count' and column:
                # Count(nullable column) and row count are different contracts.
                if column not in grading.get('nonnull_count_columns',[]):raise ValueError('Unverified count column')
                column=''
            signature=(kind,column)
            if signature in operations:raise ValueError('Duplicate metric')
            operations[signature]=alias
    columns=groups[:]
    if len(groups)!=len(grading['groups']):raise ValueError('Wrong group output')
    for metric in grading['metrics']:
        signature=(metric['aggregation'],metric.get('column',''))
        if signature not in operations:raise ValueError('Requested metric omitted')
        columns.append(operations[signature])
    return columns


def compare(frame, info, grading, expected):
    columns=result_columns(info,grading)
    if not set(columns)<=set(frame.columns) or len(frame)!=len(expected):return False
    actual=frame[columns].copy();actual.columns=expected.columns
    if grading.get('sort_metric'):
        column=grading['sort_metric']; ascending=grading.get('ascending',True)
        sorted_ok=actual[column].is_monotonic_increasing if ascending else actual[column].is_monotonic_decreasing
        if not sorted_ok:
            return False
    groups=grading['reference_groups']
    if actual.duplicated(groups).any():return False
    actual=actual.sort_values(groups).reset_index(drop=True)
    expected=expected.sort_values(groups).reset_index(drop=True)
    if not actual[groups].equals(expected[groups]):return False
    for metric in grading['metrics']:
        name=metric['reference'];digits=metric.get('round',8)
        if not np.allclose(actual[name].astype(float).round(digits),expected[name].astype(float),rtol=1e-8,atol=10**(-digits)/2,equal_nan=True):return False
    return True


def replay_summary(parent, query):
    """Replay recorded declarative reductions in a fresh store on counterfactual input."""
    from utils.analysis_datasets import DatasetStore
    from utils.analysis_group_summary import summarize_groups
    spec=json.loads(query[len('GROUP SUMMARY '):])
    store=DatasetStore();root=store.register(parent,source='counterfactual',coverage='complete',predicate_known=True)
    result=summarize_groups(store,root.id,**spec)
    return store.frames[result['dataset']['id']]
