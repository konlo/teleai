"""Coalesce redundant declared obligations, without interpreting user prose."""
from copy import deepcopy


def normalize(goal,selection=None):
    if not isinstance(goal,dict) or not isinstance(goal.get('tasks'),list):return goal
    # The question is an output-only clarification field. Models sometimes
    # copy the current input there while declaring executable obligations.
    # Clear that irrelevant field; never change mode or result obligations.
    if goal.get('mode') in {'execute','explain'} and isinstance(goal.get('question'),str) and goal['question']:
        goal=deepcopy(goal);goal['question']=''
    tasks=goal['tasks']
    if any(not isinstance(t,dict) for t in tasks):return goal
    # Remove only redundant root fields already superseded by an independent
    # scalar subject. Never add a missing measure or alter grouping/predicates.
    measures=(selection or {}).get('output_columns',[])
    if (isinstance(goal.get('columns'),list) and measures and all(c in goal.get('columns',[]) for c in measures)
            and any(t.get('capability')=='calculation' for t in tasks)
            and all(t.get('capability') in {'calculation','chart'} for t in tasks)):
        used=set(measures)
        for task in tasks:
            options=task.get('options',{})
            if not isinstance(options,dict):return goal
            used.update(options.get('group_columns',[]))
            axes=options.get('axes',{})
            if isinstance(axes,dict):used.update(axes.values())
            if options.get('category'):used.add(options['category'])
        goal=deepcopy(goal);goal['columns']=[c for c in goal['columns'] if c in used]
    if goal.get('columns')==[] and len(tasks)==1 and tasks[0].get('capability')=='chart':
        options=tasks[0].get('options')
        axes=options.get('axes') if isinstance(options,dict) else None
        if isinstance(axes,dict) and not set(axes)-{'x','y'}:
            columns=[axes[k] for k in ('x','y') if isinstance(axes.get(k),str) and axes[k]]
            if columns:
                goal=deepcopy(goal);goal['columns']=list(dict.fromkeys(columns))
    # Nonempty conflicting columns remain invalid. No axis/subject is guessed.
    metadata=[t for t in tasks if isinstance(t,dict) and t.get('capability')=='metadata']
    if len(metadata)<2:return goal
    if any(set(t)!={'capability','options'} or not isinstance(t.get('options'),dict)
           or set(t['options'])!={'kind'} or not isinstance(t['options']['kind'],str) for t in metadata):return goal
    kinds={t['options']['kind'] for t in metadata}
    if not kinds.issubset({'columns','dtypes'}) or 'dtypes' not in kinds:return goal
    # One dtypes view contains every column name as well as its DB type. It
    # fulfills both obligations without discarding either requested output.
    result=deepcopy(goal);merged=[];added=False
    for task in result['tasks']:
        if task.get('capability')=='metadata':
            if not added:merged.append({'capability':'metadata','options':{'kind':'dtypes'}})
            added=True
        else:merged.append(task)
    result['tasks']=merged
    return result
