"""Coalesce redundant declared obligations, without interpreting user prose."""
from copy import deepcopy


def normalize(goal):
    if not isinstance(goal,dict) or not isinstance(goal.get('tasks'),list):return goal
    tasks=goal['tasks']
    if any(not isinstance(t,dict) for t in tasks):return goal
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
