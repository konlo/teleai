"""Bound visible decision evidence to user text or observed context, never CoT.

Literal support is necessary evidence, not proof of linguistic entailment. The
semantic role still judges meaning; execution independently verifies results.
"""
from copy import deepcopy
from hashlib import sha256
import json

FIELD = 'decision_evidence'
ORIGINS = ('request', 'verified_previous', 'requested_previous_analysis',
           'selected_dataset', 'tables', 'requested_previous_subject',
           'visible_output_references', 'schema_subject')
SCHEMA = {'type': 'array', 'maxItems': 160, 'items': {
    'type': 'object', 'additionalProperties': False,
    'required': ['path', 'origin', 'quote', 'reference'], 'properties': {
        'path': {'type': 'string'}, 'origin': {'type': 'string', 'enum': list(ORIGINS)},
        'quote': {'type': 'string', 'maxLength': 384},
        'reference': {'type': 'string', 'maxLength': 256}}}}

INSTRUCTIONS = '''decision_evidence is an OBJECT keyed by exact goal JSON pointers;
each value is {origin,quote,reference}. It contains support, NOT hidden reasoning.
Provide one entry for EVERY task (/tasks/0 etc), source (/sources/0), column
(/columns/0), predicate (/conditions/0, /any_conditions/0, /measure_conditions/0),
non-null /ratio, and true /current_result_only or /fresh_source_required.
Each task must have origin=request, quote copied verbatim from CURRENT request,
reference="". Check that the quoted request positively supports that task and all
its options, including ordering, window, operations, negation and exclusions.
For other decisions use origin=request with a verbatim supporting quote, OR an
observed context origin with quote="" and a JSON pointer reference within that
context whose value EXACTLY matches the decision. For example a retained
predicate may reference /scope/conditions/0 in verified_previous. Never cite
stale context that CURRENT overrides. Missing information requires clarification,
not fabricated quotes/context.
Use the exact keys/choices from the grammar. Explain/clarify uses {}.
This support is not execution evidence or permission.'''


def schema_with_evidence(schema,goal=None,data=None):
    schema = deepcopy(schema)
    schema['properties'][FIELD] = deepcopy(SCHEMA)
    if goal is not None:
        paths=required_paths(goal)
        bindings={}
        for path in paths:
            binding=deepcopy(SCHEMA['items'])
            binding['properties'].pop('path')
            binding['required'].remove('path')
            # Required object keys work across provider JSON grammars without
            # depending on support for array prefixItems/unique path strings.
            if path.startswith('/tasks/'):
                binding['properties']['origin']={'const':'request'}
                binding['properties']['reference']={'const':''}
            if data is not None:
                value=pointer(goal,path);request=data['request']
                quote=value if isinstance(value,str) and value and value in request else request
                current=deepcopy(binding)
                current['properties']['origin']={'const':'request'}
                current['properties']['reference']={'const':''}
                if len(quote)<=384:current['properties']['quote']={'const':quote}
                choices=[current]
                if not path.startswith('/tasks/'):
                    for origin in ORIGINS[1:]:
                        matches=references(data.get(origin),value)
                        if not matches:continue
                        inherited=deepcopy(binding)
                        inherited['properties']['origin']={'const':origin}
                        inherited['properties']['quote']={'const':''}
                        inherited['properties']['reference']={'type':'string','enum':matches}
                        choices.append(inherited)
                binding={'anyOf':choices}
            bindings[path]=binding
        schema['properties'][FIELD]={'type':'object','additionalProperties':False,
                                     'required':paths,'properties':bindings}
    schema['required'] = list(dict.fromkeys([*schema['required'], FIELD]))
    return schema


def references(context,target):
    """Enumerate exact observed values, not interpretations of user language."""
    found=[];target=canonical(target)
    def walk(value,path,depth):
        if len(found)>=8 or depth>8:return
        if path and canonical(value)==target:found.append(path)
        if isinstance(value,dict):
            for key,item in value.items():
                walk(item,path+'/'+str(key).replace('~','~0').replace('/','~1'),depth+1)
        elif isinstance(value,list):
            for i,item in enumerate(value):walk(item,path+'/'+str(i),depth+1)
    walk(context,'',0)
    return found


def required_paths(goal):
    if goal.get('mode') != 'execute':
        return []
    paths = [f'/{key}/{index}' for key in ('tasks', 'sources', 'columns',
             'conditions', 'any_conditions', 'measure_conditions')
             for index in range(len(goal.get(key, [])))]
    if goal.get('ratio') is not None:
        paths.append('/ratio')
    paths.extend('/'+key for key in ('current_result_only', 'fresh_source_required') if goal.get(key))
    return paths


def pointer(value, path):
    if not isinstance(path, str) or not path.startswith('/') or len(path) > 256:
        raise ValueError('Decision evidence needs a JSON pointer')
    try:
        for part in path[1:].split('/'):
            part = part.replace('~1', '/').replace('~0', '~')
            if isinstance(value, list):
                if not part.isdigit(): raise ValueError('Invalid array reference')
                value = value[int(part)]
            elif isinstance(value, dict):
                value = value[part]
            else:
                raise ValueError('Reference is not a container')
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError('Decision evidence refers to missing context') from exc
    return value


def canonical(value):
    # JSON distinguishes booleans from numeric values, unlike Python equality.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def verify(goal, entries, data):
    required = set(required_paths(goal))
    if isinstance(entries,dict):
        converted=[]
        for path,item in entries.items():
            if not isinstance(item,dict) or set(item)!={'origin','quote','reference'}:
                raise ValueError('Invalid decision evidence fields')
            converted.append({'path':path,**item})
        entries=converted
    if not isinstance(entries, list) or len(entries) > 160:
        raise ValueError('decision_evidence must be a bounded array')
    seen = set()
    for item in entries:
        if not isinstance(item, dict) or set(item) != {'path', 'origin', 'quote', 'reference'}:
            raise ValueError('Invalid decision evidence fields')
        path, origin, quote, reference = (item[k] for k in ('path', 'origin', 'quote', 'reference'))
        if not isinstance(path, str) or path not in required or path in seen:
            raise ValueError('Unexpected or repeated decision evidence path')
        if origin not in ORIGINS or not isinstance(quote, str) or len(quote) > 384:
            raise ValueError('Invalid decision evidence origin/quote')
        value = pointer(goal, path)
        if origin == 'request':
            if reference != '' or not quote.strip() or quote not in data['request']:
                raise ValueError('Decision quote must occur verbatim in CURRENT request: '+path)
        else:
            if path.startswith('/tasks/') or quote != '':
                raise ValueError('Output tasks require CURRENT request support')
            if canonical(pointer(data.get(origin), reference)) != canonical(value):
                raise ValueError('Inherited decision differs from referenced context: '+path)
        seen.add(path)
    if seen != required:
        raise ValueError('Missing decision evidence: '+', '.join(sorted(required-seen)))
    return {'status': 'supported', 'request_sha256': sha256(data['request'].encode()).hexdigest(),
            'goal_sha256': sha256(canonical(goal).encode()).hexdigest(),
            'bindings': deepcopy(entries), 'linguistic_entailment_proven': False}


def bound_to_current(current):
    support=current.get('goal_decision_evidence') or {}
    request=current.get('request_text')
    return (support.get('status')=='supported' and isinstance(request,str)
            and support.get('request_sha256')==sha256(request.encode()).hexdigest()
            and support.get('executable_goal_sha256')==sha256(canonical(current.get('goal')).encode()).hexdigest())
