"""Expose the same output contract to JSON-object and grammar providers.

Azure JSON-object mode guarantees JSON syntax, not our application fields.
Include its schema in the budgeted system input; keep validation at admission.
Repeated schema fragments are factored without dropping any constraints.
"""
from collections import Counter
from copy import deepcopy
import json
from langchain_core.messages import SystemMessage


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def schema_children(value, transform):
    """Visit schema positions only, never properties maps or const user data."""
    result=deepcopy(value)
    for key in ('properties', 'patternProperties', 'dependentSchemas'):
        if isinstance(result.get(key),dict):
            result[key]={k:transform(v) for k,v in result[key].items()}
    for key in ('items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else', 'propertyNames'):
        if isinstance(result.get(key),dict): result[key]=transform(result[key])
    for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
        if isinstance(result.get(key),list): result[key]=[transform(v) for v in result[key]]
    return result


def compact_schema(schema):
    if '$defs' in schema or 'definitions' in schema:
        return deepcopy(schema)
    def share_branches(value):
        if not isinstance(value, dict): return value
        value = schema_children(value,share_branches)
        branches = value.get('anyOf', [])
        # Hoist common object constraints and keep only varying properties in
        # branches. additionalProperties remains at the declaring object level.
        if (set(value) == {'anyOf'} and len(branches) > 1
                and all(isinstance(b,dict) and b.get('type')=='object'
                        and isinstance(b.get('properties'),dict) for b in branches)):
            first = branches[0]
            common = {k:v for k,v in first.items() if k!='properties'}
            keys = set(first['properties'])
            if all({k:v for k,v in b.items() if k!='properties'}==common
                   and set(b['properties'])==keys for b in branches):
                varying = {k for k in keys if any(b['properties'][k]!=first['properties'][k]
                                                for b in branches[1:])}
                return {**common, 'properties':{k:({} if k in varying else v)
                        for k,v in first['properties'].items()},
                        'anyOf':[{'properties':{k:b['properties'][k] for k in sorted(varying)}}
                                 for b in branches]}
        return value
    schema = share_branches(deepcopy(schema))
    counts = Counter()
    def count(value):
        if isinstance(value, dict):
            counts[encoded(value)] += 1
            schema_children(value,count)
        return value
    count(schema)
    repeated = {value for value, n in counts.items() if n > 1 and len(value.encode()) >= 120}
    definitions = {}
    names = {}
    def project(value, root=False):
        if isinstance(value, dict):
            key = encoded(value)
            if not root and key in repeated:
                if key not in names:
                    names[key] = 's'+str(len(names))
                    definitions[names[key]] = project(value, root=True)
                return {'$ref': '#/$defs/'+names[key]}
            return schema_children(value,project)
        return value
    result = project(schema, root=True)
    if definitions: result['$defs'] = definitions
    return result if len(encoded(result).encode()) < len(encoded(schema).encode()) else deepcopy(schema)


def invoke_role(model, request, budget):
    """Add Azure's contract before measuring or invoking any model request."""
    schema = getattr(model, '_telly_output_schema', None)
    if isinstance(schema, dict):
        instruction = ('JSON output contract: Return ONE JSON object conforming to the following '
            'JSON Schema, resolving $ref from $defs. Include every required field, even empty '
            'defaults. Use only allowed keys, enum/const values and task options. '
            'This is a format contract, not evidence of user intent.\n'+encoded(schema))
        system = request.system_message
        system = (system.model_copy(update={'content':str(system.content)+'\n'+instruction})
                  if system else SystemMessage(content=instruction))
        request = request.override(system_message=system, model=model)
    else:
        request = request.override(model=model)
    return budget.wrap_model_call(request, lambda r: model.invoke([r.system_message, *r.messages]))
