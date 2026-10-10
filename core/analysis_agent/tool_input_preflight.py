"""Validate declared tool inputs before population checks can mislabel them."""
from itertools import islice
import json
from jsonschema import Draft202012Validator


def input_error(call, schemas):
    schema=schemas.get(call.get('name'))
    if not schema:return None
    arguments=call.get('args') or {}
    props=schema.get('properties',{});required=set(schema.get('required',[]))
    if isinstance(arguments,dict):
        # Match the executor's existing optional-null/default convention.
        arguments={k:v for k,v in arguments.items() if not (v is None and k not in required and k in props
            and 'null' not in (props[k].get('type') if isinstance(props[k].get('type'),list) else [props[k].get('type')]))}
    errors=list(islice(Draft202012Validator(schema).iter_errors(arguments),4))
    if not errors:return None
    issues=[{'path':list(e.absolute_path),'rule':e.validator,
             'received':e.instance if isinstance(e.instance,(str,int,float,bool,type(None))) and len(str(e.instance))<=250
                        else json.dumps(e.instance,ensure_ascii=False,default=str)[:250],
             **({'expected':e.validator_value} if e.validator in {'type','required','enum','minimum','maximum','additionalProperties','oneOf','anyOf'} else {})}
            for e in errors]
    contract={k:{a:v for a,v in p.items() if a in {'type','enum','minimum','maximum'}} for k,p in props.items()}
    return {'error_code':'invalid_tool_arguments','failure_stage':'tool_input_validation',
            'tool':call['name'],'validation_issues':issues,'required_fields':sorted(required),
            'input_fields':contract,
            'message':'Correct the rejected field values using validation_issues received/expected and the declared input_fields. Preserve the task, source and filters. '
                      +('For local_analysis_sql use dataset_id and query (SELECT ... FROM data); table names are not local table bindings.'
                        if call['name']=='local_analysis_sql' else 'Do not repeat the same rejected arguments.')}
