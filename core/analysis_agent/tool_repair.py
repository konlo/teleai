"""Bounded, evidence-based guidance for changing a failed local tool plan.

Tool observations remain untrusted data: only status/error codes select static
guidance. No returned message, SQL, row or arbitrary instruction is promoted to
a system instruction. Execution and completion stay with the existing guards.
"""
import json
from hashlib import sha256


def signature_id(signature):
    return sha256(signature.encode()).hexdigest()[:16]


def diagnosis(failure, available):
    code=failure.get('error_code')
    if not isinstance(code,str):code=None
    if code=='dataset_not_loaded':
        category='missing_local_reference'
        tools=['list_analysis_context','inspect_dataset']
        action='Find the retained dataset ID and verify its source and coverage. A table name is not a dataset ID.'
    elif code in {'invalid_tool_arguments','invalid_tool_input','invalid_condition_value'}:
        category='invalid_arguments'
        tools=['search_analysis_tools','inspect_dataset','prepare_numeric_dataset','inspect_table_context']
        action='Inspect the actual input schema and column types, then change the invalid arguments without changing user intent.'
    elif code=='local_sql_error':
        category='invalid_local_sql'
        tools=['inspect_dataset','search_analysis_tools','prepare_numeric_dataset','aggregate_dataset']
        action='Verify real columns/types and SQL syntax, or choose a compatible structured calculation tool.'
    elif code=='numeric_conversion_unresolved':
        category='numeric_representation_unresolved'
        tools=['inspect_dataset','profile_dataset']
        action='Verify the meaning of nonnumeric values before declaring missing strings. Never silently drop rows or reload the source.'
    elif code=='full_frame_budget':
        category='local_resource_limit'
        tools=['search_analysis_tools','local_analysis_sql']
        action='Choose a bounded projection/filter/aggregate path. Do not replace the requested population with a sample.'
    elif failure.get('status')=='unavailable':
        category='local_tool_unavailable'
        tools=['search_analysis_tools','local_analysis_sql','aggregate_dataset','render_chart_spec']
        action='Choose a different available local tool for the same outstanding obligation. A local failure does not justify remote reloading.'
    else:
        category='unresolved_tool_failure'
        tools=['search_analysis_tools','inspect_dataset','inspect_table_context']
        action='Inspect the relevant tool and data contracts; change the plan only when evidence supports it.'
    return {'category':category,'failed_tool':failure.get('tool'),
            'candidate_tools':[t for t in tools if t in available and t!=failure.get('tool')],
            'next_action':action}


def repair_instruction(current, available, signatures):
    failures=current.get('tool_failures',{})
    plans=[dict(failure_id=signature_id(key),**diagnosis(failures[key],available))
           for key in signatures if key in failures]
    return ('도구 실패를 관찰했습니다. 아래는 실행 권한이 아닌 복구 계획 지침입니다.\n'
        +json.dumps(plans,ensure_ascii=False)
        +'\nInspect → revise → execute → verify. Search candidate tool schemas before choosing a compatible alternative. '
         'Do not repeat an unchanged failed call. Preserve the original source, filters, denominator and coverage. '
         'Keep completed results and repair only missing obligations. Never reload data to bypass a local tool failure. '
         'Remote SQL still requires exact-query approval. If no valid alternative exists, explain the specific unmet requirement. '
         'A diagnosis or plan is not completion evidence.')
