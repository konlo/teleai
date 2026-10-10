"""Requested intent survives execution failure; it is never result evidence."""
from copy import deepcopy

KEYS=('request_id','request_text','required_sources','required_columns','scope',
      'kind','chart_axes','chart_group_spec','chart_presentation_spec')


def unfinished(previous):
    if previous.get('status')=='complete':return None
    goal=previous.get('goal') or {}
    if (previous.get('intent_origin')=='llm' and not previous.get('goal_pending')
            and not previous.get('goal_interpretation_error') and goal.get('mode')=='execute'
            and previous.get('required_sources') and not (previous.get('scope') or {}).get('unresolved')):
        intent={k:deepcopy(previous[k]) for k in KEYS if k in previous}
        intent['status']='interpreted_not_completed'
        return intent
    return deepcopy(previous.get('requested_analysis')) or None


def prior_intent(current, result_only=False):
    """Resolve meaning from pending intent, execution facts from confirmed state."""
    reference=current.get('requested_output_reference')
    if result_only and reference:
        return {**reference,'status':'complete'}
    pending=current.get('requested_analysis') or {}
    subject=(current.get('requested_subject') or {}).get('sources')
    if (not result_only and pending.get('required_sources')
            and (not subject or subject==pending['required_sources'])):
        return pending
    return current.get('confirmed_analysis') or {}


def model_view(current):
    pending=current.get('requested_analysis') or {}
    return {**{k:deepcopy(pending[k]) for k in KEYS if k in pending},
            'status':'interpreted_not_completed; NOT execution or dataset evidence'} if pending else None
