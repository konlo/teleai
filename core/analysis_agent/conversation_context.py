"""Separate verified analysis state from a temporary explanatory turn."""
from copy import deepcopy
import re


def explanation_only(text):
    if re.search(r'(?:그려줘|그려\s*줘|계산해줘|생성해줘|실행해줘)', text):
        return False
    return bool(re.search(r'설명|차이|개념|\bexplain\b', text, re.I)
        and re.search(r'설명만|차이만|(?:계산|생성|실행).{0,24}(?:하지\s*마|하지\s*말|금지)|\bexplain\s+only\b', text, re.I))


def snapshot(current, context):
    saved = deepcopy({k:v for k,v in current.items() if k != 'confirmed_analysis'})
    saved['selection_at_confirmation'] = context.selected_dataset_id if context else ''
    return saved


def prior_analysis(current, context):
    previous = current.get('confirmed_analysis') if current.get('explanation_only') else current
    previous = previous or {}
    selected = context.selected_dataset_id if context else ''
    if ('selection_at_confirmation' in previous
            and previous['selection_at_confirmation'] != selected):
        return {}  # An explicit UI selection supersedes the earlier analysis.
    return previous


def suspend_analysis(current, previous):
    from core.analysis_agent.completion import CONTRACTS
    for contract in CONTRACTS:
        current[contract.request_key] = None
    current.update(explanation_only=True, operations=[], operation_pending=False,
        kind=None, fresh_source_required=False, required_sources=[], required_columns=[],
        scope={'conditions':[], 'any_conditions':[], 'unresolved':[], 'columns':[]},
        previous_scope={}, confirmed_analysis=deepcopy(previous.get('confirmed_analysis') or previous))
