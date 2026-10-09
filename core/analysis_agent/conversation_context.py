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
    confirmed = current.get('confirmed_analysis') or {}
    # An unfinished/failed request is not a new verified analytical subject.
    # Keep its pending intent elsewhere; omitted-source follow-ups must use
    # the last completed analysis, even after exhaustion or restart.
    previous = (confirmed if (current.get('explanation_only')
                             or (current.get('status') != 'complete'
                                 and confirmed.get('status') == 'complete'))
                else current)
    previous = previous or {}
    selected = context.selected_dataset_id if context else ''
    if ('selection_at_confirmation' in previous
            and previous['selection_at_confirmation'] != selected):
        return {}  # An explicit UI selection supersedes the earlier analysis.
    proof = previous.get('table_preview_evidence') or {}
    info = context.datasets.metadata.get(proof.get('dataset_id')) if context else None
    if (previous.get('status') == 'complete' and info is not None
            and proof.get('source') == info.source and proof.get('snapshot') == info.snapshot
            and list(proof.get('columns', [])) == list(info.columns)
            and previous.get('required_sources') == [info.source]
            and (previous.get('scope') or {}).get('sources') != [info.source]):
        # Repair older confirmed preview snapshots without changing the saved
        # analytical selection or trusting an assistant's free-form prose.
        previous = deepcopy(previous)
        previous['scope'] = {'sources': [info.source], 'conditions': [],
            'any_conditions': [], 'measure_conditions': [], 'ratio': None,
            'unresolved': [], 'columns': []}
        # Unknown predicates cannot be replaced by invented empty conditions.
        # An unfiltered wildcard preview has a mechanically checked WHERE-free
        # query; other older mismatches require explicit scope resolution.
        from sqlglot import parse_one
        from sqlglot.errors import SqlglotError
        from dataclasses import asdict
        try:
            tree = parse_one(info.query, read=context.sql_dialect) if info.query else None
            if tree is not None and tree.args.get('where'):
                previous['scope']['unresolved'] = ['이전 미리보기의 필터 범위를 확인해야 합니다.']
            elif tree is None and info.predicate_known:
                previous['scope']['conditions'] = [asdict(c) for c in info.conditions]
        except (ValueError,SqlglotError):
            previous['scope']['unresolved'] = ['이전 미리보기의 필터 범위를 확인해야 합니다.']
    return previous


def suspend_analysis(current, previous):
    from core.analysis_agent.completion import CONTRACTS
    for contract in CONTRACTS:
        current[contract.request_key] = None
    current.update(explanation_only=True, operations=[], operation_pending=False,
        kind=None, fresh_source_required=False, required_sources=[], required_columns=[],
        scope={'conditions':[], 'any_conditions':[], 'unresolved':[], 'columns':[]},
        previous_scope={}, confirmed_analysis=deepcopy(previous.get('confirmed_analysis') or previous))
