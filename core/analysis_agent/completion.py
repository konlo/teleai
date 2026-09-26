"""Single contract for completion, result rendering and targeted repair feedback.

Evidence keys contain observations already accepted by RecoveryMiddleware's
scope/lineage validators. A tool's `ready` status alone never populates them.
"""
from dataclasses import dataclass
from typing import Callable

from core.analysis_agent import completion_renderers as renderers


@dataclass(frozen=True)
class CompletionContract:
    name: str
    request_key: str
    evidence_key: str
    tools: tuple[str, ...]
    render: Callable
    evidence_check: Callable | None = None

    def __post_init__(self):
        if not all((self.name, self.request_key, self.evidence_key, self.tools)) or not callable(self.render):
            raise ValueError('A capability requires intent, evidence, tools and a result renderer')
        if self.evidence_check is not None and not callable(self.evidence_check):
            raise ValueError('Evidence check must be callable')

    def satisfied(self, current):
        return bool(self.evidence_check(current) if self.evidence_check
                    else current.get(self.evidence_key))


def _outlier_aggregate_ready(current):
    evidence = current.get('outlier_aggregate_evidence') or {}
    mode = current.get('outlier_aggregate_mode')
    needed = {'comparison' if mode == 'grouped_comparison' else 'grouped'}
    if mode == 'top_frequency' and current.get('outlier_metric_aggregation'):
        needed.add('overall')
    return all(evidence.get(key) for key in needed)


def _count_rate_ready(current):
    return bool(current.get('count_rate_evidence') and current.get('artifact_ids'))


# Adding a capability makes it participate in all four consumers automatically:
# early completion, final exit validation, rendering and missing-evidence repair.
CONTRACTS = (
    CompletionContract('preview', 'preview_limit', 'preview_evidence', ('inspect_dataset',), renderers.render_preview),
    CompletionContract('data_load', 'data_load', 'load_evidence_id', ('query_databricks',), renderers.render_data_load),
    CompletionContract('metadata', 'metadata_kind', 'metadata_evidence', ('inspect_table_context',), renderers.render_metadata),
    CompletionContract('profile', 'profile_kind', 'profile_evidence', ('profile_dataset',), renderers.render_profile),
    CompletionContract('join', 'join', 'join_evidence', ('join_datasets',), renderers.render_join),
    CompletionContract('time_series', 'time_series_frequency', 'time_series_evidence', ('prepare_time_series',), renderers.render_time_series),
    CompletionContract('statistics', 'statistical_kind', 'statistical_evidence', ('statistical_test',), renderers.render_statistics),
    CompletionContract('winsorization', 'winsor_spec', 'winsor_evidence', ('winsorize_numeric',), renderers.render_winsorization),
    CompletionContract('pivot', 'pivot_requested', 'pivot_evidence', ('pivot_dataset',), renderers.render_pivot),
    CompletionContract('group_summary', 'group_summary_requested', 'group_summary_evidence', ('summarize_groups',), renderers.render_group_summary),
    CompletionContract('outliers', 'outlier_spec', 'outlier_evidence', ('detect_outliers', 'select_outlier_rows'), renderers.render_outliers),
    CompletionContract('outlier_aggregate', 'outlier_aggregate_requested', 'outlier_aggregate_evidence', ('aggregate_dataset', 'compare_group_aggregates'), renderers.render_outlier_aggregate, _outlier_aggregate_ready),
    CompletionContract('count_rate', 'count_rate_layout', 'count_rate_evidence', ('render_count_rate_chart',), renderers.render_chart, _count_rate_ready),
    CompletionContract('chart', 'chart', 'artifact_ids', ('render_chart_spec', 'recommend_chart_images', 'render_histogram', 'prepare_histogram', 'show_chart', 'render_count_rate_chart'), renderers.render_chart),
    CompletionContract('calculation', 'calculation', 'evidence_ids', ('local_analysis_sql', 'aggregate_dataset', 'query_databricks'), renderers.render_calculation),
)
if len({contract.name for contract in CONTRACTS}) != len(CONTRACTS):
    raise ValueError('Duplicate completion capability')


def active_contracts(current):
    return tuple(contract for contract in CONTRACTS if current.get(contract.request_key))


def missing_contracts(current):
    return tuple(contract for contract in active_contracts(current) if not contract.satisfied(current))


def completion_ready(current):
    active = active_contracts(current)
    return all(contract.satisfied(current) for contract in active) if active else not current.get('failed')


class CompletionOutputError(ValueError):
    def __init__(self, capability, error_type):
        super().__init__(f'Cannot render verified output for {capability}: {error_type}')
        self.capability, self.error_type = capability, error_type


def render_completion(runtime, current):
    parts, rendered = [], set()
    for contract in active_contracts(current):
        if not contract.satisfied(current):
            raise CompletionOutputError(contract.name, 'missing_evidence')
        if contract.render in rendered:
            continue
        try:
            text = contract.render(runtime, current)
        except (KeyError, ValueError, TypeError, OSError, IndexError) as error:
            raise CompletionOutputError(contract.name, type(error).__name__) from error
        if not isinstance(text, str) or not text.strip():
            raise CompletionOutputError(contract.name, 'empty_output')
        parts.append(text)
        rendered.add(contract.render)
    return '\n\n'.join(parts)


def recovery_instruction(current):
    missing = missing_contracts(current)
    tasks = '\n'.join(f'- {c.name}: 검증된 {c.evidence_key} 필요; 사용 가능한 도구: {", ".join(c.tools)}'
                      for c in missing)
    return ('이전 응답은 완료 증거가 없어 채택되지 않았습니다. 원래 요청에서 남은 작업을 수행하세요.\n'
            + tasks + '\n'
            '도구의 ready 표시만으로 완료하지 말고 요청 출처·컬럼·조건·집계·범위에 맞는 결과를 확인하세요. '
            '이미 검증된 결과와 원본을 보존하고 미완료 작업만 복구하세요. '
            '컬럼이나 관계 정보가 부족하면 inspect_table_context 또는 inspect_dataset으로 확인하세요. '
            '원격 데이터가 필요한 경우 query_databricks로 정확한 SQL 승인을 요청하고 사용자 승인을 기다리세요. '
            '대체 도구나 올바른 인자 schema가 필요하면 search_analysis_tools로 기능 또는 도구명을 검색하세요. '
            '같은 실패 호출을 반복하거나 검증되지 않은 수치·차트를 완료했다고 말하지 마세요.')
