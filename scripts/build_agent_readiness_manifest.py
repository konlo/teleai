"""Build an auditable requirements/journeys/reference-case inventory, not scores.

Only category-level capability mapping is automatic. Oracle gaps and unresolved
meaning conflicts stay visible; mapping coverage is not implementation coverage.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_analysis_agent import load_grading, load_specs

REQUIREMENT_JOURNEYS = {
    'A01': ['J01', 'J19', 'J22'], 'A02': ['J05', 'J13', 'J22'],
    'A03': ['J04', 'J14', 'J17'], 'A04': ['J02', 'J03', 'J21', 'J24'],
    'A05': ['J03', 'J10', 'J22', 'J24'], 'A06': ['J07', 'J10', 'J16'],
    'A07': ['J07', 'J12', 'J16', 'J23'], 'A08': ['J01', 'J07', 'J13', 'J20'],
    'A09': ['J06', 'J07', 'J20'], 'D01': ['J08', 'J15'],
    'D02': ['J08', 'J24'], 'D03': ['J08', 'J20', 'J23'],
    'D04': ['J08', 'J15'], 'D05': ['J09', 'J18', 'J21'],
    'D06': ['J08', 'J15', 'J20'], 'D07': ['J22', 'J23'],
    'D08': ['J21', 'J22', 'J23', 'J24'], 'N01': ['J05', 'J19', 'J24'],
    'N02': ['J04', 'J05', 'J19', 'J21'], 'N03': ['J06', 'J16'],
    'V01': ['J02', 'J12', 'J16', 'J21'], 'V02': ['J08', 'J15'],
    'V03': ['J11', 'J13'], 'V04': ['J12'], 'V05': ['J13', 'J15'],
    'V06': ['J13', 'J16'], 'V07': ['J14', 'J17', 'J21'],
    'V08': ['J15', 'J19', 'J24'], 'O01': ['J09', 'J17', 'J20'],
    'O02': ['J07', 'J10', 'J20'], 'O03': ['J10', 'J17', 'J20', 'J23'],
    'E01': ['J01', 'J21', 'J22', 'J23', 'J24'],
    'E02': ['J01', 'J06', 'J13'], 'E03': ['J01', 'J04', 'J07', 'J13', 'J22'],
}

CATEGORIES = {
    '테이블 스키마 탐색': ['A01', 'D01'],
    '유사어 기반 컬럼 탐색': ['A01', 'N01'],
    '단순 필터링 및 텍스트/표 답변': ['A02', 'N01', 'N02'],
    '단일 그룹 집계 및 요약표': ['N01', 'N02'],
    '단일 차트 시각화': ['V01', 'V04'],
    '복합 조건 필터링 및 유사어 매핑': ['A01', 'A02', 'N01', 'N02'],
    '다차원 피벗 및 소계 집계': ['N01', 'N02', 'D06'],
    '이상치 탐지 및 조건부 분석': ['N02', 'N03', 'V01'],
    '통계적 가설 검정 및 추론': ['N03'],
    '캠페인 전환율 및 이중 축 시각화': ['N01', 'V01', 'V04'],
    '다중 패널 대시보드 및 고급 시각화': ['V01', 'V04', 'V08'],
    '예외 대응 및 스키마 경계 검증': ['A01', 'A06', 'A07', 'N02'],
}

CONFLICTS = {
    'L2_026': 'Age bins in prose/reference disagree; do not copy reference boundaries silently.',
    'L2_032': 'Passenger count versus count(Age); missing Age changes denominator.',
}

# Confirmed existing entry points cover parts of these journeys, not full acceptance.
PARTIAL_TESTS = {
    'J01': ['migration.test_completion_contracts.CompletionTests.test_schema_dtype_and_type_subsets_are_table_neutral_and_model_free'],
    'J02': ['migration.test_completion_contracts.CompletionTests.test_repeated_histogram_reuses_verified_png_without_model_call'],
    'J04': ['migration.test_completion_contracts.CompletionTests.test_followup_cannot_silently_drop_inherited_month'],
    'J07': ['migration.test_completion_contracts.CompletionTests.test_bad_sql_observation_allows_corrected_sql_without_restart'],
    'J10': ['migration.test_approval_rollout.LedgerTests.test_process_crash_after_claim_never_resubmits'],
    'J21': ['tests.test_data_preservation_acceptance.DataPreservationAcceptanceTests.test_derive_transform_aggregate_chart_restart_keeps_root'],
    'J22': ['migration.test_completion_contracts.CompletionTests.test_wrong_remote_scope_never_creates_approval_before_corrected_query'],
    'J23': ['tests.test_data_preservation_acceptance.DataPreservationAcceptanceTests.test_candidate_quota_failure_preserves_all_ready_assets'],
    'J24': ['tests.test_data_preservation_acceptance.DataPreservationAcceptanceTests.test_derive_transform_aggregate_chart_restart_keeps_root'],
}


def build():
    document = ROOT / 'docs/data_agent_requirements_and_evaluation_2026-09-24.md'
    lines = document.read_text().splitlines()
    requirements, journeys = [], []
    for line in lines:
        cells = [part.strip() for part in line.split('|')[1:-1]]
        if cells and re.fullmatch(r'[ADVNOE]\d{2}', cells[0]):
            requirements.append({'id': cells[0], 'priority': cells[1], 'requirement': cells[2],
                'acceptance': cells[3], 'journeys': REQUIREMENT_JOURNEYS[cells[0]],
                'verification_status': 'NOT_RUN', 'acceptance_readiness': 'MAPPED_NOT_COMPLETE'})
        if cells and re.fullmatch(r'J\d{2}', cells[0]):
            journeys.append({'id': cells[0], 'scenario': cells[1], 'acceptance': cells[2],
                'requirements': [key for key, values in REQUIREMENT_JOURNEYS.items() if cells[0] in values],
                'partial_test_entrypoints': PARTIAL_TESTS.get(cells[0], []),
                'full_acceptance_status': 'NOT_RUN',
                'gap': 'Full journey oracle/fault/UI coverage is not yet executable as one test.'})
    grading = load_grading()
    cases = []
    for spec in load_specs():
        cases.append({'id': spec['id'], 'category': spec['category'], 'type': spec['type'],
            'requirements': sorted(set(CATEGORIES[spec['category']] + ['A07', 'E01'])),
            'mapping_basis': 'definition_category; capability-level only',
            'prompt_sha256': sha256(spec['prompt'].encode()).hexdigest(),
            'reference_sha256': sha256(spec['python_code'].encode()).hexdigest(),
            'oracle_kind': grading.get(spec['id'], {}).get('kind'),
            'oracle_status': 'DEFINED' if spec['id'] in grading else 'UNGRADED',
            'semantic_conflict': CONFLICTS.get(spec['id']),
            'execution_status': 'NOT_RUN',
            'launch_support': 'UNDECIDED_UNTIL_ACCEPTANCE'})
    return {'schema_version': 1, 'scope': 'Implementation preparation, not release certification',
        'requirements': requirements, 'journeys': journeys, 'reference_cases': cases,
        'summary': {'requirements': len(requirements), 'journeys': len(journeys),
                    'reference_cases': len(cases), 'defined_reference_oracles': len(grading),
                    'ungraded_reference_cases': len(cases)-len(grading)},
        'core_slice': {'required_journeys': ['J02', 'J03', 'J04', 'J07', 'J09', 'J10', 'J21', 'J22', 'J23', 'J24'],
            'capabilities': ['schema inspection', 'approved bounded raw load', 'protected root reuse',
                'filter/derive', 'count/sum/mean/group', 'histogram/bar/line/scatter/boxplot',
                'root restoration/restart', 'approved remote aggregate'],
            'status': 'IMPLEMENTATION_TARGET_NOT_RELEASE_CLAIM',
            'deferred': ['arbitrary Python until isolation proven', 'advanced chart families without oracle',
                         'incremental merge without stable key/version', 'full Spider cloud benchmark']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tests/fixtures/agent_readiness_manifest.json')
    args = parser.parse_args(argv)
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
