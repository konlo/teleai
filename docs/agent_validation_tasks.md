# 분석 agent 전환 작업 현황

최신 상태: 2026-09-07. 새 agent를 기본 실행 경로로 연결했습니다. 개발용 연결 검사에는 별도 사용자 승인을 요구하지 않습니다. 제품 agent가 데이터를 로딩·재로딩할 때의 개별 조회 승인은 유지합니다.

| Task | 작업 | 상태 | 증거 / 남은 조건 |
|---|---|---|---|
| T01–T02 | 기준선·외부 정답 fixture·평가 도구 | 완료 | 기존 회귀 및 통계 정답 비교 |
| T03–T05 | 승인·재사용·대화 단절 진단 | 완료 | 초기 실패 기록은 analysis_acceptance_review.md에 보존 |
| T06a | 도구와 runtime 경계 분리 | 완료 | 독립 ToolContext와 controller |
| T06b | 격리 LangChain v1 환경 | 완료 | 버전 lock과 pip check 통과 |
| T06c | 영속 graph·DataFrame·차트 | 완료 | 프로세스 재생성 후 복원 검증 |
| T07 | 승인 ledger·HITL·UI | 구현 및 계약 검증 완료 | 변경·거절·중복 승인·불명 제출 차단 |
| T08a | 로컬 회귀·실제 모델·긴 문맥 | 검증 완료 | 기존 32개, 새 환경 18개; 실제 4턴 및 요약 후 계산 통과 |
| T08b | 실제 Databricks 연결·조회 검증 | 연결 검사 실행, 조회 검증 미완료 | OpenSession HTTP 403, 진단 SELECT 실행 0회 |
| T08c | 기본 실행 경로 전환 | 구현 완료 | main.py → ui/analysis_page.py, scripts/run_telly.py; 구형 호환 경로 보존 |

T08 전체 수용 검증 완료를 선언하지 않습니다. 현재 자격 증명/권한으로 Databricks 세션이 열리지 않으며, 실사용 대용량·지연시간 기준은 아직 확인하지 못했습니다.

실행 방법은 [README](../README.md), 상세 결과는 [검증 보고서](t07_t08_validation.md)에 있습니다. 이전 문서의 개발 조회 ‘사용자 승인 대기’와 ‘자동 요약 미구현’ 상태는 이 문서로 대체합니다.
