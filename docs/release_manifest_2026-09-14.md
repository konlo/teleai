# TeleAI agentic analysis release candidate — 2026-09-14

최신 고정 참조: Git tag `agentic-analysis-rc3-2026-09-14`. 이전 기준선과 RC2는 보존한다.

## 포함 범위

- LangGraph 기반 영속 agent loop, 완료 증거 검증과 제한된 복구
- Databricks SQL별 사용자 승인, 중복 실행 방지와 실패 보존
- 동적 테이블·스키마 계약 및 현재 DataFrame 재사용/최신 원본 구분
- 기본 count/avg/sum/min/max/ratio와 AND·IN·한 개 OR 그룹
- histogram/bar/line/scatter/단일 수치 boxplot과 실제 PNG 증거
- 대화, DataFrame, 차트, 승인 ledger의 재시작 복원
- 원격 결과·컬럼·메모리 cache·대화 저장공간·보존 후보 운영 한도
- turn SLO, 모델·도구 시간, process peak RSS와 cache 바이트 진단

## 고정 검증

- migration: 111/111 PASS
- tests: 76/76 PASS
- 전체: 187/187 PASS
- compileall PASS
- `git diff --check` PASS
- 변경 후보 32개 파일의 토큰·비밀키 패턴 검사: 0건
- 실제 Databricks: 승인 전 0회, 승인 후 1회, 10,000행·18열 저장
- 실제 화면: 현재 표본 histogram 0.114초, 모델 0회, 원격 0회, PNG 표시
- 실제 10,000행 복제본 30회: p50 0.074초, p95 0.146초, 최대 0.185초
- 동시 로컬 histogram 20회·4 workers: 20/20 PASS, p95 0.562초, 6.864 req/s
- 피어슨 상관계수: 실제 모델 5/5 PASS·p95 80.704초, 결정적 로컬 전이 후 30/30 PASS·p95 0.046초
- 실제 화면 현재 표본 상관계수: 0.05918371025192562, 0.224초, 모델 0회, 원격 0회
- Level 3 production agentic recovery 계약: 17/17 PASS, placeholder 85개 제거

## 출시 판정

보유 데이터 재사용, 메타데이터, 검증된 기본 집계·필터·차트를 포함한 제한 범위의 release candidate다. 200문항 전체와 조인·가설 검정·고급 복합 차트의 일반 출시는 승인하지 않는다.

RC3 tag와 branch는 원격 `origin`에 push했고 draft PR #68을 열었다. 리뷰·병합, 배포와 배포 환경 smoke test는 별도 게이트다.

PR에는 비밀키나 Databricks 연결 없이 pinned 환경에서 migration·tests·Level 3·전체 참조 runner·compileall을 실행하는 `.github/workflows/agent-release-gate.yml`을 추가했다. 첫 원격 실행 `34846840045`는 모든 step이 성공했다.

상세 근거: [출시 수용 기록](release_acceptance_2026-09-13.md), [Databricks 승인 여정](databricks_approval_journey_2026-09-14.md), [성능·메모리 측정](runtime_performance_2026-09-14.md).
