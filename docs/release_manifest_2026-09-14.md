# TeleAI agentic analysis release candidate — 2026-09-14

최신 고정 참조는 tag `agentic-analysis-rc6-2026-09-16`, commit `6defd2a`다. GitHub Actions run `35029959063`, job `104585889158`이 성공한 뒤 같은 코드 commit에 고정했다. 이전 기준선과 RC2·RC3·RC4·RC5는 보존한다.

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

- migration: 125/125 PASS
- tests: 78/78 PASS
- 전체: 203/203 PASS
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
- 배포 preflight: 단위 계약 6/6 PASS, 현재 local-desktop READY, multi-user 차단
- 로컬 용량 gate: 단일 p95 0.162초, 4 worker·20/20 동시 p95 0.549초, peak RSS 약 361 MiB, READY
- 실제 agent 독립 oracle: 37/200 PASS. `L1_001` metadata는 0.161초, 모델·원격 0회
- 실제 화면 보유 스키마: 전체 18개 컬럼, 0.160초, 모델 0회, 로컬 도구 1회, 추가 조회 0회
- 실제 화면 보유 10,000행 복합 AVG·MAX: 평균 40.931, 최대 86, 0.177초, 모델 0회, 로컬 도구 1회, 추가 조회 0회

## 출시 판정

보유 데이터 재사용, 메타데이터, 검증된 기본 집계·필터·차트를 포함한 제한 범위의 release candidate다. 200문항 전체와 조인·가설 검정·고급 복합 차트의 일반 출시는 승인하지 않는다.

RC3·RC4·RC5 tag는 원격에 보존돼 있고 draft PR #68은 RC6 변경을 포함한다. RC6는 복합 집계, 배포 preflight, 컬럼 metadata 복구와 용량 gate를 포함한다. 리뷰·병합, 배포와 배포 환경 smoke test는 별도 게이트다.

PR에는 비밀키나 Databricks 연결 없이 pinned 환경에서 migration·tests·Level 3·전체 참조 runner·compileall을 실행하는 `.github/workflows/agent-release-gate.yml`을 추가했다. 첫 원격 실행 `34846840045`는 모든 step이 성공했다.

`scripts/deployment_preflight.py`는 네트워크나 SQL을 실행하지 않고 Ollama·Databricks 설정, 코드와 분리된 영속 저장소, 운영 한도와 사용자 범위를 검사한다. 현재 revision은 `local-owner`를 사용하므로 로컬 또는 접근제어된 1인용 배포만 허용하며 multi-user profile은 실패한다. 실제 플랫폼과 사용자 범위를 정한 뒤 [제한 배포 계약](deployment_contract_2026-09-15.md)의 smoke/rollback을 실행해야 한다.

상세 근거: [출시 수용 기록](release_acceptance_2026-09-13.md), [Databricks 승인 여정](databricks_approval_journey_2026-09-14.md), [성능·메모리 측정](runtime_performance_2026-09-14.md).
