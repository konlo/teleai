# 남은 작업 목록 — 2026-09-22

기준 시각: 2026-09-22 KST. 상세 평가와 근거는 [chatbot agentic 평가](chatbot_agentic_evaluation_2026-09-13.md)에 기록했다.

| ID | 우선순위 | 작업 | 상태 | 완료 증거 또는 남은 조건 |
|---|---|---|---|---|
| R01 | P0 | 요청 조건과 계산·차트 범위 검증 | 완료 | 잘못된 로컬·원격·차트 범위 실행 전 차단, 후속 조건과 재시작 보존 검증 |
| R02 | P0 | 실제 모델 실패 수정·재검증 | 완료 | Level 1 지원 20/20, Level 2 지원 4/4 실제 agent PASS |
| R03 | P0 | 기존 긴 대화와 실제 화면 검증 | 완료 | 최신 서버에서 같은 `age` histogram 요청이 기존 PNG와 750,000건 범위를 표시. 모델·원격 조회 0회, 승인 카드 0개 |
| R04 | P0 | 최종 회귀와 서버 반영 | 완료 | 최신 회귀 205/205, compileall·diff PASS. 서버 실제 화면에서 18개 컬럼 전체를 0.160초에 표시, 모델 0회·로컬 도구 1회·추가 조회 0회 |
| R05 | P1 | 응답 지연과 실행 한도 | 부분 완료 | 초기 용량 gate READY: 단일 p95 0.162초, 4 worker·20/20 p95 0.549초, peak RSS 약 361 MiB. 1 GiB/768 MiB 경고/896 MiB 위험 계약과 preflight 검증 적용. 실제 배포 호스트 재측정 남음 |
| R06 | P1 | 대용량 DataFrame·캐시·저장소 | 검증 완료 | 750,000행 복구와 함께 100k행·256열·512MiB·대화별 2GiB 한도, 30일 정리 후보 보고 적용. 자동 삭제 없음 |
| R07 | P2 | 실제 agent 독립 채점 확대 | 진행 중 | 누적 72/200문항 지원. 변경하지 않은 `L2_061`·`L2_064`·`L2_065`·`L2_066`의 그룹별 건수·성공률 이중축이 실제 PNG·분모·성공수·digest와 독립 reference로 PASS. 모델·원격 0회이며 나머지 128문항은 PASS로 계산하지 않음 |
| R08 | P2 | 고급 분석·차트·스킬 범위 | 진행 중 | 전체 행·IN·공통조건+OR·백분율·피어슨 상관, 5종 차트와 그룹 박스플롯, 숫자축 빈도 선·영문 월 누적 곡선, bounded join, 6개 구조화 통계 방법, 이상치 탐지와 bounded cohort 후속 계산을 검증. `prepare_time_series`, `aggregate_dataset`, `compare_group_aggregates`, `render_count_rate_chart`로 datetime 재집계, cohort 비교와 그룹별 건수·성공률 dual-axis/split-panel을 검증. winsorization과 범용 다중 패널 대시보드는 미검증 |
| R09 | P1 | 운영 승인 여정과 출시 판정 | 완료 | 실제 승인·1회 조회·10,000행 저장·로컬 분석 재사용 완료. RC3 보존, 원격 gate 성공한 `cd40480`에 RC4 tag 고정·push |
| R10 | P0 | 특정 테이블·스키마 의존 제거 | 완료 | production 테이블 하드코딩 제거, stale schema 차단, 동적 loader, schema fingerprint, 최신 원본 캐시 우회와 승인 대기 검증 |
| R11 | P0 | 제한 출시 후보 승격 | 진행 중 | RC7 tag `e31150f`, draft PR #68, GitHub run `35095497759` 성공. secret-safe preflight와 1인용 배포·용량·smoke·rollback 계약 준비. 대상 확정, 코드 리뷰/병합과 실제 환경 검증 남음 |
| R12 | P1 | 운영 TableContext 준비 | 부분 완료 | bank_loan 18/18·titanic 12/12 alias 준비. 네 테이블 모두 stale여서 각각 승인형 schema refresh 필요 |
| R13 | P1 | 평가 환경 정비 | 완료 | SciPy/NumPy 호환 수정 후 Level 1·2 참고 코드 200/200 PASS. 임의 placeholder를 제거하고 production agentic recovery 계약 17/17 PASS를 `--level 3`에 연결 |
| R14 | P0 | agent tool 계약·기능 공백 보강 | P0·P1 핵심 완료 | 공통 출력 schema, `profile_dataset`, 승인형 source discovery, bounded chart·join·`statistical_test`·`detect_outliers`·`select_outlier_rows` 완료 |

## 바로 이어서 수행할 순서

1. 1인용 배포 대상·secret manager·영속 볼륨·Ollama 배치를 확정한 뒤 후보를 리뷰·병합·배포하고, 승인형 Databricks 적재와 로컬 재사용 smoke test 및 rollback을 확인한다.
2. 실제 배포 호스트에서 용량 gate를 재실행해 p95·RSS 기준을 보정하고 Ollama 동시 추론 표본을 수집한다.
3. 출시 대상 테이블의 최신 TableContext와 업무 alias를 준비하고 schema 변경 운영 절차를 확인한다.
4. 남은 128문항의 oracle을 우선순위별로 확대하며 범용 다중 패널과 winsorization 범위를 결정한다. 그룹별 건수·성공률 이중축은 `L2_061`·`L2_064`·`L2_065`·`L2_066`을 주 평가에 편입했다.
