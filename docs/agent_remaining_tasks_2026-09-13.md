# 남은 작업 목록 — 2026-09-24

> 최신 판정: [2026-09-24 출시 판정](release_readiness_2026-09-24.md). 이번 구현은 승인 SQL의 실제 출처/후보 검증, 영속 선택 상태, 집계가 원본 기준을 덮지 않는 규칙, 배치 후보 파일 발행, 지정 컬럼 프로파일, 필터 히스토그램의 결정적 로컬 복구, UI 미리보기, 실제 모델 완료 근거 오류를 보강했다. localhost 실검증은 통과했으나 운영 출시는 NO-GO다. 아래 기존 R01~R14 완료 이력은 이번 T00~T13 전체 완료를 뜻하지 않는다.

> 2026-09-24 작업 방식 전환: [전체 요구사항·DeepEval/Spider·대용량 평가안](data_agent_requirements_and_evaluation_2026-09-24.md)의 T00~T13을 신규 구현의 기준으로 사용한다. 아래 표는 이전 검증 이력을 보존한다. 113문항은 미채점이며 전부 기능 미지원이라는 의미는 아니다. 이전 PASS와 이번 준비 단계 재검증 결과는 위 보고서에서 구분한다.

> EDA 보강: [연속 탐색 계약](eda_exploration_contract_2026-09-24.md)의 V05~V08·J13~J16을 T01~T13에 연결했다. 자동 이미지 추천→발견→후속 탐색→비교·되돌리기와 대용량 공유 집계는 설계 상태이며 구현 완료가 아니다.

> 설계 재리뷰: [R01~R08 보강 계약](design_review_2026-09-24.md)을 적용한다. 공통 상태/도구 관찰/복구 계약과 작은 실제 모델 기준선을 T02/T05로 앞당겼다. J17~J20의 동시성·일관성·집계 의미·자원/발행 경계를 추가했다.

> 데이터 손실 방지 우선: [로딩·원본 보존 계약](data_loading_and_preservation_contract_2026-09-24.md)의 D07/D08·J21~J24를 추가했다. T06과 T07의 보존/후보 적재 부분은 P0이며, 잘못된 적재·오해석 재조회·EDA 후 원본 소실을 먼저 검증한다.

기준 시각: 2026-09-22 KST. 상세 평가와 근거는 [chatbot agentic 평가](chatbot_agentic_evaluation_2026-09-13.md)에 기록했다.

| ID | 우선순위 | 작업 | 상태 | 완료 증거 또는 남은 조건 |
|---|---|---|---|---|
| R01 | P0 | 요청 조건과 계산·차트 범위 검증 | 완료 | 잘못된 로컬·원격·차트 범위 실행 전 차단, 후속 조건과 재시작 보존 검증 |
| R02 | P0 | 실제 모델 실패 수정·재검증 | 완료 | Level 1 지원 20/20, Level 2 지원 4/4 실제 agent PASS |
| R03 | P0 | 기존 긴 대화와 실제 화면 검증 | 완료 | 최신 서버에서 같은 `age` histogram 요청이 기존 PNG와 750,000건 범위를 표시. 모델·원격 조회 0회, 승인 카드 0개 |
| R04 | P0 | 최종 회귀와 서버 반영 | 완료 | 최신 회귀 205/205, compileall·diff PASS. 서버 실제 화면에서 18개 컬럼 전체를 0.160초에 표시, 모델 0회·로컬 도구 1회·추가 조회 0회 |
| R05 | P1 | 응답 지연과 실행 한도 | 부분 완료 | 초기 용량 gate READY: 단일 p95 0.162초, 4 worker·20/20 p95 0.549초, peak RSS 약 361 MiB. 1 GiB/768 MiB 경고/896 MiB 위험 계약과 preflight 검증 적용. 실제 배포 호스트 재측정 남음 |
| R06 | P1 | 대용량 DataFrame·캐시·저장소 | 검증 완료 | 750,000행 복구와 함께 100k행·256열·512MiB·대화별 2GiB 한도, 30일 정리 후보 보고 적용. 자동 삭제 없음 |
| R07 | P2 | 실제 agent 독립 채점 확대 | 진행 중 | 누적 87/200문항 지원. 피벗 13문항까지 독립 reference로 PASS. 모델·원격 0회이며 나머지 113문항은 PASS로 계산하지 않음 |
| R08 | P2 | 고급 분석·차트·스킬 범위 | 진행 중 | 전체 행·IN·공통조건+OR·백분율·피어슨 상관, 5종 차트와 그룹 박스플롯, 숫자축 빈도 선·영문 월 누적 곡선, bounded join, 6개 구조화 통계 방법, 이상치 탐지와 bounded cohort 후속 계산을 검증. datetime 재집계, cohort 비교, dual-axis/split-panel, winsorization, bounded 피벗·교차표를 검증. 범용 다중 패널, 다중 지표·구간화 소계와 결과 내보내기는 미검증 |
| R09 | P1 | 운영 승인 여정과 출시 판정 | 완료 | 실제 승인·1회 조회·10,000행 저장·로컬 분석 재사용 완료. RC3 보존, 원격 gate 성공한 `cd40480`에 RC4 tag 고정·push |
| R10 | P0 | 특정 테이블·스키마 의존 제거 | 완료 | production 테이블 하드코딩 제거, stale schema 차단, 동적 loader, schema fingerprint, 최신 원본 캐시 우회와 승인 대기 검증 |
| R11 | P0 | 제한 출시 후보 승격 | 진행 중 | draft PR #68의 후속 scope 수정 commit `1286e29`, GitHub run `35931785519` 성공. secret-safe preflight와 1인용 배포·용량·smoke·rollback 계약 준비. 대상 확정, 코드 리뷰/병합과 실제 환경 검증 남음 |
| R12 | P1 | 운영 TableContext 준비 | 부분 완료 | bank_loan 18/18·titanic 12/12 alias 준비. 네 테이블 모두 stale여서 각각 승인형 schema refresh 필요 |
| R13 | P1 | 평가 환경 정비 | 완료 | SciPy/NumPy 호환 수정 후 Level 1·2 참고 코드 200/200 PASS. 임의 placeholder를 제거하고 production agentic recovery 계약 17/17 PASS를 `--level 3`에 연결 |
| R14 | P0 | agent tool 계약·기능 공백 보강 | P0·P1 핵심 완료 | 공통 출력 schema, `profile_dataset`, 승인형 source discovery, bounded chart·join·`statistical_test`·`detect_outliers`·`select_outlier_rows` 완료 |

## 바로 이어서 수행할 순서

1. J22~J24를 실제 Databricks에 연결해 **정확한 SQL별 사용자 승인**→후보 검증→기존 원본/선택 보존→후속 로컬 분석까지 확인한다. 거절/403/부분 결과/재시작도 같은 여정으로 검증한다.
2. 원격 적재의 디스크 staging, 지정 컬럼 프로파일, 명시 컬럼 차트·집계·통계·시계열·피벗·그룹 비교/요약·이상치 요약 및 단순 로컬 SQL의 부분 읽기를 적용했다. 남은 wildcard/복잡 SQL·자동 추천·join·행 파생 경로의 전체 복원을 줄이고, 여러 세션의 peak RSS·저장량·지연과 보호 원본 quota 정책을 실제 호스트에서 측정한다.
3. 실제 모델의 held-out 반복 여정에서 PRES_03 timeout/편차, 동적 schema·후속 범위·오류 복구를 정량화한다. DeepEval 과정 평가와 Spider 2.0 공식 SQLite subset은 별도 점수로 연결하고, 남은 113문항의 독립 oracle을 우선순위별로 채운다.
4. 1인용 배포 대상·외부 접근제어·영속 볼륨·Ollama 배치를 확정한 뒤 PR #68 리뷰/병합·배포, 실제 환경 smoke/rollback을 수행한다. 현재 `private-single-user` preflight는 NOT READY다.
