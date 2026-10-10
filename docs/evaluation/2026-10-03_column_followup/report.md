# 컬럼 목록 후속 요청 진단·수정 — 2026-10-03

## 실패 근거
- 실제 웹 요청: `테이블의 column 다시 보여줘`, 대화 `0728dee0-d555-4aa2-853b-1101679346da`.
- run `f1d8896b136444288cfdb0ad3f28d795`, error `00849ca53a9b`, 221.109초.
- 컬럼 조회 의도는 단어 감지와 표시 동작의 결합이 협소해 누락됐다. 테이블명을 생략한 후속 요청을 현재 출처에 연결하는 결정적 경로도 부족했다.
- 모델·요약 반복 후 Ollama HTTP 스트림 ReadTimeout. `inspect_column_definitions`는 계획만 만들었고 데이터 조회 SQL은 실행되지 않았다. DB 결과 반환 실패가 아니다.

## 변경
- 단수·복수 영어와 한국어 조사, 컬럼 표시 동작을 공통 schema 요청으로 분류한다. 컬럼 값 미리보기/고유값/결측/설명과 구분한다.
- 생략 요청은 최근 확정 출처를 사용하고, 명시적 UI 선택 변경은 이전 분석을 대체한다. explicit table 요청은 우선한다. 주제를 설명한 요청은 선택 테이블로 임의 대체하지 않는다.
- 집계 결과는 출처 식별에만 사용하며 실제 전체 schema는 inspect_table_context의 관측·freshness 검증으로 확인한다. 오래된 schema는 LIMIT 0으로 갱신한다. 테이블이 모호하면 질문하며 임의 테이블을 고르지 않는다.
- 기존 실패 checkpoint도 원문에 따라 metadata 계약으로 보강하고 로컬 schema inspection으로 재개한다. 모델·tool 예산/원격 실행 이력/원본·선택 결과를 초기화하지 않는다.

## 실제 검증
- 기존 웹 실패 재개: run `02e037f5f05840fb82a7f11166ecf6e8`, 0.032초, 기존 모델 호출 이력 5회 보존, 이번 재개 모델 호출 0회. bank_loan 18컬럼 표시.
- 같은 문장 신규 제출: run `6722c1325b8d43c7b3b62d22319d6f07`, 0.298초, inspect_table_context 1회, 모델 0회, 데이터 재로딩 0회. 보유 결과 7개와 선택 결과 6 유지.
- 별도 MySQL·Databricks 연결에서 실제 SELECT * LIMIT 0 스키마를 확인한 뒤 동일 요청: 두 backend 모두 18컬럼, 모델 0회, 추가 데이터 조회 0회. 모델 대역은 추론 호출 시 실패하는 NoInference이며 자연어 실모델 점수로 계산하지 않는다.
- 회귀: 양 backend/영어 조사/명시적 테이블 변경/재시작/모호한 테이블/과거 실패 재개/stale LIMIT 0 갱신/원본·집계 선택 보존을 검증한다. 앞 두 행 미리보기와 서술형 테이블 탐색의 회귀 영향 3건도 수정·재검증했다.

## 증거 및 한계
- live.json, web.json, web.png, validation.json.
- 8504 MySQL 평가 서버 실행 유지. .env의 Databricks 기본 설정은 바꾸지 않았다.
- 이 요청의 수정과 전체 GO 판정은 구분한다. 이번 화면에서 별도로 발견한 stack/stackbin 요청의 배치 설정 누락·기존 차트 재사용은 남은 개선 항목으로 기록한다.

## 최종 결과
전체 `pytest tests migration -q`: 683 PASS, 0 FAIL, 4 SKIP, 477 subtests, 96.53초. 관련 회귀28 PASS/15 subtests. compile/diff-check/8504 health PASS. 최종 서버 동일 요청도 0.179초, 모델·데이터 조회0회.
