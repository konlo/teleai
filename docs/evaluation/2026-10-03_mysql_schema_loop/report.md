# MySQL 컬럼 조회 반복 실패 진단·수정

## 최초 실패 증거

- 실제 웹: `http://127.0.0.1:8504/Telly?conversation=0728dee0-d555-4aa2-853b-1101679346da`.
- 요청: `bank_loan 컬럼들을 보여줘`.
- 실행 ID: `0efbe7b224454636ba6e438534e13758`, 요청 ID: `7fb4a677-1bcc-428c-8d49-f14cf4651788`.
- 모델은 `inspect_column_definitions(table="teleai_default.bank_loan")`를 두 차례 선택했다.
- 두 결과 모두 `needs_context`, 메시지 `catalog.schema.table 전체 이름이 필요합니다.`.
- 종료 이유는 `repeated_failed_tool`. 그래프 recursion ceiling 소진 또는 DB timeout이 아니다. 해당 요청에 SQL 실행 이벤트가 없다. 소요 시간 119.171초.
- 최초 테이블 목록 요청은 MySQL 조회 성공, 5행 저장·출력 완료.

## 원인과 수정

1. 컬럼 설명 도구가 Databricks 세 단계 이름과 Unity Catalog SQL만 지원했다. 연결된 SQL dialect에 따라 MySQL `database.table`과 `information_schema.columns.column_comment`를 사용하도록 분리했다. 정확한 SQL/테이블/시점이 일치하는 저장 결과만 정의로 재사용하며 65행 상한은 유지한다.
2. `컬럼들을 보여줘`가 컬럼 목록 완료 의무로 분류되지 않았다. 구조 컬럼 목록의 복수 표현 및 영문 show/display columns를 인식한다. 차트·고유값·결측 등 통계 요청은 기존 구분을 유지한다.
3. 동일 도구 실패를 일반 실행 횟수 한도 소진처럼 설명했다. 반복 실패 종료 문구를 분리했다. 실패·SQL 검증·데이터 보존·중복 실행 제한은 유지한다.
4. 의미 해석 서비스의 정의 재사용도 연결된 SQL dialect를 전달한다. 도구 안내의 오래된 매번 승인 문구를 현재 실행 정책 준수 안내로 교체했다.

## 검증

- 집중 실행: `TELLY_TEST_MYSQL=1 .telly_runtime/v1-venv/bin/python -m pytest -q tests/test_mysql_metadata_contract.py tests/test_mysql_backend_live.py tests/test_analysis_metadata_discovery.py tests/test_analysis_schema_refresh_recovery.py`: 최초 17 PASS / 15 subtests PASS. 고유값 요청의 오분류 차단 검사를 추가한 뒤 `tests/test_mysql_metadata_contract.py`는 3 PASS / 11 subtests PASS. 추가 검사의 첫 실행은 결과를 일반 evidence_ids로 찾은 테스트 가정 오류로 실패했으며, 실제 profile_evidence와 출력 값을 확인하도록 수정했다.
- 관련 회귀: `tests/test_tool_repair_loop.py tests/test_analysis_full_read_budget.py tests/test_analysis_selection_state.py tests/test_analysis_selection_ui.py tests/test_remote_latest_histogram.py`: 25 PASS / 8 subtests PASS.
- 실 DB: 동적으로 찾은 MySQL 테이블의 컬럼 메타데이터 계획·실행·저장 재사용 성공. 정보 원문 외 raw 로딩 없음.
- AppTest: 실제 MySQL 테이블 목록 → 동적으로 찾은 테이블의 `컬럼들을 보여줘` 완료. 모델 호출 0회.
- 보호 원본 유지, 두 SQL engine의 컬럼 목록 요청, 다른 테이블·변형 SQL·오래된 결과·65행 가득 찬 페이지 차단 검사 포함.
- `git diff --check`, `git diff --cached --check` 통과.

## 실제 웹 재검증

- 8504 서버 PID 3184를 종료하고 MySQL 설정 유지한 PID 4242로 재시작. health `ok`; 저장 대화/기존 결과 보존.
- 같은 대화에서 같은 요청 재실행. 실행 ID `64e5777edba543acaf70f0c4dcaf1646`, 요청 ID `98987f23-53e8-41cf-b64a-106153a7ce22`.
- `inspect_table_context` 1회 `ready`, `completion_checked=complete`, `required_capabilities=[metadata]`, 모델 호출 0회, 소요 0.120초.
- 화면: `teleai_default.bank_loan에는 컬럼이 18개 있습니다`와 실제 18개 컬럼 출력.
- 스키마는 런타임의 실제 MySQL information_schema 조회로 확인한다. 목록 확인을 위해 대용량 raw 데이터를 다시 로딩하지 않는다.
- 과거 오류 메시지는 대화 이력으로 유지하며 성공한 새 응답을 덧붙였다.
- 이 요청의 원인을 수정·검증한 결과이며 전체 제품 GO 또는 자유 Python 실행 통합 완료 판정이 아니다. 변경은 현재 로컬 작업 트리에 반영; 이번 요청에서는 commit/push하지 않았다.
