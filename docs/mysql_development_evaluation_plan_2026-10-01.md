# 로컬 MySQL 개발 평가와 Databricks 최종 검증 계획

작성: 2026-10-01. **판정: 조건부 채택.** MySQL은 실제 DB와 왕복하는 agent의 빠른 통합 평가 환경으로 사용한다. 출시 대상과 최종 합격 환경은 Databricks로 고정한다. MySQL 통과율을 Databricks 또는 Spider 2.0 공식 점수로 표시하지 않는다.

## 판단 근거

- 최근 `age` 히스토그램의 최초 실패는 Databricks에 SQL을 보내기 전 LLM `ReadTimeout`과 계획·재개 문제였다. MySQL은 이 오류를 직접 해결하지 않는다. 모델 공급자 장애, agent 계획과 복구, DB 실행을 분리해서 채점해야 한다.
- 현재 DuckDB/fixture 검사는 빠른 계산·계약 회귀에 이미 사용된다. MySQL을 추가하면 독립 프로세스와 실제 SQL 서버의 메타데이터, 연결, 지연, 배치 읽기, 실패 분류를 반복 시험할 수 있다. DuckDB 검사를 대체하지 않는다.
- 현재 호스트에서 `mysql`/`mysqld` 실행 파일과 기본 TCP 3306 리스너는 확인되지 않았다. 실제 설치 위치·대체 포트·기존 서버는 별도 확인이 필요하며, 이 문서 작성 시 데이터 복사나 서버 설치는 수행하지 않았다.
- Databricks Unity Catalog는 `catalog.schema.table` 3단계 이름을 사용한다. MySQL의 `INFORMATION_SCHEMA.COLUMNS`는 `TABLE_CATALOG` 값이 `def`이며 `TABLE_SCHEMA`가 데이터베이스를 가리킨다. 이름만 치환해서 같은 schema 탐색이라고 평가할 수 없다. [Databricks namespace](https://docs.databricks.com/aws/en/catalogs/), [MySQL column metadata](https://dev.mysql.com/doc/refman/8.4/en/information-schema-columns-table.html).
- 대용량 Databricks 전송·Arrow 스키마 보존·세션 오류는 MySQL로 재현되지 않는다. Databricks SQL Connector는 큰 결과에 `fetchmany_arrow` 배치 읽기를 제공한다. [Databricks connector](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector).

## 현재 코드에서 분리해야 할 경계

| 경계 | 현재 상태 | 변경 계약 |
|---|---|---|
| 실행 선택 | `ui/analysis_page.py`가 `ConnectionConfig`와 Databricks executor를 직접 생성 | 명시적 `TELLY_DATA_BACKEND=mysql|databricks`, 기본값 `databricks`. 환경별 저장소·연결 fingerprint를 분리해 다른 DB의 체크포인트/결과를 재개하지 않음 |
| SQL 제출 | `core/analysis_agent/runtime.py`의 `query_databricks`, `ApprovalLedger.envelope`와 `source_plan`의 Databricks 기본 dialect | 공통 읽기 전용 query gateway와 backend별 dialect·namespace·실행기. 기존 Databricks 경로는 회귀 검증 전까지 보존; tool 이름 전환은 호환 테스트와 함께 수행 |
| 계획·검증 | `core/analysis_sql.py`, `core/analysis_load_plan.py`, `core/analysis_agent/recovery.py`에 Databricks 기본값·직접 파싱이 남음 | 모든 SQL 생성→AST 검증→출처 대조→실행→provenance가 같은 활성 dialect를 사용하도록 전달. 다른 dialect의 SQL을 임의 문자열 치환하지 않음 |
| 탐색·스키마 | `core/analysis_catalog.py`, `core/analysis_metadata_discovery.py`, `core/analysis_runtime_tools.py`가 Unity Catalog, `catalog.information_schema`와 3단계 이름을 전제 | backend별 table/column/relationship discovery가 공통 관측 schema로 반환. agent는 실행 시 확인한 실제 테이블·컬럼만 사용 |
| 결과·보존 | `PersistentDatasets`와 차트/ledger는 재사용 가능하지만 0행 schema, raw, aggregate의 의미 구분이 필수 | 서버별 커넥터가 동일한 rows/columns/dtypes/coverage/grain/snapshot/lineage 계약을 만족. schema probe는 원본 EDA 모집단이 아님 |

## 실행 순서와 완료 조건

1. **기준 고정**: 현재 Databricks 경로의 핵심 사용자 여정과 실패·성능 baseline을 기록한다. SQL이 아예 제출되지 않은 LLM 오류, 제출 전 연결 실패, 제출 후 상태 불명, SQL 실행 오류를 별도 코드로 분류한다. 기존 5개 자동 정책 테스트 불일치도 기준에서 분리한다.
2. **공통 DB 계약**: 조회 실행, catalog/schema 탐색, dialect, 안전한 source identity, connection fingerprint, 배치 결과와 오류 상태를 하나의 backend 인터페이스로 정의한다. 읽기 전용 계정, 단일 SELECT AST, 출처 대조, 행·열·byte·시간 한도, 중복 제출 방지와 불명확한 제출 차단은 양쪽에서 동일하게 강제한다. agent 루프·데이터 저장소·차트 도구를 두 벌로 복제하지 않는다.
3. **MySQL 실행기와 개발 환경**: 프로젝트 전용 로컬 MySQL 인스턴스, 적재용 계정과 별도의 agent 읽기 전용 계정, 별도 Python driver, 격리된 DB를 구성한다. schema probe의 실제 dtype, 커서 배치 읽기, NULL/DECIMAL/datetime/boolean 취급, 예외 분류, 재시작과 중단을 테스트한다. UI에는 현재 backend를 명확하게 표시한다.
4. **평가 데이터 이관**: 먼저 작은 합성 테이블로 agent 계약을 확인한다. 이어 필요한 Databricks 테이블만 승인된 로컬 평가 snapshot으로 배치 추출·적재한다. 데이터와 인증 정보는 Git에 넣지 않는다. 민감 컬럼은 제외/비식별화하고, 원본 FQN·추출 조건·관측 시각·타입·행 수·NULL/중복/분포 checksum을 manifest로 보존한다. 시간순서와 키 중복이 필요한 EDA 시나리오도 포함한다. 10k → 대표 대량(현재 750k 규모) → 자원 한계/부분 결과 순으로 확대한다.
5. **MySQL 사용자 여정 평가**: 테이블 탐색 → 컬럼 확인 → 제한 로딩/서버 집계 → 여러 EDA 이미지 → 필터·집계·차트 후속 수정 → 원본 복귀를 실제 모델과 MySQL에서 연속 실행한다. 스키마 변경, 잘못된 컬럼, 렌더 실패, 모델 timeout, DB 접속 실패, 결과 제한, 재시작을 주입한다. 독립 oracle이 조건·모집단·수치·PNG/plot data·계보를 검사한다. 모델 없이 성공한 결정적 경로와 모델의 탐색·계획 성공을 따로 집계한다.
6. **Databricks 교차 검증과 출시 gate**: 같은 의미의 요청을 동등한 snapshot/스키마의 Databricks에서 다시 실행하고 수치와 범위·차트 근거를 비교한다. SQL 문자열 일치가 아니라 독립 oracle의 결과를 비교하며 NULL, collation, 시간대, decimal/정수 나눗셈, 정렬, window 함수 차이를 별도 검증한다. Unity Catalog 탐색·권한·Arrow/배치·대용량 전송·원격 실패 복구는 반드시 실제 Databricks에서 판정한다. 최종 GO는 기존 `docs/agent_release_criteria.md`의 실제 Databricks 여정과 차단 조건을 통과해야 한다.

## 평가 결과 표기

각 case는 `backend`, 코드/모델 revision, schema fingerprint, dataset snapshot, 계획·도구 trace, SQL 제출 여부, 결과 digest, 차트 ID, 재조회 횟수, 지연·peak RSS, 실패 단계를 남긴다. `PASS`, `FAIL`, `BLOCKED_INFRA`, `UNSUPPORTED`, `UNGRADED`를 구분한다. MySQL에서는 agent 기능·복구·대화 연결의 개발 점수를, Databricks에서는 실제 운영 적합성을 판정한다.

DeepEval은 도구 선택·후속 대화·답변 품질의 보조 지표로 사용하고 정확한 수치/이미지·보존 실패를 통과시키지 않는다. Spider 2.0 공식 트랙은 원래 DB와 공식 평가기를 유지한다. MySQL로 옮긴 변형 문제는 별도의 **TeleAI MySQL 적응 평가**로 표기하고 공식 Spider 점수에 합산하지 않는다. 기존 실제 평가의 복합 SQL 최종 완료 0/10과 미채점 oracle은 MySQL로 바꾸는 것만으로 해결됐다고 간주하지 않는다.

## 착수 전 확인할 입력

- 평가용 로컬 MySQL 서버의 설치/접속 방식과 저장 공간. 현재 기본 설치·3306 리스너는 확인되지 않았다.
- 옮길 Databricks 테이블 범위와 민감 컬럼 처리, 보존 기간, 최대 크기. 처음에는 한 개의 대표 테이블과 합성 다중 테이블 fixture로 시작한다.
- MySQL snapshot과 Databricks 최종 검증에 사용할 같은 질문 묶음 및 독립 기대 결과. 요구사항을 낮추지 않고 기능별 지원 범위를 명시한다.
