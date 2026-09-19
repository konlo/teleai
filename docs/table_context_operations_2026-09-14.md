# 운영 TableContext 준비 상태 — 2026-09-14

운영 코드에는 특정 테이블명이나 컬럼 의미를 넣지 않는다. 테이블별 schema, profile, 업무 alias는 `.telly_table_context`의 외부 파일에서 읽고, 실행 시점에 보유 DataFrame의 실제 컬럼과 대조한다.

## 현재 상태

| 테이블 | 저장 컬럼 | alias가 있는 컬럼 | 최신성 | 실행 전 조치 |
|---|---:|---:|---|---|
| `workspace.default.bank_loan` | 18 | 18 | stale | 승인형 `SELECT * ... LIMIT 0` schema 확인 |
| `workspace.default.titanic` | 12 | 12 | stale | 승인형 `SELECT * ... LIMIT 0` schema 확인 |
| `workspace.default.ncr_ride` | 21 | 19 | stale | 승인형 `SELECT * ... LIMIT 0` schema 확인 |
| `workspace.default.stormtrooper` | 13 | 7 | stale | 승인형 `SELECT * ... LIMIT 0` schema 확인 |

`bank_loan`과 `titanic`의 한국어 업무 alias를 외부 override에 준비했다. alias 파일에는 컬럼명과 별칭만 있으며 행 데이터는 없다. 오래된 alias는 승인 후 로딩된 실제 컬럼과 이름이 일치할 때만 사용할 수 있다.

## 변경 운영 절차

1. 선택한 테이블의 TableContext와 `observed_at`을 읽는다.
2. stale이면 현재 schema를 추측하지 않고 `SELECT * FROM <table> LIMIT 0` 승인 카드를 표시한다.
3. 사용자가 승인한 경우에만 한 번 조회하고, 반환된 실제 컬럼과 dtype을 과거 snapshot보다 우선한다.
4. 컬럼 추가·삭제·dtype 변경을 fingerprint로 기록하고 alias를 실제 컬럼과 다시 대조한다.
5. profile 재학습은 별도 승인 쿼리로 수행하고, raw rows와 비밀정보를 TableContext에 저장하지 않는다.
6. 거절 또는 실패 시 과거 결과를 보존하고 자동 재조회하지 않는다.

현재 네 테이블 모두 schema 확인 승인이 필요하다. 이번 점검에서는 Databricks 조회를 실행하지 않았다. 기계 판독 결과는 [TableContext readiness JSON](table_context_readiness_2026-09-14.json)에 있다.
