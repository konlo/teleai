# 동적 테이블·스키마 계약 — 2026-09-13

## 원칙

운영 agent는 특정 테이블명, 컬럼명, 업무 값에 의존하지 않는다. 테이블별 이름·별칭·대표값은 외부 TableContext에만 있으며, TableContext도 영구 정답이 아니라 관측 시각이 있는 스키마 스냅샷이다. `test_set`의 `bank_loan`, `titanic` 등은 평가 fixture와 oracle일 뿐 production 판단 규칙이 아니다.

## 실행 시 우선순위

1. 사용자가 현재 로딩된 결과를 가리키면 해당 DatasetInfo의 source, columns, query, coverage, grain, conditions, snapshot을 사용한다.
2. 승인 후 실행된 `SELECT *` 결과가 있으면 그 결과의 실제 컬럼명과 dtype을 같은 테이블의 과거 TableContext보다 우선한다.
3. 저장된 TableContext는 기본 24시간(`TELLY_TABLE_CONTEXT_MAX_AGE_SECONDS`) 이후 stale이다. stale 컬럼·별칭·대표값은 실제 로딩 결과에 같은 컬럼이 확인되지 않는 한 조건 해석이나 새 SQL 생성에 사용하지 않는다.
4. 현재 테이블의 컬럼을 묻는데 스키마 스냅샷이 stale이면 `SELECT * FROM catalog.schema.table LIMIT 0`을 승인 카드로 제안한다. 승인 전 원격 실행은 없다.
5. 사용자가 최신·현재 원본 테이블·오늘 기준을 명시하면 이전 DataFrame·집계·PNG를 재사용하지 않는다. 새 원격 조회를 제안하고 승인을 기다린다. 일반적인 후속 분석은 기존 snapshot을 유지한다.

## 변경 유형별 처리

| 변경 | 처리 |
|---|---|
| 컬럼 추가·삭제·이름 변경 | 승인된 `SELECT *` 결과의 실제 컬럼 집합으로 교체하고 schema fingerprint 차이를 기록 |
| dtype 변경 | 실제 DataFrame dtype으로 fingerprint를 다시 계산하고 과거 dtype을 이어 쓰지 않음 |
| 테이블 추가 | 실행 중인 runtime도 컨텍스트 loader를 매 요청·inspect 때 다시 읽어 재시작 없이 반영 |
| 테이블 삭제·이동 | stale context만으로 존재한다고 확정하지 않음. 승인된 갱신 조회 실패를 관찰로 남기고 자동 재시도하지 않음 |
| 같은 짧은 테이블명 | 하나일 때만 짧은 이름을 해석하고, 둘 이상이면 `catalog.schema.table` 전체 이름 요구 |
| 행 데이터 변경 | 기존 DataFrame은 불변 snapshot으로 유지. 최신 원본 요청은 새 승인 필요 |
| 집계·부분 컬럼 결과 | 그 Dataset 자체의 분석에는 사용하지만 전체 테이블 스키마 증거로 승격하지 않음 |

## 검증

`migration/test_dynamic_table_context.py`는 업무 테이블과 무관한 임의 이름으로 stale 차단, 컬럼 추가·삭제, dtype 변경, 짧은 이름 충돌, 실행 중 컨텍스트 교체, 승인 전 원격 0회, 최신 원본 요청의 캐시 우회 계약을 검사한다.

최종 회귀 결과는 migration 98개와 tests 73개, 총 171개 PASS다. 테이블별 참조 코드 실행 결과는 이 계약의 agent 평가 점수에 포함하지 않는다.
