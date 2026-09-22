# 실제 Chatbot Scalar Recovery 검증

검증 일시: 2026-09-22 KST

## 발견한 결함

보유 `DataFrame`에서 단일 평균을 계산하는 요청이 결정적 로컬 경로로 연결되지 않아 Ollama 모델 호출로 넘어갔다. 첫 실제 화면 실행은 60초 `ReadTimeout`으로 끝났고 오류 ID는 `a30d7ae636cf`였다. 원격 Databricks 조회는 실행되지 않았고 기존 데이터는 보존됐다.

후속 요청에서는 새 조건 컬럼이 기존 측정 컬럼을 대체할 수 있었다. 예를 들어 `그중 segment A`에서 `segment`가 `value`를 밀어내는 구조였다.

## 수정

- 보유 데이터 요청의 `AVG`, `MEDIAN`, `SUM`, `MIN`, `MAX`를 로딩된 dataset metadata와 실제 dtype으로 검증한 뒤 bounded `local_analysis_sql`로 실행한다.
- 후속 조건 컬럼은 predicate로 분리하고 이전 측정 컬럼을 유지한다.
- 조건과 OR 범위는 구조화된 request scope에서 SQL로 만들며 임의 SQL이나 특정 테이블 지식을 사용하지 않는다.
- 일반 모델 경로의 잘못된 SQL 복구 테스트는 그대로 유지한다.

## 스키마 비의존성

production 경로에는 `bank_loan`, `age`, `segment`, `value` 같은 테이블별 지식을 넣지 않았다. 런타임에 등록된 source, columns, dtype, coverage, freshness를 확인한다. 필요한 metadata가 없거나 stale이면 로컬 계산을 추측해 실행하지 않고 schema inspection 또는 사용자 승인형 discovery 경로를 사용한다.

별도 회귀 테스트는 임의 source `arbitrary.runtime_table`과 임의 컬럼 `batch_code`, `cohort_key`, `metric_amount`를 사용했다. W1 평균 6.0과 alpha 후속 평균 3.0을 모델 호출 없이 계산했다.

## 실제 화면 검증

대화 ID: `24ddb8bf-ab41-42a9-a3f9-8e9ec65af3e8`

| 순서 | 실제 요청 | 결과 | 실행 시간 |
|---:|---|---:|---:|
| 1 | 보유 데이터에서 2026-08의 value 평균을 계산해줘. | 40.0 | 0.187초 |
| 2 | 그중 segment A만 계산해줘. | 40.0 | 0.031초 |
| 3 | 평균 말고 중앙값으로 바꿔줘. | 20.0 | 0.029초 |
| 4 | 같은 그룹의 이전 달 중앙값도 계산해줘. | 4.0 | 0.037초 |

네 요청은 모두 `recovery_transition -> local_analysis_sql` 한 번으로 `answered`가 됐다. 각 turn은 모델 호출 0회, 로컬 tool 1회였고 Databricks 및 다른 원격 실행은 0회였다.

## 자동 검증

- application: 135/135 PASS
- migration: 126/126 PASS
- actual-agent evaluation harness: 40/40 PASS
- Level 3 agentic contracts: 17/17 PASS
- 전체 참고 runner: 217/217 PASS
- `compileall` 및 `git diff --check`: PASS

이번 검증은 로컬 합성 fixture를 사용했다. Databricks 조회나 schema refresh는 요청·승인·실행하지 않았다.
