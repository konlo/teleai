# 자율 탐색·복구 보강 결과 — 2026-09-26

**범용 자율 분석 agent 출시는 NO-GO 유지.** 등록된 도구를 검색하고 실패 후 다른 도구를 선택하는 공통 구조를 추가했지만, 복잡한 SQL 정답률과 미채점 분석 범위가 남았다. 별도 서버 배포는 사용자 요청에 따라 제외한다.

## 구현과 발견한 결함

- 실제 registry에서 schema·제약·권한·관련 skill을 검색하는 `search_analysis_tools`와 `autonomous-recovery` skill을 추가했다. 검색은 권한을 추가하거나 계산 완료의 증거가 되지 않는다.
- 로컬 도구의 `unavailable`을 원격 실패와 구분했다. 실제 모델이 원본을 유지하며 SQL 등 다른 로컬 도구로 복구할 수 있다. 원격 거절·실패의 자동 재조회 금지는 유지한다.
- `inspect_column_definitions`가 알려진 테이블의 저장된 fresh 업무 설명을 찾고, 없으면 Unity Catalog `information_schema.columns`의 제한된 SELECT를 제안한다. 실행은 기존 정확한 SQL 승인 gateway를 거친다. query/source/자료형/관측 시각을 검증하며 metadata 조회 결과가 선택된 raw 원본을 대체하지 않도록 수정했다. [공식 COLUMNS 문서](https://docs.databricks.com/aws/en/sql/language-manual/information-schema/columns).
- 필터 컬럼과 측정값이 같은 후속 요청에서 측정값이 사라지던 문제를 수정했다. 컬럼과 연산이 명확한 로컬 평균·중앙값 등은 모델 호출 전에 실행하며, 과거 timeout 체크포인트도 새 로딩 없이 재개한다.
- 대화 요약 호출이 분석 호출 한도에서 빠졌던 문제를 수정했다. 요약을 같은 호출 예산에 포함하고 마지막 한 번은 분석에 남긴다.

## 실행한 평가와 한계

| 평가 | 결과 | 해석 |
|---|---|---|
| application 회귀 | 309/309 PASS | 승인 거절·재시작·metadata/raw 분리·로컬 복구·호출 예산 포함 |
| migration | 133/133 PASS | 상태·완료·보존 계약 |
| reference + Level 3 | 217/217 PASS, 58 figures | reference 코드 성공은 실제 agent 성공과 별개 |
| 고정 분석 질문 | 97 PASS / 103 UNGRADED | 이전 실행 96 PASS/1 NOT_COMPLETE도 보존. 입력은 외부 업무 설명을 보강한 fixture; 미채점을 통과로 합산하지 않음 |
| 실제 모델 복구 | 5개 여정·6턴 PASS | 결정적 계획/차트 복구를 꺼서 실제 모델의 도구 검색·SQL 수정·대체 도구·복합 결과·재시작 후속 요청을 검사 |
| 공식 Spider scorer | 원문 0/3 제출, 고정 5문항 기준 0/5 | SQL 지시 별도 track도 0/1 제출, 고정 5문항 기준 0/5. 전체 547개 실행 결과가 아님 |
| DeepEval GEval | 6턴 평균 0.7667, threshold 0.8 통과 4/6 | 로컬 Gemma judge. 검증된 수치에도 출처/범위 서식을 벌점 처리하여 보정되지 않은 보조 점수. 제품 정확도·출시 점수로 사용하지 않음 |
| 100만 행 로컬 저장 시험 | 모든 계약 PASS | 저장·읽기·캐시·차트·재시작·중단 복구. 최대 fresh-process RSS 약 400 MB. 원격 전송이나 out-of-core 전체 분석의 증거가 아님 |

실모델 복구 시험 중 잘못된 첫 도구 호출과 로컬 worker 장애는 시험 장치가 주입했다. 이후 결정은 실제 모델이다. 도구 검색 문항은 검색하라는 요청을 포함한다. SQL 원격 executor는 연결하지 않았으므로 승인 안전성은 별도의 production graph 계약으로 검증한다. 이 작은 고정 묶음을 범용 성능으로 일반화하지 않는다.

DeepEval 첫 실행은 내부 JSON을 참조 답으로 주어 judge가 JSON 출력 의무를 만들어냈다. 참조 수치를 일반 문장으로 바꾼 재평가도 서식 판단 문제가 남아 있다. 별도 정답/오답 통제군을 통한 judge calibration 전에는 release gate로 사용하지 않는다. 점수를 높이려고 출처·표본 범위 표시를 제거하지 않았다.

## 실제 웹 및 저장 검증

기존 승인 10,000행에서 `age >= 60` 건수 **327** → 같은 조건의 평균 **64.97553516819572** → 중앙값 **62**가 독립 Parquet 계산과 일치했다. 건수 0.443초, 평균 재개 0.134초, 중앙값 0.324초였다. 평균 최초 실행은 긴 대화 요약 뒤 로컬 모델 `ReadTimeout`으로 중단됐다. 실패를 숨기지 않고 원인을 수정한 뒤 같은 체크포인트에서 로컬 계산으로 복구했다. 평균 재개 시 추가 모델 호출은 없었고 이전 요약 호출 1건이 상태에 남아 있다. 중앙값은 모델 0회다.

원본 SHA256 `7c4a1c20588261932629087d091d43e1ad2d80f4e36cb3d1acb20eb8462fe10d`, 승인 장부 `completed=1`은 불변이다. 신규 Databricks SQL은 실행하지 않았다. 최신 앱은 loopback 8502에서 실행 중이며 브라우저에 실제 결과를 확인했다.

## 남은 필수 작업

1. 운영 컬럼 설명: 네 테이블의 저장 metadata는 stale이고 description이 없다. `bank_loan` 컬럼 metadata 1회 SELECT의 정확한 SQL 승인을 요청했으며 아직 실행하지 않았다. DB comment가 비어 있으면 실제 업무 정의가 추가로 필요하다. fixture 설명을 운영에 주입하지 않는다.
2. 복잡한 SQL: 실제 관계 metadata 탐색과 join 의도/역할 계약, SQLite dialect 대체식, 기간 조건 해석, 중간 탐색 SELECT와 최종 분석 SELECT 구분을 묶어 보강해야 한다. 현 join 검증은 사용자가 지정한 관계만 인정해 자율적인 다중 테이블 과제를 막는다. 안전 검사를 단순 제거해 점수를 높이지 않는다.
3. 평가 범위: 미채점 103문항의 독립 oracle, 더 넓은 held-out/repeated-run 복구, DeepEval judge calibration, 공식 Spider 분모 확대가 남았다.
4. 대규모 데이터: 최종 코드의 승인형 원격 신규 적재·전송 실패/재시작·실제 peak RSS와 모든 분석 도구의 부분 scan 검증이 남았다. 100만 행 로컬 저장 성공으로 대체하지 않는다.
5. 일반 코드 실행: 등록된 선언형 도구 밖의 분석을 수행할 격리 코드 실행기는 아직 미완료다. 앱 프로세스에 임의 Python 실행을 허용하는 우회는 하지 않았다.

## 증거 파일

- `2026-09-26_autonomy_scores.json`: 200문항 분모, 채점 결과, 앞선 미완료, 모델 호출 수.
- `2026-09-26_autonomy_live.json`: 실모델 6턴, 실제 값·도구·원본 보존·시간.
- `2026-09-26_autonomy_spider.json`: 동일 5문항의 두 SQL 제안 track과 공식 채점.
- `2026-09-26_autonomy_deepeval.json`: 실제 trace에 대한 judge 점수와 이유, 해석 한계.
- `2026-09-26_autonomy_large.json`, `2026-09-26_autonomy_web.json`: 용량·실 웹·승인/원본 검증.
