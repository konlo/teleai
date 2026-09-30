# 연결 복구 후 GO 평가 및 개선 — 2026-09-30

**판정: 범용 자율 데이터 분석 agent는 NO-GO.** 로컬 단일 사용자 환경에서 검증된 EDA는 사용 가능하다. 별도 서버 운영은 사용자 지시로 범위 밖이다. 연결 복구를 전체 agent 완성으로 계산하지 않는다.

## 이번에 수정한 공통 결함

1. **집계가 원본 스키마를 덮어쓰는 오류**: `COUNT(*)`의 별표까지 전체 컬럼 조회로 취급했다. 이제 단일 원본의 단독·무변형 `SELECT *`만 스키마 권한을 갖는다. 집계·부분 projection·EXCEPT/REPLACE·파생/조인 결과는 원본 스키마를 대체하지 않는다. 오래된 스키마가 건수 조회만으로 최신 상태로 바뀌는 경로도 차단했다.
2. **테이블 주소를 분석 컬럼으로 오인**: 관측된 qualified table 주소의 schema 이름이 실제 컬럼명과 같으면 계산에 그 컬럼까지 요구했다. 컬럼/조건 해석에서 알려진 테이블 주소를 구분한다. 같은 컬럼이 따로 언급되거나 주소가 조건의 문자열 값인 경우는 보존한다. 실제 `workspace.default.bank_loan`의 `age` 평균에서 불필요한 `default` 요구가 사라졌다.
3. 영어 `How many … for/in each …`를 개수 및 명시적 그룹으로 연결했다. 평균이나 그룹이 없는 총건수를 대신 완료시키지 않는 음성 검사 포함. 범용 영어 의미 해석이 완성됐다는 뜻은 아니다.
4. Spider 보고서에 실제 agent의 해석 계약(출처·컬럼·연산·범위)을 남겨, API·모델 SQL·완료 검증 실패를 구분한다. 생산 코드에 benchmark 테이블명/정답은 추가하지 않았다.

## 실제 데이터와 자연어 검증

- [최신행 실DB 여정](latest_live.json): 실제 750,000행, 명시적 키/최대 ID/추가 동률 기준/선택 전 결측 제외. 12키의 8구간 빈도·경계가 독립 ROW_NUMBER SQL+NumPy 정답과 일치. 내려받은 결과9행, 25.888초, schema/집계/oracle SQL3회. 반복 요청 추가 조회0회. 이 명시적 여정의 모델 호출은0이며 실제 자연어 모델 평가로 세지 않는다. [실제 PNG](latest_live.png)도 확인했다.
- [스키마 오염 실DB 회귀](schema_authority_after.json): scripted model이 실제 Databricks COUNT→AVG 두 쿼리를 실행, answered 및 원본18컬럼 보존. 처음 [하네스 중단](schema_authority_live.json)과 [원인 진단](schema_authority_diagnosis.json)은 보존했다. 첫 하네스는 세 번째 모델 요청을 AssertionError로 중단했다. 진단 하네스로 바꿔 실제 결함인 schema 이름/컬럼 충돌을 찾아 수정한 것이다.
- [자연어·후속 대화](language_journeys.json): 실제 Databricks 모델을 연결한 합성 로컬 데이터10여정/12턴 PASS, 실제 모델 호출6회, warehouse0회. 조건·연산·후속·차트·원본 보존을 독립 기대값과 비교했다.
- [원래200문항](reference_200.json): **105 PASS / 95 UNGRADED**, 실제 모델 호출14회, warehouse0회. 대부분 결정적 도구 경로다. 원래 reference 계산은 바꾸지 않았으며 새 oracle3개는 L1_096/097/100의 실제 PNG·차트 데이터 검증이다. [신규3개 별도 실행](new_oracles.json)도 PASS. 미채점을 통과 또는 제품 결함으로 단정하지 않는다.

## 복합 SQL의 현재 한계

[최초2문항](spider_initial/proposals.json), [나머지8문항](spider_remaining_initial/proposals.json), [수정 후 같은10문항](spider_after/proposals.json)을 분리 보존했다. 실제 설정 모델과 production graph, 공개 SQLite를 사용했다. Gold는 agent에 제공하지 않았다. 수정 후 모델 호출77회, 검증된 최종 SQL **0/10**. 제출할 최종 SQL이 없어 공식 최종답 점수는 미산출이다.

원인을 구분하려고 각 문항의 **마지막 모델 SQL 초안**을 별도 [진단 제출물](spider_diagnostic_drafts_manifest.json)로 저장하고 설치된 공식 scorer를 변경 없이 실행했다. [공식 결과](spider_diagnostic_scoring.log)는 **초안1/10 정답(local004)**. 이는 agent 성공률이 아니다. 공식 출력의 전체547 분모 값도 전체 benchmark를 실행한 점수로 사용하지 않는다. 나머지9개 초안은 탐색 단계/중단 후 초안일 수 있어 전부 LLM 자체의 오답이라고 단정할 수 없다.

| 문항 | 종료 시 확인된 원인 | 모델 호출 |
|---|---|---|
| local007 | model_call_budget | 10 |
| local358 | missing_evidence | 5 |
| local002 | unverified_join_relationship | 5 |
| local003 | unverified_join_relationship | 7 |
| local004 | unverified_join_relationship | 10 |
| local008 | unverified_join_relationship | 10 |
| local009 | request_scope_unresolved | 5 |
| local198 | unverified_join_relationship | 5 |
| local210 | request_scope_unresolved | 10 |
| local221 | unverified_join_relationship | 10 |

구체적으로 여러 테이블의 일반 단어 컬럼이 요청의 일반 문장과 충돌하며, 파생 계산식의 scalar 덧셈을 행 집계 SUM으로 취급한다. FK가 없는 관계를 검증하는 경로가 없고, 명시적 조인 요청·전체 출처가 없으면 FK가 있어도 계획을 제한한다. 필요한 해결은 **출처별 요청 의미 연결 → 계산식/모집단/조인 역할 계약 → 제한된 탐색·관계 검증 → 실행 결과와 완료 계약 대조**다. 단순 재시도 증가나 완료 검증 해제로 정답 차단 문제를 해결하지 않는다.

## 남은 완료 기준

1. 복합 SQL: 위10개 최초 실패를 고정하고, 일반 단어 충돌·파생 식·명시적 키 없는 관계의 탐색 및 검증·JSON/OR/CTE/HAVING/DISTINCT를 함께 보강한다. 정답 초안을 실행/완료까지 연결하되 잘못된 모집단·역할은 계속 차단해야 한다.
2. 최신행 EDA: 추가 통계·비균등 경계·혼합 필터 단계/복합 조건. 현재 단일 단계 AND 필터와 기본 분포의 성공을 전체 완료로 바꾸지 않는다.
3. 대용량: D07/D08·J21~J24 미검증 혼합 여정, 남은 전체 컬럼 읽기·실제 ENOSPC/UI 경합. 기존102만행 및 실제75만행 근거는 해당 범위에서만 사용한다.
4. 독립 평가: 남은95문항 oracle 및 새 schema/자연어/다회 대화의 성공·실패 조건. DeepEval 보조 judge 점수로 실제 SQL/계보/이미지 실패를 상쇄하지 않는다.

## 검사 및 적용 범위

최초 수정 제품503개, migration136개 PASS. Oracle 추가 후 검사에서 선언 oracle 개수를102로 고정한 검사가 실패했고105로 갱신한 뒤50개 관련 검사 PASS. 마지막 테이블 주소 분리 후 관련21개 PASS 및 전체 제품 재검증을 수행한다. 실제 브라우저/사용자 운영 서버 배포는 이번 실행으로 확인한 것이 아니다. 커밋·CI 결과는 아래에 추가한다.


## 최종 실모델·실DB 확인

[qualified table 평균](qualified_mean_live_model.json): 실제 설정 Databricks 모델3회, SQL1회로 `age` 평균 answered, source 스키마18컬럼 보존, 요구 컬럼은 age 하나로 확인. 최초 count-probe2 SQL 회귀와 구분한다. 모델 결과를 독립 평균 oracle로 재채점한 검사는 아니며, 이 검사는 관측 스키마와 완료 연결의 회귀 검사다.

이번 warehouse 조회는 최신행3회 + count→평균 회귀 세 번×2회 + 실제 모델 평균1회 = 총10회다. 공개 SQLite 실행 및 모델 API 호출과 구분한다. 앞선 연결 확인 SELECT1은 별도 요청의 근거다.

마지막 전체 제품504검사에서503개 통과,1개는 새 oracle105개와 저장된 readiness manifest102개 불일치였다. manifest 생성기로 갱신하고 해당 검사를 재실행했다. 최초 실패 로그도 보존했다. 이 manifest 갱신은 제품의 통과 기준을 낮추지 않는다.
