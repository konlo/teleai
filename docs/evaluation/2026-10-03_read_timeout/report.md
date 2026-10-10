# ReadTimeout 92cebb8130f2 진단

실행 ID: `5d7f376643444b5c87524af71cceacc9`. 오류 발생: 2026-10-03 13:38:50 KST. 당시 로컬 MySQL 모드, PID 4242, 모델 `ChatOllama`. 아래 진단은 저장된 로그·checkpoint를 읽은 결과이며, 이후 사용자 요청으로 수행한 수정·실검증은 마지막 절에 기록했다.

사용자 요청은 “education은 어떤 값들로 되어 있지 ?”였다.

| 단계 | 로그 근거 |
|---|---|
| 첫 모델 호출 | 74.59초, 도구 호출 1개 생성 |
| 실제 DB SQL | `SELECT DISTINCT education FROM teleai_default.bank_loan` |
| DB 실행 | 13:37:23.741 → 13:37:24.181, 약 0.440초, completed/ready |
| 결과 검증·저장 | `remote_result_verified`, 4행, Parquet 및 receipt 존재 |
| 대화 요약 | 25.822초, 성공 |
| 두 번째 모델 호출 | 13:37:50.016 → 13:38:50.038, 60.022초 후 ReadTimeout |
| 전체 요청 | 161.267초, incomplete |

직접 원인은 Ollama 응답 대기 시간 초과다. 스택은 langchain_ollama의 `_create_chat_stream` → ollama client → httpx transport에 위치한다. DB 연결·SQL 생성 오류나 DB 결과 미수신이 아니다. Ollama server 로그의 같은 시각 task 1064는 입력 13,751토큰, prompt cache 재사용 불가로 전체 입력 재처리를 기록하며, 60초 후 `/api/chat` 요청이 종료되었다. 과거 시점의 CPU/메모리 사용률을 측정하지 않았으므로 자원 부족을 확정하지 않는다.

agent 구조의 추가 원인도 확인했다. checkpoint는 calculation=True, operations=[], required_columns=[education]이며 DISTINCT 4행 receipt는 있지만 calculation evidence가 없다. `_valid_calculation`은 집계 연산 또는 비율 근거를 요구하므로 범주값 목록을 이 완료 계약으로 인정하지 않는다. 실제 목록 결과를 바로 렌더링하지 못하고 요약·추론을 다시 호출했다. timeout local continuation의 허용 작업에도 범주값 목록 표시가 포함되지 않는다.

`retry_scheduled=false`, retries=0이다. 첫 모델·요약·실패 호출에 약 160초를 썼으며 현재 180초 예산 안에 추가 60초 모델 호출이 들어가지 않아 재시도하지 않았다. 재개는 같은 모델 지연·완료 계약 문제가 재현될 수 있으므로 성공을 보장하지 않는다.

## 필요한 보강

1. 범주값 목록 의도를 숫자 계산·고유값 개수와 분리한다. 특정 컬럼/테이블에 의존하지 않는다.
2. 일치하는 실제 SELECT DISTINCT receipt·출처·조건·컬럼·결과 크기를 검증한 뒤 bounded 목록을 모델 없이 출력하는 완료 계약을 추가한다. 잘린 결과는 전체 목록이라고 하지 않는다.
3. 저장된 목록 결과의 재사용·후속 대화·모델 timeout 후 출력·원본 선택 보존·잘못된 조건의 완료 차단을 회귀로 검증한다. DB를 다시 조회하거나 원본을 대체하여 우회하지 않는다.
4. 모델에 전달하는 도구 schema/문맥 양과 실제 입력 처리 지연을 줄이고 관측한다. timeout 연장만으로 구조 문제를 해결했다고 선언하지 않는다.

최초 진단에서는 실제 실패 결과를 유지했고 SQL 또는 모델을 새로 호출하지 않았다.

## 수정 및 실제 검증 — 2026-10-03

`inspect_value_list`와 `value_list` 완료 계약을 추가했다. 범주 목록을 숫자 계산·고유값 개수와 구분하고, 실제 출처·컬럼·요청 필터·현재 연결의 completed receipt가 일치하는 결과만 표시한다. 결과가 이미 저장되어 있으면 추가 추론/조회 없이 완료한다. 이름은 실제 schema에서 확인하며 특정 테이블/컬럼 상수에 의존하지 않는다.

원격 경로는 DISTINCT/ORDER BY/LIMIT 101로 최대 100개와 초과 여부를 확인한다. LIMIT와 반환 건수가 같은 결과는 SQL 결과가 완전히 수신되었어도 모집단 전체 목록이라고 하지 않는다. 로컬 완전 원본은 필요한 컬럼만 Arrow 배치로 읽고 별도 목록을 만들며, 표본이나 조건이 다른 결과를 전체 모집단으로 재사용하지 않는다. 긴 값은 256자로 제한한다. 이번 자동 조건 계획 범위는 검증 가능한 단순 AND 조건이며 복합 OR/조인 조건은 완료로 추측하지 않는다.

저장된 DISTINCT 결과가 raw 원본 선택을 대체하지 않도록 막았고, timeout 후 재개도 같은 완료 계약을 통과한다. 재개 시 controller가 생성한 결과를 LLM 호출로 잘못 집계하고 대화 중단 시간을 모델 실행 시간에 포함하던 로그 오류도 수정했다.

| 검증 | 실제 근거 |
|---|---|
| 사용자 실패 대화 복구 | 같은 웹 대화의 미완료 분석 재개. run `3baca101e0b646c291b8fc1ca5c22e43`, answered, 0.007초. 저장된 4행 사용, 해당 재개에서 DB/모델 호출 없음 |
| 표시 결과 | education: secondary, primary, tertiary, unknown. 기존 age histogram 보존, 오류/재개 안내 제거. [화면](fixed_web.png) |
| 실제 Databricks | 현 schema 0행 확인 → 실제 문자열 컬럼 DISTINCT 목록 12개 → 후속 재사용. SQL 총 2회, 반복 0회, 모델 0회. [측정 JSON](databricks_value_list.json) |
| 회귀 | 관련 15개 파일 115 PASS, 131 subtests PASS. 조건 변경·restart·raw digest·수동 승인 모드·부분 목록·위조/미수신 결과 거부·모델 timeout 저장 결과 복구 포함 |
| 적용 | MySQL 챗봇 8504 최종 프로세스 PID 6939, health ok. 실제 .env의 Databricks 기본 설정은 유지 |

이 수정은 범주값 목록의 불필요한 재추론과 timeout 뒤 결과 미출력을 해결한다. 다른 자유형 LLM 계획의 입력 처리 지연이나 모든 ReadTimeout이 해결됐다는 근거는 아니며, 회사 서버 및 범용 agent GO 판정으로 확대하지 않는다. 기존 모델 문맥/도구 입력량 최적화와 독립 실모델 평가 항목은 계속 남아 있다. 이번 요청에서 commit/push하지 않았다.
