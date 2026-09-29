# 연속 평가·복구 기능 보강 결과

2026-09-30 · 대상: 로컬 단일 사용자 chatbot. **전체 완료/범용 GO가 아니다.** 로컬 검증과 웹 복구 여정은 통과했지만 실제 Databricks 모델·SQL 실행은 공급자 오류로 중단됐다. 기존 고급 SQL 실패와 독립 oracle 미완료를 환경 장애로 지우지 않는다.

## 이번에 수정한 공통 기능

- 최신행 동률 질문 → 추가 내림차순 컬럼 답변 → 재계산 → 실제 차트까지 연결했다. 재시작과 다른 출처 선택을 구별하며, 완료 후 짧은 bins 변경도 선택 데이터/스냅샷이 바뀌면 이전 출처를 상속하지 않는다. 기준을 바꾼 도구 호출은 실행/완료 전에 거부한다. 로컬 Parquet와 원격 SQL 모두 같은 정렬 계약을 저장한다.
- 명시적인 **키·정렬·분포 컬럼의 결측 행을 최신행 선택 전에 제외**하는 정책을 추가했다. 기본값은 자동 제외하지 않는다. 최신행이 결측이면 이전 유효 행이 선택되는지 독립 정답과 대조했다. 원본·제외 건수·파생 모집단을 보존한다. 선택 후 제외, 불명확한 단계, 부정문을 이 정책으로 바꾸지 않는다. 무한대는 결측 정책으로 임의 삭제하지 않는다.
- `동률`을 `비율`로 잘못 읽어 동일 요청 재사용을 막던 의도 해석을 수정했다. 최초 실패는 [initial_tie_failure.txt](initial_tie_failure.txt)에 남겼다.
- Databricks가 특정 임시 모델 서비스 장애를 HTTP400으로 반환하는 경우를 구조화된 응답 전체로 좁게 구별한다. 기존 요청별 최대 2회 재시도/시간/호출 한도를 유지하며 일반400·401·403에는 적용하지 않는다. 사용자에게 공급자 장애와 미완료를 알리고 로그에 분류·오류 ID를 남긴다. 원문 응답·토큰은 기록하지 않는다.
- Spider에 `--interactive` 모드를 추가했다. 공개 SQLite에서 탐색 SQL → 관찰 → 최종 계산을 실제 production graph로 실행한다. 첫 미리보기나 임의 마지막 SQL을 정답으로 제출하지 않고, 검증된 계산 evidence의 SQL만 내보낸다. 읽기 전용·출처·행/byte/시간 한도를 적용한다. 골드는 agent에 전달하지 않는다.

## 검증 근거

| 종류 | 결과와 범위 |
|---|---|
| 제품 전체 | 461/461 PASS, 55.794초. 이후 추가한 평가기 1개는 별도 관련 검사와 최종 CI로 확인한다. [로그](application_tests.txt) |
| migration | 136/136 PASS, 17.900초. [로그](migration_tests.txt) |
| reference·복구 | 217/217 PASS, 58개 그림. 마지막 경계 보강 전 결과이며 최종 CI에서 다시 실행한다. agent 자유계획 성공률이 아니다. |
| 최신행 정책 | 로컬/원격 모의 실행, 독립 pandas·NumPy 정답, 원본 digest, 재시작, source 변경, 이중 동률, 결측 전체, 부정문, 무한대 차단 검증. 실제 모델/warehouse 점수와 분리한다. |
| 100만 행 최신행 | 10,000키 독립 정답 일치, 전체 pandas 읽기0, cache0, 8.884초, 프로세스 peak RSS 418,234,368 bytes. 원본·재시작 보존. [결과](large_latest/result.json) |
| 100만 행 적재·EDA | batch 최대1024, 평균·조건부 평균·차트 PASS, 전송중단/byte한도에서 부분 결과 발행0, 원본·선택·재시작 보존. peak RSS 467,451,904 bytes. 합성 cursor 검사로 실제 네트워크 성능을 뜻하지 않는다. [결과](streaming_scale.json) |
| 실제 웹 | 별도 합성12행 → 결측 정책 확인 → 동률 기준 확인 → 5키·4범주(1,2,1,1) 실제 PNG660×385. 원본 digest 유지, 모델0/SQL0. [근거](browser_journey.json), [이미지](browser_chart.png) |

## 실제 실행에서 확인한 외부 차단

1. Interactive Spider `local007`, `local358`은 첫 모델 요청에서 HTTP400이 발생했다. 분류 보강 후 `local007`은 자동 재시도2회까지 **총3번** 같은 응답을 받았다: `Cannot create or query foundation model endpoints, please try again later.` SQL 실행은0. [최초 결과](interactive/proposals.json), [재시도 결과](interactive_retry/proposals.json). 실패 시 checkpoint의 `model_calls=0`은 성공 응답이 없어서 생긴 옛 표시이며 실제 시도3회는 diagnostics에 남아 있다. 새 평가기는 실패 ledger까지 합산한다.
2. 실제 최신행 SQL 검증도 schema 조회 연결 단계에서 HTTP400 `Cannot create the resource, please try again later.`로 중단됐다. agent 분석 쿼리는 실행되지 않았다. [결과](live_latest_policies.json).
3. 같은 설정의 warehouse 조회 API는 HTTP200을 반환했고 상태는 `STOPPED`였다. [상태](warehouse_status.json). 리소스를 생성/시작하지 못하는 내부 원인은 확정하지 못했다. 이 근거로 토큰 재발급이 필요하다고 단정하지 않는다. 조회를 불명확한 상태에서 자동 반복하지 않았다.

이번 interactive 실모델 평가는 **채점 불가/BLOCKED_PROVIDER**이며 정답률0%로 새로 계산하지 않는다. 이전 proposal-only Spider 공식0/10은 그대로 보존한다. DeepEval 실모델 재채점도 이 상태에서 성공했다고 보고하지 않는다.

## 남은 범위와 재개 기준

미완료 기능 묶음은 여전히4개다. 개수는 남은 소요시간이나 결함 수가 아니다.

1. 최신행 EDA: 일반 조건의 선택 전후 적용, 추가 통계, 비균등 구간, 다른 결측/동률 정책. 이번에 내림차순 추가 기준과 선택 전 결측행 제외의 후속 연결은 완료했다.
2. 복합 SQL: 역할별 조인·JSON·OR/NULL·다단계 CTE/HAVING/DISTINCT 의미 검증·복구. Interactive 실모델 실행 및 공식 scorer 비교가 필요하다.
3. 대용량: [D07/D08·J21~J24 근거 매핑](data_journey_coverage.md)의 부분/미검증 경계, 남은 도구의 부분 scan·혼합 실행.
4. 독립 oracle: 기존200문항 중102 PASS/98 UNGRADED. 이번 새 계약·여정 테스트를 그98문항의 채점 완료로 바꾸지 않는다.

재개 시 외부 서비스의 실제 한 요청이 정상 응답하는지 확인한 뒤 동일 고정 Spider10문항의 interactive 실행·공식 채점과 실제 SQL 정책 여정을 계속한다. 최초 실패/수정 후 근거를 덮어쓰지 않고 출력 디렉터리를 분리한다. 로컬 코드 통과만으로 GO를 선언하지 않는다.
