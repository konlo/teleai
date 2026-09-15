# TeleAI chatbot agentic 평가 — 2026-09-13

## 판정

현재 production 경로는 `GraphAnalysisRuntime` 안에서 목표 확인, 도구 실행, 결과 범위 검증, 제한된 복구, 승인 대기, 영속 상태 복원을 수행한다. 따라서 구조상 단순 chatbot이 아니라 agent loop를 가진 분석 agent다. 검증된 핵심 범위에서는 잘못된 결과를 완료로 채택하지 않고 로컬 대안을 실행하는 동작까지 확인했다.

다만 전체 200문항을 처리하는 범용 데이터 분석 agent라고 출시할 증거는 아직 부족하다. 독립 정답으로 실제 agent를 채점한 누적 범위는 37/200문항이다. 따라서 판정은 **검증된 핵심 범위의 제한 출시 후보 / 전체 Level 1·2 기능은 NO-GO**다.

## 평가 결과

| 층 | 실행 대상 | 결과 | 의미 |
|---|---:|---:|---|
| Level 1 참고 코드 | 체크인된 `python_code` 100개 | 100/100 PASS, figure 25개 | 정답 코드와 fixture의 실행 가능성. chatbot 평가는 아님 |
| Level 1 실제 agent | production graph + 실제 로컬 모델 + 독립 oracle | 누적 31/31 PASS | 컬럼 목록·개수 metadata 문항 추가 PASS |
| Level 2 참고 코드 | 체크인된 `python_code` 100개 | 100/100 PASS, figure 33개 | SciPy 1.16.3 고정과 `np.trapezoid` 호환 수정. chatbot 평가는 아님 |
| Level 2 실제 agent | production graph + 실제 로컬 모델 + 독립 oracle | 누적 6/6 PASS | 이번 확대 OR/IN 문항 2/2 PASS |
| Level 3 agentic 복구 | production graph에 실패를 주입한 시나리오 | 17/17 PASS | 잘못된 SQL/범위/도구, 승인 거절, 재시작, 중복 실행, 현재 표본 로컬 계산, bounded stop 검증 |
| 전체 회귀 | `migration` + `tests` | 203/203 PASS | metadata·용량 계약 반영 기준 125 + 78, compileall과 `git diff --check`도 PASS |

원본 증거는 [기존 Level 1 결과](actual_agent_evaluation_20_latest_2026-09-13.json), [기존 Level 2 결과](actual_agent_evaluation_level2_4_repaired_2026-09-13.json), [신규 10문항 결과](actual_agent_evaluation_expanded_10_final_2026-09-13.json), [최신 Level 3 복구 결과](agentic_recovery_evaluation.json), [운영 저장소 보고](operational_storage_report_2026-09-13.json)에 보존했다.

## 이번 평가에서 발견하고 수정한 결함

`L2_005`는 학력, 혼인 상태, 20대 범위, 가입 여부의 다섯 조건을 요구한다. 모델은 조건을 정확히 적용한 7행짜리 로컬 데이터셋을 만들었지만, 원본과 필터 결과를 모두 후보로 본 복구기가 계산 대상을 고르지 못했다. 두 번째 모델 호출이 각각 130.689초와 185.520초에 `NOT_COMPLETE`로 끝났다.

복구기는 이제 요청 조건과 정확히 일치하는 필터 결과를 다른 캐시 후보보다 우선한다. 또 따옴표가 포함된 “가입한 사람들의 수” 표현을 COUNT 의무로 인식해, 데이터와 조건이 충분하면 모델 재호출 없이 로컬 SQL을 실행한다. 수정 후 같은 요청은 0.197초에 PASS했고, 네 개 Level 2 묶음 재실행은 4/4 PASS, 총 0.367초, 최대 0.194초였다. 원격 조회는 0회였다.

실제 기존 대화에서 `age에 대한 histogram을 보여줘`를 최신 서버에 다시 제출했다. 복구기는 검증된 기존 PNG를 `show_chart` 도구로 선택했고, 모델 호출 0회·원격 조회 0회·0.292초에 완료했다. 브라우저에는 `age 분포` 이미지와 `빈도 합계 750,000`이 표시됐으며 활성 재조회 승인 카드는 0개였다. 동시에 들어온 중복 제출은 새 오류 ID를 만들지 않고 실행 중 안내로 처리한다.

`L1_093` 타이타닉 `Age` histogram의 첫 실제 모델 평가는 정확한 PNG를 만들었지만 63.680초 중 모델 도구 선택에 63.264초를 썼다. 이미 완전한 로컬 데이터, 출처, 단일 수치 컬럼, histogram 의도가 확정된 경우를 결정적 로컬 전이로 옮겼다. 수정 후 같은 독립 oracle은 0.238초에 PASS했고 모델·원격 호출은 0회였다. 실제 agent 독립 채점은 35/200으로 늘었다.

`L1_017`의 “평균 나이와 최고령 나이” 요청은 평균만 의무로 인식해 로컬 데이터가 충분한데도 모델 경로에서 완료 증거 없이 중단했다. `최고령`을 MAX로 grounding하고, 한 수치 컬럼의 AVG·MEDIAN·SUM·MIN·MAX 조합을 하나의 table-neutral DuckDB query로 계산하게 했다. 독립 multi-scalar oracle은 두 결과를 모두 확인하고 변형 fixture 두 개에서 계산 계보를 재실행한다. 수정 후 평균 `41.064`, 최대 `83`을 0.673초에 PASS했으며 모델·원격 호출은 0회였다. 재시작한 실제 화면의 보유 10,000행에서도 평균 `40.931`, 최대 `86`을 0.177초에 함께 표시했고 모델 0회·로컬 도구 1회·추가 조회 0회였다. 실제 agent 독립 채점은 36/200이 됐다. 원본은 [L1_017 평가 결과](actual_agent_evaluation_L1_017_2026-09-15.json)에 보존했다.

`L1_001`의 “전체 컬럼 목록과 총 개수” 요청은 TableContext가 하나로 확정돼도 모델이 검사 도구를 선택해야 했다. 이제 fresh 또는 승인 후 실제 스키마가 `ready`일 때만 table-neutral 로컬 전이로 `inspect_table_context`를 호출한다. stale 결과는 기존 승인형 갱신 loop를 유지한다. 독립 metadata oracle은 응답 문장을 신뢰하지 않고 도구가 반환한 컬럼 배열·개수를 reference code의 `cols`와 비교한다. fixture는 18개 컬럼을 0.161초에 PASS했고 실제 화면도 0.160초에 같은 18개를 표시했다. 양쪽 모두 모델·원격 호출은 0회였다. 원본은 [L1_001 평가 결과](actual_agent_evaluation_L1_001_2026-09-15.json)에 보존했다.

신규 대화에서 `SELECT * FROM workspace.default.bank_loan LIMIT 10000` 승인 카드를 실제로 승인했다. 승인 전 Databricks 실행은 0회였고 승인 후 정확히 1회 조회해 10,000행·18열을 영속 저장했다. 이어서 `현재 로딩된 10,000행 표본의 age 히스토그램을 보여줘`를 실행하자 저장된 DataFrame만 사용해 0.215초에 실제 PNG를 표시했으며 모델 호출과 추가 원격 조회는 모두 0회였다. 이 여정에서 발견한 controller load 오분류와 숫자가 포함된 표본 표현 누락을 회귀 테스트로 고정했다. 상세 증거는 [Databricks 승인 여정](databricks_approval_journey_2026-09-14.md)에 보존했다.

같은 실제 대화에서 `현재 보유한 bank_loan 데이터에서 age와 balance의 피어슨 상관계수를 계산해줘`를 실행했다. 처음에는 `현재 보유한 <table> 데이터` 표현과 `coverage=unknown`, `predicate_known=false` 표본을 현재 결과로 인식하지 못해 느린 모델 경로로 진입했다. 현재 결과 명시를 원문에서 확정하고, 그 프레임 자체를 모집단으로 삼는 무필터 수치 상관분석만 로컬 `CORR`로 허용했다. 최종 화면은 `0.05918371025192562`를 0.224초에 표시했고 모델·원격 호출은 모두 0회였다.

추가 독립 평가 10문항의 첫 실행은 6 PASS, 2 FAIL, 2 NOT_COMPLETE였다. 완전한 로컬 DataFrame을 두고 전체 행 수를 원격 조회하려 한 문제, 백분율의 0–1/0–100 단위 문제, OR을 AND로 축소한 문제, 명시된 월 목록을 누락한 문제를 발견했다. 요청 scope에 전체 행, IN, 공통 조건과 한 개 OR 그룹, 백분율 단위를 보존하고 DuckDB 식별자 quoting을 분리했다. 수정 후 같은 10문항은 10/10 PASS, 최대 0.237초, 원격 실행 0회였다.

## agentic 동작으로 인정한 기준

- 모델이 자연스러운 문장을 출력했다는 사실만으로 완료 처리하지 않는다. 계산 데이터셋이나 실제 PNG가 있어야 한다.
- 사용자 조건과 SQL·필터·차트의 실제 데이터 범위가 다르면 실행 또는 완료 전에 거절한다.
- 완전한 로컬 데이터가 있으면 Databricks를 다시 조회하지 않고 계산하거나 시각화한다.
- 새 데이터나 더 넓은 범위가 필요하면 구체 SQL을 승인 카드로 만들며, 승인 전에는 실행하지 않는다.
- 잘못된 SQL·도구 ID·범위는 제한된 횟수 안에서 수정한다. 같은 실패를 반복하면 안전하게 중단하고 기존 결과를 보존한다.
- 요청 조건, 승인 상태, DataFrame, PNG를 재시작 뒤에도 복원한다.
- 특정 테이블의 정적 스키마를 코드에 넣지 않는다. 오래된 TableContext는 실행 근거에서 제외하고 승인 후 관측된 실제 스키마를 우선한다.

## 평가 세트의 구조적 문제

`test_set/run_test_set.py`는 질문을 chatbot에 보내지 않고 각 문항의 정답 `python_code`를 `exec`한다. 이 실행의 PASS를 agent 이해도나 복구 성능으로 합산하면 안 된다.

`test_set/level3/definitions_part5.py`에 있던 무작위 placeholder는 제거했다. 이제 이 파일은 production `GraphAnalysisRuntime`에 고장을 주입하는 `A3_001`~`A3_017` 계약만 등록하며, `run_test_set.py --level 3`과 `--all`이 실제 unittest를 실행한다. Level 1·2 참조 코드 PASS와 Level 3 제어 계약 PASS는 지표를 분리해 표시한다.

## 남은 출시 위험

- 전체 200문항 중 163문항은 실제 agent 독립 채점이 없다. 조인, 가설 검정, 고급 그룹 분석, 복합 시각화를 지원한다고 선언할 수 없다.
- 평가용 테이블 동의어는 `test_set`의 외부 TableContext에만 있다. 운영 agent는 오래된 alias를 실행 근거로 사용하지 않으며, 새 업무 용어 해석 품질은 최신 외부 TableContext의 품질에 좌우된다.
- 초기 운영 한도(원격 100,000행, 256열, DataFrame 512 MiB, cache 64 MiB, 대화별 2 GiB, 정리 후보 30일)를 적용했다. 60초 model timeout은 streaming 전체 deadline이 아니라 읽기 비활성 기준이며, 180초는 turn SLO와 모델 호출 사이 누적 예산이다. 실제 모델 상관계수는 5/5 정확·p95 80.704초였고 결정적 로컬 전이 후 30/30·p95 0.046초가 됐다. 최신 단일 p95 0.162초, 동시 4 worker·20/20 p95 0.549초, peak RSS 약 361 MiB는 초기 1 GiB/768 MiB 경고/896 MiB 위험 용량 gate를 통과했다. 다른 모델 질문과 실제 배포·Ollama 동시 부하는 남았다.
- RC3는 원격에 보존돼 있다. 복합 집계 수정과 배포 preflight를 포함한 commit `cd40480`은 원격 release gate 성공 후 `agentic-analysis-rc4-2026-09-15` tag로 고정했다. 배포는 별도 작업이다.

## 다음 출시 게이트

제한 출시 범위는 보유 데이터 재사용, 기본 집계, 조건 필터, histogram/bar/line/scatter/단일 수치 boxplot으로 명시해야 한다. 최신 서버와 실제 브라우저 확인, 실제 승인형 Databricks 적재와 후속 로컬 시각화까지 통과했으므로 이 범위는 release candidate로 볼 수 있다. 전체 기능 출시는 우선순위 문항의 독립 oracle 확대와 모델 질문·동시 부하의 실제 p95·process RSS 보정이 필요하다.
