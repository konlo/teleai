# 미완료 요청이 새 테스트를 막는 문제 수정

모델 ReadTimeout으로 중단된 그래프가 남아 있을 때 `submit()`이 새 요청을 ValueError로 거절했다. 재개만 제공해 모델이 계속 실패하면 같은 대화에서 테스트를 이어갈 수 없었다.

## 동작 변경

- 미완료 요청에 **중단된 요청 종료하고 계속하기** 버튼을 제공한다. 실패한 작업의 완료를 주장하지 않고 `cancelled` 상태와 종료 사유를 기록한다.
- 다른 요청을 입력하면 비활성 미완료 작업을 종료하고 새 요청을 처리한다. 동일한 요청의 재입력은 재개/종료 안내를 반환하고 기존 요청 ID 및 실패 예산을 유지한다.
- 취소된 작업의 조건을 새 요청에 상속하지 않는다. 마지막 검증된 분석과 현재 선택을 기준으로 맥락을 복원한다.
- DataFrame/Parquet/PNG/분석 선택/완료 도구 결과 및 실패 예산을 지우지 않는다. 미실행 도구 호출은 취소된 ToolMessage로 짝을 맞추며 실행하지 않는다.
- 실제 실행 중에는 conversation lock이 종료를 거절한다. 승인 대기/불확실 원격 제출 상태는 이 기능으로 우회하지 않는다.
- 취소 결정을 먼저 체크포인트에 보존하고 대기 노드를 END로 정리한다. 두 단계 사이에 프로세스가 종료돼도 다음 controller가 모델/도구 재실행 없이 정리한다.

순수 취소 상태 생성은 `core/analysis_agent/interruption.py`, graph/lock 처리는 runtime, 확정 맥락 선택은 conversation_context, 버튼은 ui/analysis_page에서 담당한다. 기존 approval용 `cancel(request_id)`와 기능을 구분한다.

## 검증

관련 29개 PASS. 실제 ReadTimeout 주입→다른 컬럼 조회, 동일 요청 재입력, 실패 조건의 누출 방지, 도구 취소/SQL 미실행, 자산·선택·실패 예산 보존, 별도 interpreter 재시작, 취소 두 checkpoint 사이의 중단, concurrent/승인/불확실 제출 차단, 실제 페이지 AppTest를 포함한다.

전체 회귀 **704 PASS / 0 FAIL / 4 SKIP · 491 subtests PASS**, 128.72초. compileall 및 diff-check PASS. 건너뛴 외부 환경 검증을 성공으로 계산하지 않는다. [검증 기록](validation.json).

실제 기존 대화에서 종료 후 재개 버튼이 사라졌고 새 `테이블의 column 다시 보여줘` 요청이 **0.279초**에 18개 컬럼을 출력했다. 모델 호출 0, 분석용 원격 SQL 0. metadata의 현재 문맥 확인은 inspect_table_context로 수행했다. 저장된 데이터 8개/차트 9개의 metadata·payload SHA256와 선택이 모두 변경 전과 동일하다. [실제 실행 근거](web.json), [현재 화면](final_web.png).

서버 전환 중 구 프로세스의 resume 실행이 lock을 점유한 구간이 있어 첫 종료는 busy로 거절됐다. 해당 실행이 끝난 뒤 종료에 성공했다. 실행 중인 추론을 이 기능으로 취소하거나 중복 SQL을 실행하지 않았다. 이 구간은 web.json에 별도로 기록했다.

## 현재 한계

현재 대화에서 테스트를 이어갈 수 있다. **전체 데이터 age/balance 산점도는 생성되지 않았다.** 범용 모델 지연·큰 출처 산점도 계획은 별도 미완료 항목이며, 이번 테스트 성공은 전체 agent GO를 뜻하지 않는다. MySQL 평가용 8504 서버를 실행 상태로 남겼으며 기본 Databricks 설정을 바꾸지 않았다. 커밋/push/회사 서버 배포는 수행하지 않았다.
