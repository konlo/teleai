# 승인 후 “결과가 오면 안내” 응답 수정

2026-09-28. 보고된 최종 응답을 동기 실행 경로에서 재현하고 수정했다. 최초 사용자 대화의 실행 로그는 이 호스트에서 찾지 못했지만, 실제 production graph에 같은 유형의 응답을 내는 모델을 연결하여 두 오류를 모두 재현했다: **조회가 이미 완료된 경우에도 대기 문구를 반환**, **조회하지 않은 경우에도 해당 문구로 answered 처리**. 수정 전 회귀 2건 모두 실패했다.

## 원인과 수정

1. 테이블 목록 조회는 기존 통계/차트/컬럼설명 계약에 속하지 않을 수 있었다. 계약이 비어 있고 도구 오류도 없으면 `completion_ready()`가 성공으로 간주해 자유 응답을 그대로 보여줬다. 이제 원격 도구 호출 자체가 완료 의무를 생성한다. 승인 ledger의 completed receipt, 정확한 SQL/출처/이유, ToolMessage, 저장된 dataset의 ID/출처/SQL/행·컬럼이 일치해야 완료한다.
2. 완료된 목록 결과를 반드시 출력하는 경로가 없었다. 이제 검증된 저장 결과에서 목록을 직접 렌더링한다. 정보 스키마 결과는 최대 200행/20열, 일반 단독 조회는 10행의 제한된 미리보기이며 일부 표시·빈 결과·잘림 범위를 명시한다. 차트나 통계가 추가로 요청됐다면 기존 완료 의무를 유지한다. 사용자 원본 전체를 무조건 출력하거나 다시 로딩하지 않는다.
3. 실제 모델 추가 검증에서 알려진 catalog의 이름/타입 목록을 미확정 수치 연산으로 분류했다. 이 경우 평균/최솟값 해석 경로로 들어가지 않도록 분리하고, 목록 조회 자체에 실행 결과 계약을 부여했다. 테이블/컬럼별 업무 이름은 코드에 추가하지 않았다.
4. 승인 도구 설명에 “먼저 도구를 호출하면 승인 카드가 생성된다”는 의미가 불명확했다. 기존 승인 규칙을 다시 작성해 승인 카드 생성과 실제 SQL 실행을 구분했다. 알려진 catalog 조회에는 탐색/승인 도구만 집중 노출한다. 실제 모델이 SQL만 쓰고 종료하던 중간 실패도 저장했다.
5. 실행 없이 나중에 결과를 알려주겠다는 답변은 제한된 재계획 후에도 해결되지 않으면 미완료로 반환한다. 실제 완료 receipt가 있으면 LLM 문구를 쓰지 않는다. 지원하지 않는 자동 알림 기능을 명시적으로 설명하는 정상 답변은 허용한다.
6. 구버전의 잘못된 마지막 대기 답변은 대화 재개 시 완료된 receipt와 저장 결과가 모두 검증되는 경우에만 정정한다. SQL·모델 호출 없이 처리하며 원문은 rejected_transcript에 보존한다. 대기 중/진행 중인 graph에는 이 정정을 적용하지 않는다.

진단 이벤트: `remote_result_verified`, `unsupported_deferred_reply_rejected`, `completed_query_reply_reconciled`. 이벤트에는 자격증명·데이터 원문을 기록하지 않는다.

## 검증 결과

- 관련 계약/승인/SQL 범위/스키마 갱신/복구/연산 해석 회귀 **109개 통과**. [로그](regression_tests.txt)
- 신규 회귀는 정상 2행 목록, 0행, 승인 취소, 403 연결 실패, timeout/unknown 제출, 실행 기록 불일치, 기록 없는 dataset, 중복 승인, 재시작, 구버전 대기 답변 정정, 205행 표시 제한, 일반 설명, catalog 목록 의도, 계획만 출력하는 응답, 실제 Streamlit 페이지 승인 버튼→목록 표시를 포함한다.
- 실제 Databricks 모델 + 합성 SQL 실행기의 최종 동일 여정 **2/2 통과**. 승인 전 실행 0회, 승인 후 각각 1회, 실제 저장된 테이블 이름/타입 출력. [첫 최종 실행](live_model_focused.json), [반복](live_model_repeat.json)
- 중간 실행에서 발견한 통계 오분류와 도구 미호출 실패를 [최초](live_model.json), [연산 분리 후](live_model_after.json), [완료 계약 후](live_model_final.json), [승인 설명 정리 후](live_model_verified.json)에 보존했다. 이들은 최종 성공으로 덮어쓰지 않았다.
- Streamlit AppTest는 실제 ui/analysis_page.py를 사용하며 모델과 원격 SQL은 합성 대역이다. 실제 모델 평가도 Databricks **SQL Warehouse** 쿼리를 실행한 것이 아니다. 이번 warehouse SQL 실행은 **0회**다.
- 사용 중인 localhost:8502 앱은 실행 중 작업이 없는 것을 확인한 후 재시작했다. PID 50503, / 및 /_stcore/health HTTP 200. 다른 호스트의 배포 상태는 확인하지 않았다.

이번 수정은 해당 응답/승인 결과 전달 결함에 대한 것이다. 별도 최신행 중복 제거, 맥락 보존 등의 미해결 평가를 포함한 전체 agent GO 판정은 변경하지 않는다.

## 재실행

```sh
.telly_runtime/v1-venv/bin/python -m unittest tests.test_remote_result_completion -v
.telly_runtime/v1-venv/bin/python scripts/evaluate_remote_result_completion.py --output /tmp/remote-result-eval.json
```
