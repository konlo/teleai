# 입력 예산 재발 원인과 호출 전 예방

## 판정

`0a3f64804e0f`는 새 실패다. 실패한 요청은 `alibaba_ssd 이 테이블에서 찾아줘야지`이며 앞선 시간·날짜 관련 컬럼 질문을 정정한 발화다. 최초 분석 모델 호출은 성공했으나 107개 컬럼 스키마 수신 후 다음 모델 호출 직전에 입력 예산 검사에서 중단됐다. MySQL 데이터 조회 실패가 아니다. 저장된 실제 요청을 웹에서 재개하여 32.249초에 입력 오류 없이 응답을 받았다. 이 재개에는 분석 SQL 제출이 없었으며 기존 31개 asset의 metadata/payload SHA256과 선택 상태가 모두 동일하다.

이번 검증은 입력 예산 복구·출처/목표 연결을 확인한다. 실제 모델이 응답한 날짜 관련 컬럼 후보의 의미까지 독립적으로 확정한 검증은 아니다. 모델 응답의 `ds`·`disk_id` 등 가능성 표현은 추정이며, 특히 ID에 시간 의미가 있다는 근거는 확인하지 않았다. 타입/설명/값 검증 없는 추정 후보를 확인된 시간 컬럼으로 평가하지 않는다. 범용 agent GO/Databricks 운영 판정도 갱신하지 않는다.

## 실제 실패

- 최초 run `8ed04892d4174efa93d4a87feed6326a`: 총 44.956초.
- 첫 모델 입력: 10,789 byte + 여유 1,216 = 12,005, 예산 12,288 이내. 실제 입력 3,183토큰/출력 189토큰, 18.127초, inspect_table_context 1회.
- 실제 tool 결과: 대상 `teleai_default.alibaba_ssd`의 107개 컬럼 스키마. 저장된 메시지 UTF-8 4,675 byte.
- 이후 요약 모델: 26.250초. 메시지 16개를 압축했지만 분석 모델 입력에는 현재 스키마 관찰과 시스템 환경·도구 계약이 계속 포함됐다.
- 다음 모델 입력: 17,314 byte + 여유 1,344 = **18,658**, 예산 12,288 초과. 이 두 번째 분석 모델 HTTP 요청은 제출되지 않았다. `model_call_started`는 middleware span 시작이며 HTTP 제출 횟수와 다르다.
- 화면의 앞선 시간/date 의미 질문을 단순 컬럼 목록으로 처리하고 선택된 bank_loan으로 돌아간 문제도 발견했다. 단순 listing 의도와 schema 의미 질문을 분리하고, 확인된 metadata가 대화의 테이블을 유지하도록 보강했다.

## 공통 예방 처리

특정 테이블명·컬럼명 예외나 문맥 창 확대를 사용하지 않는다. 모델 num_ctx=16,384/출력 예약=4,096은 유지한다.

1. **현재 도구 관찰도 압축**: 넓은 스키마의 반복 `{name,dtype}` 항목을 `columns_by_dtype`로 묶는다. 전체 컬럼 이름·타입과 authority/freshness/scope, 추가 컬럼 설명은 보존한다. 저장된 ToolMessage·원본·call ID는 바꾸지 않는다.
2. **시스템 환경 자체를 축소**: 동적 catalog 블록을 내부 태그로 식별한다. 예산이 부족하면 현재 출처 관련 dataset/table identity만 남기며 다른 출처는 list_analysis_context로 탐색한다. 사용자/custom 지침·실행 정책·원문·필터를 임의로 잘라내지 않는다. 추가 middleware에서도 이 내부 태그를 유지한다.
3. **확인된 맥락과 현재 목표 유지**: 이전 증거뿐 아니라 현재 출처·조건·목표를 모델 뷰에 유지한다. 검증된 metadata 증거가 있으면 별도 요약 추론을 반복하지 않는다. 시간/date 의미 질문을 행 필터 OR 조건으로 오인하지 않는다. 테이블 정정 발화는 이전 의미 질문의 목표를 이어받는다.
4. **도구 메뉴를 단계적으로 제공**: 일반 축소 후에도 예산을 초과하면 검색/skill 및 우선 발견 도구로 메뉴를 더 줄인다. 다른 기능은 정확한 이름으로 검색하면 등록된 실행 계약을 제공한다. 검색 결과와 제공 메뉴에 tool schema를 중복해서 넣지 않는다. 실행기 등록/권한·SQL 검증·원본 보호는 유지한다.
5. **끝난 내부 추론 중복 제거**: 예산 부족 시 이미 발행된 tool arguments/관찰은 보존하고 과거 AI reasoning_content의 모델 뷰만 제거한다. 사용자 조건이나 실행 receipt를 제거하지 않는다.
6. **구성별 크기 진단**: projection 전후 payload, 시스템·메시지·tool calls·reasoning·tool schema의 byte 수, 제공 tool 이름, 압축 여부를 기록한다. 원문/SQL/데이터/credentials는 이 크기 진단에 기록하지 않는다. 공개 진단 요약에 허용된 숫자/플래그만 노출하고 입력 추정량도 표시한다.

바이트 검사는 모델 토큰 수의 보수적 대용값이다. 실제 token 사용량과 구분해 기록한다. 보호해야 할 사용자 요청/custom 지침/조건 자체가 모델 창에 맞지 않거나 극단적으로 큰 관찰을 손실 없이 표현할 수 없는 경우에는 여전히 안전하게 중단한다. 무제한 크기를 처리한다고 주장하지 않는다. 매우 넓은 스키마/긴 컬럼 설명의 페이지별 탐색·고급 자유계획은 추가 검증 대상이다.

## 검증 근거

- **실제 실패 checkpoint 격리 복사**: 새로운 보호 목표를 포함한 projection 전 23,161 byte를 최종 10,549 byte + 여유 1,088 = **11,637**로 줄였다. 실제 생산 runtime의 middleware를 통과했다. 대역 모델 응답 자체는 분석 정확도 성공으로 계산하지 않는다. 초기 격리 시도는 12,313+1,344로 여전히 실패했고, 이를 보고 추가 discovery/끝난 reasoning 축소를 구현했다(`checkpoint_replay.json`에 실패와 최종 결과 모두 보존).
- **실제 웹/Ollama**: run `97424e250ace4c98b6521d07a7fca48b`, 32.249초, HTTP 모델 응답 성공. 실제 input 3,170토큰/output 383토큰, 추가 분석 SQL/데이터 재로딩 0. 대상 alibaba_ssd와 앞선 시간/date 질문을 이어서 응답했다. 종료 시 미완료 버튼이 해제됐다.
- **전체 회귀**: `pytest tests migration -q` 728 PASS/0 FAIL/4 SKIP/500 subtests, 114.83초. 기본 SKIP은 별도 환경의 opt-in 테스트다. 최초 검사 2개 실패(관련 데이터의 일반 컬럼 목록을 의미 질문으로 과대 분류)를 수정하고 해당 정상 여정과 전체를 재검사했다(`regression_initial.txt`, `regression_final.txt`).
- **마지막 소규모 보강**: 전체 회귀 후 discovery schema 중복 제거/공개 진단 크기와 새 시험을 추가하고 관련 27 PASS(2.57초)로 검증했다. 마지막 전체와 이 변경의 검증 범위를 구분한다.
- **불변성**: `before.json`/`validation.json`: 기존 31개 asset 해시 및 선택 동일, 신규 asset 0. 수정은 inference view와 대화 연결이며 데이터 프레임을 덮어쓰지 않았다.
- **비밀정보 제외**: 공개 크기 진단에 임의 prompt/schema content/credentials를 넣은 음성 사례에서 노출되지 않음을 검증했다.

## 파일과 실행 상태

핵심 분리 모듈: `core/analysis_agent/input_views.py`, `schema_questions.py`; 연결 변경: model_context/runtime/tool_focus/memory/recovery/support_report. 신규/보강 회귀: test_model_context_budget.py, test_schema_question_context.py.

실행 내역은 `execution.json`, 읽기 쉬운 요약은 `original_failure_brief.txt`/`web_resume_brief.txt`에 저장했다. 최종 화면은 `final_web.png`. 8504의 MySQL 개발 서버를 최신 코드로 재시작해 테스트 가능 상태로 유지했다. `.env`의 기본 Databricks 설정과 원격 실행 정책은 변경하지 않았다. 커밋/push하지 않았다.
