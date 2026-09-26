# T00~T02 구현 착수 준비 결과

범위: 데이터 손실 방지·불필요 재조회 방지의 첫 구현 묶음. 전체 agent 구현 완료 또는 출시 승인 문서가 아니다.

## 작업 상태

| 작업 | 이번에 확보한 결과 | 남은 범위 |
|---|---|---|
| T00 기준선 | HEAD `aa01c28` 별도 추출, 당시 변경 13파일/patch/hash 보존, 기준본과 작업본 검사 비교, 피벗 회귀 수정 | 미완료 그룹 요약/구간화 전체 agent 연결은 별도 구현 대상 |
| T01 범위·추적 | 요구 34개→여정 24개, 참고 문항 200개→기능군/요구/기존 oracle 매핑, 첫 구현 범위 명시 | 문항별 의미 검토와 독립 oracle 113개 미정의; 매핑은 지원/성공 증거가 아님 |
| T02 계약·평가 | 데이터 역할/LoadPlan/Observation/평가 JSON schema, 정상·금지 상태 검사, 원본 보존 도구 여정, 실제 로컬 모델 기준선 | schema의 운영 실행 계층 연결, 전체 여정 fault/정답/UI 평가, DeepEval/Spider adapter |

스냅샷은 `.telly_runtime/baselines/20260924-100921`에 있다. [보존 manifest](evaluation/preparation_2026-09-24/baseline_snapshot.json)에 SHA와 파일별 hash를 기록했다. 원래 작업 폴더를 초기화하거나 미완료 변경을 삭제하지 않았다. 별도 HEAD 추출본은 검증용이며 실행 서버를 교체하지 않았다.

## 기준선 비교와 회귀 수정

[변경 전 검사](evaluation/preparation_2026-09-24/before_gates.json): HEAD migration 126/126·application 167/167 통과. 당시 작업본은 migration 126개 통과, application 171개 실행 중 10개 실패 보고(피벗 subtest 포함)였다.

원인은 미완료 피벗 확장의 조건식이었다. 단일 mean/success_rate에 값 컬럼이 있어도 유효 명세에서 배제했고, 여러 집계가 list일 때 set membership이 TypeError를 일으킬 수 있었다. 집계의 값 컬럼 필요 조건과 list 비교를 수정했다. 기존 10개 실패 회귀와 추가 다중 집계 production graph 검사를 연결했다. 그룹 요약 도구가 전체 자연어 요청까지 완성됐다고 주장하지 않는다.

수정 후 gate는 [실행 결과](evaluation/preparation_2026-09-24/prepared_gates.json), 추가 계약/평가 코드까지 포함한 최종 application 검사는 [최종 결과](evaluation/preparation_2026-09-24/final_application.json)로 구분한다. 최종 application **183/183**, migration **126/126**, Level 3 **17/17**, 전체 참고 runner **217/217** 통과와 compile 성공을 확인했다. 참고 runner 통과는 agent의 200문항 자연어 성공률이 아니다. 한번 잘못된 interpreter 경로를 사용해 발생한 import 오류는 [환경 오류 기록](evaluation/preparation_2026-09-24/application_wrong_interpreter.json)에 별도 보존했다. 이는 제품 회귀 판정에서 구분하며 최종 검사는 v1 venv 경로를 유지한다.

## 첫 구현 범위

schema 확인, 승인형 제한 raw 적재, 보존 원본 재사용, 필터/파생, count/sum/mean/그룹 집계, histogram/bar/line/scatter/boxplot, 원본 복귀/재시작, 승인형 DB 집계가 첫 구현 목표다. 이는 이미 모두 출시 가능하다는 선언이 아니다. 원본 보존과 로딩 필요성 판단을 먼저 완성한다.

임의 Python 실행은 격리 backend 검증 뒤, 고급 차트는 각각의 oracle 준비 뒤 연결한다. 증분 결합은 안정 key/version을 확인할 수 있을 때만 지원한다. 기존 통계·고급 도구는 제거하지 않지만 검증된 지원 범위를 확대 해석하지 않는다.

[평가 manifest](../tests/fixtures/agent_readiness_manifest.json)는 문항 ID·prompt/reference hash·요구 매핑·oracle 유무·의미 충돌·기존 부분 검사 경로를 보존한다. 현재 87개는 기존 oracle 정의가 있고 113개는 UNGRADED다. L2_026 구간 경계와 L2_032 건수/결측 분모 충돌은 미해결로 명시했다. category 기반 매핑을 200개 문장의 의미 검토 완료로 계산하지 않는다.

## 실행 계약과 독립 평가

- [기계 검증용 schema](../tests/fixtures/agent_contracts/execution.schema.json): 보호 root는 ready/raw/불변 보호 상태, 원격 적재는 fingerprint와 승인 상태, 제출 상태 unknown은 자동 재시도 금지. PASS에는 원본 보존·정답·실제 증거가 필요하다.
- [fixture와 로컬 모델 질문](../tests/fixtures/data_preservation_v1.json): 음수·중복 수치·NULL·그룹을 가진 합성 데이터 및 이름 변경 변형. schema/데이터는 알려주되 oracle와 기대 행동은 모델에 주지 않는다.
- [원본 보존 acceptance](../tests/test_data_preservation_acceptance.py): 필터→작업 복사본 변환→집계→실제 PNG→재시작, cache 변형 격리, quota 초과 적재 실패에서 실제 영속 저장소의 원본·metadata·계보를 확인한다.
- [평가 계약 검사](../tests/test_agent_readiness_contract.py): 금지 상태·허위 PASS·잘못된 histogram 모집단을 거절하고, 임시 저장소 삭제 후에도 진단/PNG 증거가 남는지 확인한다.

이 JSON schema는 평가·설계 계약이며 운영 runtime이 아직 강제하지 않는다. SQL 의미 대조, 자산 권한·참조 일치, 상태 전이와 실제 파일 무결성은 schema만으로 입증되지 않는다. 도구 acceptance가 통과해도 자연어로 root를 자동 선택하는 전체 agent 여정 통과와 구분한다.

## 실제 모델 초기 측정

환경: 현재 개발 호스트의 localhost Ollama `gemma4:e4b`; fixture만 사용, DB 실행기 미연결, 승인 자동 제출 없음. [환경 기록](evaluation/preparation_2026-09-24/environment.json), [60초 기본 제한 결과](evaluation/preparation_2026-09-24/live_preservation.json).

| 사례 | 결과 | 해석 |
|---|---|---|
| PRES_01: 전체 histogram→그룹 필터→원본 복귀→재시작 | 4단계 PASS | 원본 보존, 실제 histogram 입력/count 일치, 추가 원격/승인 0. 필터 단계만 모델 1회, 나머지는 결정적 경로 |
| PRES_02: 평균→원본 histogram | 첫 단계 ReadTimeout, 다음 단계 NOT_RUN | 첫 단계 약 117초, 모델 시도 2회 중 1회 오류. 원본은 보존되었지만 분석 실패 |
| PRES_03: 차트명을 지정하지 않은 도수분포 요청 | ReadTimeout | 모델 시도 1회, 약 60초. 원본 보존·원격 0이어도 분석 성공이 아님 |

첫 30초 제한의 시도도 [초기 기록](evaluation/preparation_2026-09-24/live_preservation_initial_30s.json)에 남겼다. 이후 운영 기본 model timeout 60초·turn budget 180초로 맞춘 결과를 기준선으로 사용한다. timeout을 숨기거나 성공 실행만 선택하지 않았다. runtime의 model_calls는 실패한 호출을 제외할 수 있어 callback 기반 model_attempts/model_errors를 별도 추가했다.

PRES_01 필터 단계는 약 42초였다. 이번 측정은 작은 합성 데이터의 단회 실행이고 일부 회귀 검사와 겹쳤으므로 p95·운영 SLO·대용량 성능 증거가 아니다. PRES_03의 [진단 포함 재실행](evaluation/preparation_2026-09-24/live_preservation_trace.json)은 **73.028초, 모델 1회, PASS**였다. 원래 60초 ReadTimeout 실패도 유지하며, 반복 실행의 편차가 있으므로 안정화됐다고 판단하지 않는다. HTTP read timeout은 전체 응답 wall time 상한과 다르므로 73초 응답이 발생할 수 있다. 초기 전체 실행의 상세 임시 trace는 보존하지 못했으며 turn별 정답/보존/호출 결과와 오류 ID만 남아 있다. 평가기를 보강해 이후 실행은 진단 metadata와 PNG를 영속화하고 그 보존 자체도 검사했다. 이번에는 UI 가독성이나 전체 Databricks 로딩 여정을 실행하지 않았다.

## 다음 구현의 순서

1. **T05/T06**: 아래의 메타데이터 기반 부모/registry 탐색을 전체 분석 계획과 차트 도구까지 확장한다. root/branch/candidate 역할과 LoadPlan·사용자 선택 상태는 아직 runtime에 연결해야 한다.
2. **T07 P0**: 후보 적재의 검증·발행·선택과 보호 원본 수명을 강제. 디스크 부족/실패/재시작에서도 원본 보존.
3. **실제 모델 실패 추적**: timeout 단계의 prompt/tool 크기·모델 지연·도구 선택을 관측하고, 동일 fixture·모델 설정에서 실패 원인을 분리한다. timeout 연장만으로 해결됐다고 처리하지 않는다.
4. 각 변경을 J21~J24로 연결하고 기존 여정 회귀 후, SQL 집계/raw/hybrid 및 EDA 추천을 확대한다.

이제 첫 구현 묶음을 시작할 기준과 실행 가능한 부분 검사는 마련됐다. 전체 T01/T02의 모든 oracle와 여정이 완성된 것은 아니며, 새 schema를 문서에 두는 것만으로 데이터 보존 기능이 구현된 것도 아니다.

## T05/T06 로컬 자산 선택 1차 구현 (10:39 KST)

`select_reusable_dataset`은 현재 dataset을 먼저 판정하고, 부족하면 metadata만으로 같은 source·snapshot의 부모 계보를 가장 가까운 단계부터 탐색한다. 계보 밖 registry 자산은 snapshot이 명시되고 충분한 후보가 정확히 하나일 때만 선택한다. 여러 후보, 다른 snapshot, 불완전한 데이터는 원격 필요 상태로 남기며 `current_result_only`는 현재 결과를 벗어나지 않는다. `use_dataset`과 `local_analysis_sql`은 선택한 부모 ID·선택 이유를 결과에 남긴다. SQL 결과의 `parent_id`도 실제 계산에 사용한 자산을 가리킨다. 존재하지 않는 SQL 컬럼은 재조회 사유 대신 수정 가능한 SQL 오류로 취급한다.

[재사용 회귀](../tests/test_analysis_dataset_selection.py)는 필터 해제, 집계에서 raw 복귀, SQL 출력 별칭, 같은 버전 registry 단일/중복, snapshot 불일치, 불완전 부모, 현재 결과 한정, 영속 저장소 재시작을 확인한다. 도구 수준에서 원격 제안 0회와 원본 불변을 확인했다. 이 범위는 전체 agent의 자연어 의도 판단, LoadPlan 후보 검증, 대규모 실제 데이터 성능, Databricks 승인을 검증한 결과가 아니다.

`prepare_histogram`에도 선택한 `dataset_id`를 전달할 수 있게 했다. 서로 독립적인 원본이 여러 개일 때 ID가 없으면 `needs_context`로 멈춘다. ID가 있으면 해당 원본/부모의 직접 계산 결과만 재사용하며, 두 원본의 기존 PNG가 모두 있을 때 자동 캐시 경로가 마지막 이미지를 임의로 띄우지 않는다. `current_result_only`와 `where_sql`을 함께 쓸 때는 필터를 실제 로컬 결과에 적용한다. 값이 한 건만 남아 히스토그램이 부적합하면 `no_valid_chart`를 반환한다. 임의 schema의 두 snapshot·기존 이미지·명시 ID 복구를 도구와 production graph 양쪽에서 검증했고 원격 실행 0회였다.

남은 구조적 공백: 현재 선택한 root/branch의 영속 상태와 자연어의 “이전 결과/그중/원본으로” 참조 해석은 아직 통합되지 않았다. 모델이 `show_chart`를 직접 호출하는 경로까지 선택 의미가 강제된 것은 아니다. 차트 외의 모든 데이터 도구도 공통 resolver에 연결해야 한다.

## 재실행 명령

프로젝트 루트에서 v1 venv의 경로를 그대로 사용한다. 아래 마지막 명령만 localhost 모델을 호출하며 운영 DB는 연결하지 않는다.

```bash
.telly_runtime/v1-venv/bin/python scripts/build_agent_readiness_manifest.py
.telly_runtime/v1-venv/bin/python -m unittest tests.test_agent_readiness_contract tests.test_data_preservation_acceptance
.telly_runtime/v1-venv/bin/python scripts/evaluate_data_preservation.py --live-local-model --output .telly_runtime/evaluations/preservation.json
```

## T05~T07 보강·실제 검증 (12:20 KST)

현재 선택 dataset ID를 대화별 SQLite에 저장하고 재시작 뒤 prompt·차트 도구·UI에서 복원한다. 새 raw 결과가 답변으로 검증되면 기준을 이동하지만, SQL 집계는 기존 raw EDA 기준을 대체하지 않는다. 새로운 asset의 `role`과 `root_id`를 기록하고 legacy asset은 알 수 없는 계보를 추정하지 않는다. `source_plan`은 승인 전 SQL AST의 실제 테이블과 제시한 출처를 대조하고, 실행기는 결과 컬럼·행 폭·컬럼 한도·배치별 프레임 크기를 검증한 후에만 발행한다. 짧게 반환된 fetch batch를 EOF로 오인하지 않는다. 이것은 전체 LoadPlan 의미 대조·디스크 staging·대용량 부분 scan의 완료가 아니다.

UI 목록은 저장 시 만든 최대 5행의 작은 미리보기를 사용해 화면 갱신 때 전체 Parquet DataFrame을 읽지 않는다. 이전 형식 자산은 무리하게 전체 복원하지 않고 미리보기를 생략한다. localhost:8502 프로세스를 최신 코드로 재시작한 뒤 합성 예제 7행을 선택하고 실제 histogram PNG가 표시됨을 확인했다. 재시작 전에는 이전 세션 객체와 새 UI 코드가 섞여 `selected_dataset_id` AttributeError가 발생했다. 기존 대화의 4턴 계산 결과는 재시작 후 화면에 복원됐다. [출시 판정](release_readiness_2026-09-24.md)에 재시작 요구를 기록했다.

실제 모델 실패를 새로 재현했다. `PRES_02` 평균 요청은 `aggregate_dataset` 도구가 ready 결과를 만들었지만 완료 판정이 그 근거를 누락해 126.538초/모델 4회 뒤 exhausted였다. 도구 결과를 공통 계산 검증에 연결한 후 같은 모델·fixture에서 평균→원본 histogram 2/2 PASS, 원격/승인 0회였다. 첫 요청 55.274초라 지연 문제는 남는다. `PRES_03`은 한 번 60초 ReadTimeout으로 실패했고 다음 단회 재검사에서는 60.569초 PASS였다. [실패](evaluation/preparation_2026-09-24/live_preservation_completion_failure.json), [수정 후](evaluation/preparation_2026-09-24/live_preservation_aggregate_fix.json), [변동 재검사](evaluation/preparation_2026-09-24/live_preservation_recheck.json)를 모두 보존했다.

최종 계약 회귀와 실제 모델 결과는 분리한다. application **201/201**, migration **128/128**, Level 3 **17/17**, 전체 참고 runner **217/217**, compileall·diff 검사가 통과했다. 참고 runner의 200문항은 생성된 agent의 자연어 성공률이 아니다. 운영 DB 재조회, DeepEval·Spider 공식 평가, 배포 호스트 성능/접근제어는 아직 확인하지 않았다.
