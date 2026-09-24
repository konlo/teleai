# 데이터 분석 agent 출시 판정 — 2026-09-24

판정: **운영 출시 NO-GO, localhost 개발 화면은 사용 가능.** 현재 브랜치의 계약 회귀 통과는 실제 모델·대용량·Databricks·배포 환경을 포함한 출시 성공률이 아니다. 운영 조회를 새로 실행하지 않았으며 사용자별 정확한 SQL 승인을 유지했다. [draft PR #68](https://github.com/konlo/teleai/pull/68)은 리뷰·병합·배포하지 않았다.

## 이번 실행으로 확인한 것

| 항목 | 증거 | 판정 범위 |
|---|---|---|
| 조회 출처와 승인 SQL | AST 출처 결합, 잘못된 출처 승인 전 거절, 결과 컬럼/행 구조·배치 크기 확인. 후보 오류에서 기존 root 보존 | mock connector/합성 fixture |
| 원본과 현재 선택 | root/파생/집계 역할과 root ID, 선택 상태 재시작 복원. 승인형 집계가 선택된 raw EDA 기준을 덮지 않음 | 영속 로컬 저장소/합성 fixture |
| 로컬 분석 완료 | 실제 `gemma4:e4b`의 `aggregate_dataset` 평균 결과가 처음에는 저장되고도 완료 근거로 인정되지 않아 126.538초 후 exhausted. 공통 근거 검증에 연결한 재실행은 평균→원본 histogram 2/2 PASS, 승인·원격 실행 0회. `PRES_03`은 한 번 ReadTimeout, 이후 60.569초 PASS. [실패 기록](evaluation/preparation_2026-09-24/live_preservation_completion_failure.json), [수정 후 기록](evaluation/preparation_2026-09-24/live_preservation_aggregate_fix.json), [변동 재검사](evaluation/preparation_2026-09-24/live_preservation_recheck.json) | 단회 성공은 안정 성공률 증거가 아님. 첫 평균 55.274초 |
| 화면과 대화 보존 | localhost:8502 프로세스를 최신 코드로 재시작. 새 합성 대화에서 선택한 7행 원본의 실제 histogram PNG를 표시하고, 이전 4턴 대화의 40→40→20→4 결과가 복원됨. 목록 미리보기는 저장된 5행만 읽어 전체 DataFrame을 화면 갱신 때 복원하지 않음 | 개발 호스트, 합성 데이터 |
| 회귀 | application 201/201, migration 128/128, Level 3 17/17, 전체 참고 runner 217/217, compileall·`git diff --check` PASS | 참고 runner 200문항은 실제 자연어 agent 성공률이 아님 |
| 배포 사전 점검 | `local-desktop` READY(프로젝트 내부 저장소 경고), `private-single-user` NOT READY(영속 볼륨·외부 접근제어 미설정) | 설정 검사만 수행. Databricks 연결·SQL 실행 미포함 |

## 13:15 KST 추가 점검

- 승인된 원격 결과를 1,024행 이하 배치로 Parquet 후보 파일에 쓰고, 모든 배치·quota 검증 후 메타데이터를 발행한다. 신규 원격 자산은 SQLite BLOB 대신 파일을 사용한다. 이전 BLOB 자산은 계속 읽힌다. 검사·지정 컬럼 프로파일은 신규 파일에서 전체 프레임을 복원하지 않는다. 중간 배치 타입 불일치·용량 초과 시 후보 파일을 폐기하고 기존 원본/선택을 유지하는 회귀를 추가했다.
- 로컬 모의 커서 100,000행 × 64열 정수 결과: 적재 1.185초, Parquet 32.79 MiB, 프로세스 최고 RSS 178.1 MiB. 이 수치는 합성 단일 프로세스 측정으로 Databricks 전송·동시 세션·실제 데이터형 성능을 대표하지 않는다.
- 실제 `gemma4:e4b` 재검사 [PRES_02/PRES_03](evaluation/preparation_2026-09-24/live_preservation_streaming_recheck.json): 3/3턴 PASS, 각 45.007/8.324/51.748초, 추가 승인·원격 실행 0회, 원본 불변. 이전 PRES_03 ReadTimeout 기록은 남아 있다.
- 신규 코드에 대한 application 206/206, migration 128/128, Level 3 17/17, 전체 참고 runner 217/217, compileall·`git diff --check` PASS. 파일 staging commit `e00a370`의 [GitHub Actions run 35955184372](https://github.com/konlo/teleai/actions/runs/35955184372), 문서 commit `c3b88b0`의 [run 35955354672](https://github.com/konlo/teleai/actions/runs/35955354672), 필터 복구 commit `356ebcd`의 [run 35956329709](https://github.com/konlo/teleai/actions/runs/35956329709)가 deterministic-validation 전체 PASS. 참고 runner 200문항은 실제 자연어 성공률로 계산하지 않는다.
- 추가 실제 모델 [PRES_01 실패](evaluation/preparation_2026-09-24/live_preservation_root_roundtrip_final.json)에서 필터 조건 `zone=east`의 히스토그램을 요청했으나 Databricks 방언에서 이중 따옴표 식별자가 문자열로 해석돼 3회 모델 호출/77.783초 후 `exhausted`였다. 범위를 틀리게 실행하거나 원본을 변경하지는 않았다. 방언에 맞는 백틱 조건과 검증된 단일 로컬 원본의 결정적 계획을 추가했다. [동일 4턴 재검증](evaluation/preparation_2026-09-24/live_preservation_root_roundtrip_fixed.json)은 4/4 PASS; 실패했던 필터 턴 0.070초·모델/원격 0회, 원본 복귀와 재시작 복귀도 PASS. 임의 schema명 두 종류에서 정확한 필터 분포·원본 보존을 자동 회귀로 확인했다.
- 최신 localhost 화면에서 별도 `period=2026-08` 필터 히스토그램 요청을 제출해 실제 PNG와 **빈도 합계 4**를 확인했다. 진단 로그는 8.145초, 로컬 도구 1회, 모델 0회, 원격 조회/승인 0회를 기록했다. 기존 7행 원본은 유지됐다. 캐시 후보 검사에서 실제 거절이 아닌 `request_scope_rejected` 로그가 생기던 잡음도 제거하고 두 schema 회귀로 확인했다.

## 출시를 막는 일

1. **실제 원격 적재 검증**: 변경된 승인 SQL 출처·범위 확인, 승인→Databricks 1회 실행→후보 저장→차트/후속 분석→거절·실패·재시작의 J22~J24를 운영 연결에서 재검증해야 한다. 이 단계는 조회별 사용자 승인이 필요하다. 현 시점에는 새 원격 SQL을 제안/실행하지 않았다.
2. **실제 모델 안정성**: PRES_02가 수정 전 실제 오류를 냈고 PRES_03은 이번 반복에서 60초 ReadTimeout이었다. 수정 후 PRES_02 2턴은 통과했으나 모델 지연과 실패 편차가 있다. 반복/held-out 핵심 여정의 성공률, 복구율, p95를 아직 계산할 수 없다.
3. **대용량 실행층**: 원격 배치의 행·컬럼·메모리 제한, 파일 staging, 지정 컬럼 프로파일과 UI 미리보기의 전체 복원 방지를 보강했다. 명시 컬럼 차트·집계·통계·시계열·피벗·그룹 비교/요약·이상치 요약과 단일 명시 컬럼 로컬 SQL은 필요한 Parquet 컬럼만 읽는다. 자동 추천은 전체 파일을 스캔하되 배치별 표본만 복원한다. `SELECT *`, 복잡한 SQL, join 및 원본 행을 파생하는 이상치 선택은 한도 안에서 여전히 전체 DataFrame을 복원한다. 이 경로에는 Parquet metadata 기반의 사전 추정 한도(기본 128 MiB)를 적용했고 초과 시 원본과 선택을 보존하며 거절한다. 기존 로컬/파생 자산은 SQLite BLOB에 저장한다. 여러 사용자/쿼리의 합산 메모리·시간 예산, 복잡한 데이터형/희소 null 배치, 실제 배포 호스트 측정은 미완료다. 추정치는 엄격한 peak 메모리 상한이 아니며 큰 단일 값·동시 세션의 할당을 제한하지 못한다.
4. **agent/평가 범위**: 34개 요구/24개 여정의 전체 독립 수용 결과와 113개 미채점 참고 문항이 남았다. DeepEval judge 보정과 Spider 2.0 공식 SQLite subset 실행·점수는 아직 없다. 임의 분석 코드의 격리 실행과 범용 다중 패널/연속 EDA 발견 검증도 출시 기능으로 주장할 수 없다.
5. **배포 환경**: 현재 `local-owner`는 loopback 단일 사용자 전제다. 목표 호스트·영속 저장소·접근제어·Ollama 배치, PR #68 리뷰/병합, 배포 후 smoke/rollback이 확정되지 않았다. 개발 프로세스의 Streamlit hot reload는 이전 런타임 객체를 유지해 새 UI 속성 오류를 만들었고 전체 프로세스 재시작으로 복구했다. 배포는 세션 객체를 포함한 프로세스 재시작이 필요하다.

운영 자료와 승인되지 않은 query를 평가기나 외부 judge로 전송하지 않는다. 출시 범위를 좁힐 경우에도 위 해당 경로의 독립 oracle·실제 모델·실제 화면/연결 검증이 먼저 필요하다.

## 13:55 KST 추가 구현·검증

- 파일 기반 원본에서 필요한 컬럼만 읽는 공통 projection 경로를 차트·집계·통계·시계열·피벗·그룹 비교/요약·이상치 요약과 단일 명시 컬럼 로컬 SQL에 연결했다. 전체 건수 집계는 저장된 행 수로 영 컬럼 프레임을 구성하고, 로컬 `COUNT(*)`는 내부 행 마커로 계산한다. `SELECT *`는 전체 스키마를 유지한다. agent의 열 이름 식별도 프레임 전체 대신 영속 metadata를 사용한다.
- 파일 기반 임의 schema에서 전체 프레임 decode를 금지한 채 histogram PNG·평균/건수·통계·피벗·시계열·그룹 비교/요약·이상치 요약과 실제 agent histogram을 검증했다. 원본 파일 바이트·선택 ID는 불변이다. 잘못된 컬럼은 decode 전에 거절하고 wildcard SQL은 전체 컬럼을 보존한다.
- 합성 100,000행 × 64열 정수 데이터를 100회 배치로 파일에 저장하고 실제 agent에 `signal` 히스토그램을 요청했다. 적재 0.877초, 차트 응답 7.86초, 모델 0회, 전체 프레임 decode 0회, 파일 1.38 MiB, 프로세스 peak RSS 310.3 MiB. 반복·외부 서비스·동시 세션 측정이 아니며 작은 정수 반복값의 압축률이 실제 데이터와 다르다.
- 최종 로컬 회귀: application 210/210, migration 128/128, Level 3 17/17, 참고 runner 217/217, compileall·diff check PASS. 참고 runner의 200문항은 자연어 agent 성공률이 아니다. 이번 projection 변경 후 실제 모델 반복·운영 Databricks는 아직 실행하지 않았다.

## 14:11 KST 실제 화면 실패와 복구

- 저장된 합성 대화에서 `value의 boxplot을 보여줘`를 실제 UI로 실행했을 때 `boxplot을`의 한국어 조사가 차트명 정규식의 영문 단어 경계와 충돌해 결정적 계획을 놓쳤다. 모델은 58.633초 뒤 `render_chart_spec`를 호출했으나 이전 4행 집계 결과로 박스플롯을 만들려고 해 유효값 최소 5개 조건에서 거절됐다. 이어진 모델 호출은 60초 ReadTimeout이었고 재개 후에도 누적 모델 시간 한도로 exhausted였다. 이 실패는 보존된 원본/선택을 바꾸거나 원격 조회를 실행하지 않았다.
- 영문 차트명에 한국어 조사가 붙은 요청을 해석하고, 이전 코드가 분류하지 못한 미완료 요청도 다음 단계에서 재분류하도록 수정했다. 단일 원본의 결정적 차트 계획은 전체 프레임 대신 요청 컬럼만 읽는다. 박스플롯은 유효값 2개부터 허용하되 5개 미만이면 사분위수 불안정 경고를 붙이고, 집계 결과를 원본 분포 박스플롯으로 잘못 그리지 못하게 거절한다.
- 같은 UI 대화에서 새 요청은 **7행 raw 원본**의 박스플롯을 모델 0회·원격 0회·7.881초에 생성했다. 처음에는 `render_chart_spec` 성공과 답변만 표시되고 PNG 카드가 누락됐다. UI의 도구명 고정 목록이 이 도구를 빠뜨린 것이 원인이었다. 저장된 카드 ID가 있는 모든 정상 도구 결과의 이미지를 표시하도록 바꾸고, 재시작 후 같은 대화에서 실제 박스플롯 PNG·범위·답변이 함께 복원됨을 브라우저에서 확인했다. AppTest가 이미지 요소와 PNG payload를 함께 검증한다.
- 이 수정까지 포함한 로컬 최종 gate는 application 215/215, migration 128/128, Level 3 17/17, 참고 runner 217/217, compileall·diff check PASS. 실제 운영 DB와 배포 환경의 출시 gate는 여전히 미완료다.

## 14:37 KST 대용량 자동 추천과 반복 여정

- 파일 기반 자동 차트 추천은 필요한 모든 행을 메모리에 복원한 다음 20,000행을 표본 추출하던 경로를 바꿨다. Parquet 행 수와 자산 메타데이터를 대조하고, 고정 seed의 균등 무작위 행 위치를 선택한 뒤 256행 배치에서 해당 행만 읽는다. 추천은 여전히 파일 전체를 **스캔**하지만 전체 DataFrame은 복원하지 않는다. 표본 20,000행과 모집단 행 수는 카드 scope에 그대로 표시한다.
- 합성 100,000행×64 정수열·33.48 MiB 파일의 자동 추천 3장: 적재 0.448초, 추천 8.104초, 전체 프레임 decode/전체 projection 0회, 프로세스 peak RSS 176.4 MiB. 합성 단일 프로세스 결과이며 실제 문자열/복잡 타입·동시 세션·호스트 성능은 대표하지 않는다. 원본 파일 바이트와 선택 ID 불변을 회귀로 확인했다.
- 실제 로컬 모델의 PRES_03 자연어 분포 요청 반복은 변경 전 2/3 PASS(60.954초, 69.050초), 1회 60초 ReadTimeout이었다. 실패 보고서의 `KeyError`는 평가기 도구 호출 계측에서 발생한 2차 오류였고, 진단 로그가 원래의 ReadTimeout을 확인한다. 평가기를 실제 `tool_started` 진단 이벤트 기준으로 수정했다. 이름 없는 명확한 분포 그림 요청은 스키마에 근거한 로컬 histogram 경로로 복구하도록 보강했고, 변경 후 3/3 PASS(7.818/7.960/7.761초), 모델·원격 0회, 실제 도구 1회, 원본 불변이었다. 이는 이 표현의 결정적 경로 근거이지 일반적인 모델 성공률 근거가 아니다.
- 추가 보존 여정: PRES_01 원본→필터→원본 복귀→재시작 4/4 PASS(모델/원격 0회), PRES_02 평균→원본 histogram 2/2 PASS(첫 턴 모델 1회·63.798초, 둘째 턴 9.188초). 단일 평균의 모델 지연은 여전히 출시 위험이다. 평가 보고서는 `docs/evaluation/preparation_2026-09-24/`에 보존했다.
- 범용 상관·단일 수치 집계의 로컬 계획 검증은 파일 기반 원본 전체 대신 요청 컬럼만 읽게 했다. 임의 schema의 파일 기반 평균→상관 후속 여정을 전체 원본 decode 금지, 모델/원격 0회, 원본 바이트 불변으로 검증했다. 실제 UI AppTest에서는 boxplot 다음의 자연어 분포 요청이 두 번째 PNG로 표시됐다.
- 최종 로컬 gate는 application 220/220, migration 128/128, 참고/Level 3 runner 217/217(그중 Level 3 17/17), compileall·diff check PASS다. 이전 SQLite BLOB 자산의 자동 추천도 전체 프레임 decode 없이 동작한다. 운영 Databricks, 공식 DeepEval/Spider 2.0, 실제 배포 호스트는 이 gate에 포함되지 않는다.
- 코드 commit `f85a3bd`의 [GitHub Actions run 35961029122](https://github.com/konlo/teleai/actions/runs/35961029122)가 전체 deterministic-validation PASS. 새 프로세스로 localhost:8502를 재시작한 뒤 저장된 대화에서 `value 값들이 어느 구간에 얼마나 모여 있는지 그림으로 보여줘. 보유 데이터만 사용해줘.`를 제출했고 실제 7행 원본 PNG와 답변을 화면에서 확인했다. 진단 로그는 7.718초, 로컬 도구 1회, 모델 0회, 추가 원격 조회 0회를 기록한다.
- 배포 preflight 재실행: `local-desktop` READY(저장소 위치 경고), `private-single-user` NOT READY(외부 영속 저장소 및 접근제어 미설정). 별도 설치된 `qwen3:8b`의 PRES_02 첫 평균 요청은 60초 ReadTimeout으로 실패했고 원본/원격 경계는 유지됐다. 현재 `gemma4:e4b` 대체가 지연·정확성을 개선한다는 증거가 없으므로 모델을 교체하지 않았다.

## 16:17 KST 전체 프레임 복원 경계

- 파일 기반 `SELECT *`/복잡한 로컬 SQL, join, 이상치 행 선택은 Parquet metadata에서 원본 전체 복원량을 추정하고, join·행 선택은 예상 출력량도 더한 다음 `TELLY_MAX_FULL_READ_BYTES`(기본 128 MiB)를 넘으면 `full_frame_budget`으로 거절한다. 조인 key·복구 계획의 필요한 컬럼은 전체 원본 대신 부분 읽기로 확인한다. 한도 이하에서는 기존 동작이 유지된다.
- 300행×32열 합성 원본 두 개에 5 KiB 한도를 적용해 전체 decode를 강제로 막은 상태에서 wildcard SQL, join, 이상치 inlier 파생이 모두 원본 decode 전에 거절되고, 명시 컬럼 SQL은 성공하며 원본 바이트·선택 ID가 유지됨을 확인했다. 합성 100,000행×128 수치열(Parquet 66.95 MiB)은 `SELECT * LIMIT 2`를 추정 228.1 MiB로 거절하고 명시 컬럼 평균은 성공했다. 이 단일 프로세스 도구 호출은 0.091초, 전체 decode 0회, peak RSS 267.4 MiB였다. 적재 비용을 포함한 RSS이며 모델·실제 DB·동시 세션 성능은 나타내지 않는다.
- 거절 사유가 끝까지 해소되지 않으면 agent 응답은 완료를 주장하지 않고 `blocked/full_frame_budget`으로 분류하며 필요한 컬럼·조건 또는 선행 집계를 안내한다. 실제 모델이 이 안내를 보고 스스로 대안을 선택하는지에 대한 여정 검증은 남아 있다.
- 변경 후 로컬 application 222/222, migration 128/128, 전체 참고 runner 217/217(그중 Level 3 17/17), compileall·`git diff --check` PASS. 참고 runner는 실제 자연어 agent 성공률이 아니다.
- 구현 commit `b0b56ed`의 [GitHub Actions deterministic-validation](https://github.com/konlo/teleai/actions/runs/35969152630) PASS. localhost:8502 프로세스를 새 코드로 재시작하고 health 응답 `ok`를 확인했다. 브라우저 대화 재검증과 운영 연결 검증은 이 단계에 포함되지 않았다.
- 이 검사는 일부 전체 복원 도구의 사전 경계다. 모든 `FrameCache` 호출을 중앙에서 차단하는 강제 한도는 아니고, 메타데이터 추정과 실제 pandas/DuckDB peak가 다를 수 있다. 다른 전체 복원 경로의 인벤토리, 전역 자원 예산과 동시 요청 부하 검증은 계속 남는다. 운영 출시 NO-GO 판정은 유지한다.

## 17:20 KST metadata·답변·근거 읽기 보강

- 승인된 `SELECT *` 관측의 스키마 확인은 데이터 전체 대신 Parquet schema에서 dtype을 얻는다. 영속 결과를 답변에 붙일 때는 앞 15행만 제한된 배치로 읽고, 피벗·집계·cohort 결과의 증거 해시는 256행 배치로 재계산한다. 이전에는 각 경로가 저장된 DataFrame 전체를 복원했다. 원본·파생·집계 역할과 선택 상태는 바꾸지 않았다.
- 파일 기반 300행×32열에서 전체 `FrameCache.__getitem__`을 금지한 채 스키마·15행 답변 미리보기를 확인했다. 파일 기반과 이전 SQLite BLOB 자산의 2,500행 숫자·문자열·결측 데이터에서 배치 해시와 기존 전체 프레임 해시가 일치했다. 테스트 모델의 실제 graph 여정에서는 과대한 wildcard 조회 거절→명시 컬럼 로컬 조회→정확한 2행 응답으로 복구했고 원본/선택은 유지됐다. 모의 원격 connector를 연결해도 승인 요청·실행은 0회였다. 이 테스트 모델 결과는 실제 모델 능력의 증거가 아니다.
- [실제 로컬 모델 단회](evaluation/preparation_2026-09-24/live_full_read_preview_2026-09-24.json)는 보유된 합성 데이터에서 `event_key` 앞 두 값을 요청하자 `inspect_dataset` 미리보기로 0, 1을 정확히 답했다. 87.781초가 걸렸고 원본·선택은 불변이었다. 이 실행은 큰 읽기 거절을 만나지 않았으므로 실제 모델의 거절 후 대안 선택 성공률로 계산하지 않는다. 지연도 출시 위험이다.
- 이 변경은 확인된 schema/답변/근거 읽기 경로를 줄였지만 중앙 전체 읽기 강제 한도와 모든 호출의 동시 메모리 한도를 구현하지 않았다. 실제 운영 조회·다양한 데이터형·동시 부하·반복 모델 성공률 검증 전에는 운영 NO-GO다.
- 최종 로컬 gate: application 225/225, migration 128/128, 참고/agentic runner 217/217(그중 Level 3 17/17), compileall·diff check PASS. 모의 원격 connector 연결 경계 변경 후 관련 집중 회귀 5/5 PASS.

## 20:10 KST 선택 원본의 빠른 로컬 경로

- 선택한 원본의 단일 컬럼 앞 1~5행 요청은 저장된 `inspect_dataset` 미리보기에서 값·출처·선택 ID를 검증해 답한다. 정렬·필터·상위값·6행 이상 요청에는 적용하지 않는다. 합성 300행×32열의 앞 2행은 0/1로 일치했고 0.091초, 모델·원격 호출 0회, 원본 파일·선택 불변이었다. 이전 같은 합성 질문은 실제 모델 경유 87.781초였다. [새 경로 근거](evaluation/preparation_2026-09-24/go_local_fast_paths_2026-09-24.json).
- **명시적으로 선택한** 완전한 raw 원본의 단일 수치 집계는 출처·스키마·dtype·범위를 검증한 뒤 로컬 SQL로 실행한다. 선택이 없는 평가기의 모델 도구 오류 주입은 그대로 관찰된다. 합성 PRES_02 평균→원본 histogram은 2/2 PASS, 첫 턴 0.101초, 모델·추가 원격 0회, 원본 불변이었다. 이는 결정적 경로의 개선이며 모델의 자율 복구 성공률 증거가 아니다.
- 현재 코드의 application 229/229, migration 128/128, Level 3 17/17 PASS. `local-desktop` preflight READY(영속 저장소 경고), `private-single-user` NOT READY(영속 볼륨·외부 접근제어 미설정). 실제 Databricks 승인→조회와 배포 호스트 검증이 없어 **운영 출시 NO-GO**를 유지한다.
- commit `9a111fa`의 [GitHub Actions run 35991339107](https://github.com/konlo/teleai/actions/runs/35991339107)은 migration/application/agentic/reference/compile 전 단계 PASS. 전체 참고 runner도 로컬 217/217 PASS다.

## 다음 운영 검증 승인 단위

실제 연결 smoke의 첫 승인 대상은 정확히 `SELECT * FROM workspace.default.bank_loan LIMIT 10` 한 번이다. 이는 현재 저장된 스키마 힌트를 신뢰하지 않고 실제 반환 컬럼과 최대 10행의 후보 staging·검증·발행을 확인하는 작은 조회다. 테이블이 변경·이동됐다면 실패로 기록하고 임의의 다른 SQL을 자동 실행하지 않는다. 반환된 10행을 전체 모집단 통계나 운영 histogram의 정확성 근거로 사용하지 않는다. 추가 조회는 매번 별도 SQL 승인으로 진행한다. 이후 기존 원본/선택 보존, 실패/거절/재시작 복구와 후속 로컬 분석을 실제 연결에서 확인해야 한다.

현재 호스트는 loopback 단일 사용자 개발 환경만 READY다. 별도 1인용 호스트의 영속 볼륨·외부 접근제어가 제공되고 그 호스트에서 preflight·smoke·rollback이 성공해야 운영 판정을 갱신할 수 있다.
