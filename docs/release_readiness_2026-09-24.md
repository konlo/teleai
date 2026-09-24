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
3. **대용량 실행층**: 원격 배치의 행·컬럼·메모리 제한, 파일 staging, 지정 컬럼 프로파일과 UI 미리보기의 전체 복원 방지는 보강했다. 하지만 차트·로컬 SQL·통계 등 많은 분석 도구는 여전히 전체 DataFrame을 복원한다. 기존 로컬/파생 자산은 SQLite BLOB에 저장한다. 여러 사용자/쿼리의 합산 메모리·시간 예산, 복잡한 데이터형/희소 null 배치, 실제 배포 호스트 측정은 미완료다. 큰 값을 가진 단일 배치의 peak allocation까지 현재 계약으로 제한하지 못한다.
4. **agent/평가 범위**: 34개 요구/24개 여정의 전체 독립 수용 결과와 113개 미채점 참고 문항이 남았다. DeepEval judge 보정과 Spider 2.0 공식 SQLite subset 실행·점수는 아직 없다. 임의 분석 코드의 격리 실행과 범용 다중 패널/연속 EDA 발견 검증도 출시 기능으로 주장할 수 없다.
5. **배포 환경**: 현재 `local-owner`는 loopback 단일 사용자 전제다. 목표 호스트·영속 저장소·접근제어·Ollama 배치, PR #68 리뷰/병합, 배포 후 smoke/rollback이 확정되지 않았다. 개발 프로세스의 Streamlit hot reload는 이전 런타임 객체를 유지해 새 UI 속성 오류를 만들었고 전체 프로세스 재시작으로 복구했다. 배포는 세션 객체를 포함한 프로세스 재시작이 필요하다.

운영 자료와 승인되지 않은 query를 평가기나 외부 judge로 전송하지 않는다. 출시 범위를 좁힐 경우에도 위 해당 경로의 독립 oracle·실제 모델·실제 화면/연결 검증이 먼저 필요하다.
