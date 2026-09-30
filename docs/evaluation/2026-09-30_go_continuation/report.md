# GO 잔여 작업 후속 결과

대용량 그룹 EDA와 로컬 자원 실패 복구를 보강했다. **범용 GO 판정은 아직 NO-GO**다. 실제 사용자 서버는 접속·로그 반출이 불가능하며 이번 검증 대상이 아니다. 이 개발 환경의 Databricks 추론과 SQL 연결도 HTTP 400으로 막혀 실모델 재평가는 진행하지 못했다.

## 완료한 변경

| 작업 | 변경과 검증 |
| --- | --- |
| 대용량 그룹 통계 | 20,000행을 넘는 영속 dataset에서 필요한 컬럼만 1,024행씩 읽는다. 로컬 DuckDB로 count·sum·mean·median·min·max·조건부 count/percent/mean을 집계한다. 원격 재조회와 원본 전체 pandas 복원을 사용하지 않는다. |
| 실행 한도 | SQL 엔진 메모리128 MB, 임시공간512 MB, 시간30초, 입력 batch8 MB, 그룹/출력 최대1,000개를 적용한다. 모든 입력 행을 읽었는지 검증한 뒤 결과를 발행한다. 엔진 메모리 제한은 Python 프로세스 전체 RSS 제한과 다르다. |
| 정확성·보존 | 조건부 지표의 분모, 결측 그룹 제외, 빈 조건부 그룹0/empty_value, 2^53 이상 정수 합계의 정밀도를 검증했다. 입력 중단·불완전 stream·그룹/출력 한도 실패에서 파생 결과를 발행하지 않고 원본·선택 상태를 보존한다. |
| 자원 실패 복구 | MemoryError와 DuckDB OutOfMemory를 구체적인 도구 실패 관찰로 반환한다. agent가 같은 실패 호출을 반복하지 않고 다른 로컬 집계 도구를 선택할 수 있도록 후보와 제약을 제공한다. |
| 실행 정책 충돌 | 복구 지침의 오래된 무조건 SQL 승인 요구를 제거했다. 현재 runtime 정책·읽기 전용 검사·실행 장부를 따르고 자동 조회 환경에서는 승인을 다시 묻지 않도록 정리했다. 수동 승인 설정은 유지한다. |

## 확인한 근거

- [100만행 집계·차트](million_groups_final.json): 필요한3컬럼, batch최대1,024행, 977batch, 7그룹×8지표. 별도 행 생성 규칙으로 계산한 정답과 일치. 집계1.188초, 프로세스 peak RSS282,214,400 bytes. 원본 해시·선택·재시작 후 결과 보존. PNG 생성/입력 dataset 연결/이미지 실물 확인. 합성 데이터의 한 실행 측정이며 운영 서버 성능이나 SLA가 아니다.
- [agent 및 복구 관련 검사](repair_and_group_tests.txt): 17 PASS. 자연어 그룹 평균·중앙값 요청이 기존 raw dataset에서 완료되는 production graph 경로, MemoryError/OutOfMemory 주입→동일 실패 실행1회→중복 차단→대체 도구 결과9.0·원본 digest 보존을 확인했다. 모델은 대역이며 실제 모델의 자유계획 성공률이 아니다.
- [제품 전체](application_final.txt): **490/490 PASS**. [migration](migration_final.txt): **136/136 PASS**.
- [reference 및 agentic](reference_tests.txt): **217/217 PASS, 58그림**. 이는 200문항 실모델 재채점이 아니다. 이후 복구 정책 문구 변경은 제품/migration 전체 검사로 다시 확인했다.
- [최초 제약 재현](group_baseline.txt): 기존 경로가 전체 컬럼 projection을 사용해 배치 전용 검증을 통과하지 못했다. 원본 손실을 재현했다는 의미는 아니다. [중간 검사](repair_policy_initial_failure.txt)는 오래된 승인 문구를 기대한 테스트 실패로 보존했고, 현재 정책을 확인하도록 고쳤다.
- [실서비스 재점검](provider_probe.json): 2026-09-30 14:50 KST, warehouse 제어 API200/STOPPED, 모델 추론400, SQL OpenSession400. 최소 모델 요청1회와 SELECT1 연결 시도1회이며 분석 SQL은 실행되지 않았다. 단기 장애/한도 초과/계정 제한의 원인은 확정하지 않았다.

## 여전히 남은 GO 차단 항목

1. **최신행 후속 EDA**: 일반 조건을 최신행 선택 전/후 중 어디에 적용할지 구분하고, 추가 통계·비균등 구간까지 정확한 범위를 유지해야 한다.
2. **복합 SQL**: 역할별 조인, JSON, OR/NOT/NULL, CTE/HAVING/DISTINCT 의미 계약·복구와 Spider interactive 실모델 검증이 남았다. 이전 공식0/10을 이번 작업으로 갱신하지 않는다.
3. **대용량 전체 여정**: 그룹 통계의 전체 컬럼 읽기는 해결했다. J21~J24의 혼합 raw/집계/추가 로딩 여정, 나머지 전체 읽기 도구, 실제 UI 경합·장치 ENOSPC는 별도 검증이 필요하다.
4. **독립 평가98문항**: 정답 비교기와 새 schema/표현/다회 대화의 실모델 평가가 남았다. 이번 새 회귀 검사로 해당 문항의 미채점을 통과로 바꾸지 않는다.
5. **실환경 재평가**: 설정된 공급자 연결이 회복된 후 동일 고정 여정을 다시 실행해야 한다. 사용자 서버 검증은 [서버 실패 진단 안내](../../server_failure_diagnostics.md)의 공유 가능한 상태만으로 확인한다. 서버 접속 정보나 원본 로그 반출은 요구하지 않는다.

다음 구현은 최신행 선택 전후 조건과 역할별 SQL 의미 검증을 우선한다. 외부 연결 차단은 이 구현 공백을 해결한 근거가 될 수 없다.

## Push와 Linux CI

구현 `5b7c7a3`, float64 정확 정수 범위 초과 경계 테스트 보강 `ac07806`을 작업 브랜치 `codex/agentic-analysis-rc-2026-09-14`에 push했다. [Linux CI 36676855817](https://github.com/konlo/teleai/actions/runs/36676855817)이 `ac0780688dedbd564ecc5f57692254c057bbd25e`의 migration·제품·agentic recovery·reference 전체·compile을 모두 통과했다. 이전 실행은 새 커밋 검증으로 대체됐다. 최종 CI는 승인 정책 수정과 정밀도 경계 검사를 포함한다. [검증 메타데이터](ci.json). 후속 기록 커밋은 문서만 변경하며 main 병합·실제 서버 배포를 의미하지 않는다.
