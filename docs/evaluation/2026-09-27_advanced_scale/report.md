# 고급 분석·대규모 적재 검증

2026-09-27. 고급 분석과 대용량 검증을 실제 실행했다. **전체 GO는 NO-GO 유지**한다. 로컬 대용량·지원 분석 계약은 통과했지만, 어려운 자연어 SQL의 의미 정확도와 검증 과잉 차단 문제가 확인됐다. 사용자의 추가 SQL 승인 면제 지시 후 실제 100,000행 적재와 EDA를 완료했다. [후속 수정·검증 보고](warehouse_agent_repair.md).

## 실행 결과

| 평가 | 결과 | 범위 |
|---|---|---|
| 통계 검정 | PASS, Level 2 고정 10/10 및 별도 Welch 검정 일치 | 실제 graph, 독립 SciPy oracle; 모델 호출 0 |
| 조인 | PASS, 5,000행 값·행 수·카디널리티 일치 | 실제 graph, 로컬 fixture, 모델 호출 0 |
| 시계열 | PASS, 일별 합계·시간대·빈 구간·PNG 일치 | 실제 graph, 독립 기준, 모델 호출 0 |
| Spider 2.0 고정 5문항 | SQL 제안 1/5 | 실제 Databricks serving 모델; 로컬 공개 SQLite; warehouse SQL 0 |
| 제안 SQL 공식 채점 | 정답 0/1 | local358. 547개 전체 benchmark 점수가 아님 |
| 차단된 초안 별도 공식 채점 | 정답 1/4 | local009 정답. 진단용이며 agent 성공에 합산하지 않음 |
| 로컬 100만 행 저장 | PASS | 변경 격리·LRU·재시작·SIGKILL 후 원본/차트 보존 |
| 로컬 100만 행 배치 적재→EDA | PASS | 실제 ingestion/graph 코드 + 합성 cursor, network/model 0 |
| 실제 Databricks 10만 행 적재 | 적재 및 복구·EDA PASS | SQL 1회, 21컬럼, 추가 재조회 0; 타입 전달 결함 수정 |

## 대용량 수치와 한계

최종 [배치 적재 결과](streaming_million_verified.json)는 100만 행, 5컬럼을 한 번에 최대 1,024행씩 978번 fetch하여 파일로 저장했다. 적재 2.652초, 적재 시 peak RSS 301,072,384 bytes(약 287MiB), 저장 Parquet 11,845,826 bytes, 적재 후 DataFrame cache 0 bytes다. 원본 전체를 RAM cache에 붙잡아 두지 않았다.

평균 49.95 → measurement >= 90 조건 평균 94.95 → 조건 해제 후 평균 49.95+히스토그램까지 독립 주기 oracle과 일치했다. 원본 digest와 선택 ID를 유지했고 신규 원격 호출 없이 이어갔다. EDA·장애·재시작 검사를 포함한 전체 18.590초, peak RSS 507,904,000 bytes(약 484MiB)다. 이 peak는 인터프리터·import와 누적 실행을 포함하며 할당량 증가 또는 RSS 상한을 뜻하지 않는다.

50,176행 수신 후 연결 중단 주입: 미완성 후보 발행 0, staging 정리, 기존 원본/선택 유지. 128KiB 수신 메모리 제한 주입: 첫 배치에서 MemoryError, 부분 발행 0, 원본/선택 유지. SIGKILL 과정은 [별도 저장소 시험](storage_million_verified.json)에 기록했다.

이는 실제 Databricks 네트워크 전송이나 21컬럼 업무 데이터의 메모리 측정이 아니다. 합성 시험 runtime만 100만 행 상한을 사용했고 제품 기본 10만 행 상한은 변경하지 않았다. LIMIT 결과는 coverage=unknown으로 남겨 전체 모집단으로 오인하지 않았다. 넓은 문자열 데이터·동시 세션·대규모 실제 원격 전송 검증은 아직 남아 있다.

## 고급 분석 실패 원인

[고정 문항 결과](spider/proposals.json), [채점 분모·원인 요약](spider_score_summary.json).

| 문항 | 관찰 | 필요한 공통 agent 보강 |
|---|---|---|
| local009 | 공식 정답과 일치하는 SQL을 조건 미확정으로 차단 | 출발/도착처럼 같은 테이블의 서로 다른 역할, JSON 필드 접근, OR 모집단을 표현·검증하는 요청 계약 |
| local221 | 홈 승리만 집계해 원정 승리를 누락; 관계 검증도 차단 | 역할별 집계 후 통합하는 다단계 계획과 목표 누락 검사 |
| local210 | 질문에 없는 2026년 조건 삽입 | 기간 기준·분모·증가율을 독립 명세로 고정하고 임의 조건 추가 차단 |
| local198 | 중앙값 단계 누락, 조인 후 고객 행 수를 고객 수로 사용 | DISTINCT 개체 수 → 조건별 그룹 선택 → 합계 → 중앙값의 중간 결과 계약 |
| local358 | 실행 가능한 연령대 SQL이 공식 정답과 불일치 | 기준일·나이 계산·NULL/기타 처리의 지표 정의 확인 |

LLM만 교체하거나 모든 scope/관계 guard를 해제하면 해결되는 문제가 아니다. 실제 정답 차단 1건과 잘못된 계산 4건(제안 1+차단 초안 3)이 공존한다. 다음 구현은 구조화된 다단계 요청 계약, 역할/조건 검증, 미완료 목표별 재계획을 중심으로 해야 한다. 이번에는 이 기능을 구현했다고 주장하지 않는다. 최초 결과·SQL·공식 진단을 고정해 후속 회귀로 사용한다. 금지된 재조회나 데이터 손실은 발생하지 않았다.

## 실제 적재 실행 (완료)

[검토 계획](plan.md), [승인 checkpoint](warehouse_prepared.json).

```sql
SELECT * FROM workspace.default.ncr_ride LIMIT 100000
```

최초에는 승인 대기였으나, 사용자의 후속 포괄 승인으로 1회 실행했다. 실제 저장·실패 원인·재조회 없는 복구·실모델 EDA 결과는 [후속 보고](warehouse_agent_repair.md)에 기록했다. 대기 checkpoint는 과거 준비 상태의 증거다.

## 재현

teleai 루트의 `.telly_runtime/v1-venv/bin/python`으로 아래 scripts를 실행한다.

- `scripts/evaluate_analysis_statistics.py --output <report>`
- `scripts/evaluate_analysis_join.py --output <report>`
- `scripts/evaluate_analysis_time_series.py --output <report>`
- `scripts/check_large_data_storage.py --rows 1000000 --output <report>`
- `scripts/evaluate_streaming_scale.py --rows 1000000 --output <report>`
- `scripts/evaluate_spider2_teleai.py --spider-root /Users/najongseong/git_repository/ai_agent_eval/Spider2-main --benchmark-instruction --provider databricks --id local221 --id local009 --id local210 --id local358 --id local198 --output-dir <directory>`

공식 scorer는 설치된 Spider2-Lite `evaluation_suite/evaluate.py`를 변경 없이 사용했다. 제안과 차단 초안 디렉터리를 분리하여 실행했다. 금번에는 DeepEval judge를 재실행/보정하지 않았다. 정확한 실행 결과를 LLM judge 점수로 대체하지 않는다.

관련 적재/부분읽기/메모리 계약 테스트 28/28, compileall 및 diff 검사를 통과했다. 이번 변경은 검증 도구와 증거이며, 위 고급 계획 문제를 이미 수정했다는 의미는 아니다.
