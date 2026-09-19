# Agent tool contract matrix — 2026-09-18

## 공통 계약

`ToolDefinition`은 입력 schema와 함께 공통 출력 schema 및 허용 status를 선언한다. 모든 registry tool 결과와 production adapter 오류는 다음 필드를 갖는다.

| 필드 | 의미 |
|---|---|
| `status` | `ready`, `planned`, `awaiting_approval`, `needs_*`, `unavailable`, `rejected`, `error` 중 하나 |
| `evidence_ids` | dataset, chart, schema 또는 skill version처럼 결과를 다시 확인할 수 있는 ID 목록 |
| `error_code` | 성공은 `null`, 실패는 기계 판독 가능한 원인 |
| `retryable` | 같은 호출을 자동 반복해도 되는지 여부. 입력 오류와 원격 실패는 기본적으로 `false` |
| `user_action` | 사용자가 해야 하는 조치. 원격 조회 계획은 승인 필요성을 명시 |
| `scope` | 모집단, snapshot, coverage 또는 도구 결과의 적용 범위 |

기존 업무 필드는 중첩하거나 제거하지 않았다. 따라서 이전 recovery와 UI가 읽던 `dataset`, `cards`, `preview`, `histogram_plan`은 그대로 유지된다.

## 활성 tool 검증표

기호: **완료**는 자동 회귀가 있는 계약, **부분**은 인접 경로로 검증되지만 해당 tool 전용 조합이 부족한 계약, **공백**은 후속 테스트가 필요한 계약이다.

| Tool | 성공 | 입력/복구 오류 | 승인 안전성 | 재시작 | 중복/재사용 | 판정 |
|---|---|---|---|---|---|---|
| `list_analysis_context` | 완료 | 부분 | 해당 없음 | 완료 | 완료 | 공통 envelope 적용 |
| `read_analysis_skill` | 완료 | 완료 | 해당 없음 | 부분 | 완료 | 임의 경로·크기·manifest 차단 유지 |
| `inspect_table_context` | 완료 | 완료 | stale이면 조회하지 않고 refresh SQL만 반환 | 완료 | 완료 | 실제 승인 schema 우선 |
| `inspect_dataset` | 완료 | 완료 | 해당 없음 | 완료 | 완료 | 로딩 ID만 허용 |
| `profile_dataset` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 결측·고유값·요약 통계 결정적 복구 연결 |
| `use_dataset` | 완료 | 완료 | 부족한 범위는 `needs_data` | 완료 | 완료 | 조건/lineage 보수적 검사 |
| `plan_source_discovery` | 완료 | 완료 | 계획 후 `query_databricks`에서 별도 승인 | 완료 | 부분 | 승인 전 원격 호출 0회 graph 검증 |
| `recommend_chart_images` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 실제 PNG 저장 |
| `prepare_histogram` | 완료 | 완료 | 데이터 부족 시 계획만 반환 | 완료 | 완료 | 완전한 raw/frequency/PNG 재사용 |
| `render_histogram` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 빈도 lineage 검증 |
| `render_chart_spec` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 5개 kind와 제한된 축·집계·정렬·표시 수·라벨, 실제 PNG와 입력 digest |
| `show_chart` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 실제 PNG 재표시·missing ID 오류 직접 검증 |
| `local_analysis_sql` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 단일 `data`와 CTE, 외부 접근 차단 |
| `join_datasets` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | key dtype·NULL·cardinality·행 증가 사전 검사, 양쪽 parent lineage 보존 |
| `statistical_test` | 완료 | 완료 | 원격 실행 없음 | 완료 | 완료 | 6개 선언형 방법, raw grain·dtype·그룹 수·분산·범위 fail-closed |
| `query_databricks` | 완료 | 완료 | 정확한 fingerprint별 HITL, 불명 제출 자동 재실행 금지 | 완료 | 완료 | 원격 연결이 있을 때만 등록 |

`propose_databricks_query`는 registry 호환 정의에는 남아 있지만 production LangGraph에서는 등록하지 않는다. production은 `query_databricks`와 `HumanInTheLoopMiddleware`를 사용한다.

## 이번 보강의 검증 근거

- `profile_dataset`은 raw 행을 반환하지 않고 최대 64개 컬럼씩 프로파일링한다. 고유값이 50개를 넘는 컬럼은 값 자체를 생략한다.
- source discovery는 현재 문맥에 확인된 catalog만 사용한다. catalog가 없거나 둘 이상인데 지정되지 않으면 `needs_context`이며 SQL을 만들거나 실행하지 않는다.
- source discovery graph 테스트는 `information_schema.tables` SELECT를 만든 뒤 정확한 승인 카드에서 멈추며 remote executor 호출이 0회임을 확인한다.
- 결측치와 범주형 고유값 문항 `L1_006`, `L1_008`, `L1_009`는 production graph와 독립 reference oracle로 3/3 PASS했다. 모델 호출과 원격 실행은 모두 0회다.
- `render_chart_spec`은 임의 Python·파일·URL 없이 histogram/bar/line/scatter/boxplot을 실제 PNG로 만든다. `L1_077`, `L1_078`, `L1_086`의 데이터 digest·표시 범위·lineage가 독립 oracle과 일치했고 모델·원격 호출은 0회였다. 재시작 후 PNG 복구와 한국어 글꼴 선택도 검증했다.
- `join_datasets`는 1,000행 고객과 5,000행 거래 fixture를 `customer_id`로 one-to-many inner join해 5,000행을 만들었다. 실제 값과 독립 pandas merge digest가 일치했고 모델·원격 호출은 0회였다. many-to-many·출력 한도·dtype 불일치는 새 dataset 생성 전에 차단하며, 명시적 key별 집계 후 안전한 재조인, outer join의 오른쪽 전용 key, 후속 집계와 재시작도 검증했다.
- `statistical_test`는 독립·대응 t, 카이제곱, 일원 ANOVA, Mann–Whitney, 평균 CI의 표본 수·결측·가정·통계량·p-value·효과크기·CI를 구조화한다. 변경하지 않은 `L2_051`–`L2_060` 10문항이 독립 SciPy oracle와 값 단위로 일치했고 모델·원격 호출은 0회였다.
- application tests 115/115, migration tests 126/126, Level 3 17/17, 전체 참고 runner 217/217, compileall과 `git diff --check`가 통과했다.

## 남은 tool 공백

P0의 공통 계약, dataset profile, 승인형 source discovery와 P1의 제한된 chart spec·다중 dataset join·구조화 통계 검정은 구현됐다. 범용 분석 범위를 넓히려면 다음을 별도 release candidate로 구현해야 한다.

1. 이상치 탐지, 시계열 준비, 다중 패널 차트와 결과 내보내기 중 미채점 사용자 의도에 필요한 tool.
2. source discovery의 승인 완료·거절·재시작 전용 사용자 여정 확대. 현재 공통 `query_databricks` 계약은 검증됐지만 discovery 결과 전용 승인 후 실행 테스트는 실제 원격 없이 추가할 수 있다.
