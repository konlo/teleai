# TeleAI agent tool 진단 — 2026-09-16

## 판정

현재 production `GraphAnalysisRuntime`의 tool 구조는 **승인형 데이터 로딩, 보유 DataFrame 재사용, 기본 SQL 계산, 기본 차트 생성과 복구**에는 적합하다. 중앙 registry와 실행 wrapper, 승인 ledger, lineage 검사, 영속 artifact가 연결돼 있어 단순 tool 모음은 아니다.

2026-09-21까지 P0 공백인 공통 결과 계약, 데이터셋 프로파일, 승인형 source discovery와 P1의 제한된 차트 사양·다중 dataset join·구조화 통계 검정, 직접 이상치 탐지를 구현했다. 현재 남은 tool 범위는 이상치 행의 후속 분석, 시계열, 다중 패널 차트와 결과 내보내기다.

따라서 현재 상태는 **검증된 제한 범위에 적합**하며, 범용 분석 agent로 승격하려면 남은 의도와 배포 환경 검증이 필요하다.

## 실제 활성 tool 구성

2026-09-21 이상치 탐지 보강 후 `build_analysis_tools()`는 17개 정의를 만든다. production agent에는 `propose_databricks_query`를 제외한 로컬 tool 16개가 등록되고, 원격 연결이 있을 때 승인 middleware가 적용된 `query_databricks` 1개가 추가된다.

| 영역 | 활성 tool | 현재 역할 | 진단 |
|---|---|---|---|
| 발견 | `list_analysis_context`, `plan_source_discovery` | 보유 dataset, 저장된 table, skill 목록과 승인형 catalog 탐색 계획 | 현재 문맥에서 확인된 catalog만 사용하며 실제 목록 조회는 기존 승인을 거친다. |
| skill | `read_analysis_skill` | 등록된 안전한 skill 읽기 | 경로·크기·frontmatter 검증은 좋다. 분석 성공과 연결된 평가가 적다. |
| metadata | `inspect_table_context`, `inspect_dataset`, `profile_dataset` | 저장 schema, 로딩 결과, 결측·고유값·요약 통계 확인 | stale 차단, 실제 schema 우선, bounded profile과 고카디널리티 값 생략을 적용한다. |
| reuse | `use_dataset` | 조건을 좁혀 로컬 재사용 | lineage 검사가 보수적이다. OR와 다중 dataset은 지원하지 않는다. |
| calculation | `local_analysis_sql` | 한 DataFrame을 DuckDB `data`로 계산 | 외부 접근 차단, 15초 중단, 20,000행 결과 제한이 있다. 한 dataset만 바인딩한다. |
| join | `join_datasets` | 보유 raw dataset 두 개를 명시적 key와 방식으로 결합 | dtype·NULL·cardinality·예상 행 수·증가율을 실행 전에 검사하고 두 부모 lineage를 보존한다. many-to-many는 실행하지 않는다. |
| statistics | `statistical_test` | 보유 raw dataset의 선언된 가설 검정·평균 신뢰구간 | 독립·대응 t, 카이제곱, 일원 ANOVA, Mann–Whitney, 평균 CI를 enum으로 제한하고 표본·결측·가정·통계량·p-value·효과크기·CI를 반환한다. |
| outliers | `detect_outliers` | 보유 raw dataset의 단일 수치 컬럼 이상치 기준·건수 계산 | IQR·Z-score·MAD·분위수와 tail을 제한하고 임계값·결측·상하한 건수·비율·범위를 원시 행 없이 반환한다. |
| visualization | `recommend_chart_images`, `prepare_histogram`, `render_histogram`, `render_chart_spec`, `show_chart` | 실제 PNG 추천·생성·재표시 | histogram/bar/line/scatter/boxplot의 축·집계·정렬·top-N·bins·제목·라벨을 제한된 schema로 실행한다. 임의 코드·파일·URL·style dictionary는 받지 않는다. |
| remote | `query_databricks` | 정확한 SELECT를 승인 후 1회 실행 | fingerprint·연결 identity·제출 불명 상태 차단이 강하다. 테이블 탐색용 전용 계획 tool은 없다. |

과거 `core/tools.py`에는 join, 이상치, 시계열, heatmap 등 27개가량의 legacy tool이 있으나 현재 v1 production 경로에는 등록되지 않는다. 이들을 그대로 되살리면 Streamlit session state 의존, 문자열 결과, 불균일한 오류 계약이 다시 유입된다. 필요한 기능만 현재의 `ToolDefinition`·lineage·artifact 계약으로 재구현해야 한다.

## 잘 설계된 부분

- tool 입력은 `additionalProperties: false`로 제한되고 필요한 필드가 명시된다.
- 로컬 SQL은 등록된 `data`와 CTE만 참조하며 DuckDB 외부 접근이 비활성화된다.
- Databricks는 변경 SQL을 거절하고 정확한 query fingerprint별로 매번 승인받는다.
- 원격 제출 상태가 불명확하면 자동 재실행하지 않는다.
- dataset에는 source, coverage, predicate, grain, aggregation, snapshot, parent lineage가 붙는다.
- 차트 완료는 실제 PNG와 dataset binding을 근거로 판정한다.
- 진단 로그는 tool명, 단계, 시간, 오류 유형을 기록하면서 prompt·row·credential은 기록하지 않는다.
- 반복 실패, model/tool budget, 완료 증거 누락을 recovery middleware가 제한한다.

## 우선 보강할 tool

### P0 — 출시 전에 계약을 먼저 고칠 것

1. **공통 `ToolResult` 출력 계약 — 구현 완료**
   - 모든 tool이 `status`, `evidence_ids`, `error_code`, `retryable`, `user_action`, `scope`를 같은 의미로 반환한다.
   - `ToolDefinition`에 `output_schema`와 허용 status를 추가한다.
   - 현재 `list_analysis_context`, `read_analysis_skill`, `inspect_dataset`은 공통 `status`가 없고, 오류 코드는 wrapper 일부에만 있다.

2. **`profile_dataset` — 구현 완료**
   - 행·열 수, dtype, null count/ratio, distinct count, min/max, 평균·중앙값·표준편차·quantile, capped top values를 구조화해 반환한다.
   - 결과 범위와 표본 여부를 포함하고 고카디널리티 값은 노출하지 않는다.
   - 결측·고유값·요약통계 같은 반복 의도를 임의 SQL 생성에서 분리한다.

3. **승인형 source discovery — 구현 완료**
   - `plan_source_discovery(catalog?, schema?, pattern?)`가 읽기 전용 `information_schema` SELECT와 범위를 만든다.
   - 실제 실행은 기존 `query_databricks` 승인 ledger를 그대로 통과시킨다.
   - 현재 UI는 FQN 직접 입력만 받고 agent는 저장된 `available_tables` 밖의 테이블을 찾을 수 없다.

### P1 — 일반 분석 범위를 넓힐 때 필요

4. **`render_chart_spec` — 구현 완료**
   - `kind`, x/y/category, aggregation, sort, top_n, bins, title, axis label을 bounded schema로 받는다.
   - histogram/bar/line/scatter/boxplot부터 지원하고 실제 PNG·dataset lineage·표본 범위를 반환한다.
   - 기존 추천 tool은 유지하되 사용자가 지정한 차트나 수정 요청은 이 tool로 처리한다.

5. **`join_datasets` — 구현 완료**
   - 여러 dataset ID를 명시적 alias에 바인딩하고 join key, cardinality, 예상 행 증가를 사전 검사한다.
   - source/snapshot/coverage를 결과 lineage에 보존하고 many-to-many 폭증 한도를 둔다.
   - inner/left/right/full outer, 복합 key 1~4개, NULL SQL 의미, 같은 이름 key와 충돌 컬럼 suffix를 검증한다. 결과는 `parent_ids` 두 개로 영속화되고 후속 로컬 집계에 사용할 수 있다.

6. **`statistical_test` — 구현 완료**
   - t-test, paired t-test, chi-square, ANOVA를 명시적 enum으로 제한한다.
   - 표본 수·결측 처리·가정 검사·통계량·p-value·효과크기·신뢰구간을 구조화해 반환한다.
   - 모델 문장이나 임의 Python 실행을 완료 증거로 사용하지 않는다.

### P2 — 사용 범위를 선언한 뒤 추가

7. **`detect_outliers` — 직접 탐지 구현 완료**: IQR/Z-score/MAD/분위수 결과와 threshold·행 수·범위를 구조화한다. 이상치 행 후속 집계와 변환은 별도 lineage 계약이 필요하다.
8. **`prepare_time_series`**: datetime 검증, timezone, 빈도, gap, 중복 시각, resampling 정의를 명시한다.
9. **`export_result`**: 검증된 dataset/chart만 CSV·PNG로 내보내고 provenance manifest를 함께 만든다.

## 추가하지 말아야 할 것

- 무제한 Python REPL을 production agent에 다시 연결하지 않는다.
- 차트 종류마다 별도 tool을 계속 늘리지 않는다. 하나의 제한된 chart spec으로 묶는다.
- table별 컬럼·업무 의미·대표값을 tool 코드나 prompt에 넣지 않는다.
- skill 파일이 실행 권한이나 Databricks 승인으로 작동하게 만들지 않는다.
- legacy tool의 문자열 성공 메시지를 구조화된 evidence 없이 완료로 채택하지 않는다.

## 평가 근거와 공백

독립 oracle은 200문항 중 60문항을 지원한다. 미채점 140문항의 구성은 다음과 같다.

| 구분 | 미채점 수 |
|---|---:|
| table/계산 | 79 |
| chart | 52 |
| schema | 9 |
| 단일 그룹 집계·요약표 | 25 |
| 단일 차트 | 22 |
| 피벗·소계 | 15 |
| 전환율·이중축 | 15 |
| 다중 패널·고급 시각화 | 15 |
| 이상치 | 11 |
| 통계 검정 | 0 |

`show_chart`의 실제 PNG 재사용과 missing ID 오류 계약을 추가했다. 전체 활성 tool의 성공, 입력/복구 오류, 승인 안전성, 재시작, 중복/재사용 상태는 `agent_tool_contract_matrix_2026-09-18.md`에 기록했다. `use_dataset`과 `read_analysis_skill`의 전용 조합 테스트 확대는 후속 보강 대상이다.

## 권장 실행 순서

1. 공통 출력 schema와 tool contract matrix를 먼저 만든다.
2. `profile_dataset`과 승인형 source discovery를 추가한다.
3. 실제 agent 독립 oracle 중 schema/결측/고유값/기본 그룹 집계를 우선 확대한다.
4. 제한된 `render_chart_spec`과 지정 차트·수정·재시작 검증을 완료한다.
5. multi-dataset join, statistical test와 직접 outlier detection을 완료했다. 다음 tool 보강은 이상치 행 후속 분석·시계열·다중 패널 중에서 선택한다.

이 순서라면 tool 수를 통제하면서도 자주 실패하는 사용자 의도를 결정적이고 검증 가능한 실행으로 옮길 수 있다.
