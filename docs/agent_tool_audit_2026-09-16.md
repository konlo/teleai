# TeleAI agent tool 진단 — 2026-09-16

## 판정

현재 production `GraphAnalysisRuntime`의 tool 구조는 **승인형 데이터 로딩, 보유 DataFrame 재사용, 기본 SQL 계산, 기본 차트 생성과 복구**에는 적합하다. 중앙 registry와 실행 wrapper, 승인 ledger, lineage 검사, 영속 artifact가 연결돼 있어 단순 tool 모음은 아니다.

2026-09-24까지 P0 공백인 공통 결과 계약, 데이터셋 프로파일, 승인형 source discovery와 P1의 제한된 차트 사양·다중 dataset join·구조화 통계 검정, 직접 이상치 탐지, 계보가 있는 이상치 cohort 후속 계산, bounded datetime 시계열 준비, 단일 그룹·TOP-N 집계, 그룹별 건수·성공률 dual-axis/split-panel, winsorization과 bounded 피벗을 구현했다. 현재 남은 tool 범위는 범용 다중 패널, 구간화·다중 지표 소계와 결과 내보내기다.

따라서 현재 상태는 **검증된 제한 범위에 적합**하며, 범용 분석 agent로 승격하려면 남은 의도와 배포 환경 검증이 필요하다.

## 실제 활성 tool 구성

2026-09-24 피벗 보강 후 `build_analysis_tools()`는 24개 정의를 만든다. production agent에는 `propose_databricks_query`를 제외한 로컬 tool 23개가 등록되고, 원격 연결이 있을 때 승인 middleware가 적용된 `query_databricks` 1개가 추가된다.

| 영역 | 활성 tool | 현재 역할 | 진단 |
|---|---|---|---|
| 발견 | `list_analysis_context`, `plan_source_discovery` | 보유 dataset, 저장된 table, skill 목록과 승인형 catalog 탐색 계획 | 현재 문맥에서 확인된 catalog만 사용하며 실제 목록 조회는 기존 승인을 거친다. |
| skill | `read_analysis_skill` | 등록된 안전한 skill 읽기 | 경로·크기·frontmatter 검증은 좋다. 분석 성공과 연결된 평가가 적다. |
| metadata | `inspect_table_context`, `inspect_dataset`, `profile_dataset` | 저장 schema, 로딩 결과, 결측·고유값·요약 통계 확인 | stale 차단, 실제 schema 우선, bounded profile과 고카디널리티 값 생략을 적용한다. |
| reuse | `use_dataset` | 조건을 좁혀 로컬 재사용 | lineage 검사가 보수적이다. OR와 다중 dataset은 지원하지 않는다. |
| calculation | `local_analysis_sql` | 한 DataFrame을 DuckDB `data`로 계산 | 외부 접근 차단, 15초 중단, 20,000행 결과 제한이 있다. 한 dataset만 바인딩한다. |
| join | `join_datasets` | 보유 raw dataset 두 개를 명시적 key와 방식으로 결합 | dtype·NULL·cardinality·예상 행 수·증가율을 실행 전에 검사하고 두 부모 lineage를 보존한다. many-to-many는 실행하지 않는다. |
| statistics | `statistical_test` | 보유 raw dataset의 선언된 가설 검정·평균 신뢰구간 | 독립·대응 t, 카이제곱, 일원 ANOVA, Mann–Whitney, 평균 CI를 enum으로 제한하고 표본·결측·가정·통계량·p-value·효과크기·CI를 반환한다. |
| outliers | `detect_outliers`, `select_outlier_rows`, `winsorize_numeric` | 보유 raw dataset의 이상치 기준·건수 계산, 후속 cohort 파생과 원본/clip 통계 비교 | IQR·Z-score·MAD·분위수와 tail을 제한한다. cohort는 원시 행을 반환하지 않고 lineage를 보존하며, winsorization은 원본을 변경하지 않고 aggregate evidence만 반환한다. |
| aggregation | `aggregate_dataset` | 보유 raw 또는 파생 cohort의 전체·단일 그룹·TOP-N 집계 | count/mean/sum/median/min/max, 단일 group, 정렬, TOP-N, group/output 한도를 제한하고 parent·snapshot·digest가 있는 aggregate child를 만든다. |
| pivot | `pivot_dataset` | 보유 raw dataset의 bounded 피벗·교차표 | 실제 runtime schema에서 행 1~3개·열 1~2개를 선택하고 count/mean/sum/median/min/max/success-rate/overall-percent, 조건, margins, 월 순서를 제한한다. 최대 4축·축당 100값·1,000셀과 parent·snapshot·digest를 강제한다. |
| visualization | `recommend_chart_images`, `prepare_histogram`, `render_histogram`, `render_chart_spec`, `render_count_rate_chart`, `show_chart` | 실제 PNG 추천·생성·재표시 | 기본 5종 차트와 그룹 박스플롯, 그룹별 전체 건수+명시적 성공률의 dual-axis/split-panel을 제한된 schema로 실행한다. 임의 코드·파일·URL·style dictionary는 받지 않는다. |
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

7. **`detect_outliers`·`select_outlier_rows`·`aggregate_dataset`·`compare_group_aggregates`·`winsorize_numeric` — 직접 탐지와 bounded 후속 계산 구현 완료**: IQR/Z-score/MAD/분위수 결과와 threshold·행 수·범위를 구조화한다. 선택 cohort의 lineage를 보존하며 원본-versus-cohort 비교와 원본 불변 quantile clipping 비교를 구조화한다.
8. **`prepare_time_series`**: 완료. 실제 dtype·95% 파싱 성공률, IANA timezone, hour/day/week/month, gap omit/zero/nan, 중복 시각, 최대 20개 series와 5,000행 한도를 검증하고 parent lineage가 있는 aggregate dataset을 만든다.
9. **`pivot_dataset` — 기본 피벗 구현 완료**: 단일·복수 축, count/수치 집계/성공률/전체 백분율, 필터, margins와 영문 월 달력 정렬을 지원한다. 구간화 축과 한 요청의 다중 지표 요약은 후속 범위다.
10. **`export_result`**: 검증된 dataset/chart만 CSV·PNG로 내보내고 provenance manifest를 함께 만든다.

## 추가하지 말아야 할 것

- 무제한 Python REPL을 production agent에 다시 연결하지 않는다.
- 차트 종류마다 별도 tool을 계속 늘리지 않는다. 하나의 제한된 chart spec으로 묶는다.
- table별 컬럼·업무 의미·대표값을 tool 코드나 prompt에 넣지 않는다.
- skill 파일이 실행 권한이나 Databricks 승인으로 작동하게 만들지 않는다.
- legacy tool의 문자열 성공 메시지를 구조화된 evidence 없이 완료로 채택하지 않는다.

## 평가 근거와 공백

독립 oracle은 200문항 중 87문항을 지원한다. 미채점 113문항의 구성은 다음과 같다. 아래 기능별 수는 서로 중첩될 수 있다.

| 구분 | 미채점 수 |
|---|---:|
| table/계산 | 62 |
| chart | 42 |
| schema | 9 |
| 단일 그룹 집계·요약표 | 25 |
| 단일 차트 | 17 |
| 피벗·소계 | 5 |
| 전환율·이중축 | 11 |
| 다중 패널·고급 시각화 | 14 |
| 이상치 | 7 |
| 통계 검정 | 0 |

변경하지 않은 `L2_037`은 `bank_loan` fixture에서 잔액 IQR 상한 이상치 137행을 선택한 뒤 평균 나이 41.7153과 직업 TOP 3(technician 34, blue-collar 29, management 19)를 두 aggregate child로 계산했다. 독립 reference, parent lineage, snapshot, digest가 모두 일치했고 모델·원격 호출은 0회였다.

변경하지 않은 `L2_038`은 같은 fixture에서 원본 2,000행과 IQR 상한 이상치를 제외한 1,863행을 직업별 평균 잔액으로 비교했다. 12개 그룹의 전체·cohort 평균, 차이, 변화율과 두 부모 lineage·snapshot·digest가 독립 reference와 일치했고 모델·원격 호출은 0회였다.

변경하지 않은 `L2_061`·`L2_064`·`L2_065`·`L2_066`은 월·직업·객실 등급·날짜별 전체 건수와 명시된 성공률을 dual-axis PNG로 만들었다. 각 문항의 명시적 성공값, 비결측 분모, 성공수, 정렬과 plotted data digest가 독립 pandas reference에 일치했고 모델·원격 호출은 0회였다. 임의 source와 임의 컬럼에서도 같은 계약을 검증했다.

변경하지 않은 `L2_089`는 같은 count/rate 계약으로 월별 건수와 전환율 2열 패널을 만들었다. `L2_096`은 balance 상하위 1% 경계, clip 건수, 원본/보정 평균·최솟값·최댓값이 독립 reference와 일치했다. 임의 수치 컬럼에서도 원본 DataFrame 불변과 재시작 복원을 확인했고 모델·원격 호출은 0회였다.

`pivot_dataset`은 변경하지 않은 `L1_062`–`L1_064`와 `L2_021`–`L2_025`·`L2_027`–`L2_029`·`L2_031`·`L2_034`를 count, 수치 집계, 성공률, 전체 백분율, 조건, 복수 열 축, margins와 월 달력 순서로 실행했다. 13/13 결과가 독립 pandas oracle과 셀 단위로 일치했고 임의 schema에서도 lineage·digest·재시작 복원을 확인했다. 모델·원격 호출은 0회였다.

`show_chart`의 실제 PNG 재사용과 missing ID 오류 계약을 추가했다. 전체 활성 tool의 성공, 입력/복구 오류, 승인 안전성, 재시작, 중복/재사용 상태는 `agent_tool_contract_matrix_2026-09-18.md`에 기록했다. `use_dataset`과 `read_analysis_skill`의 전용 조합 테스트 확대는 후속 보강 대상이다.

## 권장 실행 순서

1. 공통 출력 schema와 tool contract matrix를 먼저 만든다.
2. `profile_dataset`과 승인형 source discovery를 추가한다.
3. 실제 agent 독립 oracle 중 schema/결측/고유값/기본 그룹 집계를 우선 확대한다.
4. 제한된 `render_chart_spec`과 지정 차트·수정·재시작 검증을 완료한다.
5. multi-dataset join, statistical test, 직접 outlier detection과 bounded cohort 후속 계산, 숫자축 빈도 선·영문 월 누적 곡선, datetime resampling·gap·다중 series와 기본 피벗을 완료했다. 다음 tool 보강은 구간화·다중 지표 소계, 범용 다중 패널 또는 결과 내보내기 중에서 선택한다.

이 순서라면 tool 수를 통제하면서도 자주 실패하는 사용자 의도를 결정적이고 검증 가능한 실행으로 옮길 수 있다.
