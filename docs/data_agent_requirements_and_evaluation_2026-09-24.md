# 데이터 분석·시각화 Agent 요구사항 및 통합 평가안

작성: 2026-09-24 · 상태: **설계 보강 및 T00~T02 부분 실행 / 전체 통합 평가 미완료**
초기 점검 기준: HEAD `aa01c28` 및 미커밋 변경. 후속 T00~T02의 코드 수정·검증 결과는 [착수 준비 보고서](implementation_readiness_2026-09-24.md)에 분리 기록한다. 운영 DB 조회와 서버 재시작은 수행하지 않았다.

## 1. 결론과 제품 목표

목표는 사용자의 분석 목적을 이해하고, 필요한 스키마를 확인한 다음, 보유 데이터 재사용 여부와 실행 위치를 결정하고, 분석·시각화 결과를 검증하며, 복구 가능한 실패에는 방법을 바꿔 목적을 완수하는 agent다.

현재는 LangChain `create_agent`와 LangGraph checkpoint, 도구 호출, 승인 중단/재개, 제한 복구가 있는 **실제 agent 구조**다. 다만 일반적인 분석 요청을 일관되게 계획·검증하는 계층과 메모리보다 큰 데이터를 다루는 실행 계층은 보강이 필요하다. 지금 증거로 범용 분석 agent의 완성도나 출시 가능 비율을 산정할 수 없다.

- 독립 채점 정의: **87/200 = 43.5%의 oracle coverage**. 완성도·전체 성공률이 아니다.
- 나머지 **113개는 미채점**이다. 구현 없음, 부분 지원, 기존 도구로 가능하지만 평가 없음으로 다시 분류해야 한다.
- 이전 application 167/167 등의 결과는 이전 검증 시점의 기록이다. 현재 미커밋 변경의 통과 증거로 사용할 수 없다.
- DeepEval·Spider 2.0은 이번에 공식 자료와 적용 방법을 조사했으며 설치·실행·점수 산출은 하지 않았다.
- 전수 요구를 먼저 묶어 설계하되, 전체 시스템을 한 번에 갈아엎는 배포는 하지 않는다. 평가 계약을 고정하고 공통 원인별 변경 묶음을 순서대로 검증한다.

## 2. 현재 구조 점검

운영 v1 경로는 `pages/Telly.py → ui/analysis_page.py → GraphAnalysisRuntime → create_agent + middleware → tools → datasets/artifacts`다. `core/chat_flow.py` 중심의 과거 경로와 구분해야 한다. 현재 진입점은 LangChain major version에 따라 v1/legacy UI를 선택한다.

| 영역 | 코드에서 확인한 기반 | 보강할 사항·영향 | 근거 |
|---|---|---|---|
| Agent 실행 | 모델↔도구 loop, checkpoint, 실행 횟수·시간 예산, 승인 중단/재개 | 계획과 완료 의무를 공통 구조로 관리하고 도구 교체·재계획을 검증해야 함 | `core/analysis_agent/runtime.py`, `recovery.py` |
| 의도·문맥 | 외부 TableContext alias, 조건 해석, 일부 후속 질문 유지 | `_state`, `_next_local`, `_complete`, 응답 조립에 기능별 분기가 결합됨. 복합 요청·조건 변경에 교차 영향 가능 | `core/analysis_agent/recovery.py`, `intent_scope.py` |
| 데이터 재사용 | source, columns, conditions, coverage, grain, snapshot, lineage 및 보수적 reuse 판단 | 경로별 규칙 통합, 실제 원본 버전과 적재 시각 분리, 요청 범위 충족 증명 필요 | `utils/analysis_datasets.py`, `core/analysis_databricks.py` |
| 원격 조회 | 승인 ledger·fingerprint·중복 방지, 최대 행수 초과 표시 | `fetchmany(max_rows+1)`는 수신 행수 제한. 서버 scan/cost 제한이나 byte 단위 streaming은 아님 | `core/analysis_databricks.py:22` |
| 저장·캐시 | SQLite metadata+Parquet BLOB, immutable 결과, LRU 및 대화 quota | 전체 BLOB 읽기→전체 DataFrame 복원, 캐시 hit 시 deep copy. byte quota가 peak RSS를 보장하지 않음 | `core/analysis_agent/assets.py:62` |
| 로컬 SQL | DuckDB SELECT 제한, 외부 접근 차단, 512MB 설정·15초 interrupt | 실행 전에 DataFrame이 이미 메모리에 존재. 파일을 부분 scan하는 out-of-core 경로가 아님 | `core/analysis_sql.py:20` |
| 스키마·preview | dataset metadata 및 bounded preview | preview의 `frame.head(5)` 이전 전체 frame 로딩, dtype 추론의 `frames.values()` 순회가 불필요한 복원을 유발할 수 있음 | `core/analysis_runtime_tools.py:51`, `recovery.py` |
| 집계·히스토그램 | 원본 재사용과 값별 `COUNT(*)` 원격 계획, 구조화 통계·join·피벗 도구 | 연속형 고유값이 많으면 값별 집계도 거대함. bin별 집계, 피벗 출력 규모 사전 계산, 복수 지표 의미 계약 필요 | `core/analysis_runtime_tools.py:403`, `utils/analysis_pivot.py` |
| 시각화 | 실제 PNG, 입력 digest, 원본/집계 구분, 추천 이미지, 표본 표시 | 추천은 주로 dtype와 첫 후보 컬럼을 선택. 사용자 목적·분포·희소 범주 기반 순위, 다양한 chart 조합, 실제 화면 판독 평가 필요 | `utils/analysis_charts.py` |
| 관측·평가 | run ID, 오류 위치, tool/event 기록, 독립 pandas/SciPy oracle, CI | 통합 trace와 재계획 원인, scan/transfer/cache 비용, 실제 모델 반복 및 미지 schema 평가 부족 | `diagnostics.py`, `scripts/evaluate_analysis_agent.py`, CI |

### 미완료 변경에 대한 분리 판정

현재 `recovery.py`, `analysis_runtime_tools.py`, `analysis_pivot.py` 변경과 새 `analysis_group_summary.py`는 **작업 중**이다. `summarize_groups` 도구 정의는 있으나 recovery의 실행·완료 증거 경로 연결은 끝나지 않았다. 초기 정적 점검에서 복수 집계 list의 set membership과 단일 mean 피벗의 `valid_spec` 문제를 발견했다. 후속 T00 비교 검사에서 피벗 10개 실패 보고로 확인했고 해당 두 조건을 최소 수정해 기존 회귀 및 다중 집계 production graph 검사를 통과했다. 최종 application 183/183 통과는 미완료 그룹 요약의 자연어 연결 완료를 뜻하지 않는다.

초기 diff는 별도 snapshot/patch/hash로 보존했고 HEAD 추출본과 작업본을 구분해 검사했다. 새 계약의 운영 연결과 전체 EDA 구현은 아직 남아 있으며 배포하지 않았다.

## 3. 공통 아키텍처 요구사항

모델은 목적 해석과 도구 선택·재계획을 맡고, 실행 가능성·정확성·승인·예산은 코드 계약으로 강제한다. 규칙 기반 빠른 경로는 정확히 지원하는 입력에만 적용하고, 임의 문장을 정규식에 맞춰 의도를 바꾸지 않는다.

처리 흐름:

`요청 + 대화 상태 → 스키마/업무 의미 확인 → AnalysisPlan → 데이터 충분성·비용 검사 → 로컬 재사용 또는 승인 대기 → 실행 → 결과 검증 → 응답 또는 제한 재계획`

`AnalysisPlan`은 아직 구현된 기능이 아니라 제안하는 공통 계약이다. 다음을 포함한다.

- 요청 ID·원문 참조, 목적, 완료해야 하는 산출물 목록, 계획 버전.
- 출처와 schema fingerprint, dataset/snapshot 참조, 컬럼·업무 의미의 근거.
- 전체 모집단 필터, 지표별 분자·분모 조건, 그룹·시간대·정렬·결측 처리·집계 수준.
- 결과 정확성 모드(exact/approximate/sample), 오차 또는 sampling 정책.
- 단계별 입력·출력 artifact ID, 의존 관계, 실행 위치(memory/local file/Databricks), 예상 자원.
- 승인 대상 SQL/fingerprint, 예산, 실패 분류와 허용 대체 경로, 각 단계 검증 의무.

후속 질문은 이전 텍스트를 다시 추측하는 대신 계획에 명시적인 변경을 적용한다. 유지할 필터, 교체할 축, 해제할 조건, 새로운 snapshot을 구분한다. “지난달 대신 이번 달”, “그중”, “전체로 돌아가”, “차트만 변경”을 별도 시험한다.

실패 후에는 실제 도구 관찰을 읽고 계획을 수정한다. 계획 준수 점수 때문에 잘못된 초기 계획을 고수하지 않도록 변경 사유·전후 계획·검증 근거를 기록한다. 내부 사고 과정 전문은 로그나 평가의 필수 입력으로 요구하지 않는다.

### 3.1 목표 수준: 사용자가 해결 절차를 지시하지 않아도 완수하는 agent

사용자 추가 요구: **Claude Code 정도로 필요한 것을 스스로 찾아 문제를 해결하는 자율성**. 제품 목표는 데이터 분석·시각화 영역에서 이 작업 방식을 구현하는 것이다. 동일 성능 달성을 선언하는 것이 아니며, 비교 주장은 동일 과제·데이터·접근권·예산으로 별도 측정해야 한다.

Claude Code 공식 설명은 정보 수집→행동→검증을 반복하고, 도구 관찰에 따라 다음 행동을 바꾸는 구조다. 필요한 skill 본문도 사용 시점에 읽는다. 이 설계에서 참고하는 것은 공개된 행동 구조이며 내부 구현을 복제했다고 주장하지 않는다. [공식 agent loop 설명](https://code.claude.com/docs/en/how-claude-code-works), [공식 skills 설명](https://code.claude.com/docs/en/skills).

TeleAI가 충족해야 할 관찰 가능한 행동은 다음과 같다.

| 능력 | 기대 동작 | 실패로 판정할 행동 | 연결 요구 |
|---|---|---|---|
| 부족한 정보 발견·탐색 | 필요한 schema·컬럼 의미·기존 dataset·lineage·관련 지침을 검색하고 근거를 기록 | 조회 가능한 정보를 사용자에게 반복 입력시키거나 컬럼을 추측 | A01,A08 |
| 목표 중심 계획 | 산출물·완료 조건을 정하고 여러 도구를 조합, 관찰에 따라 계획 변경 | 정규식에 맞는 단일 도구 결과만 반환하고 복합 요청 완료 선언 | A02,A06 |
| 도구·스킬 선택 | capability 검색→입출력/제약 확인→관련 skill 필요한 부분 읽기→실행 | 모든 지침을 prompt에 넣거나 없는 도구를 호출 | A08 |
| 새로운 분석 구성 | 검증된 도구 조합/로컬 SQL을 우선하고 부족하면 격리된 분석 코드 실행 | 미리 만든 함수가 없다는 이유만으로 가능한 분석을 포기 | A09,N01~N03 |
| 오류 원인별 복구 | 실패 관찰로 인자·표현식·실행 위치·renderer를 수정하고 다시 검증 | 동일 호출 반복, 사용자에게 디버깅 전가, 조건을 몰래 바꿔 성공 처리 | A06,D06,V01 |
| 긴 작업·대화 연속성 | 목표/계획/승인/검증 결과는 구조화 저장, 상세 결과는 참조로 필요할 때 읽기 | 요약 후 필터·분모·승인을 잃거나 완료 단계 중복 실행 | A03,O02,O03 |
| 스스로 완료 확인 | 실제 수치·scope·chart 입력·생성 파일을 완료 조건과 대조 | 그림 없이 “분포를 확인했다”, 일부 결과만으로 완료 선언 | A07,V01 |
| 필요한 때만 사용자 개입 | 업무 의미가 결정 불가하거나 원격 조회 승인이 필요할 때 구체적으로 질문 | 통상적인 로컬 재시도마다 허락 요청, 승인 없이 Databricks 조회 | A05,A06 |

공통 loop는 `목표/완료 의무 → 부족한 근거 탐색 → 계획/도구 선택 → 정책 검증 → 실행 → 관찰 → 결과 검증 → 완료 또는 재계획`이다. 모델은 관찰을 보고 다음 행동을 선택한다. 결정적 validator는 권한·데이터 범위·예산·결과 계약을 검사한다. 새로운 관찰이나 수정 없이 같은 실패가 반복되면 순환으로 감지하고 중단한다. 최대 step/시간/token/쿼리/메모리 예산은 실행 시작 시 고정하며, 복구 성공은 원래 목적과 정확성 모드를 유지해야 한다.

**도구·스킬 탐색 계약(A08)**: registry에는 capability ID/version, 설명, 입력·출력 schema, 적합한 grain, 비용, 권한, 오류 분류, 관련 skill ID를 둔다. 검색으로 후보를 좁히고 선택한 도구 정의·skill만 로딩한다. 스킬에는 적용 조건·절차·검증 방법을 두고 테이블별 사실은 외부 TableContext로 분리한다. DB 설명문·문서·도구 결과의 텍스트가 승인 규칙을 바꾸는 명령으로 해석되지 않도록 출처와 신뢰 경계를 유지한다. 자동으로 찾는 범위는 등록·허용된 자료/connector이며 임의 패키지나 외부 서비스를 자동 설치·연결하지 않는다.

**분석 코드 실행 계약(A09)**: 고정 도구로 표현할 수 없는 지원 범위 분석을 SQL 또는 제한 Python으로 구성하고, 오류 관찰에 따라 수정할 수 있어야 한다. 생성 코드는 운영 앱 프로세스 안에서 `exec`하지 않는다. 별도 격리 worker에 승인된 dataset 참조와 허용 라이브러리만 제공하고 네트워크/DB credentials/임의 파일 접근을 차단한다. CPU/RAM/시간/출력 byte 한도를 실행 계층에서 강제하며 AST 검사만으로 격리를 주장하지 않는다. 입력은 불변, 결과는 schema·scope·lineage·수치 검증을 통과한 뒤 등록한다. 새로운 원격 데이터가 필요하면 기존 approval gateway를 거친다. 격리 실행기가 준비되지 않은 동안 이 능력을 지원 완료로 표시하지 않는다.

예를 들어 히스토그램 실패 시 dataset 참조·범위·dtype·NULL·오류를 확인하고, 적절한 기존 도구/로컬 SQL로 bin count를 계산해 다른 renderer로 재시도할 수 있어야 한다. bin 합계·경계·실제 이미지까지 확인한 뒤 완료한다. 데이터 자체가 부족하면 새 조회 SQL과 이유를 제시하고 승인 대기한다. 렌더링 실패를 이유로 데이터를 자동 재로딩하지 않는다.

### 3.2 자율성 전용 검증

J01~J24 중 해당 동작이 있는 여정에 다음 변형을 교차 적용한다. 동일 문장을 맞추는 하드코딩을 막기 위해 평가용 schema·표현·실패 지점은 개발용과 분리한다.

- **정보 공백**: task에 테이블/컬럼 정답을 주지 않고 검색 가능한 schema·문서만 제공. 발견→검증→완료 경로를 확인한다.
- **복구 가능한 장애**: 도구 인자 오류, 로컬 SQL 오류, chart renderer 1회 실패를 주입. 사용자에게 수정 방법을 받지 않고 원래 목표를 완수하는지 확인한다.
- **도구 경로 변경**: 선호 도구 하나를 unavailable로 만들고 동등한 대체 경로를 제공. 특정 tool call 순서가 아닌 정답·scope·비용·artifact로 채점한다.
- **새 조합**: 개별 도구는 알고 있지만 사전 정의하지 않은 복합 분석을 계획하고 실행. 단일 정규식 fast path 통과와 별도 집계한다.
- **문맥 압축·재시작**: 여러 단계 후 상세 tool 출력이 context에서 빠져도 저장 참조로 필터·snapshot·산출물을 복원한다.
- **차단 조건**: schema 설명에 승인 우회 문구, 코드의 외부 접근 시도, 예산 초과, 동일 실패 반복을 주입. 올바른 차단과 분석 완료를 구분한다.

보고 항목에는 사람의 **해결 방법 개입 없이 완료한 비율**, 복구율, 불필요한 질문 수, 반복 실패 횟수, 계획 변경의 유효성, 비용과 결과 정확도를 포함한다. 정책상 필요한 승인과 해결 불가능한 업무 의미 확인은 디버깅 개입과 따로 센다. 실제 모델이 탐색·계획한 trace가 없는 실행은 자율 계획의 증거로 계산하지 않는다. DeepEval은 과정 품질을 보조 평가하고 수치·범위·승인·산출물은 독립 계약으로 판정한다. 이 추가 평가는 아직 실행하지 않았다.

### 3.3 EDA를 위한 연속 탐색

EDA는 단일 chart 생성이 아니라 **프로파일→여러 관점의 이미지→근거 있는 관찰→후속 탐색→비교·해석**을 수행하는 핵심 여정이다. 일반적인 “EDA 해줘” 요청에서도 agent가 제한된 예산 안에서 탐색을 진행하고, 사용자는 이미지 선택·수정으로 방향을 바꿀 수 있어야 한다. 자세한 행동·상태·대용량·평가 계약은 [EDA 연속 탐색 설계](eda_exploration_contract_2026-09-24.md)에 정의한다.

기존 `recommend_charts`는 dtype 기반 후보와 제한 표본을 사용한다. 이것만으로 발견 기반 후속 탐색이나 전체 EDA 완료를 입증하지 못한다. V05~V08 및 J13~J16을 기존 공통 계획·집계·시각화·복구 작업에 연결하고 별도의 테이블별 분기로 구현하지 않는다.

### 3.4 재리뷰로 확정한 실행 계약

[설계 재리뷰 R01~R08](design_review_2026-09-24.md)의 실행 상태/revision, snapshot 일관성, 지표 재집계, 공통 Observation, 모델 context·전역 자원, 산출물 수명, EDA 완료·예산, 실제 모델 조기 평가 계약을 이 설계의 상세 기준으로 적용한다. 새 요구 ID를 늘리기보다 아래 기존 요구의 합격 조건에 연결한다. 구현 완료를 뜻하지 않는다.

| 리뷰 항목 | 연결 요구 | 검증 여정 |
|---|---|---|
| R01 요청 변경·늦은 결과 | A03,A05,O01,O03 | J10,J14,J17 |
| R02 source 일관성 | A04,D05,V01 | J06,J18 |
| R03 의미·재집계 | A01,A04,N01,N02,V08 | J05,J15,J19 |
| R04 Observation·복구 책임 | A02,A06,A07,A08,O02 | J07,J10,J16 |
| R05 context·공유 예산 | A08,D06,O01,O03 | J08,J15,J20 |
| R06 발행·보존 | A07,D03,D05,O03,V04 | J09,J10,J12,J20 |
| R07 EDA 완료·예산 | A02,A07,V05,V06 | J13,J15,J16 |
| R08 실제 모델 조기 검증 | E01,E02,E03 | J01,J04,J07,J13 및 T12 전체 |

공통 상태/Observation/완료 계약과 최소 복구 loop는 T02/T05에서 먼저 연결한다. T11까지 복구 구조 설계를 미루지 않는다. 실제 모델의 작은 기준선도 T02에서 측정하며 T12는 후반 종합 평가다.

## 4. 요구사항 추적표

P0는 핵심 정확성·승인·일관성, P1은 일반 분석과 대용량 경로, P2는 고급 범위다. 출시 시 지원한다고 선언하는 기능은 우선순위와 무관하게 해당 완료 기준을 충족해야 한다.

| ID | 우선 | 요구사항 | 구현 전에 고정할 합격 기준 |
|---|---|---|---|
| A01 | P0 | 런타임 schema 탐색과 업무 의미 grounding | 임의 테이블/컬럼명 변경에도 동일 결과. 없는 컬럼·stale schema는 추측하지 않음. 의미 없는 ID에서 업무 지식 생성 금지 |
| A02 | P0 | 복합 목적을 구조화 계획과 완료 의무로 변환 | “건수+평균+차트” 중 하나만 만들고 완료 처리하지 않음. 모호한 분모·join key는 확인 |
| A03 | P0 | 대화 상태·후속 변경 | 필터 유지/교체/해제, table 전환, 재시작 후 문맥 대조. 늦은 이전 revision 결과의 현재 상태 덮어쓰기 0 |
| A04 | P0 | 데이터 충분성·재사용 단일 정책 | 동일 범위+snapshot은 재사용, 없는 컬럼·더 넓은 범위·최신 원본 요구는 새 계획. raw/aggregate/sample/truncated 혼용 금지 |
| A05 | P0 | Databricks 조회·재로딩 승인 | 새로운 원격 실행은 정확한 SQL 승인 후에만. 거절 0회 실행, 중복 클릭·재시작에도 한 번. SQL/연결 변경은 재승인 |
| A06 | P0 | 자율 복구와 안전한 중단 | 잘못된 인자/로컬 SQL/차트 실패 시 유효 대체 도구로 복구. 동일 오류 무한 재시도·묵시적 조건 완화 금지 |
| A07 | P0 | 결과 기반 완료·응답 | 수치·표·이미지·범위가 실제 저장 artifact와 일치. ready 산출물만 완료 근거로 사용. 미완료·빈 결과·승인 거절·연결 실패를 서로 다른 상태로 표시 |
| A08 | P0 | 필요한 도구·스킬·근거의 동적 탐색 | 미지 schema/도구 대체 시험에서 필요한 정의·지침을 찾아 검증 후 실행. 데이터별 하드코딩·지침에 의한 승인 우회 0 |
| A09 | P1 | 격리된 범용 분석 코드 구성·실행·수정 | 고정 도구 밖 지원 분석을 불변 dataset으로 수행, 오류 수정 후 독립 oracle 통과. 외부 접근/자원 초과 차단과 lineage 증명 |
| D01 | P1 | 메타데이터로 읽기 계획 수립 | dtype·행수·필요 컬럼 파악 때문에 전체 payload를 읽지 않음. 필요한 컬럼·파티션만 선택 |
| D02 | P1 | 원격 집계 우선 실행 | sum/count/빈도/bin/시계열 등 축약 가능한 작업은 DB 집계 계획. 반환 행수와 scan bytes/cost를 따로 제한·기록 |
| D03 | P1 | Arrow batch 적재·영속화 | 전체 Row list→DataFrame 중복 materialization을 피함. byte·시간·디스크 예산 초과 전에 중단, 부분 적재는 complete 아님 |
| D04 | P1 | 메모리 초과 로컬 데이터 분석 | 통제된 Parquet/Arrow 저장 참조를 DuckDB 등으로 scan, spill quota와 worker 예산 적용. 임의 경로/URL 접근은 계속 차단 |
| D05 | P0 | 재현 가능한 캐시·버전 | source/권한 범위/schema/원본 버전/predicate/projection/집계/근사 정책으로 key 결정. 적재 시간과 DB version 구분, 별도 조회의 일관성 unknown 처리·참조 만료 정책 |
| D06 | P1 | 비용 예측·실행 예산 | join fanout·피벗 셀·그룹 cardinality·메모리·디스크·시간을 계산 전 검사. 모델 context/도구 결과 한도와 호스트 전역 예산을 포함하고 실제 peak RSS도 측정 |
| D07 | P0 | 의도에 맞는 적재 계획·후보 검증 | 원문/기존 계획과 source·컬럼·필터·grain을 대조. 새 결과 검증 전 현재 root 변경 0, 잘못된 계획/부분 로딩은 완료 금지 |
| D08 | P0 | 보호 원본·파생 계보·전체 보유 자산 재사용 | EDA/재로딩/실패/재시작 후 원본 불변·읽기 가능. 현재 결과 부족 시 부모/root 탐색, quota로 보호 원본 자동 삭제 0 |
| N01 | P1 | 스칼라·그룹·복수 지표·비율·피벗 공통 명세 | count(*)/count(column), weighted mean, group/overall denominator, 빈 그룹, NULL, 소수/반올림·부분 집계 병합 가능성을 독립 oracle로 검증 |
| N02 | P1 | 필터·구간·시간·안전한 파생 컬럼 | 중첩 AND/OR, 경계 포함 여부, 단위·timezone·DST·정밀도·변환 실패 건수, 명시적 mapping과 수식. 원본 불변·lineage 유지 |
| N03 | P1 | join·통계·이상치·품질 점검 | 타입·key 유일성·중복·표본 수·분산·검정 가정·이상치 제외 범위 검증. SQL/DuckDB/pandas 결과 비교 |
| V01 | P0 | 차트 데이터 정확성 계약 | 축·집계·필터·분모·단위·정렬·범례가 계획과 일치. 단순 PNG 존재로 성공 판정하지 않음 |
| V02 | P1 | 대용량용 차트 입력 축약 | histogram은 bin count, 시계열은 bucket, 범주는 Top-N+기타, scatter는 밀도/표본. 분석용 정확 집계와 표시용 축약 구분 |
| V03 | P1 | 통계·목적 기반 이미지 추천 | 상위 3개 내 적절한 후보, 추천 사유·scope·표본/근사 여부 표시. 선택·차트 수정 시 가능한 한 같은 결과 재사용 |
| V04 | P1/P2 | 기본·고급 시각화와 산출물 내보내기 | 기본 bar/line/hist/scatter/boxplot 우선. heatmap·grouped/stacked·multi-panel 등은 별도 capability와 검증. PNG/CSV 및 근거 metadata 연결 |
| V05 | P1 | 목적 기반 자동 EDA와 유한 탐색 계획 | 데이터 품질·분포·관계 중 적합한 관점의 실제 이미지를 선택, 후속 탐색·차트 수·시간 예산 준수. 목표 변수 추측 금지 |
| V06 | P0 | 근거 있는 발견과 자율 후속 탐색 | 관찰→질문→도구→검증의 연결 증거. 패턴 없는 대조군에서 허위 발견, 상관의 인과 해석, 묵시적 이상치 삭제 0 |
| V07 | P1 | 시각화 선택·분기·비교·되돌리기 | 기본 모집단과 분기 필터·분모·snapshot 보존, 재시작 복원. 비교 가능한 축·bin·단위 및 범위 차이 표시 |
| V08 | P1 | 여러 차트의 공유 집계·캐시·비용 관리 | 스타일 변경 시 집계 재실행 0, 로컬 데이터 충분 시 원격 0. 후보 수 사전 제한·sample/exact 표시·scan/RSS 측정 |
| O01 | P0 | 사용자별 격리·승인·schema 변경 연속성 | 소유자 간 캐시 누출 0, 요청 중 schema 변경 및 프로세스 재시작 시 기존 승인 재해석 금지 |
| O02 | P1 | 단계별 trace·진행 화면·오류 ID | 계획→도구→관찰→복구→산출물 추적, 데이터 원문/secret 최소화. 오류 ID로 재현 가능 |
| O03 | P1 | 장시간 작업·취소·운영 복구 | 타임아웃·취소가 실제 worker/query까지 전달되는지 확인. 제출 여부 불명 원격 쿼리 자동 재제출 금지 |
| E01 | P0 | 평가 정의와 결과 연결 | 모든 출시 요구에 fixture·oracle/평가기·실패 주입·예산·상태 정의가 먼저 존재 |
| E02 | P1 | DeepEval + Spider + 제품 평가 연결 | 동일 run ID의 수치/trace/화면/성능을 수집하되 벤치마크별 점수를 별도 보고 |
| E03 | P0 | 실제 모델·미지 schema·재현성 | 모델/도구/프롬프트/데이터/evaluator 버전 고정. held-out 평가, 반복 실행, 회귀 비교. 대역 모델과 분리 |

### 주요 의미 계약을 먼저 정할 것

- “승객 수”의 기본은 전체 행수인가, 나이 결측을 제외한 건수인가? 기존 L2_032 reference의 count(Age)는 문장상 전체 승객 수와 다를 수 있다.
- L2_026 문장은 20대부터 시작하지만 reference는 20대 이하와 상한 120을 포함한다. 이 차이를 정답에 맞춘 하드코딩으로 숨기지 않고 ambiguity로 등록한다.
- 조건부 평균에서 대상이 없으면 NULL/계산 불가가 기본이다. reference에 0 대입이 있어도 사용자 요청의 의미와 일치하는지 명시해야 한다.
- 비율의 결측 분모 처리, 표본과 전체 값 구분, 원본 데이터의 음수를 임의 제거하는 행위 등을 명시한다.

## 5. 대규모 데이터 처리 설계

DataFrame 재사용은 유용하다. 다만 **모든 데이터를 항상 pandas로 적재해야 한다는 제약을 없애고** 논리적 dataset ID 뒤에서 저장·실행 방식을 선택한다.

사용자가 보고한 잘못된 적재·오해석 재로딩·EDA 원본 소실은 P0 문제로 다룬다. [데이터 로딩·원본 보존 계약](data_loading_and_preservation_contract_2026-09-24.md)을 적용한다. 검증된 raw snapshot은 보호하고, 파생·집계·차트·로딩 후보와 현재 선택을 분리한다. 새 로딩은 원문 기반 LoadPlan→승인→별도 staging→검증→발행→현재 revision에 따른 선택 순서다. 새 결과 저장이나 선택은 기존 원본 삭제를 뜻하지 않는다.

재사용은 현재 결과 하나만 검사하지 않고 권한 있는 registry의 부모/root·파일·집계를 metadata로 탐색해 판단한다. 로컬 자료로 충분하면 재조회 없이 처리하고, 부족하면 원격 집계·raw 적재·혼합 실행 중 목적과 비용에 맞는 계획을 승인 요청한다. 보호 원본을 quota/TTL 정리로 삭제하지 않으며 공간 부족이면 새 적재를 중단하거나 다른 범위를 제안한다. 명시 삭제·조직 보존 의무·권한 회수는 상세 계약의 별도 정책을 따른다.

| 데이터/요청 상태 | 우선 경로 | 결과와 확인 기준 |
|---|---|---|
| 작은 보유 DataFrame으로 충분 | 로컬 재사용·필터/집계 | 새 Databricks 실행 0. 동일 조건·version 확인 |
| 필요한 범위가 로컬 파일에 있음 | 컬럼·필터 pushdown한 파일 scan | 전체 DataFrame 생성 없이 aggregate/plot data만 반환 |
| 데이터가 원격에만 있고 크거나 로컬 자원이 부족 | DB에서 filter/join/group/bin 후 작은 결과 적재 | SQL 승인→실행. 전송량을 줄이고 정확한 범위 표시 |
| 필요한 결과·차트가 이미 저장됨 | artifact cache | 차트 스타일 변경은 집계 재실행 없이 재렌더링 |
| 정확 계산이 예산 초과 | 실행 계획 변경 또는 명시적 근사 제안 | exact에서 approximate로 몰래 전환하지 않음 |
| 데이터가 부족하고 재조회 거절 | 기존 결과 보존·범위 내 대안 | 전체 요청 완료로 처리하지 않음 |

구현 후보는 Arrow batch 전송, manifest를 가진 분할 Parquet 저장, DuckDB 선택 scan이다. Databricks는 대용량 결과에서 `fetchmany_arrow` 사용을 안내하고, DuckDB는 Parquet projection/filter pushdown을 지원한다. out-of-core도 모든 연산의 성공이나 RSS 상한을 보장하지 않으므로 spill·동시 worker·실측 RSS 예산을 함께 둔다. [Databricks connector](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector), [DuckDB Parquet](https://duckdb.org/docs/current/data/parquet/overview), [DuckDB 메모리 한계](https://duckdb.org/docs/current/guides/performance/oom).

보안 경계는 유지한다. 모델이 지정한 파일 경로를 그대로 SQL에 넣는 방식 대신, 신뢰한 dataset ID를 실행기가 승인된 로컬 scan으로 해석한다. schema refresh/EXPLAIN/COUNT 등이 원격 SQL을 실행한다면 승인 정책 대상이다. 승인된 단일 쿼리의 batch fetch는 별도 재조회가 아니지만 SQL 재실행·범위 변경은 새 승인이 필요하다.

### 차트별 큰 데이터 처리 기준

| 차트 | 집계·표시 방식 | 정확성 시험 |
|---|---|---|
| 히스토그램 | 공통 bin 경계를 결정하고 bin별 count만 전송. NULL·범위 밖 count도 별도 보존 | bin count 합계=대상 유효 건수. 음수/최솟값/최댓값 경계 검증 |
| 막대·비율 | 그룹 집계 후 Top-N 및 기타, 지표별 분모 | 전체와 표시 건수 reconcile. Top-N이 전체 분모를 바꾸지 않음 |
| 시계열 | 시간 bucket 및 필요한 구간만 scan. 빈 구간·timezone 명시 | sum/count 보존. 일 평균들의 단순 평균으로 월 평균을 만들지 않음 |
| 산점도 | density/hexbin 또는 재현 가능한 표본, 희소 그룹 고려 | 표시점 수·seed·표본율 공개, 전체 상관계수와 표본 상관계수 혼용 금지 |
| 박스플롯 | 정확/근사 quantile과 whisker/outlier 정책 분리 | 근사 알고리즘·오차 표시. 백분위 요약만으로 모든 이상치가 보존됐다고 주장하지 않음 |
| 다중 패널 | 공통 필터·snapshot의 중간 집계를 공유 | 패널마다 별도 원격 scan 반복 여부, 각 패널의 데이터와 범례 확인 |

### 성능 시험 매트릭스

기존 750,000행 저장/복원 시험은 합성 5컬럼이고, 기존 동시 성능 보고는 10,000행 로컬 계산이다. 이를 대규모 운영 데이터의 효율성 증거로 확대 해석하지 않는다.

- 규모: 10만/100만/1천만 행과, 압축 해제 후 데이터가 **실행 메모리 예산의 최소 2배**인 out-of-core 조건. 1억 행·TB급은 원격 pushdown 시험으로 별도 실행.
- 너비: 10/100/256컬럼, 수치/문자열/decimal/timestamp, 긴 문자열·높은 cardinality·skew·NULL·많은 파티션.
- 경로: cold/warm cache, 같은 요청 반복, 기간 확대/축소, snapshot 변경, 여러 conversation, worker 1/4/8.
- 부하: profile, count/sum/mean, histogram, 복수 그룹 집계, window SQL, join fanout, 차트 추천·수정·다중 패널.
- 측정: 정확도, wall time p50/p95, DB scan bytes, transfer bytes, 로컬 read bytes, peak RSS, spill/disk, cache hit, 원격 횟수, 모델 tokens/cost. 얻을 수 없는 DB 비용은 0이 아닌 unavailable.
- 합격 초안: OOM·무승인 재조회·원본 훼손 0, 동일 범위 반복 시 새 원격 조회 0, 선언한 exact 결과의 명시적 수치 오차 허용치 만족, 메모리 예산 밖 데이터에서도 bounded 실행 또는 실행 전 명시적 중단.
- 성능 개선 목표 초안: 축약 가능한 fixture에서 원본 전송 대비 전송량 90% 이상 감소, 동일 hardware의 기존 baseline 대비 p95 20% 초과 악화 금지. p95 20% 비교는 표본수·반복 변동을 함께 보고하며 기준 확정 전에는 제품 SLO로 주장하지 않는다.
- UI 진행 알림 2초 이내, warm 로컬 핵심 여정 p95 5초 이내를 **제안값**으로 둔다. 실제 모델 포함/제외, 사용자 승인 대기/DB cold start를 따로 측정한 뒤 배포 호스트에 맞춰 확정한다.

## 6. DeepEval과 Spider 2.0을 결합하는 방법

두 가지를 단일 성공률로 합산하지 않는다. **동일 실행의 서로 다른 측면**을 측정한다. Spider는 SQL 문제·정답 검증, DeepEval은 실행 과정·응답/대화 품질, 자체 평가기는 수치/이미지·승인·대용량 자원을 담당한다.

### 6.1 평가 층

| 층 | 평가 대상 | 판정 방법 | 적용 한계 |
|---|---|---|---|
| 실행 정답 | SQL/집계/통계 결과 | Spider official evaluator 또는 독립 SQL/pandas/SciPy oracle | SQL 문자열 일치가 아니라 결과 의미. order/NULL/중복/정밀도를 task 계약에 맞춤 |
| Agent 행동 | 계획·도구 선택·인자·단계 효율·완료 | DeepEval `TaskCompletionMetric`, `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric`, `StepEfficiencyMetric`; 계획이 기록되면 `PlanQualityMetric`/`PlanAdherenceMetric` | Judge 점수로 잘못된 수치나 승인 위반을 통과시키지 않음 |
| 대화 연결 | 후속 변경·목적 완료·조건 보존 | DeepEval `ConversationCompletenessMetric`, `KnowledgeRetentionMetric`, `TurnRelevancyMetric` + 계획 상태 assertion | 자연스러운 답변만으로 실제 조건 보존을 입증할 수 없음 |
| 시각화 | 데이터·차트 종류·렌더링·가독성 | plot input/table/bin oracle + PNG/UI 검사 + 보조 rubric judge | 이미지 존재·유사도만으로 정답 판정하지 않음 |
| 대용량·운영 | 메모리·시간·조회량·승인·격리·취소·재시작 | instrumentation, fault injection, performance harness | Spider/DeepEval의 기본 점수만으로 측정되지 않음 |

DeepEval 공식 문서는 trace 전체를 보는 지표와 component별 도구/인자 지표를 구분한다. ToolCorrectness는 기본적으로 expected_tools 대조이며, available_tools를 이용한 최적성 판정에는 LLM이 추가될 수 있다. 유효한 대체 복구 경로를 실패로 만들지 않도록 필수 동작·금지 동작·허용 대안 집합을 먼저 정의한다. [Agent 지표](https://deepeval.com/guides/guides-ai-agent-evaluation-metrics), [ToolCorrectness](https://deepeval.com/docs/metrics-tool-correctness).

TaskCompletion은 LLM judge와 trace를 사용한다. 평가할 사용자 목적을 명시적으로 고정하고, 실제 산출물/수치 판정 결과를 함께 제공한다. Judge가 agent 자신의 “완료했다”라는 말을 근거로 통과시키지 않게 한다. [TaskCompletion](https://deepeval.com/docs/metrics-task-completion). 대화 지표는 전체 turn 기록과 함께 적용한다. [Multi-turn 지표](https://deepeval.com/guides/guides-multi-turn-evaluation-metrics).

### 6.2 Spider 선택과 적용 경계

공식 README의 현재 Lite 구성은 547문항(BigQuery 214, Snowflake 198, SQLite 135)이다. 첫 도입은 SQLite subset으로 하되 task/DB/gold availability를 pinned revision에서 검증한다. 공식 평가기는 SQL 또는 CSV 실행 결과를 입력받는 모드를 제공한다. [공식 저장소](https://github.com/xlang-ai/Spider2), [Lite 평가기](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite/evaluation_suite).

- 공식 비교 트랙: 문제·DB·dialect·평가기를 유지하고 실제 완료한 subset만 명시한다. SQLite subset 점수를 전체 Spider 2.0 점수라고 부르지 않는다.
- 제품 적응 트랙: Databricks dialect/승인 UX/한국어 후속 질문/시각화를 붙인 별도 문제군을 만든다. 이는 **Spider 기반 TeleAI 평가**이며 공식 leaderboard와 비교하지 않는다.
- 기존 production remote tool을 우회해 benchmark 접근권을 넣지 않는다. 평가 환경 전용 source adapter를 별도 경계로 설계한다.
- 외부 benchmark의 정답 SQL·정답 테이블명은 evaluator만 읽는다. oracle table을 planner에 주는 설정은 schema 탐색 평가와 분리 표기한다.
- full cloud 트랙에는 접근권·비용·query budget이 필요하다. 현재 공식 저장소에는 2026-08-12 Snowflake 계정 정지에 따른 접근 장애 공지가 있으므로, 이 의존성으로 local 요구사항 작업을 막지 않는다.
- Spider는 시각화, 재조회 승인 UX, RAM/전송 최적화를 종합 검증하는 benchmark가 아니다. 별도 제품 시험이 필수다.
- DBT는 repository transformation 과제다. 현재 read-only 분석 제품의 필수 출시 범위와 분리한다. task 수는 문서 간 차이가 있으므로 pinned manifest로 확정한다.

### 6.3 공통 실행 기록과 재현성

각 run은 다음 참조를 남긴다: case/requirement ID, dataset·schema fingerprint, 코드 revision/dirty 여부, 모델·prompt·tool/evaluator 버전, seed, 계획/수정 이력, 실제 tool calls, 승인 이벤트, 결과 digest·차트 ID, 비용·시간·자원, 최종 상태.

LangGraph tool messages와 diagnostics를 DeepEval trace로 연결하는 adapter를 설계한다. 평가 전용 코드와 dependency는 운영 agent와 분리하고 버전을 고정한다. Agent가 SQL을 실행했으면 가능한 한 CSV 결과 모드로 채점하여 evaluator의 불필요한 원격 재실행을 피한다.

Judge는 수치 oracle과 분리하고 모델·rubric을 고정한다. 한국어 사용자 표현을 포함한 성공/실패 표본 30개 이상을 사람이 검토해 false-pass를 보정한다. 중요 불일치는 재검토 대상으로 남긴다. 평가 입력의 원시 행·credentials는 제거하고, 승인되지 않은 외부 judge/telemetry로 운영 자료를 보내지 않는다.

### 6.4 완료도 측정표

하나의 “완성도 %” 대신 아래를 함께 보고한다.

1. **요구사항 검증 커버리지**: 해당 출시 요구 중 독립 acceptance가 준비된 비율.
2. **실행 성공률**: 정확 결과+필요 산출물+범위 일치로 완료한 실행 / 채점 가능 실행. first attempt와 budget 내 최종 성공 구분.
3. **Agent 품질**: DeepEval 항목별 점수/분포/실패 이유. 결정적 fast path와 실제 모델이 계획한 경로를 분리.
4. **자율 복구율**: 복구 가능 오류를 주입한 여정 중 도움 없이 원래 목표를 완료한 비율. 원격 승인 요청은 정상 정책이며 자율성 감점 아님.
5. **재사용 효율**: 재사용 가능한 후속 요청 중 실제 재사용 비율, 불필요한 원격 실행 수, 전송·scan 비용.
6. **시각화 정확도·가독성**과 **성능 p95/peak RSS**.
7. **차단 지표**: 무승인 실행·틀린 범위의 완료·산출물 없는 성공·데이터 훼손/격리 위반 건수.

`PASS / FAIL / BLOCKED_INFRA / UNSUPPORTED / UNGRADED / NOT_RUN / AWAITING_APPROVAL`을 구분한다. 정당한 승인 거절 시 올바른 종료 행동은 safety PASS일 수 있지만 분석 성공은 아니다. 미실행·인프라 차단 건수와 전체 계획 대비 비율도 노출해 분모를 줄여 성능이 높아 보이지 않게 한다.

제품 핵심 여정 합격 기준 초안: 중요 계약은 전건 통과·차단 지표 0, 실제 모델 핵심 여정 end-to-end 성공률 ≥95%, 복구 가능 오류 recovery ≥90%, DeepEval 주요 지표 ≥0.8. 이는 외부 benchmark 기준이 아닌 **제품 제안값**이며 baseline·비용·표본/95% 신뢰구간을 확인한 뒤 확정한다. Spider 점수에는 임의로 95% 기준을 적용하지 않는다.

## 7. 코딩 전에 준비할 평가 묶음

| ID | 대표 여정/경계 | 반드시 확인할 내용 |
|---|---|---|
| J01 | 알려지지 않은 테이블→schema→컬럼 의미→분석 | discovery 근거와 SQL 컬럼 일치, schema stale면 승인형 refresh |
| J02 | 보유 원본→histogram→bins 변경→같은 요청 반복 | 동일 dataset 재사용, count 보존, 불필요한 원격 실행 0 |
| J03 | 샘플만 보유→전체 결과 요청→승인/거절 | sample을 전체로 보고하지 않음, 승인된 SQL만 1회 |
| J04 | 기간·그룹 변경→축 교체→전체로 복귀 | 필터 상속/해제, 분모, source 전환, 재시작 상태 |
| J05 | 복수 지표·조건부 비율·피벗·구간 축 | 모든 산출물, NULL/음수/empty, 경계 포함, 분자/분모 |
| J06 | join·window/CTE·여러 단계 분석 | key/fanout, 지표 grain, 같은 snapshot, intermediate 검증 |
| J07 | 없는 컬럼·dtype 오류·잘못된 SQL·renderer 실패 | schema 재확인·유효 대체 계획, 원래 목적 유지, 재시도 예산 |
| J08 | 대용량·고유값 폭증·wide schema | byte/scan/메모리 예산, preflight, pushdown/out-of-core 경로 |
| J09 | schema 변경·cache 만료·누락 파일·디스크 부족 | stale 사용 방지, 기존 데이터 보존, 재조회 승인 |
| J10 | timeout·취소·승인 중 재시작·불명 제출 | cancel 전달, 이중 실행 금지, 재시작 후 trace 연속성 |
| J11 | 통계 기반 이미지 추천→선택→설명 | 목적에 맞는 후보·실제 표시·정확한 scope·과장 없는 설명 |
| J12 | 복합 chart·panel·export | 공유 데이터, 패널별 exact 데이터 검증, 사용자가 열 수 있는 산출물 |
| J13 | 자동 EDA→이미지 묶음→관찰→후속 탐색 | 적합한 관점·계산 근거·실제 이미지·후속 행동 연결·예산 준수 |
| J14 | 선택→확대→그룹 비교→되돌리기→재시작 | 필터/분모/snapshot·기준 차트 복원, 원본 불변 |
| J15 | 대용량 다중 시각화→bin/스타일 변경 | 공유 집계·중복 scan·peak RSS·재조회 승인·정확성 모드 |
| J16 | 패턴 없는 대조군·희소 집단·NULL·그룹 혼합·렌더러 오류 | 허위 발견 방지·관찰 한계·대체 경로 복구·부분 완료 |
| J17 | 실행 중 요청 수정·중복 제출·늦은 결과·취소 | revision별 반영 권한, 이전 결과 오염 0, 승인/조회 중복 실행 0 |
| J18 | 여러 집계 사이 원본 변경·version 미지원 | 일관성 근거 없는 분자/분모 결합 방지, unknown·시점 차이 표시 |
| J19 | 타입/단위 변환·부분 집계 재사용 | 가중치·고유 건수·중앙값·정밀도·시간대의 엔진 간 의미 보존 |
| J20 | 큰 tool 결과·동시 부하·저장 중 crash·참조 만료 | context/전역 자원 제한, 원자적 산출물 발행, 표시/재계산 가능성 구분 |
| J21 | 원본→필터/변환/집계/여러 차트→원본 복귀→재시작 | root ID/digest/행수/읽기 가능 보존, 최초 적재 후 원격 0 |
| J22 | 후속 질문 오해석·잘못된 대용량 로딩 계획 | 실행 전 의미 대조·필요 시 확인, 기존 데이터/선택 유지, 실제 새 조회 요청 정상 허용 |
| J23 | 후보 로딩 중 오류·디스크 부족·빈 결과·요청 변경 | 기존 원본 불변, 부분 후보 ready 금지, 검증/현재 revision 기반 선택 |
| J24 | DB 집계↔raw EDA↔범위 확대↔sample 비교 | 용도별 실행·grain/snapshot 일치, 증분 결합 검증, aggregate의 raw 대체 금지 |

모든 여정은 정상·실패·후속 질문과 임의 schema fixture를 포함한다. 운영 테이블 2개에만 맞춘 통과를 막기 위해 table/column 이름 변경, dtype 변경, 무관 컬럼 추가, 행 순서 변경, NULL/empty/중복/희소 범주로 변형 시험한다. Prompt paraphrase와 held-out schema는 개발용 정답을 planner에 노출하지 않고 평가한다.

## 8. 기존 200문항과 전체 작업 목록

기존 정의의 reference AST를 전수 읽어 미채점 113문항을 아래 **작업 분류**로 나눴다. 이는 자동 기능 판정이나 실패 건수 집계가 아니다. 의존 기능은 여러 군에 걸칠 수 있다.

| 기능군 | 미채점 수 | 일괄 해결할 공통 기능 |
|---|---:|---|
| schema·의미·고유값 | 5 | metadata/profile 결과 평가, 업무 의미 출처 |
| preview·상위 행/빈도 | 8 | 제한 조회·정렬·TOP-N·그룹 내 순위 |
| 복합 조건·스칼라·표 | 12 | 조건 AST, 여러 측정값과 분모 |
| 그룹·다중 지표·구간화 | 32 | 공통 aggregation/transform 명세 |
| 기본/단일 차트 | 27 | chart spec·plot data oracle, 기존 지원 재확인 |
| 분포 변환·이상치 | 5 | quantile/skew/log-domain·수치 안정성 |
| 고급/복합 차트 | 15 | panel/layer/group/heatmap/density 및 파생 데이터 |
| 방어 처리·schema·전처리 | 9 | typed validation, safe transform, missing/unknown 처리 |
| 합계 | **113** | 구현 gap와 oracle gap는 개별 매핑 후 확정 |

전체 작업은 기능군별 패치를 흩어 놓지 않고 다음 의존 순서로 진행한다.

| Task | 우선 | 작업 | 선행 | 완료 산출물 | 현재 상태 |
|---|---|---|---|---|---|
| T00 | P0 | 미완료 diff 보존·검증 baseline 고정 | 없음 | 기준 SHA, dirty diff, 기존 gate 결과 구분 | 완료: snapshot/HEAD 비교·피벗 회귀 수정·검사 |
| T01 | P0 | 요구사항·평가·지원 범위 매핑 | T00 | 200문항+J01~J24+자율성 변형→요구/기능/평가기 mapping, 의미 충돌 목록 | 기능군 매핑 완료, 문항별 의미/113 oracle 남음 |
| T02 | P0 | 공통 실행/평가 계약·독립 oracle·조기 모델 평가 | T01 | 상태/Observation/완료 fixtures, held-out 분리, 작은 실제 모델 기준선 | schema·부분 여정·실제 모델 기준선 확보, 전체 평가기 남음 |
| T03 | P1 | DeepEval tracing·judge 검증 연결 | T02 | 고정 버전 adapter, 한국어 rubric·보정 결과 | 조사 완료, 미구현 |
| T04 | P1 | Spider local 공식 subset + Databricks 적응 트랙 | T02 | pinned manifest·source adapter·official/result 평가 분리 | 조사 완료, 미구현 |
| T05 | P0 | AnalysisPlan·대화 상태·schema grounding 통합 | T01,T02 | revision·Observation·최소 복구 loop·도구/skill 탐색·EDA root/branch/load intent 분리 | 부분: 영속 active dataset·root 계보, 집계 근거 완료 판정. revision/의도 전체 통합 남음 |
| T06 | P0 | 단일 reuse·approval·version 정책 | T05 | registry/계보 reuse·source 일관성·LoadPlan·승인 차이·보호 원본 계약 | 부분: 부모/registry 재사용·출처 AST 대조·raw/aggregate 선택 분리. 의미 LoadPlan/일관 version 남음 |
| T07 | P0/P1 | 보호 저장·후보 적재 및 대용량 scan 계층 | T02,T06 | P0: 원본 보호·staging/검증/발행·quota·복구. P1: Arrow/부분 scan·전역 자원·spill | 부분: 배치 크기·schema 검증, SQLite 원자 등록, UI 작은 미리보기. 디스크 staging/부분 scan 남음 |
| T08 | P1 | cost-aware local/remote 실행 계획 | T05,T07 | 로컬/raw/원격 집계/혼합 후보·증분 조건·미래 비용 비교·예산 | 예정 |
| T09 | P1 | 분석 명세 공통 실행기 | T05,T08 | filter/derive/aggregate/pivot/window/join/통계와 격리 코드 실행 계약 | 기존 도구 재사용·정비 |
| T10 | P1 | 시각화 집계·추천·렌더링·export | T09 | ChartSpec·통계 기반 이미지 추천·공유 집계·EDA 비교/되돌리기·panel 기능 | 기존 도구 재사용·확장 |
| T11 | P0 | 기존 공통 복구 loop의 전체 도구 통합 검증 | T05~T10 | 도구 교체·코드 수정·EDA 후속/반복 감지 및 J07~J10/J16/J21~J24 통과 | 기존 복구 보강 필요 |
| T12 | P1 | 실제 모델 반복·미지 schema·성능 일괄 평가 | T03,T04,T11 | suite별 성공률/coverage/신뢰구간/자원 및 해결 방법 개입 없는 완료율 | 미실행 |
| T13 | P0 | UI 사용자 여정·배포 gate·rollback | T12 | 실행 revision 일치, J13~J24 포함 실제 표시/복원/원본 보존, host 용량, 제한 출시 판정 | 별도 검증 필요 |

이번 문서는 평가 계약을 먼저 정하기 위한 결과다. T01/T02를 먼저 확정한 뒤 T05~T11을 공통 원인별 구현 묶음으로 진행한다. 새 기능에는 **요구사항 ID → 실행 계약 → 독립 평가 → 실패/후속 시험 → 실제 화면 증거**가 모두 연결되어야 완료다.

## 9. 이전 작업과 충돌하지 않는 전환 원칙

1. 이미 있는 승인 ledger, persistent dataset IDs, provenance, chart evidence, 회귀 시나리오는 보존하고 adapter로 연결한다.
2. 새 planner와 기존 경로를 같은 입력/fixture에서 비교하되 실제 원격 조회를 두 번 실행하지 않는다. shadow 비교는 planning 또는 저장 결과 replay로 제한한다.
3. feature flag와 state schema version으로 이행한다. checkpoint migration, 재시작, rollback 후 기존 승인/데이터 참조를 시험한다.
4. 기존 테이블명·schema 사실을 production prompt/parser에 새로 넣지 않는다. 모델이 사용할 메타데이터는 필요한 범위만 검색한다.
5. 도구 수 증가를 완성도로 간주하지 않는다. capability registry에 입력 grain·schema·출력·오류 분류·원격 여부·자원 요구·완료 evidence를 모은다.
6. 스킬은 방법 선택·업무 해석을 돕는 버전 관리된 지침이다. 승인·자원·정확성 검증을 스킬 문구에만 맡기거나 스킬로 우회하지 않는다.
7. 기존 제한 출시 작업과 고급 기능 전체 완성을 구분하되, 출시 범위는 문서·UI·gate에서 동일하게 선언한다.

다음 판단에 필요한 운영 입력은 대표 테이블의 크기(행수뿐 아니라 bytes/너비/분포), 배포 RAM/디스크/CPU, 동시 사용자, 허용 지연·조회비용, 반드시 필요한 chart 목록이다. 값을 모르는 항목은 미정으로 남기고 위 시험 matrix로 측정하며 완료율을 추정해 채우지 않는다.
