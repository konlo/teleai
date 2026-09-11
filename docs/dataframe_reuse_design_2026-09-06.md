# DataFrame 재사용과 재조회 판단 설계

진단 근거는 기존 경로의 작성 당시 상태다. 이후 부분 구현을 유지하며 실행 기반은 [LangChain/LangGraph 전환 설계](langchain_migration_design_2026-09-06.md)를 따른다.

## 확인한 결함
`utils/data_context.py:141`의 evaluate_data_readiness는 source_table, preview, columns, row_count 및 ranked task를 검사한다. DataRequirement.filters는 전달받지만 실제 재사용 판정에서 비교하지 않는다. DataFrameState.query도 조건 범위/집계/잘림 판정에 사용하지 않는다. 비교 filter_conditions는 requirement 생성 시 컬럼만 추가되고 조건 표현 자체는 보존되지 않는다.

`utils/session.py:784` 부근 SQL 로딩 경로는 query가 있으면 is_preview=False로 저장한다. 이는 LIMIT 포함 쿼리 결과의 완전성을 뜻하지 않는다. 전체행/부분행/표본/집계 결과를 구분할 메타데이터가 필요하다.

오프라인 재현: diagnostic.events에서 model='A' LIMIT 1000인 query_result 상태, columns=(event_time,model,temperature), row_count=1000을 구성했다. model='B' 요구와 전체 모델 요구가 모두 USE_CURRENT를 반환했다. 실제 DB나 모델은 호출하지 않았다.

## 권장 역할 분담
Agent는 대화 맥락에서 분석 대상/조건/기간/지표/정확성 요구를 해석한다. 데이터 도구가 현재 자산의 출처와 범위를 검증해 재사용 가능 여부 및 이유를 반환한다. 자연어 분류기로 agent 입구를 고정하지 않으며, 구조화된 요구는 데이터 접근 도구의 계약으로 사용한다.

## 데이터 자산 메타데이터
- dataset_id/result_id 및 parent_id: 데이터와 파생 결과의 참조
- source identity: canonical table 또는 다중 source/join lineage, schema/source version 또는 snapshot
- columns와 타입, grain: 원본 관측행인지 일/월/장비별 집계인지
- predicate: 실제 적용 조건의 구조화 표현, 시간 범위, timezone 및 interval 경계
- aggregation: group keys, 지표 식, 집계함수, 필요한 중간 통계
- coverage: 해당 조건 범위에서 complete / truncated / sampled / unknown
- limit와 실제 로딩 결과: 단순 len(df)로 완전성을 단정하지 않음
- loaded_at 및 freshness 정책

metadata는 실행한 조회/변환에서 생성한다. 모델의 주장만 저장하지 않는다. 임의 SQL을 완전히 해석할 수 없으면 unknown으로 표시한다. 프로파일과 컬럼 존재는 행 커버리지를 증명하지 않는다.

## 판정
동일 출처/호환 snapshot이고 요청 컬럼을 확보했으며 요청 모집단이 보유 데이터의 완전한 범위 안에 있고 분석에 필요한 세밀도가 보존되어야 재사용 가능하다. predicate 포함 관계는 지원하는 단순 식에 대해 증명하며, OR/NOT/복잡한 join 등 판단 불가를 무리하게 참으로 처리하지 않는다.

- REUSE: 같은 범위/지표/형태, 또는 명시적으로 현재 표본만 다시 시각화
- DERIVE_LOCAL: 완전한 원본 범위에서 필터 축소/새 집계를 로컬 계산할 수 있음
- QUERY_SOURCE: 기간/모집단 확대, 없는 컬럼, 손실된 원본 세밀도, 원본 기준 정확한 결과에 불충분한 표본/잘림
- NEED_SCOPE: 사용자가 현재 표본과 원본 전체 중 무엇을 말하는지 맥락으로 결정할 수 없음

요구가 명확해도 데이터가 불충분하면 필요한 원본 조회를 제안하고, Databricks 실행 전 반드시 사용자 승인을 받는다. QUERY_SOURCE는 조회 제안이 필요하다는 판정이며 실행 권한이 아니다. 범위가 모호하면 의미를 먼저 확인한다. 의미 확인과 실행 승인은 별개다. 기존 데이터의 로컬 재사용·계산은 이 원격 승인 대상이 아니다.

예외와 주의:
- 전체 데이터가 아니라 요청 범위 전체면 충분하다.
- LIMIT는 잘림 가능성이다. 별도 probe/쿼리 완료 정보 등 증거 없이 complete로 표시하지 않는다. 필터링한 표본도 보통 complete가 아니다.
- 빈 결과도 해당 조건을 완전히 조회했다면 유효한 결과다. 0행만 보고 계속 재조회하지 않는다.
- 평균들의 단순 평균은 원본 평균과 일반적으로 다르다. 집계 재사용은 sum/count 등 충분한 통계와 집계 정의가 일치할 때만 허용한다.
- 현재 표본 자체의 시각화와 원본 모집단의 정확한 통계는 서로 다른 요구다.
- 데이터 증분 결합은 안정된 키, 중복 제거, 동일 snapshot 및 범위 증명이 가능할 때만 사용한다.

## 대용량 실행
재조회가 필요해도 전체 원본을 pandas로 가져올 필요는 없다. 필터/컬럼 선택/집계를 원본 DB에서 수행하고 작은 결과를 DataFrame으로 받는다. 원본 점이 많은 차트는 의도에 맞는 표본/집계 표시를 선택하되 근사임을 명시한다. 필요한 세밀도를 잃는 집계를 임의로 적용하지 않는다.

전역 df_A 하나를 덮어쓰기보다 자산 참조 목록과 제한된 캐시를 둔다. 큰 자산은 파일에 보관하고 메모리 예산을 초과하면 내보내되 id/metadata를 유지한다. 동시 업데이트는 데이터와 metadata를 함께 갱신한다.

## 구현 순서
1. loader가 coverage/predicate/grain을 기록하고 데이터 도구 공통 gate를 거치게 한다. legacy agent도 동일 gate 적용.
2. 알려진 단순 조건의 포함 관계와 local derive를 지원하고 unknown은 재조회/범위 확인으로 처리한다.
3. dataset/result ID를 세션과 연결해 '그중', '전체로', '지난 결과'를 해석할 근거를 제공한다.
4. A→B, 기간 축소/확대, LIMIT 표본→전체 통계, 집계→원본 분포, 차트 스타일만 변경, 완전한 빈 결과를 회귀 평가한다.

최초 작성 시에는 설계와 기존 코드 재현만 수행했다. 이후 AnalysisSession용 자산 저장소와 단순 재사용 판정이 부분 구현되었다. 전체 도구의 범위 검증 및 영속 자산 저장은 전환 단계에서 보완한다.
