# 전체 원본 histogram 범위 계약 재현·수정

2026-10-12 KST. Databricks + 로컬 Ollama gemma4:e4b. 회사 Azure/Windows 및 전체 GO 판정과 구분한다.

**최신 후속 결과**: DB bin 도구를 연결해 원래 요청의 실제 브라우저 완료와 전체 9,524,806개 빈도를 검증했다. [후속 검증 기록](source_bins.md)을 참조한다. 아래는 06:53까지의 실패 재현·수정 기록이다.

## 확인한 원인

과거 19f8ef3e 실행의 검토 JSON은 저장되어 있지 않다. 이번에는 동일 요청과 관측 컬럼으로 실제 모델의 population_audit 역할을 다시 실행했다. 20.048초, 2회 호출에서 `longitude`를 predicate 컬럼으로 선언했으나 실행 조건은 비어 있었다. 첫 응답은 resolved predicate와 빈 조건의 불일치, 두 번째 응답은 unresolved와 빈 확인 질문의 불일치로 거절됐다. 따라서 모델이 출력 축을 모집단 조건으로 오분류하는 실패를 실제로 재현했다. 이 새 응답을 과거 JSON 원문이라고 주장하지 않는다.

수정 후 같은 역할 재실행은 10.236초, 1회 호출로 `source_scope/unfiltered_source`, 컬럼 없음, `change=clear`, 조건 없음으로 통과했다. 비공개 사고 과정이나 자격증명은 보고서에 저장하지 않았다.

## 공통 수정

- 출력 컬럼·조건·원본 범위를 구조화 계약에서 구분한다. 전체 원본 요청은 source_scope이며, 조건을 만족하는 모든 행은 실제 필터를 유지한다.
- 모델이 전체 원본으로 판단했으나 delta를 modify/keep으로 표현한 경우, 조건이 정말 비어 있고 인용·스키마 검증을 통과할 때만 clear로 정규화한다. 실제 조건 삭제나 자연어 키워드 강제 분기는 하지 않는다.
- source_population에 현재 요청의 불필요한 quote가 붙은 형식 오류를 정규화해 중복 추론을 줄인다. 범위 변경 판단은 유지한다.
- 검토의 kind/status/관측 컬럼/조건 수/quote 해시를 제한된 진단 요약에 남긴다. SQL·행·조건값·인용 원문은 기록하지 않는다.
- 0행 LIMIT 0 스키마 자산은 컬럼 근거로만 사용한다. 원본 분석 요청에서는 이를 데이터로 고정하지 않으며, 보유 결과만 분석하는 요청은 원격으로 확대하지 않는다.
- 단순 histogram 행동 메뉴는 관련 도구에 집중하되 검색·계약 조회를 통해 다른 등록 도구를 발견할 수 있다.
- 검증된 행동 단계에서는 소유한 기본 실행 지침을 간결한 형태로 투영한다. 사용자/custom 지침·모집단·실행 계약은 유지하며 call/time 한도를 늘리지 않는다.
- 원본 histogram 계획에서 dataset_id가 source와 정확히 같은 테이블명인 경우에만 빈 로컬 handle로 정규화한다. 임의 ID와 보유 결과만 분석하는 요청은 계속 거절한다. 실행 지침에서도 local ID와 원격 SQL 테이블을 구분한다.

## 실제 웹에서 추가 발견한 문제

기존 447563e5 대화와 원본 자산을 유지해 브라우저에서 재실행했다. 098b326c 실행은 redundant quote/delta 교정으로 호출 한도를 소진했다. 이를 수정한 41f797c0 실행은 범위 검토 1회 통과, 거절 0회였으나 모델이 0행 스키마 dataset을 prepare_histogram에 전달했고, 관찰 후 재계획 중 시간 예산을 소진했다(191.825초). 두 실행 모두 SQL/새 차트 0회이며 성공으로 채점하지 않았다. 이에 스키마 자산 취급과 도구 메뉴를 추가 수정했다.

독립 Databricks 조회: stormtrooper 전체 9,524,806행, longitude 유효값 9,524,806개, 최솟값 -179.99983, 최댓값 179.99963. 이 조회는 agent 실행 횟수와 분리한다.

## 최종 검증

관련 최종 회귀: **161 PASS, 64 subtests PASS (14.54초)**. `git diff --check` 통과. 서로 겹치는 실행 횟수를 합산하지 않았다.

실제 웹 b4e303d1: source_scope/clear 검토 1회 통과, prepare_histogram은 스키마 handle을 데이터로 취급하지 않고 planned를 반환했다. 모델이 query_databricks를 선택했지만 시간 제한 때문에 실행 전 중단됐다(188.744초, 모델 9회, 도구 1회, SQL/차트 0회).

기본 지침을 간결하게 투영한 실제 웹 75fe63c9: scope 검토는 다시 1회 통과했다. 그러나 모델이 테이블명을 dataset_id로 넣어 unknown_dataset_id로 거절됐고, 다음 응답은 로컬 ID를 SQL 테이블명으로 UNION하려 했다. 시간 한도에 의해 DB 실행 전에 중단됐으며, 그 SQL은 성공한 조회로 기록하지 않았다(219.591초, 모델 9회, 실행 도구/SQL/차트 0회). 이 결과로 정확한 source alias 정규화와 local ID/SQL 경계 지침을 마지막으로 보강했다.

마지막 수정 후 **실제 실패 도구 입력의 재실행**: 원래 대화의 영속 자산을 읽기 전용으로 열고 모델이 반환했던 prepare_histogram 인자를 실행했다. source_name_is_not_dataset_id 교정과 함께 `SELECT longitude, COUNT(*) AS __frequency FROM workspace.default.stormtrooper WHERE longitude IS NOT NULL GROUP BY longitude`의 planned 결과를 확인했다. 원격 호출 0회. 이것은 실제 모델 응답 재실행이며, 신규 모델 전체 여정 성공으로 합산하지 않는다.

기존 자산 3개의 metadata/payload SHA256, 원본 Parquet 2개의 SHA256, 선택 상태 모두 불변이었다. 최종 앱은 공식 launcher/PID38917/8504로 재시작했다. 설정과 자격증명은 변경하지 않았다.

**판정**: 최초 범위 계약의 오분류를 재현·수정했다. 전체 원본 histogram의 브라우저 종단 완료는 아직 입증하지 못했다. 범용 GO는 갱신하지 않는다. 마지막 source alias 수정은 실제 입력 재실행·회귀로 확인했으며 그 뒤의 신규 자연어 전체 실행은 미검증이다.

기존 대규모 연속형 고유값 집계의 100,000행 제한·DB bin 집계, 모델 지연, 장기 맥락과 회사 실제 환경은 별도 남은 평가 범위다.
