# 관계 탐색과 승인형 조인 집계 검증 — 2026-09-26

**범용 자율 분석 agent 출시는 NO-GO다.** 이번 변경은 관측한 FK 관계로 SQL을 검증하고, 필요한 집계만 가져와 완료하는 경로를 보강했다. 성공한 합성 여정과 재발한 모델 탐색·시간 한도·API 호출 제한 실패를 모두 보존했다. 특정 테이블·업무 값은 production 코드에 넣지 않았다.

## 문제와 구현

1. 기존에는 사용자가 조인 키를 직접 지정하지 않으면 SQL JOIN을 검증하지 못했고, 공개 SQLite 평가 adapter도 실제 FK 선언을 버렸다. `inspect_table_relationships`가 fresh DB metadata를 읽거나 정확한 승인형 metadata SELECT를 계획한다. SQLite에서는 실제 PRAGMA를 사용한다. Unity Catalog 계획은 같은 catalog의 `referential_constraints`와 `key_column_usage`를 연결한다. 컬럼 및 키 순서 근거는 [Databricks key_column_usage](https://docs.databricks.com/aws/en/sql/language-manual/information-schema/key_column_usage), [referential_constraints](https://docs.databricks.com/aws/en/sql/language-manual/information-schema/referential_constraints)다.
2. 관계 선언은 업무 의도를 대신하지 않는다. 요청에서 확인한 전체 출처 집합, 명시적 JOIN 의도, 최신 스키마, 유일한 관계, 복합 키 전체, 기존 필터를 모두 검증한다. 모델이 `join_datasets`를 선택했다는 이유만으로 사용자 JOIN 의도가 생기지 않도록 별도 상태를 둔다. 잘못된 조건·추가 테이블·부분 복합 키·여러 FK 역할·오래된 metadata는 차단한다.
3. SQL에서 이미 조인 집계가 검증됐어도 기존 완료 계약은 로컬 joined raw dataset 생성을 추가 요구했다. 통계만 요청한 경우 검증된 승인 SQL 집계로 완료하며, 원본 저장/내보내기 요청은 별도 결과가 있어야 완료한다. FK 선언이 실제 데이터의 유일성과 참조 무결성을 보증한다고 답하지 않는다.
4. 일부 테이블이 미로딩인 조인 통계 요청에서 모델이 로컬 조인과 불필요한 설명 탐색을 반복했다. fresh 출처가 확인되고 일부 raw가 없는 명시적 요청에 한해 7개 탐색·승인 도구로 메뉴를 좁히고 실행 경로를 알려준다. 양쪽 raw가 준비되거나 요청이 복합/모호하면 기존 메뉴를 유지한다. 실행 권한과 SQL 검증은 축소하지 않는다.
5. 승인된 metadata SQL이 여러 information_schema 테이블을 참조해도 기존 raw 선택을 유지한다. 원본 SHA256, 선택 ID, 재시작 후 metadata 재사용을 회귀로 검사한다.

## 검증 결과

| 검사 | 결과 | 해석 |
|---|---|---|
| 최종 application unittest | 320/320 PASS | 관계·완료·승인·원본·도구 선택 계약 포함 |
| migration | 133/133 PASS | 상태·보존 회귀; 최종 원격 CI에서도 재검사 |
| reference + Level 3 | 217/217 PASS, 58 figures | reference 코드 성공과 실제 agent 성공은 별개 |
| 고정 200문항 production graph | 97 PASS / 103 UNGRADED | 실모델을 호출한 문항 6개; 나머지 지원 문항은 로컬 결정 경로 |
| 최종 Spider 고정 5문항, 두 track | 원문 0/5, SQL 지시 0/5 | 각 track 제출 SQL 1개도 정답 아님; SQL 지시 track 2문항은 RateLimitError |
| 실제 Qwen + 공개 합성 SQLite | 성공 시 정답 3, 원본 불변, 승인 후 집계 SQL 1회 | 개발 중 모든 시도·실패는 live JSON에 보존 |
| 실제 웹 보유 표본 후속 요청 | age ≥ 60의 MAX(age)=86, 0.458초 | 독립 Parquet 계산 일치, 모델 0회, 새 DB 조회 0회 |
| compileall / diff / pip | PASS | 정적·의존성 검사 |

실모델 초기에는 호출 한도 10회 및 시간 한도에서 중단됐다. 도구 선택 보강 후 연속 두 실행은 모델 4회씩, 8.550초와 8.349초에 정답 3으로 완료했다. 같은 단계의 다른 실행 두 개는 `RateLimitError`로 실패했고, 마지막 엄격한 사용자 의도 분리 적용 후 실행도 그중 하나다. 모델 독립 graph 회귀는 최신 코드에서 통과했지만, 이 결과로 반복 신뢰성이나 일반 SQL 성능이 완성됐다고 판정하지 않는다. 한 번의 성공만 골라 성공률을 제시하지 않는다.

Spider는 기존과 같은 `local221`, `local009`, `local210`, `local358`, `local198`을 두 track으로 각각 재실행했다. 원문 track은 local221만 첫 SQL을 제안했지만 공식 실행 비교에서 오답, 나머지는 범위 미확정·출력 없음·모델 timeout·JOIN 관계 미검증이었다. SQL 지시 track은 local358 SQL이 오답, 두 문항은 공급자 RateLimitError, 두 문항은 범위/관계 검증에서 멈췄다. 호출 제한 실패를 SQL 추론 오답으로 분류하지 않는다. 이 adapter는 첫 승인 제안에서 멈추므로 중간 탐색 SQL과 최종 정답 SQL을 구분하는 종단 평가가 아직 필요하다. 공식 전체 547문항 성적을 측정한 것은 아니다.

합성 여정의 `query_databricks`는 production 승인 gateway 이름이며 executor는 공개 로컬 SQLite에만 연결했다. Databricks model serving은 사용했으나 warehouse SQL은 0회다. 승인 전 executor 호출은 0회, 승인 뒤 실제 SQLite 집계는 1회였다. 첫 두 개발 실행의 adapter에는 별도 공개 SQLite proposal probe가 있었으므로 그 실행의 executor 카운터를 모든 승인 전 SQL 실행 횟수로 해석하면 안 된다. 이후 실행에서 해당 probe를 제거했다. fixture 설치·PRAGMA 스키마 읽기는 별도 로컬 준비 작업이다.

실제 웹의 기존 10,000행 raw SHA256은 `7c4a1c20588261932629087d091d43e1ad2d80f4e36cb3d1acb20eb8462fe10d`, 승인 장부는 `completed=1`로 유지됐다. 새 앱 PID는 45374이며 `127.0.0.1:8502/_stcore/health`는 `ok`다. 이 웹 검증은 보유 데이터 회귀이며 새 FK 조회나 조인 검증을 대신하지 않는다.

## 지원 경계와 남은 작업

- 자동 관계 검증은 요청에 명시된 2~4개 서로 다른 물리 테이블, 단순 SELECT, INNER JOIN, 유일한 FK 관계에 한정한다. CTE/서브쿼리/자기 조인/외부 조인/다중 역할 FK의 자동 관계 선택과 암묵적 업무 질문의 출처 결정은 미완료다. 이름이 유사한 컬럼으로 관계를 추측하지 않는다.
- metadata SELECT는 같은 catalog에 있는 참조 키만 포함한다. 최대 65행이 반환되면 불완전할 수 있어 사용하지 않는다. 실제 Unity Catalog 권한·schema·조회 결과를 검증하려면 정확한 SQL의 승인이 필요하다. 이번에는 실행하지 않았다.
- 실제 데이터 키 중복·미일치·조인 행 증가를 원격 계획에 반영하는 비용/카디널리티 검증은 남았다. SQL 집계는 joined 행 기준이며 DB FK 선언만으로 사실상 1:N을 보증하지 않는다.
- 103개 독립 oracle, held-out 자연어/다단계 여정, DeepEval judge calibration, 복잡한 Spider SQL과 최종 결과까지 실행하는 평가 adapter가 남았다.
- 최종 빌드의 대규모 원격 신규 적재·전송 실패/재시작·RSS, 격리 코드 실행기는 미완료다. 별도 서버 배포는 사용자 요청에 따라 제외한다.

## 증거 파일

- `2026-09-26_relationship_live.json`: 개발 중 성공·실패, 도구 관찰, SQL, 승인 카운터, 시간.
- `2026-09-26_relationship_scores.json`: 고정 200문항 결과와 문항별 모델 호출 수.
- `2026-09-26_relationship_web.json`: 실제 웹 결과, 독립 정답, 원본 hash, 승인 수.
- `2026-09-26_relationship_spider.json`: 중간/최종 두 track의 고정 5문항 공식 채점, 제안 SQL, 실패 원인.
- `tests/test_analysis_relationships.py`: 실제 PRAGMA, metadata 승인/거절·재시작, 부정확 JOIN 차단, SQL 집계 완료 및 raw 저장 미완료 구분.
- `tests/test_analysis_tool_focus.py`: 미로딩/로딩/모호/복합/연결 없음의 도구 메뉴 경계.

## 원격 검증

코드와 평가 commit `49b7a6e312f538e1ca9161f988fa9dde4c6cf799`를 push했고 [GitHub Actions 36238260204](https://github.com/konlo/teleai/actions/runs/36238260204)의 migration/application/agentic/reference/compile 단계가 모두 성공했다(2분 12초). [PR #68](https://github.com/konlo/teleai/pull/68)은 draft·미병합 상태다. 실제 모델/warehouse/복잡 SQL의 미완료는 이 CI 성공과 별개다.
