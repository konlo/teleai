# 최근 분석 대상과 생략된 차트 요청 연결 — 2026-10-03

기대 결과는 `age histogram → education 값 목록 → histogram을 그려줘`에서 education의 범주별 실제 빈도 막대다. education은 문자열 범주이므로 수치 구간 histogram을 강제로 만들지 않는다. 고유값 네 행을 각각 한 건으로 세는 것도 오답이다.

## 원인

최초 오답 run `580b30a5544345d1bc82691eb99f55ac`에서 `cached_chart_reuse_planned`는 이전 age 차트 `8deaee62-87a7-418e-899f-44a9501f4fe0`를 선택했다. 모델 호출 0회, show_chart 1회, answered 0.341초였다. 생략된 컬럼을 최근 education 완료 결과에 연결하지 않아 required_columns가 비었고, 저장 차트 검사도 요구 축이 없으면 예전 차트를 허용했다. 이 오답은 LLM이 생성한 SQL 문제가 아니라 agent의 대상 결합·캐시 검증 문제다.

## 수정

- 대상이 생략된 짧은 차트 발화는 직전 검증된 분석의 한 컬럼·출처·조건·결과 범위를 상속한다. 최근 값 목록의 대상이 오래된 PNG보다 우선한다. 명시한 다른 컬럼/출처, 여러 대상, UI 데이터 전환은 임의로 상속하지 않는다. 알 수 없는 새 컬럼명을 생략으로 오인하지 않는다.
- 캐시 재사용은 확정된 축을 요구하며, WHERE가 없어도 다른 축/출처의 prepare_histogram과 show_chart 호출을 거부한다.
- 실제 문자열/범주/boolean schema의 histogram 요청은 categorical_distribution으로 분리한다. MySQL longtext/text/enum도 포함하며 특정 테이블/컬럼 이름은 production 코드에 넣지 않는다.
- 기존 prepare_histogram/render_histogram에 categorical 계약을 추가했다. 완전한 실제 COUNT(*) 결과만 검증하여 bar PNG를 생성한다. 50개를 넘으면 상위 범주만 표시했다고 안내하며 실제 labels/counts/전체 빈도 합계를 차트 metadata에 남긴다. 결측 제외를 명시하고 NULL이 포함된 빈도 결과를 그대로 제외된 결과라고 출력하지 않는다.
- 저장된 현재 요청의 COUNT receipt가 있으면 새 조회 없이 렌더링한다. 기존 timeout checkpoint도 타입 분류와 도구 복구 계약을 갱신하여 재개할 수 있다. 원본 선택 및 파일을 바꾸지 않는다.
- 기존 prompt와 도구 설명을 이 계약에 맞춰 수정했다.

## 검증

관련 14개 파일 **112 PASS / 111 subtests PASS**, compileall 및 diff check PASS. 원본 digest/선택 보존, 재시작 후 생략 참조, 명시한 수치 컬럼의 우선순위, 필터 유지, 기존 age 차트 거부, 고유값과 빈도 구분, longtext remote 계획, timeout 뒤 저장된 COUNT 복구를 검사했다.

실제 Databricks의 관측 schema에서 문자열 컬럼을 선택하여 schema 0행 → DISTINCT 목록 → 컬럼 생략 histogram → 반복 이미지 재사용을 확인했다. SQL 3회, 반복 0회, 모델 0회, 빈도 합계 750,000, bar PNG 18,348바이트. [실행 결과](databricks.json). 이는 실제 adapter와 결정적 대상/완료 경로 검증이며 자유형 LLM 품질 평가와 구분한다.

실제 MySQL 웹 대화의 동일 흐름도 확인했다. 검증 중 longtext 분류 누락으로 첫 시도가 모델 추론으로 넘어가 run `916a9f618550448dbec2c2e6ad933f5e`가 172.812초/ReadTimeout `5cac33ac5234`로 중단됐다. 범주 빈도 4행은 실제 조회·저장이 완료되어 있었다. 분류와 checkpoint 복구를 보강했고, 재개 경로의 누락 import도 수정하여 별도 재현 회귀로 확인했다. 실패 근거를 성공 기록으로 덮어쓰지 않았다.

최종 같은 대화에서 저장된 빈도를 이용하여 **education bar, 빈도 합계 750,000**을 표시했다. 재개 run `d4e5d05a19984f90aab6c0bcd16d7ae0`는 render_histogram으로 완료했고 추가 DB/모델 호출이 없다. [실제 화면](fixed_web.png). 최초 잘못 표시한 age 차트는 과거 기록으로 남아 있으며 새 education 결과와 구분된다. 서버 PID 7705/8504 health ok, 실제 .env 기본 Databricks 설정 유지.

범용 GO·회사 서버 적용·모든 자유형/복합 차트 참조 해결을 주장하지 않는다. 이번 요청에서 commit/push하지 않았다.
