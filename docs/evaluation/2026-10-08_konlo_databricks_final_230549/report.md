# Databricks konlo #1 최종 재평가 — NO-GO

평가 생성: 2026-10-08T23:48:19.519965+09:00. 설정은 자식 프로세스에서 Databricks로 지정했고 기존 `.env`는 변경하지 않았다. 실제 브라우저 새 대화 `8fea7041-bc7e-46c7-8697-cd4fd6c1b250`에서 원본31문항을 순서대로 입력했다. 대화·실행 ID31쌍을 대조했으며 평가 도중 제품 코드 해시는 바뀌지 않았다.

## 완료한 계획

1. goal 입력에서 관련 없는 테이블의 반복 스키마 JSON을 제거하고 이름 목록/총 개수만 제공한다. 현재 요청과 정확한 기존 조건은 보존한다. 200개 테이블 중 미리보기 밖의 명시한 테이블도 유지한다.
2. 모델 호출 전에 pending 요청을 체크포인트에 기록한다. 실패·한도 소진 후에는 현재 미완료 요청과 마지막 완료 분석을 구분하며, 재시작·재개와 UI 선택 변경을 검증했다. confirmed snapshot의 재귀 누적도 제거했다.
3. 재평가 중 발견한 Databricks LIMIT0 문자열 타입 소실 경로를 수정했다. 실제 typed Parquet footer를 사용하며 untyped null은 계속 미확인으로 남긴다. 범주형 분포는 원본 전체 재로딩 대신 COUNT(*) 빈도를 사용한다.

## 결과

- 고정 기대값 기준 **13/31 PASS, 18/31 FAIL (41.94%)**. PASS: 1, 2, 3, 4, 5, 15, 19, 20, 21, 22, 23, 24, 26.
- 런타임은 28/31을 answered로 보고했지만 요구를 충족한 것은13개다. 실행 완료율을 출시 성공률로 사용하면 안 된다.
- 입력 한도 초과 **0/31**. 최초 baseline은1PASS/30FAIL이며30건 모두 입력 한도 초과였다.
- 중간 수정본은7건(4PASS/3FAIL)에서 공통 원인 수정을 위해 중단했다. 나머지24건은 미실행으로 보존했다. 이번31건과 합산하지 않았다.
- `education` 분포 요청5: 이전218.686초/원본100,000행 조회/한도 소진 → 최종39.222초/빈도4행/실제PNG/독립 범주 빈도 일치.
- 전체 산점도19·20·23: 전체750,000행, 고유좌표132,613개. 독립 조회의 정렬 좌표·빈도 해시 `d90e8f00f37959a42a701d79bab12af1e3bfaae42da5ee63f708a8d411ae8498`와 일치.
- 직전10행 산점도22: 원격 조회0회,21번의 정확한10행과 일치.
- 이전 자산의 payload/metadata 변경0건, 삭제0건. 분석 중 선택 값도 불변이었다. 데이터 보존은 의도·필터 보존과 별개의 검증이다.
- 앱675건:671PASS/4SKIP (104.399초). migration137PASS (20.219초). compileall/diff check PASS. 단위/계약 테스트 대부분은 합성 또는 scripted model이며 실제 언어 이해 점수가 아니다.

## 추가 출시 차단 문제와 다음 묶음 작업

| 우선순위 | 근거 | 남은 수정 요구 |
| --- | --- | --- |
| P0 | 16의 goal population audit가 기존 age30~40 AND education IN(primary,secondary)를[]로 변경;17~18도전체750,000행 사용 | LLM이 scope 유지/추가/대체/해제 의도를 명시하고, 이전 조건과 변경 근거를 대조하는 독립 계약을 만든다. 미리보기 행 수는 필터 해제 사유가 아니다. metadata→preview→chart 여정을 실제 모델로 통과해야 한다. |
| P0 | 29~31에 명시된 teleai_default.bank_loan을 workspace.default.bank_loan으로 대체 | DB/schema/table 전체 식별자를 원문 증거에 바인딩한다. 짧은 이름은 명시된 다른 namespace의 suffix로 매칭하지 않는다. 존재하지 않거나 미확인인 출처는 직접 확인/명확화하고 기존 테이블로 대체하지 않는다. |
| P1 | 9~14: 범례·색상·stack 요청을 일반 bar 목표로 축소하고 완료 표시 | goal grammar→compiler→tool→renderer→completion evidence에 legend/colors/stacked를 끝까지 연결하고 PNG와 render_spec으로 확인한다. 지원하지 않는 옵션을 묵살하지 않는다. |
| P1 | 25·27·28: 없는 출처 요청에서 goal_unverified 일반 문구 | 아직 확인하지 못한 사용자 명시 subject도 결과 증거와 별도로 보존한다. catalog/schema 탐색 또는 구체적 질문으로 복구한다. |
| P1/명확화 | 6~8: 고정 oracle은 age 분포, 실제 goal은 education 분포; 필터/합계 자체는 맞음 | 필터 컬럼과 시각화 대상을 구분한다. 이 요청은 축을 명시하지 않아 해석 여지가 있으므로 기대값을 몰래 바꾸지 않고 review 근거를 함께 남긴다. |

이번 변경으로 위 새 차단 문제들이 해결됐다고 주장하지 않는다. 현재 제품은 **NO-GO**다. 회사 Azure API·Windows 서버는 이번 평가 대상이 아니다. 이 수치는 공식 DeepEval/Spider2.0 점수가 아니다.

## 개별 결과

| 턴 | 판정 | 실행초 | 모델호출 | 근거 |
| --- | --- | ---: | ---: | --- |
| 1 | PASS | 46.488 | 2 | 독립 조건·결과 검증 통과 |
| 2 | PASS | 27.530 | 3 | 독립 조건·결과 검증 통과 |
| 3 | PASS | 38.865 | 3 | 독립 조건·결과 검증 통과 |
| 4 | PASS | 31.430 | 3 | 독립 조건·결과 검증 통과 |
| 5 | PASS | 39.222 | 3 | 독립 조건·결과 검증 통과 |
| 6 | FAIL | 50.786 | 4 | contract mismatch |
| 7 | FAIL | 37.375 | 3 | contract mismatch |
| 8 | FAIL | 34.207 | 3 | contract mismatch |
| 9 | FAIL | 36.274 | 3 | contract mismatch |
| 10 | FAIL | 35.317 | 2 | contract mismatch |
| 11 | FAIL | 41.839 | 3 | contract mismatch |
| 12 | FAIL | 56.040 | 3 | contract mismatch |
| 13 | FAIL | 59.689 | 3 | contract mismatch |
| 14 | FAIL | 59.751 | 3 | contract mismatch |
| 15 | PASS | 35.389 | 2 | 독립 조건·결과 검증 통과 |
| 16 | FAIL | 74.665 | 4 | preview lost preceding population conditions |
| 17 | FAIL | 31.715 | 3 | unrequested prior-population removal |
| 18 | FAIL | 26.128 | 3 | unrequested prior-population removal |
| 19 | PASS | 25.326 | 3 | 독립 조건·결과 검증 통과 |
| 20 | PASS | 25.285 | 3 | 독립 조건·결과 검증 통과 |
| 21 | PASS | 25.518 | 3 | 독립 조건·결과 검증 통과 |
| 22 | PASS | 24.807 | 3 | 독립 조건·결과 검증 통과 |
| 23 | PASS | 25.405 | 3 | 독립 조건·결과 검증 통과 |
| 24 | PASS | 23.228 | 2 | 독립 조건·결과 검증 통과 |
| 25 | FAIL | 18.354 | 3 | goal_unverified |
| 26 | PASS | 21.815 | 2 | 독립 조건·결과 검증 통과 |
| 27 | FAIL | 33.559 | 5 | goal_unverified |
| 28 | FAIL | 18.719 | 3 | goal_unverified |
| 29 | FAIL | 27.163 | 4 | requested qualified source was silently replaced |
| 30 | FAIL | 19.367 | 2 | requested qualified source was silently replaced |
| 31 | FAIL | 25.019 | 3 | requested qualified source was silently replaced |

## 증거

Git에는 report.md/report.json/go_gate.json/build.json/plan.json과 화면 증거를 보관한다. 아래 NN 원문 체크포인트·실행 이벤트, 전체 oracle 및 재생 수집 스크립트는 로컬 평가 폴더에만 보존하며 push하지 않는다. report.json에31개 요청의 ID·판정·출처·조건·조회 횟수 요약을 함께 담았다.

`report.json`은 요청/run ID, 실제 출처·조건·조회 횟수와 판정을 담는다. `NN.json`은 각 요청의 체크포인트·자산 해시·실행 이벤트, `NN_check.json`은 독립 기대값 검증이다. `oracles.json`은 실제 DB의 별도 집계이며 `build.json`은 평가한 코드 해시다. 최종 출처 검증에서29·30·31은 namespace가 같아야 한다는 조건을 명시해, 좌표 값만 같아서 잘못 통과하는 평가기 누락을 보완했다. 프롬프트의 끝 공백만 비교 정규화했다.
