# Go 확대 검증 — 2026-10-09~10

최종 판정: **로컬 MySQL/Ollama의 지정 기본 EDA는 단일 사용자 사용 시험 GO**. **범용 agent 운영 및 회사 Windows/Azure/Databricks는 GO 미충족**이다. Codex/Claude 동등성이나 공식 benchmark 점수는 판정하지 않았다. 이번 작업은 로컬 수정이며 commit/push하지 않았다. 상세 판정은 [go_gate.json](go_gate.json).

| 확인 항목 | 실제 근거 | 범위 |
|---|---|---|
| 최신 전체 회귀 | 884PASS/4 opt-inSKIP + 실제MySQL4PASS(48.23초), 서로다른888시험의 성공 실행 | 자동화 성공 개수는 운영 판정과 별개 |
| 최신 실제 웹 복합 요청 | AVG34.8200 + 실제PNG208,699·38.140초/모델5회 | 한 복합문항의 수정 후 재시험; 최초 실패도 보존 |
| 조건 연결/재시작 | COUNT2→AVG15, 조건/원본hash 유지 | 직전 권한build의 실모델4행 여정; 최신8/8 점수로 합산하지 않음 |
| 확대 웹10문항 | DB·DOM·PNG10/10, 원본8자산 보존 | 이전 release_candidate build |
| 공식DeepEval/Spider2 | 미실행·점수null | 내부 시험을 공식 점수로 부르지 않음 |

새 웹 결과는 평균1행만 DB에서 로딩하고 기존11행 빈도/차트를 재사용했다. 기존8자산 metadata/payload와 평가 중 제품코드는 변하지 않았다. [최종 독립 검사](web_release_candidate/12_check.json), [실제 화면](web_release_candidate/latest_histogram.png). 응답 속도·반복 안정성/SLA와 아래 운영 차단 항목은 남아 있다.

현재 검증 범위는 로컬 단일 사용자 MySQL/Ollama다. 회사 Windows/Azure/Databricks 운영과 범용 자율 agent GO는 별도이며, 이전41문항 통과를 최신 수정의 전체 통과로 재사용하지 않는다.

## 이번 발견과 수정

- 단일 수치 요청의 평균/합계와 부정 표현: 독립 LLM 의무 검토로 연산과 실제 관측 컬럼을 제한한다. 데이터명/컬럼명/문장 순서로 라우팅하지 않는다.
- 평균과 차트를 동시에 요구하면 goal JSON의 작업 슬롯이 두 의무를 모두 포함하도록 검증한다. 이미 완료한 결과와 남은 의무, 사용 가능한 원본 ID를 다음 계획에 전달한다.
- NULL/BETWEEN/NOT IN: 연산별 값의 타입/길이 계약을 제공하고 실행·SQL 계보·완료 출력에서 일관되게 검증한다.
- 등록되지 않은 데이터 ID는 범위 오류와 구분해 실행 전에 차단하고, 현재 ID/출처/컬럼을 모델에 전달해 수정할 기회를 준다. 자동 ID 대체나 조건 삭제로 통과시키지 않는다.
- 마지막 허용 모델 응답의 검증된 도구 호출은 실행한다. 추가 모델 호출 한도와 전체 실행 정책은 유지한다.

## 자율 복구 실측 근거

[고정 기준선8여정](autonomous_frozen.json)은 6PASS/2FAIL이다. 반복 탐색, 도구 입력 검색, 잘못된 SQL 수정, 로컬 집계 장애1/2회, 로컬 timeout은 평균9를 대체 도구로 계산했고 원본을 보존했다. 실제 모델을 사용했으며 결정적 분석 rescue는 껐다. 단, 합성4행의 로컬 실행이고 원격 데이터 로딩 검증이 아니다. 복합 요청과 재시작 후 후속 조건 유지가 실패했다.

[후속 실패 기록](autonomous_goal_slots.json)도 보존했다. 복합 의도는 올바르게 읽었으나 모델이 실제 등록 ID 대신 UUID를 만들었다. 조건 건수 요청은 query 도구에 sql 키만 제공했고, 뒤 평균 요청에서 불필요한 NULL 조건도 추가했다. LLM 오류와 agent의 복구 계약 부족이 함께 있었으며 완전 해결로 판단하지 않는다.

## 남은 운영 차단 항목

1. 현재 모델의 복합 요청·후속 범위 자율 복구를 최종 동일 빌드에서 재현성 있게 통과시킬 것.
2. 회사 Windows/Azure/Databricks 실제 웹 여정·SQL receipt·오류 복구 확인. 별도 서버 구축은 사용자 제외 범위다.
3. 실제 대규모 DB 전송의 메모리/비용/종단 deadline/SLA 검증. 이전 합성100만행 시험은 실제 DB 대규모 전송과 다르다.
4. 생성 Python 격리 실행 및 고급 다단계 EDA/작업별 범위·의존성. 기존 선언형 분석/차트 도구 지원과 구분한다.
5. 공식 DeepEval/Spider2 프로토콜과 실제 정답 데이터 실행. 내부 사례 성적을 공식 benchmark 점수라고 표시하지 않는다.
6. recovery.py 약4,789행의 책임 분리. 새 기능 일부는 독립 모듈로 분리했으나 전체 모듈화는 완료되지 않았다.

## 실제 웹 확대 시험과 공통 실행 계약 보강

첫 새 대화의 고정 코드5문항은 2PASS/3FAIL, 뒤5문항은 NOT_RUN이다. [실패별 기록](web/frozen_results.json)을 보존했다. 3번은 테이블 참조는 맞지만 타입을 누락했고, 4번은 SUM과3조건을 올바르게 해석했으나 query/source 인자를 누락했다. 5번은 앞 조건은 유지했으나 계획 호출 예산을 소진했다. 실패를 성공으로 계산하지 않았다.

공통 보강은 단일 검증된 scalar 목표를 관측 schema에 맞춰 읽기 전용 SQL로 컴파일하고 DB에서 집계한 결과만 로딩하는 것이다. 이미 로딩된 사용 가능한 원본은 기존 로컬 경로를 먼저 사용한다. 다중 출처/그룹/미해결 조건/지원하지 않는 연산은 범용 계획 경로에 남는다. 컴파일러는 사용자 문장을 해석하지 않는다. metadata 출력 의무는 별도 소형 LLM 역할로 names+types를 확인한다. 양 DB 문법에서 독립 실제 SQL 결과를 비교한 계약 시험은 통과했다.

## 두 번째 고정 코드 웹10문항

[결과7PASS/3FAIL](web_final/frozen_results.json). 4/5/6번 SUM·AVG·추가 범주는 독립 DB 정답27,601,145 /1,005.9093 /923.0729와 일치했고, 7번 실제 PNG의 빈도 합계208,699도 맞았다. scalar 응답은41.588/38.022/38.289초로 앞서129~153초의 인자 누락/예산 실패를 해결했다. 3/9번은 단순 metadata_kind 역할이 type 의무를 계속 누락하여 실패했고,8번은 chart_adjust에 bins 옵션 자체가 없어 모델이 y_max=0을 반복 생성해 실패했다.

추가 공통 수정: metadata의 실제 요청/타입 포함 여부/컬럼 계열을 명시적 boolean·enum으로 분리. 동일 실제 모델 역할 시험5/5PASS(전체agent 점수와 구분). chart_adjust bins 지원 및 선택된 표시 필드만 JSON 생성·검증, 이전의 검증된 COUNT 빈도 자산으로 재그림을 추가. 원본·이전 그림 보존/원격 조회0회/잘못된 bin값 거절 계약까지 검증했다.


## 최종 release_candidate 실제 웹10문항

[동일 코드10/10 결과](web_release_candidate/frozen_results.json). 재제출0·제품변경0·이전8자산 변경/소실0이다. 직접 DB 정답과 실행 scope, 화면 표·이미지까지 비교했다. SUM27,601,145 / AVG1,005.9093 / 범주 확장 AVG923.0729, histogram 빈도 합208,699가 일치한다. 11행의 빈도 집계를 재사용한5bins 변경은 추가SQL0회다. 13개/107개 컬럼명+DB타입 모두 실제 DOM 표에 나온다. 응답 중앙값32.166초/최대63.714초이며 SLA 합격 판정은 하지 않았다.

9번의 초기 strict marker FAIL은 [원기록](web_release_candidate/09_strict_live_marker_check.json)에 보존했다. 16:37:49 UTC의 fresh observed_schema107개 타입은 실제 DB와 모두 일치했으며 화면도 확인된 스냅샷이라고 명시했다. 새 조회 강제 요청이 아닌 경우의 정당한 DB 관측 스냅샷까지 거절한 평가기 기준을 바로잡았다. pandas dtype/fixture/stale snapshot은 계속 거절한다. 제품 변경이나 프롬프트 재제출로 통과시킨 것이 아니다.

## 자율 복구의 마지막 공통 원인

[release_candidate 자율2사례](autonomous_release_candidate.json)는2FAIL. 복합 목표의 평균9는 정확했으나 prepare_histogram의 legend 누락을 SQL population 오류로 안내했다. semantic retry2회가 tool 수정 기회도 소진하여 실패했다. 후속 count는 실행기 없는 환경에서 원격 도구를 선택하고, sql/query 인자 계약도 맞추지 못했다. agent가 원래 목표를 임의 변경하지 않고 원본을 보존했으나, 복구에는 부족했다.

[복구 단계 보강 후 재검증](autonomous_repair_phases.json)은 복합 요청PASS·조건 후속FAIL. 평균9와 실제 PNG를 모델이 직접 도구로 완수했고 원본 불변. 165.921초/모델9회로 속도는 남은 문제다. 후속 요청에서는 local_analysis_sql까지 찾았으나 sql 키/누락 dataset_id를 범위 검사로 잘못 분류했다. 도구 JSON schema 검사가 실행 어댑터에는 있었지만 그 앞 scope 검사에는 없었던 공통 순서 결함이다.

공통 수정은 chart 입력의 expected/provided 필드 전달, 의미/입력/범위/재계획의 단계별 재시도계수(전체 모델10회 및 기존 도구/시간 예산 유지), 모든 선언된 로컬 도구의 schema preflight를 scope 검사 전에 적용, 외부 SQL 실행기가 없는 환경의 정확한 안내와 로컬 scalar 도구 노출이다. 사용자 문구/테이블명에 대한 규칙 분기는 추가하지 않았다. 실제 SQL arithmetic·PNG·원본 보존·재시작 후 COUNT2→AVG15·반복 잘못된 chart 입력의 유한 차단 회귀까지 통과했다. 최종 전체 회귀·실제 모델·웹 smoke 결과는 아래에 추가한다.


## NULL 역할의 검토 권한 보강

[최종 단계 재검증](autonomous_go_final.json)은 복합PASS / 후속FAIL이다. 이제 COUNT2는 실제 모델이70.867초에 로컬 SQL로 성공했다. 그러나 재시작 후 원래 요청의 AVG와 reading>=10 조건은 맞았는데 별도 population_audit가 IS NULL을 추가했다. 목표/SQL생성 자체뿐 아니라 검토 역할도 오류원이었고, 잘못된 검토 결과가 정상 실행을 막았다. 복합은134.621초, 후속 평균 실패는162.562초였다.

NULL 역할은 검토된 주 목표나 이전 확정 모집단에 근거가 있을 때만 감사 schema와 결과에 허용한다. 일반 평균에 새 NULL 조건을 만들 수 없으며, 요청된 NULL/NOT NULL과 null literal은 보존한다. 지원하는 NULL 연산을 제품 전체에서 제거하지 않는다. 문장에 대한 regex 분기가 아니라 두 구조화된 근거 사이의 검토 권한이다. schema를 무시한 가짜 감사의 반복 NULL 추가도 올바른 reviewed/confirmed 모집단을 덮어쓰지 못하고 실제 COUNT2→재시작→AVG15를 계산하는 회귀가 통과했다. 전체 회귀·실제 모델 재평가를 새 빌드에 고정하여 실행한다.


## 조건 변경 권한의 일반 보강

[NULL 권한 build의 실모델 재검증](autonomous_null_authority_final.json): 복합 평균9+실제PNG는91.840초에PASS, 조건COUNT2는57.804초PASS, 재시작 후AVG는121.251초FAIL이다. 이번에는 NULL이 아닌BETWEEN[10,10]을 모집단 검토가 추가했다. NULL 연산만 제한하는 것으로는 검토 역할의 조건 변경 권한 문제를 해결할 수 없었다.

기존 작은LLM population_basis 역할에 필터 변경 여부와 그 변경을 요구하는 CURRENT 원문의 정확한 절을 별도 필드로 추가했다. 같은 출처에서 변경 없음으로 해석된 후속 요청은 확정된 모집단을 사용하며, 나중의 조건 생성 역할이 새 범위를 만들 수 없다. 새 출처에는 이전 범위를 옮기지 않고, 명시적 조건 추가·제거 요청은 정상 검토를 거친다. 문구/테이블명 규칙 라우팅은 추가하지 않았다. 새 권한 필드가 없는 구형 계약은 기존 검토 경로를 유지한다. 잘못된중복범위+감사NULL을 주입한COUNT2→재시작→AVG15/원본hash 및 새 출처 격리 회귀2PASS3subtests. 이후 결과는 별도 동일build 근거로 기록한다.


## 최신 코드의 실제 재시작 후 조건 연결 결과

[동일 build 후속 재검증](autonomous_change_authority_final.json)은PASS. 실제 같은 Ollama/gemma4:e4b에서COUNT2(49.973초/모델6회)→runtime 재생성→AVG15(75.971초/모델7회)가 실제 로컬 SQL로 계산됐다. >=10 조건과 원본hash가 모두 유지됐다. 결정적분석rescue는 비활성이다. 조건 변경 권한을 LLM이 별도로 선택하고 나중 검토에 강제한 후 새BETWEEN/NULL 추가는 발생하지 않았다. 이는 합성4행의1재시작여정 결과다. 다른build의6/8 기준선이나 복합PASS와 합쳐 최신8/8로 표시하지 않는다. 반복 재현성·실원격 로딩·고급 범위의 완성도 점수는 아니다.


## 최신 전체 회귀와 MySQL 실제 검사

최신build 전체tests+ migration은882PASS/4 opt-inSKIP,591subtests,203.49초다. 이후 실제MySQL 직접검사3PASS 및Streamlit 목록→실제schema 검사에서45초AppTest 제한 초과1FAIL(전체87.48초)을 기록했다. 전체회귀와 동시실행 중이었으며, 제품 예외나DB timeout으로 판정할 근거는 없다. 다른시험 종료 후 동일코드·동일45초제한으로 해당UI검사만 재실행하여1PASS(총57.09초/두요청)였다. 최초실패는go_change_authority_mysql_live.log에 보존했으며 지연변동/SLA는 해결됐다고 보지 않는다. 서로다른886시험의 성공실행은확인했으나 실패0회·반복성합격으로 확대하지 않는다.


## 실제 웹 복합 요청의 추가 결함 보존

[최초 웹복합 실패](web_release_candidate/11_failed_initial.json): 조건30~40/primary+secondary는맞았으나 모델이scalar AVG에 education/age grouping을 반복 생성했다. 재해석경계에서 error=None을문자열처럼처리하여TypeError7364f94802d0가 발생했다(66.816초). 기존8자산은그대로였다. 이 건은앞선10/10과다른추가문항이며FAIL로보존한다.

독립선택이동일measure의scalar+ungrouped chart를요구한경우에만calculation의group_columns=[]를provider grammar에고정했다. 명시group chart/기존grouped calculation은제거하지 않는다. 의미검토전예산종료는구조화goal_unverified로차단하여TypeError를막고원본을보존한다. 필터를그룹으로바꾸는JSON거절/명시그룹허용및마지막재선택후검토예산종료의runtime회귀를추가했다. 기존실패요청은UI의종료버튼으로종료하고같은원문을새request로1회재제출했다. 이재제출은이전10문항의재제출0과별도로계산한다.


## 최신 코드의 실제 웹 완료 증거

[독립 검사PASS](web_release_candidate/12_check.json): 같은원문 재제출에서AVG34.8200(독립빈도 가중정답34.819970387975026), actualPNG의합208,699와5개구간[37,030;42,667;41,366;37,664;49,972]가모두일치했다. >=30 AND <=40 AND education IN(primary,secondary)의scope·출처bank_loan·scalar와chart의두출력의무도일치한다. 실제DOM에이미지1개와평균값모두나온다. 이전8자산의metadata/payload hash와제품source hash변경0.

DB의AVG를pushdown하여1행만새로로딩하고, 기존11행의완전한빈도와5bins이미지를재사용했다. 원본행을펼치거나표본화하지않았다. 38.140초/모델5회. 전체회귀는최신코드884PASS/4opt-inSKIP,591subtests,125.60초이다. [최종화면](web_release_candidate/latest_histogram.png),[UI원문](web_release_candidate/browser_messages_latest.json),[실행receipt](web_release_candidate/12.json),[고정제품hash](build_go_final.json)를저장했다. 같은대화에서이전107컬럼schema·원본·표·그림복원도확인했다.

이최신1복합문항PASS를이전build의10/10또는41/41에합쳐최신전체성적이라고표시하지않는다. 최초11번FAIL과수정후12번PASS를별도로보존한다.


최종 같은 제품build의 실제MySQL4검사는 단독 실행4PASS/48.23초였다. 전체884PASS/4SKIP에 이4개를 더해 서로다른888시험의 성공실행을 확인했다. 최초 동시실행UI 시간초과는 보존한다. compileall/pip check/diff check/loopback health 및 제품hash 불변도통과했다. 서버8504/MySQL/Ollama/gemma4:e4b와 최종 결과 대화를 열어두었다.
