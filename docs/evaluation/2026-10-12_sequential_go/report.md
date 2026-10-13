# 순차 1–4 구현·실제 평가 결과 — 2026-10-12/13

**판정: NO_GO. 구조 구현은 진행했지만 실사용 성공률과 복구 능력이 출시 기준을 충족하지 못했습니다.**

## 요청한 작업별 결과

| 순서 | 수행 내용 | 결과 |
|---|---|---|
| 1 | 중복된 의미 해석을 LLM outline → 선택한 capability별 검토 계획으로 통합 | 구현. 모델 선택·행동·관찰·한 번의 의미 재계획을 같은 요청 예산으로 연결. 실모델 지연/검토 모순은 남음. |
| 2 | 복합 요청의 출력별 출처·모집단·의존성·완료 근거 분리 | 영속 task DAG/개별 scope/receipt 구현. 계약 회귀 통과. 실제 후속·복합 요청 수용 기준은 미충족. |
| 3 | 저장된 원문31 + AI10 + 별도8 시나리오를 실제 모델로 재실행 | 모두 완료. 표 아래의 독립 판정 참조. 성공 사례만 선별하지 않음. |
| 4 | Databricks·대규모 전송·공식 평가·플랫폼 검증 | 실제 100k 전송/보존 통과. DeepEval 공식 지표 실행. Spider3 최종SQL 없음. 회사 Windows/Azure 실제 호스트 NOT_RUN. |

## 실제 모델과 독립 정답 평가

| 평가 | 독립 통과 | 실패/미완료 | 실행 시간 | 해석 범위 |
|---|---:|---:|---:|---|
| 원문31 최신 v3 | 4/31 (12.9%) | 27 | 4340.351초 | Databricks에 없는 원래 MySQL 명칭도 유지한 정확한 원문 여정 |
| AI10 최신 | 1/10 (10%) | 9 | 1724.723초 | 실제 Databricks, 정확한 저장 프롬프트 |
| 별도8 여정 | 3/8 (37.5%) | 5 | 1409.759초 | 5행 합성 자료, 실제 LLM, 원격 executor 없음; 개별 프롬프트는 4/10 통과 |
| DeepEval 대상4 독립 판정 | 1 | 2 실패 + 1 미완료 | 최초 534.256초 + metadata 별도 재확인 57.029초 | 잘못된 metadata 평가 경로 수정 후 별도 재실행을 명시 |
| Spider2 SQLite 선정3 | 검증된 최종 SQL 0/3 | 3 | 405.185초 | public local002/003/004 부분 평가; 전체 Spider2 정확도 아님 |

원문31의 runtime 상태는 answered8/incomplete14/blocked5/exception4입니다. answered를 통과로 세지 않았습니다. 실제 출처·조건·수치·PNG·원본 보존을 독립 기준과 대조했습니다. 원래 MySQL-only 명칭을 포함하므로 이 성공률을 일반 모델 지능 점수로 확대하지 않습니다. 존재하지 않는 출처를 기존 bank_loan으로 바꾸는 것은 실패입니다.

## DeepEval / Spider2 점수의 한계

공식 DeepEval4.2.3 GEval 보조 점수: **1.0, 0.2, 0.2, 0.2 (평균 0.40)**. 실제 답변을 native Ollama gemma4:e4b 심사 모델로 판정했으며 threshold0.8을 사용했습니다. 심사 모델만 think=false/context8192/output1024로 제한했고 agent 설정은 변경하지 않았습니다. logprob 가중은 사용하지 않았으며 로컬 심사 점수는 독립 수치·scope·artifact 판정을 대체하지 않습니다.

기존 OpenAI-compatible 로컬 심사 연결은 4건 모두 ReadTimeout/APITimeoutError로 점수를 내지 못했습니다. 오류 보고서는 보존했고 공식 지표/평가 기준을 유지하면서 심사기 transport를 수정했습니다. native judge 계약 테스트 3개 통과(모델 호출 없음), 실제 심사 4개 모두 점수를 반환했습니다.

ToolCorrectness는 사전 정의된 좁은 legacy 도구 경로에서 4건 모두0입니다. 첫 metadata 답변은 inspect_live_schema로 올바르게 완료됐지만 legacy 계획은 inspect_table_context만 인정합니다. 따라서 도구명 점수0을 실제 metadata 실패로 해석하지 않습니다. 올바른 새로운 경로를 관찰한 뒤 기대 계획을 바꿔 점수를 올리지는 않았습니다.

Spider 공식 evaluator는 올바른 별도 venv에서 실행했으나 `No overlapping prediction IDs with gold set. Nothing to evaluate.`를 반환했습니다. agent가 검증된 최종 SQL을 내지 못해 제출 파일이 없기 때문입니다. **3개 시도 중0개 최종완료**는 확인된 결과이며 공식 전체 benchmark 점수를 꾸며내지 않습니다. 실제 gold SQL은 agent 입력에 제공하지 않았습니다.

## 대규모 데이터·원본 보존

- 실제 Databricks 연결 SELECT1 통과; 독립 count/column/matrix 확인 완료.
- 실제 `workspace.default.stormtrooper` LIMIT100001 전송: 제품 상한100000 그대로, 저장100000행, coverage=truncated, 완료receipt1개. 선택 원본·Parquet hash·재열기 후 hash 보존. **31.114초 / peak RSS216580096bytes(약206.5MiB) / frame cache0bytes**.
- 합성 백만행 streaming/quota/interruption/restart 검증은 별도 통과. 실제9.5m행 raw loading으로 확대하지 않음.
- 계약 전체 회귀 **952통과 / 4제외 / 736subtests / 121.43초**. 초기939통과/10실패 결과와 중단한 실행은 보존. 겹치는 집중 테스트 수를 합산하지 않음.
- 회사 Windows/Azure 실호스트는 설정·접속 근거가 없어 **NOT_RUN**. macOS의 Windows 에뮬레이션 테스트는 회사 환경 증명이 아님. 별도 서버 배포는 사용자가 제외한 범위.

## 남은 출시 차단 문제: 다음 공통 수정 묶음

1. **새로운 출처와 미완료 의도를 영속 관리.** 미관측 `_tianchi...` 요청을 bank로 대체, 실패한 stormtrooper 전환 뒤 "그 테이블"을 bank로 해석, metadata 뒤 필터 유실이 실제 발생했습니다. 현재4개 recent request 창은 의도 수명을 대신할 수 없습니다. task/subject별 pending/resolved/superseded/cancelled 상태와 on-demand 참조가 필요합니다.
2. **의도와 실행의 외부 일치 검증.** 승인된 목록 catalog/schema를 tool이 default로 줄이고 balance 평균을 schema 결과로 완료한 사례가 있습니다. self-generated goal/SQL의 내부 일치만으로는 사용자 의도 일치를 보장하지 못합니다. outline/review 조건·변경 상태가 서로 모순되지 않도록 묶고 inherited predicate와 CURRENT 변경 근거를 분리해야 합니다.
3. **자료 역할에 맞는 도구와 복구.** LIMIT0 schema asset에 profile을 실행한 사례, scatter 준비 후 실행·렌더 전에 budget이 소진된 사례가 있습니다. schema/raw/aggregate 역할을 명시하고 source profile/value-list/composite scatter 경로를 제공해야 합니다. 완료된 partial output은 보존하며 남은 작업만 재계획하는 계약도 필요합니다.
4. **실행 시간 예약과 조회 상태 격리.** 모델 단계가 SQL 실행시간을 다 소비하고 한 과거 unknown 조회가 무관한 새 요청을 전역 차단했습니다. 늦은 모델 응답은 도구 실행 전에 폐기하되 DB 시간 예약·단계별 deadline이 필요합니다. unknown을 지우거나 자동 재조회하지 않고 해당 operation의 중복 제출만 막아야 합니다. 종료 확인 없이 원격 조회를 성공/실패로 단정하지 않습니다.

상세 case/trace: [remaining_batch_findings.md](remaining_batch_findings.md). 위 문제는 이번 평가로 확인한 **남은 수정 사항**이며 이미 해결됐다고 표시하지 않습니다. 새로운 조건별 정규식 분기나 단순 호출 한도 증가는 제안하지 않습니다.

## 재현과 평가 무결성

- Agent executable SHA256: `6bebe05e1e463bc0b3e90da02ee508089ce5712f81c0ea02aa9424773f79720d`.
- 최신31/10/heldout 동안 core/ui/utils/pages/main 고정. 예상값·oracle SQL은 agent 입력에 들어가지 않음. 실패 프롬프트 자동재제출 없음. 실제 모델 작업은 순차 실행했고 전체 회귀는 모델 평가 뒤 실행.
- Ollama gemma4:e4b / Databricks. 모델HTTP timeout60초, 요청 실행180초, 로컬 모델 호출 한도10 그대로. Azure50 정책은 유지. .env/자격증명 변경 없음.
- 최초8개 예전 replay는 heavy unit suite와 겹쳤으므로 독립 latency benchmark가 아닙니다. 최신 v3는 분리 실행했습니다.
- `.resolve()`가 eval venv Python symlink를 system Python으로 바꿔 발생했던 패키지 import 오류를 수정하고 올바른 venv로 재실행했습니다. 설치 누락/제품 오류와 구분합니다.
- 실행: `scripts/run_prompt_scenario.py --scenario 'prompt_konlo_test_scenario_#1' --run --backend databricks --provider ollama --output <새 파일>`; AI 시나리오도 동일. 각 실행은 격리 대화에 저장됩니다.
- 독립 채점: `scripts/grade_prompt_replay.py`의 실제 inventory/matrix/native schema/asset oracle. 결과 보고서의 answered와 독립 PASS를 구분.
- DeepEval native: 별도 ai_agent_eval/.venv/bin/python으로 `scripts/evaluate_deepeval_real_report.py --report <qualified.json> --output <새 파일> --judge-provider ollama --native-ollama-judge`.
- 원문 결과: konlo31_live_v3.json / konlo31_grade_v3.json. AI/heldout/DeepEval/Spider/transport/fullpytest: remaining_suites_v3/ 아래. 이전 실패 결과 유지.
- 읽기 전용 진행 화면: http://127.0.0.1:8768/ . 실사용 agent는 http://127.0.0.1:8504/Telly . commit/push/운영 배포는 수행하지 않았습니다.

## 최신 브라우저 검증

최신 frozen build로8504를 재시작하고 기존 stormtrooper 대화의 전체 longitude histogram 재사용 요청을 **1회** 입력했습니다. run_id `c50c93aead094d739414a144ea314e05`: **137.01초 / 모델3회 / 신규SQL0 / complete**. 9,524,806개 빈도20bin이 이전 독립 실DB oracle와 일치하고 실제 PNG/화면을 확인했습니다. 기존8asset의 metadata/payload hash, 3Parquet hash, 빈 selection table을 보존했습니다. 빈 selection을 활성 원본 선택으로 확대하지 않습니다.

해당 원격 snapshot을 새로 스캔한 검증이 아니라 완료된 receipt/캐시의 정확한 재사용입니다. 이 단일 성공을 복합 요청이나 신규 SQL 전체의 성공으로 표시하지 않습니다. source build/완료 로그/그림spec/보존검사: [browser_latest.json](browser_latest.json). 실제 화면: [browser_latest.jpg](browser_latest.jpg), 이미지: [browser_latest_histogram.png](browser_latest_histogram.png). 최종 진행 화면: [progress_final.jpg](progress_final.jpg).
