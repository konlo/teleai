# TeleAI 분석 agent 출시 수용 기록 — 2026-09-13

상태: **검증된 핵심 범위의 제한 출시 후보 / 전체 Level 1·2 기능은 NO-GO**.

## 확인한 동작

- 사용자 조건을 외부 TableContext와 보유 데이터로 해석하고 checkpoint에 저장한다.
- 계산, 차트, Databricks SQL의 범위가 요청과 다르면 완료 근거로 채택하지 않는다.
- 완전한 로컬 DataFrame과 PNG가 있으면 원격 재조회 없이 재사용한다.
- 새 데이터나 넓은 범위가 필요할 때만 구체 SQL 승인 카드를 만들며 승인 전 실행은 0회다.
- 잘못된 SQL·도구·범위는 bounded recovery로 수정하고, 검증된 결과가 없으면 성공 문구를 내보내지 않는다.
- 긴 대화 원문, 승인 ledger, DataFrame, PNG는 재시작 뒤에도 보존한다.
- 테이블 스키마는 동적 외부 컨텍스트로 읽으며, stale 컬럼은 새 SQL의 근거로 쓰지 않는다. 승인된 `SELECT *` 결과의 컬럼·dtype을 우선하고 최신 원본 요청은 기존 캐시를 우회한다.

## 검증 증거

- 실제 production graph + 로컬 모델 + 독립 oracle: 누적 41/41 PASS. 컬럼 metadata와 복합 집계까지 구조화된 증거로 채점했다.
- agentic 실패 복구 시나리오: 17/17 PASS. 임의 placeholder는 실행 경로에서 제거했다.
- 전체 회귀: migration 126 + tests 79 = 205 PASS. compileall과 `git diff --check` PASS.
- 기존 대화 56개 메시지 복제: 기존 histogram PNG 재사용, 원격 조회 0회.
- 최신 서버 실제 브라우저: 같은 `age` histogram 요청을 기존 PNG로 0.292초에 표시. 모델 호출 0회, 원격 조회 0회, 활성 승인 카드 0개.
- 실제 승인형 Databricks 여정: 승인 전 실행 0회, 승인 후 `workspace.default.bank_loan` 10,000행·18열을 1회 조회해 저장했다. 이어서 현재 10,000행 표본의 `age` histogram을 0.215초, 모델 호출 0회, 추가 원격 조회 0회로 생성해 화면에 표시했다.
- 실제 화면 현재 표본 상관분석: `age`와 `balance`의 피어슨 상관계수 `0.05918371025192562`를 0.224초에 계산했다. 모델 호출 0회, 원격 조회 0회이며 결과 범위를 `현재 보유한 일부 데이터`로 표시했다.
- 성능·메모리: 실제 10,000행 복제본의 최신 30회는 p50 0.074초·p95 0.146초·최대 0.185초였다. 동시 4 workers·20회는 20/20 성공, p95 0.562초였다. 최신 화면 turn은 0.114초, peak RSS 약 247 MiB, frame cache 약 6.1 MiB였다. 이후 turn은 RSS를 자동 기록한다.
- 최신 용량 gate: 단일 30회 p95 0.162초, 4 worker·20회 p95 0.549초·20/20 성공, peak RSS 약 361 MiB로 READY다. 초기 1 GiB 한도에서 768 MiB 경고·896 MiB 위험 기준을 적용했으며 실제 배포 호스트 재측정은 남았다.
- 대용량 저장: 750,000행 등록, LRU, 프로세스 재시작, commit 전 SIGKILL 복구, SQLite 무결성, PNG 보존 PASS.
- 복합 조건 `L2_005`: 기존에는 정확히 필터된 7행을 만든 뒤 계산 대상을 선택하지 못해 130.689초/185.520초에 미완료. 수정 후 모델 재호출 없이 0.194초에 정답 7 PASS.

원본: [종합 평가](chatbot_agentic_evaluation_2026-09-13.md), [Level 1 실제 평가](actual_agent_evaluation_20_latest_2026-09-13.json), [Level 2 실제 평가](actual_agent_evaluation_level2_4_repaired_2026-09-13.json), [최신 복구 평가](agentic_recovery_evaluation.json), [동적 테이블 검증](dynamic_table_validation_2026-09-13.json), [대용량 검증](large_data_validation_2026-09-13.json).

## 허용할 제한 출시 범위

보유 데이터 재사용, 메타데이터 조회, 명시적 AND·IN·한 개 OR 그룹 필터, 기본 count/avg/sum/min/max/ratio, histogram/bar/line/scatter/단일 수치 boxplot을 대상으로 한다. 운영 TableContext에 질문 언어의 alias와 대표값이 등록된 테이블만 포함한다.

## 전체 기능 출시 차단 항목

- 실제 agent 독립 채점은 41/200문항이다. 나머지 159문항과 조인·가설 검정·고급 시각화는 미검증이다.
- 결정적 로컬 및 단일 프로세스 동시 경로에 자동 용량 gate를 적용했다. 실제 모델 상관계수는 5/5 정확했지만 p95 80.704초였고, 로컬 전이 후 p95 0.046초가 됐다. 다른 모델 질문과 실제 배포 환경·Ollama 동시 부하는 남았다.
- schema metadata 확대를 포함한 commit `e31150f`의 원격 release gate가 성공했고 `agentic-analysis-rc7-2026-09-16` tag를 같은 코드 commit에 고정했다. PR 리뷰·병합, 배포와 배포 환경 smoke test는 남았다.

## 실행과 롤백

실행 환경은 `.telly_runtime/v1-venv`이며 `main.py`를 `127.0.0.1:8502`에서 한 프로세스로 실행한다. `.telly_runtime/v1`의 graph, transcript, 승인 ledger, Parquet, PNG는 코드 교체와 분리해 보존한다. 롤백 시 코드와 pinned 환경을 함께 이전 검증 revision으로 되돌리며, 제출 여부가 불명확한 원격 요청은 자동 재실행하지 않는다.
