# MySQL 평가 / Databricks 운영 경로 분리 검증

검증일: 2026-10-03 KST. 범위: 현재 Mac의 제품 코드, 로컬 MySQL 및 실제 설정된 Databricks SQL warehouse. 회사 서버에는 접속하지 않았다.

## 결과

`TELLY_DATA_BACKEND=databricks` 선택 시 MySQL 드라이버·접속 파일을 사용하지 않으며 연결 실패에도 MySQL로 전환하지 않는다. 두 backend의 adapter, SQL dialect, metadata 계획과 영속 상태를 구분했다. 회사의 설치 경로는 `requirements.txt`이며 MySQL 평가 드라이버는 `requirements-mysql-eval.txt`에만 있다. 모델 선택 `LLM_PROVIDER`와 데이터 backend 선택은 독립이다.

실제 Databricks 및 MySQL에서 저장된 테이블 profile 없이 목록 → 0행 스키마 → 컬럼 후속 → 10행 원본 → 모집단 histogram → 반복 요청 여정을 통과했다. 두 여정 모두 읽기 전용 SQL 4회이며 반복 요청은 0회 추가 조회, 선택 원본 ID와 digest가 유지되었다. PNG 10,756바이트를 확인했다. 이 결과는 지정된 여정의 실제 adapter 검증이며 실모델 자유계획 성공률이나 범용 GO 판정이 아니다.

## 발견한 문제와 수정

1. UI가 Databricks adapter를 먼저 import하고 backend별 설정을 화면 코드에서 직접 조립했다. `core/analysis_agent/backends.py`가 선택한 adapter만 지연 import하도록 분리했다. 잘못된 이름·누락된 연결 설정은 명시적으로 실패하고 자동 fallback하지 않는다. Databricks token alias와 안전한 config repr도 보강했다.
2. 도구 설명과 일부 SQL 파서·관계 조회·최신행 histogram이 Databricks 문법을 가정했다. 활성 backend의 namespace와 dialect를 도구·계획·검증·완료 판정에 전달한다. Databricks 기존 문법은 유지하며 MySQL은 별도 `information_schema` 관계 조회, CHAR/SIGNED cast와 구간 생성 SQL을 사용한다.
3. 확정 SQL의 실행 사유에 “테이블 목록”이 포함되면 목록 발견 의도가 함께 남아 agent가 다른 SQL을 다시 계획했다. 실제 최초 실패 `dd989a5951664998a51f4fea41e3b396`, 오류 ID `0218a6784acd`에서 schema 조건이 빠진 새 SQL을 실행하려 했고 완료가 막혔다. controller의 정확한 SQL 실행 경로에서는 해당 재계획 의무를 제거했다. 회귀는 전달 SQL 1회·동일 SQL 완료를 확인한다.
4. 새 회사 설치에는 ignored 저장 profile이 없어 최초 catalog 탐색과 후속 컬럼 확인이 연결되지 않았다. 설정 catalog를 탐색 시작점으로 사용하고 실제로 반환된 bounded/fresh `information_schema.tables`의 이름을 후속 문맥에 반영한다. 이름 관측은 컬럼 근거가 아니므로 `SELECT * LIMIT 0` 후 실제 스키마를 확인한다. 계산된 가짜 이름·다른 catalog·stale·다른 source·unbounded 결과는 이름 발견 근거에서 제외한다.
5. MySQL은 `base/mysql_eval/<connection identity>/`에 저장하고 기존 Databricks base는 유지한다. runtime SQLite에 backend를 기록하여 같은 저장 범위의 다른 backend 재개를 거부한다. 진단 요약에 `data_backend`를 남긴다. 운영 preflight는 MySQL 평가 설정으로 Databricks READY를 반환하지 않는다.

## 검증 근거

| 검사 | 결과 |
|---|---|
| backend 분리, metadata·schema refresh, 관계, 모델 설정, load plan, 최신행 histogram, 진단, 운영 preflight 회귀 | 72 PASS / 65 subtests PASS |
| `TELLY_TEST_MYSQL=1` 실제 MySQL 통합 | 4 PASS |
| Databricks driver 경로에서 MySQL import 차단 subprocess | PASS |
| MySQL driver 경로에서 Databricks import 차단 subprocess | PASS; optional MySQL package 미설치 CI에서는 이 한 검사는 skip |
| 실제 Databricks cold start 여정 | PASS / 17.111초 / SQL 4회 / 18컬럼 / 원본 10행 |
| 실제 MySQL cold start 여정 | PASS / 9.858초 / SQL 4회 / 18컬럼 / 원본 10행 |
| 운영 preflight, 현재 로컬 `.env` Databricks | local-desktop READY; 기본 프로젝트 저장소 경고 유지 |
| pip check / compileall / git diff --check | PASS |

실측 시간은 한 여정의 값이며 p95나 재시도 안정성을 나타내지 않는다. 실제 최신행 histogram의 MySQL SQL 실행·PNG도 live 통합에서 검증했다. 최초 진단 테스트 한 개는 catalog 문맥이 없는 이전 자연어 fixture 때문에 receipt가 생성되지 않았다. 해당 진단 검사는 확정 SQL의 receipt 검증으로 수정했고, 자연어 목록·컬럼 연결은 별도의 fresh-install 및 AppTest 검사로 검증한다. 이전의 `test_remote_result_completion.py`/`test_automatic_databricks.py` 전체는 이번 최종 묶음에 포함하지 않았다. 전체 제품 회귀·범용 LLM/Spider 평가는 완료로 선언하지 않는다.

재현 명령:

```sh
.telly_runtime/v1-venv/bin/python scripts/verify_data_backend.py --backend databricks --cold-start --output .telly_runtime/databricks-smoke.json
.telly_runtime/v1-venv/bin/python scripts/verify_data_backend.py --backend mysql --cold-start --output .telly_runtime/mysql-smoke.json
TELLY_TEST_MYSQL=1 .telly_runtime/v1-venv/bin/python -m pytest -q tests/test_mysql_backend_live.py
```

sanitized 실측 결과: [Databricks](databricks_final.json), [MySQL](mysql_final.json). 상세 checkpoint와 진단 로그는 ignored `.telly_runtime/backend-verification/`에 보존한다. 결과 JSON에 SQL 본문·원본 행·credential은 저장하지 않는다.

## 회사 적용

`.env`에 `TELLY_DATA_BACKEND=databricks`와 `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` 또는 `DATABRICKS_ACCESS_TOKEN`, `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`를 설정한다. OS 환경 변수에 남은 `TELLY_DATA_BACKEND=mysql`은 `.env`보다 우선하므로 제거하거나 수정한다. `requirements.txt`를 앱 실행 Python에 설치하고 `python3 scripts/run_telly.py --port 8501`로 새 프로세스를 실행한다. MySQL 설정·드라이버·서버는 필요하지 않다.

회사 토큰 권한·warehouse 상태·네트워크·실제 회사 테이블의 자료형은 여기서 확인할 수 없다. 회사 설치 후 위 Databricks smoke로 그 환경의 같은 여정을 확인할 수 있다. 복합 SQL·고급 EDA·임의 자연어 전체 성공률에 대한 기존 범용 NO-GO 판정은 이번 backend 분리 검증으로 바뀌지 않는다.
