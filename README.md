# Telly 데이터 분석 agent

LangChain `create_agent`와 LangGraph를 사용하는 로컬 Streamlit 분석 앱입니다. 대화와 도구 실행 상태를 저장하고, 로딩한 DataFrame으로 후속 통계·시각화를 수행합니다.

## 실행

검증 환경은 Python 3.11 / macOS arm64입니다.

```sh
python3.11 -m venv .telly_runtime/v1-venv
.telly_runtime/v1-venv/bin/python -m pip install -r requirements.txt
python3 scripts/run_telly.py
```

이미 환경이 있으면 마지막 명령만 실행합니다. 브라우저에서 http://127.0.0.1:8502 를 엽니다. 새 환경의 `main.py`와 `pages/Telly.py`는 기본 분석 agent를 엽니다.

8501 포트를 사용하려면 같은 지원 환경으로 실행합니다.

```sh
python3 scripts/run_telly.py --port 8501
```

한 포트에는 한 프로세스만 실행하세요. 기존 `.venv/bin/streamlit run main.py`가 8501을 점유 중이면 종료한 뒤 위 명령을 사용합니다.

`ModuleNotFoundError: No module named 'sqlglot'`가 나오면 실행한 Python 환경에 앱 의존성이 설치되어 있는지 확인합니다. `sqlglot`은 이미 `requirements-agent.txt`에 포함되어 있습니다. 저장소 루트에서 아래처럼 설치와 실행에 같은 앱 환경을 사용하세요.

```sh
.telly_runtime/v1-venv/bin/python -m pip install -r requirements.txt
.telly_runtime/v1-venv/bin/python -c "import sys, sqlglot; print(sys.executable); print(sqlglot.__version__)"
python3 scripts/run_telly.py
```

앱 환경이 없으면 위의 `python3.11 -m venv .telly_runtime/v1-venv` 단계부터 실행합니다. `pip`나 `streamlit` 명령만 직접 실행하면 PATH에 따라 다른 환경을 사용할 수 있습니다. Agent 평가 스크립트도 `.telly_runtime/v1-venv/bin/python scripts/<평가 스크립트>.py`로 실행합니다.

### 평가 도구 설치

앱 의존성은 `requirements.txt → app-requirements.txt → requirements-agent.txt`로 설치합니다. `requirements-agent.in`은 직접 의존성 목록이며 실제 설치에는 고정 버전 파일을 사용합니다.

DeepEval과 Spider2-lite 공식 채점 도구의 의존성은 별도 환경에 설치합니다. DeepEval 4.2.3의 `click<8.4` 조건이 앱의 `click==8.5.0`과 충돌하므로 앱 환경에 합쳐 설치하지 마세요.

```sh
python3.11 -m venv .telly_runtime/eval-venv
.telly_runtime/eval-venv/bin/python -m pip install -r requirements-eval.txt
.telly_runtime/eval-venv/bin/python -m pip check
```

이미 `../ai_agent_eval/.venv`를 사용 중이면 해당 환경의 Python으로 같은 파일을 설치할 수 있습니다. Agent 실행·대화 평가 스크립트는 앱 환경에서, `scripts/evaluate_deepeval_real_report.py`, `scripts/calibrate_deepeval_judge.py` 및 Spider2 공식 채점기는 평가 환경에서 실행합니다. Spider2 저장소·벤치마크 데이터는 pip 패키지가 아니므로 별도로 준비해야 합니다. 이 파일은 현재 사용하는 SQLite 채점 경로를 지원하며, BigQuery 실제 실행에 필요한 인증이나 다른 Spider2 실행 환경까지 설정하지는 않습니다.

현재 모델 어댑터는 Ollama입니다. 기존 `.env`의 `OLLAMA_MODEL`, `OLLAMA_BASE_URL`을 사용합니다. Databricks 설정은 `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN`(또는 `DATABRICKS_ACCESS_TOKEN`), `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`입니다. 자격 증명을 저장소에 추가하지 마세요.

`TELLY_GOAL_MODEL`을 지정하면 Ollama의 의도 해석·원문 재검토 단계만 별도 모델을 사용합니다. 비워두면 `OLLAMA_MODEL`과 같습니다. 실행 계획 모델은 기존 설정을 사용하며, SQL 엔진 선택과는 독립적입니다. 선택한 모델이 로컬에 설치되어 있어야 하고 실제 사용자 여정으로 검증해야 합니다. 진단의 `goal_model`에서 적용된 모델을 확인할 수 있습니다.

2026-10-06 로컬 MySQL 재검증은 설치된 `qwen3:8b`를 두 단계 모두에 사용했습니다 (`OLLAMA_MODEL=qwen3:8b`, `TELLY_GOAL_MODEL=qwen3:8b`). 16GB 호스트에서 서로 다른 모델을 번갈아 호출하면 모델 재로딩 비용이 생깁니다. 기존 `gemma4:e4b`는 범위 조건을 누락한 실제 사례가 있어, 모델 이름만 설정했다고 분석 품질이 보장되지는 않습니다. 회사 Databricks 및 다른 모델의 적합성은 별도로 검증해야 합니다.

### 데이터 DB 선택

기본 데이터 backend는 Databricks입니다. 회사에서 사용할 때는 각 호스트의 `.env`에 다음 설정을 명시하고 앱 프로세스를 다시 시작합니다. `.env`는 Git에 포함되지 않으므로 기존 파일의 `TELLY_DATA_BACKEND=mysql`은 호스트에서 변경해야 합니다.

```dotenv
TELLY_DATA_BACKEND=databricks
DATABRICKS_HOST=<workspace hostname>
DATABRICKS_HTTP_PATH=<SQL warehouse HTTP path>
DATABRICKS_TOKEN=<token>
DATABRICKS_CATALOG=<catalog>
DATABRICKS_SCHEMA=<schema>
```

실행 터미널에 MySQL 환경 변수가 남아 있더라도 Databricks를 명시해서 시작하려면 다음 명령을 사용합니다.

```sh
TELLY_DATA_BACKEND=databricks python3 scripts/run_telly.py --port 8501
```

Databricks 모드는 `requirements.txt`만 설치하면 됩니다. MySQL 드라이버·서버·접속 파일은 필요하지 않으며 MySQL adapter를 import하지 않습니다. 연결 실패를 MySQL로 전환해 처리하지 않습니다. 설정한 catalog는 탐색 시작점이며 컬럼은 저장된 유효한 스키마 또는 실제 조회로 확인합니다. 저장된 테이블 정보가 없는 새 환경에서도 실제 테이블 목록 → `LIMIT 0` 스키마 확인으로 진행합니다. `LLM_PROVIDER`는 모델 선택이고 데이터 DB 선택과 별개입니다. OS 환경 변수가 `.env`보다 우선하므로 실행 터미널에 남은 `TELLY_DATA_BACKEND=mysql` 설정도 확인하세요.

로컬 MySQL 평가에만 아래 패키지를 추가하고 `TELLY_DATA_BACKEND=mysql`, `TELLY_MYSQL_DATABASE`, `TELLY_MYSQL_OPTION_FILE`을 설정합니다. 접속 파일은 `[client]`의 user/password/socket을 담는 로컬 0600 파일입니다.

```sh
.telly_runtime/v1-venv/bin/python -m pip install -r requirements-mysql-eval.txt
```

MySQL의 데이터·대화·차트 저장소는 기본 저장소 아래 `mysql_eval/<connection identity>/`로 분리됩니다. 기존 Databricks 저장소 경로는 유지하며 서로의 대화를 이어서 실행하지 않습니다. 두 모드 모두 실행 명령은 `python3 scripts/run_telly.py --port 8501`입니다.

로컬 MySQL의 서버 쿼리 실행 제한은 `TELLY_MYSQL_QUERY_TIMEOUT_SECONDS`로 설정합니다(기본 120초, 허용 1~600초). 큰 테이블의 최초 집계는 전체 스캔이 필요할 수 있습니다. 제한을 늘리는 것은 성능 개선이나 요청 전체의 응답시간 보장이 아닙니다. 서버가 실행 제한(3024)·중단(1317)을 명시하면 실패로 기록하고, 결과 수신 여부를 알 수 없는 연결 오류는 불확실 상태로 유지합니다. 저장 완료된 집계는 후속 시각화에서 재사용합니다.

연결 후 실제 목록·스키마·10행 적재·히스토그램·재사용·원본 보존을 확인하려면 아래 검증을 실행합니다. 실제 읽기 전용 SQL 4회를 실행하며 실패 실행 기록도 로컬 저장소에 남깁니다. 이 검증은 LLM 품질 점수나 회사 서버 전체 출시 판정을 대신하지 않습니다.

```sh
.telly_runtime/v1-venv/bin/python scripts/verify_data_backend.py --backend databricks --cold-start --output .telly_runtime/databricks-smoke.json
```

MySQL 평가는 `--backend mysql`로 실행합니다. 운영 `deployment_preflight.py`는 Databricks 설정을 검사하며 MySQL 설정으로는 운영 READY를 반환하지 않습니다.

## 동작

- 기존 결과의 범위·컬럼·집계 상태를 확인하여 로컬 분석에 재사용합니다. 전체 범위를 보장하지 못하는 결과를 모집단으로 계산하지 않습니다.
- Databricks의 읽기 전용 조회는 agent가 필요에 따라 승인 없이 자동 실행합니다. 로딩된 데이터를 우선 재사용하며 SQL 검증·자원 한도·원본 보존은 유지합니다.
- 명시적으로 수동 승인 모드가 필요한 환경에서만 `TELLY_REQUIRE_REMOTE_APPROVAL=true`를 설정하세요(기본값 `false`). 기존 승인 대기는 자동 모드에서 재개됩니다. 제출 여부가 불명확한 조회는 자동 재실행하지 않습니다.
- 대화, DataFrame(Parquet), 차트 PNG와 실행 권한·결과 기록을 `.telly_runtime/v1`에 보존합니다. 긴 모델 문맥은 요약하고 화면의 원래 대화는 유지합니다.
- 통계 기반 차트 추천과 이미지 선택, `analysis_skills/`의 스킬 탐색·읽기를 제공합니다.

현재 실행기는 localhost 전용입니다. `main.py`와 `/Telly`는 항상 영속 분석 agent로 연결됩니다. LangChain 1 미만의 기존 `.venv`에서 실행하면 구형 화면으로 전환하지 않고 지원 환경 실행 방법을 표시합니다. 이전 구형 화면의 메모리 내 대화는 자동 이전되지 않습니다.

배포 전에는 다음의 읽기 전용 검사를 실행합니다. 환경 변수 값과 토큰은 출력하지 않으며 Databricks SQL도 실행하지 않습니다.

```sh
.telly_runtime/v1-venv/bin/python scripts/deployment_preflight.py --profile local-desktop
```

현재 revision은 로컬 또는 외부 접근제어가 적용된 1인용 배포만 지원합니다. 배포 범위, 영속 저장소, smoke test와 rollback 조건은 [제한 배포 계약](docs/deployment_contract_2026-09-15.md)에 정리되어 있습니다. 필요한 환경 변수 이름은 `.env.example`을 참고하세요.

## 구현과 검증

주요 구현: `core/analysis_agent/`, 화면: `ui/analysis_page.py`.

[작업 현황](docs/agent_validation_tasks.md)과 [검증 보고서](docs/t07_t08_validation.md)를 참고하세요. 현재 로컬 환경에서는 승인된 Databricks 10,000행 표본 적재와 후속 EDA를 실측했습니다. 전체 테이블 분석, Spider SQL 일반화 및 대용량 실행 성능은 검증되지 않았으므로 [최신 GO 판정](docs/evaluation/2026-09-25_go_decision.md)을 함께 확인하세요.
