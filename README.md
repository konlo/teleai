# Telly 데이터 분석 agent

LangChain `create_agent`와 LangGraph를 사용하는 로컬 Streamlit 분석 앱입니다. 대화와 도구 실행 상태를 저장하고, 로딩한 DataFrame으로 후속 통계·시각화를 수행합니다.

## 실행

검증 환경은 Python 3.11 / macOS arm64입니다.

```sh
python3.11 -m venv .telly_runtime/v1-venv
.telly_runtime/v1-venv/bin/python -m pip install -r requirements-agent.txt
python3 scripts/run_telly.py
```

이미 환경이 있으면 마지막 명령만 실행합니다. 브라우저에서 http://127.0.0.1:8502 를 엽니다. 새 환경의 `main.py`와 `pages/Telly.py`는 기본 분석 agent를 엽니다.

현재 모델 어댑터는 Ollama입니다. 기존 `.env`의 `OLLAMA_MODEL`, `OLLAMA_BASE_URL`을 사용합니다. Databricks 설정은 `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN`(또는 `DATABRICKS_ACCESS_TOKEN`), `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`입니다. 자격 증명을 저장소에 추가하지 마세요.

## 동작

- 기존 결과의 범위·컬럼·집계 상태를 확인하여 로컬 분석에 재사용합니다. 전체 범위를 보장하지 못하는 결과를 모집단으로 계산하지 않습니다.
- Databricks에서 데이터를 새로 가져오거나 다시 로딩할 때 정확한 SQL의 승인 카드를 표시합니다. 승인 전 실행하지 않고, 변경된 조회는 다시 승인받습니다.
- 승인/거절 후 중단된 agent loop를 이어갑니다. 제출 여부가 불명확한 조회는 자동 재실행하지 않습니다.
- 대화, DataFrame(Parquet), 차트 PNG와 승인 기록을 `.telly_runtime/v1`에 보존합니다. 긴 모델 문맥은 요약하고 화면의 원래 대화는 유지합니다.
- 통계 기반 차트 추천과 이미지 선택, `analysis_skills/`의 스킬 탐색·읽기를 제공합니다.

현재 실행기는 localhost 전용입니다. 기존 `.venv`와 구형 화면은 호환성을 위해 보존했으며 기존 세션을 자동 이전하지 않습니다. 구형 의존성은 `legacy-app-requirements.txt`에 있습니다.

배포 전에는 다음의 읽기 전용 검사를 실행합니다. 환경 변수 값과 토큰은 출력하지 않으며 Databricks SQL도 실행하지 않습니다.

```sh
.telly_runtime/v1-venv/bin/python scripts/deployment_preflight.py --profile local-desktop
```

현재 revision은 로컬 또는 외부 접근제어가 적용된 1인용 배포만 지원합니다. 배포 범위, 영속 저장소, smoke test와 rollback 조건은 [제한 배포 계약](docs/deployment_contract_2026-09-15.md)에 정리되어 있습니다. 필요한 환경 변수 이름은 `.env.example`을 참고하세요.

## 구현과 검증

주요 구현: `core/analysis_agent/`, 화면: `ui/analysis_page.py`.

[작업 현황](docs/agent_validation_tasks.md)과 [검증 보고서](docs/t07_t08_validation.md)를 참고하세요. 현재 Databricks 설정은 OpenSession HTTP 403으로 실제 조회 성공을 검증하지 못했습니다. 로컬 모델의 다중 턴·요약 후 계산은 통과했지만 응답 지연과 대용량 실행 메모리는 추가 개선 대상입니다.
