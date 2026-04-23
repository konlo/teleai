# 📝 Telly 시스템 작업 이력 (Work History)

이 문서는 Telly 에이전트 시스템에서 발생한 이슈들과 해결된 작업 내역을 관리하기 위한 이력 문서입니다. 문제 발생 시 추후 참고할 수 있도록 주요 변경 사항을 기록합니다.

---

## [2026-04-23] 의도 분석 라우팅 및 연쇄 실행(Chaining) 버그 수정

### 🚨 1. 데이터 시각화 시 표만 나오고 차트가 그려지지 않는 문제
* **현상**: 사용자가 "파이 차트로 그려줘"와 같은 시각화 요청을 했을 때, 데이터가 로드되지 않은 상태(Preview State)라면 SQL 쿼리를 통해 표만 반환하고 정작 파이 차트는 그려지지 않는 문제가 발생했습니다.
* **원인**: 
  * `core/llm_router.py`는 올바르게 `['SQL Builder', 'EDA Analyst']` 연쇄 실행을 제안했습니다.
  * 그러나 `core/sql_utils.py`의 `execute_sql_preview` 함수가 성공적으로 데이터를 불러온 후, 화면 전환을 위해 UI 상태인 `last_agent_mode`를 `"EDA Analyst"`로 덮어썼습니다.
  * `core/chat_flow.py`에서는 연쇄 실행(EDA 단계로 자동 진입)을 위한 조건으로 `last_agent_mode == "SQL Builder"`를 검사하고 있었는데, 이미 값이 `"EDA Analyst"`로 바뀌어버려 해당 조건문이 `False`가 되었습니다.
  * 이로 인해 두 번째 단계인 시각화 단계(EDA Analyst)가 조용히 무시(Skip)되고 흐름이 종료된 것입니다.
* **해결 방법**:
  * `core/chat_flow.py`에서 연쇄 실행을 트리거하는 조건 중 불필요하고 충돌을 일으키는 `last_agent_mode == "SQL Builder"` 검사 로직을 제거했습니다.
  * 무한 루프를 방지하기 위해 연쇄 실행이 트리거될 때 `st.session_state["llm_router_suggested_chaining"] = False`로 명시적으로 초기화하도록 수정했습니다.

### 🚨 2. Ollama 로컬 모델 사용 시 Pydantic 파싱 오류 및 키워드 기반 Fallback 추가
* **현상**: `gemma4:e4b` 등 로컬 모델이 LangChain의 `with_structured_output`을 지원하지 않아 우회(Fallback) 프롬프트를 동작시키는 과정에서 오류가 발생했습니다. 또한 로컬 소형 모델 특성상 JSON을 완벽하게 생성하지 못해 파싱 에러가 발생할 경우 무조건 `['SQL Builder']` 단일 에이전트만 반환하여 시각화 단계로 넘어가지 못했습니다.
* **원인**: 
  1. Langchain Core 0.3 버전의 Pydantic V2 호환성 문제로 `model_json_schema` 예외 처리가 중단되는 문제가 있었습니다.
  2. 파싱 실패(Exception) 발생 시 안전을 위해 `['SQL Builder']`만 제안하도록 설계되어 있어, 사용자가 명시적으로 차트를 그려달라고 요청했어도 EDA Analyst가 호출되지 않았습니다.
* **해결 방법**:
  * `core/llm_router.py` 파일의 상단 임포트 구문을 수정하여, `pydantic` 패키지에서 직접 `BaseModel`, `Field`를 가져오도록 업데이트했습니다.
  * 예외(Exception)를 잡아내는 Fallback 블록에서 사용자의 입력(`user_query`)에 시각화 관련 키워드("차트", "그려", "그래프", "시각화" 등)가 포함되어 있다면, 에러가 발생하더라도 `['SQL Builder', 'EDA Analyst']` 체이닝을 올바르게 제안하도록 키워드 기반 라우팅 보완 로직을 추가했습니다.

### 🧪 3. 테스트 시나리오(test_scenario.py) 추가
* **목적**: 라우터 모델의 변경이나 시스템 로직이 수정되었을 때 부작용이 없는지 빠르게 확인할 수 있는 테스트 스크립트가 필요했습니다.
* **내용**: 
  * 프로젝트 루트에 `test_scenario.py` 스크립트를 작성하여 터미널에서 `python test_scenario.py` 명령으로 라우터의 의도 파악과 에이전트 체이닝 로직을 자동 검증할 수 있도록 구성했습니다.
  * 4가지 주요 상태(데이터 미로드/로드 상태, 단순/시각화 요청)에 따른 예상되는 에이전트 제안 리스트를 검증합니다.

### 🚨 4. (핵심 근본 원인) AUTO_SQL_KEYWORDS가 연쇄 실행(Chaining) 시 EDA를 방해하던 문제
* **현상**: 위 1~2번 수정 이후에도 "나이가 42세인 사람들의 직업에 대해서 파이 차트로 그려줘"라고 입력하면 여전히 SQL 데이터 테이블만 표시되고 파이 차트는 그려지지 않았습니다.
* **원인 (2중 충돌)**:
  1. **`AUTO_SQL_KEYWORDS` 재트리거**: SQL 실행 성공 후 연쇄 실행을 위해 `st.rerun()`이 호출되면 `handle_user_query` 함수가 처음부터 다시 실행됩니다. 이때 `auto_eda_pending`에서 복구된 사용자의 원래 질문("...파이 차트로 그려줘")에 `AUTO_SQL_KEYWORDS` 목록의 키워드("데이터", "분포" 등)가 포함되어 있어 **`command_prefix = "sql"`이 강제로 다시 세팅**되었습니다.
  2. **`agent_request` 덮어쓰기**: `force_eda_due_to_chaining = True`로 EDA 모드에 올바르게 진입했지만, `agent_request` 결정 로직(라인 535-538)에서 `command_prefix == "sql"` 조건에 걸려 EDA Agent에게 전달되는 요청이 사용자의 원래 시각화 질문이 아닌 `"새로운 SQL 쿼리를 작성해줘."`로 교체되어 버렸습니다.
* **해결 방법** (`core/chat_flow.py`):
  * `AUTO_SQL_KEYWORDS` 키워드 검사 조건에 `auto_pending is None` 가드를 추가하여, 연쇄 실행(rerun) 시에는 SQL 강제 전환이 발생하지 않도록 차단했습니다.
  * `agent_request` 결정 로직에서 `force_eda_due_to_chaining`이 `True`일 경우, `command_prefix`와 무관하게 항상 `original_user_q`(사용자의 원래 시각화 요청)를 EDA Agent에 전달하도록 수정했습니다.
