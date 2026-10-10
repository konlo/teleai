# 저장된 프롬프트 시나리오

| 이름 | 입력 수 | 내용 |
| --- | ---: | --- |
| `prompt_konlo_test_scenario_#1` | 31 | 기존 화면 재평가 plan의 1~31번 원문 |
| `prompt_ai_test_scenario_#1` | 10 | Codex가 만든 기존 10문항 원문 |

JSON 파일명이 시나리오 이름과 같다. 각 파일의 `turns`에 입력 순서·원문·출처·기대 동작을 보관했다. 오타, 구두점, 공백을 교정하지 않았다. 체크섬은 문구/순서의 우발적인 변경을 감지한다. 버전을 변경하려면 새 시나리오 번호를 사용한다.

31개는 `docs/evaluation/2026-10-06_actual_prompt_replay/plan.json`의 앞31개를 선택했다. 전체46개 중 나머지15개는 이 묶음에 포함되지 않는다. 원본 대화 `0728dee0-d555-4aa2-853b-1101679346da`의 화면 기록에서 추출한 문구이며, 저장된 HumanMessage는 실제 사용자와 과거 브라우저 자동 입력의 작성자를 독립적으로 구분하지 않는다.

## 목록·불러오기·내보내기

프로젝트 루트에서 실행한다. `#`가 포함된 이름은 따옴표로 감싼다. 이 명령들은 DB/LLM에 연결하지 않는다.

```bash
python3 scripts/run_prompt_scenario.py --list
python3 scripts/run_prompt_scenario.py --scenario 'prompt_konlo_test_scenario_#1' --show
python3 scripts/run_prompt_scenario.py --scenario 'prompt_ai_test_scenario_#1' --export /tmp/prompt_ai_scenario_1.json
```

Python 평가 코드에서도 읽을 수 있다. 한 묶음의 후속 질문을 매번 새 대화로 실행하면 맥락 검증이 되지 않는다.

```python
from test_set.prompt_scenarios import load_scenario, prompt_turns

scenario = load_scenario('prompt_konlo_test_scenario_#1')
for prompt in prompt_turns(scenario['name']):
    # 하나의 새 평가 대화에서 원문 그대로 순차 제출
    result = runtime.submit(prompt)
```

## 실제 agent로 재실행

설치된 프로젝트 가상환경의 Python을 사용한다. CLI는 웹과 같은 `GraphAnalysisRuntime`의 LLM 의도 해석과 선택한 DB adapter를 사용하며, 기존 웹 대화와 분리된 새 저장소에서 실행한다. DB와 모델 제공자는 별개다. MySQL 실행이 Databricks로 대체되는 fallback은 없다.

```bash
python3 scripts/run_prompt_scenario.py --scenario 'prompt_konlo_test_scenario_#1' --run --backend mysql --provider ollama
python3 scripts/run_prompt_scenario.py --scenario 'prompt_ai_test_scenario_#1' --run --backend mysql --provider ollama
```

회사 Databricks 설정을 사용할 때:

```bash
python3 scripts/run_prompt_scenario.py --scenario 'prompt_ai_test_scenario_#1' --run --backend databricks --provider ollama
```

모델도 Databricks serving을 쓰려면 `--provider databricks`를 지정한다. `.env`와 기존 접속 설정을 사용한다. 실행하면 실제 모델 호출과 읽기 전용 SQL 조회가 발생한다. 기존 제품 정책/SQL 검증/자원 한도를 그대로 적용하며 승인 대기를 임의로 승인하지 않는다.

각 입력을 한 번씩 제출하고, 실패 후에도 다음 원문을 같은 대화에 제출한다. agent의 내부 자율 복구는 유지하지만 평가 runner가 실패 요청을 자동 재제출하거나 상태를 지우지는 않는다. `--output`으로 새 보고서 경로를 지정할 수 있으며 기존 보고서는 덮어쓰지 않는다. 기본 보고서는 Git에서 제외되는 `.telly_runtime/prompt_scenario_reports/<실행ID>/report.json`에 저장된다. 실행 checkpoint와 도구 로그는 `.telly_runtime/prompt_scenario_runs/`에 남는다. 종료/중단까지 완료된 입력 기록을 보존한다.

## 평가와 완료 판정

이 runner는 재현/증거 수집기다. `answered` 개수는 정답률이 아니다. 출력·실행ID·오류ID·출처·조건·모델 호출 수·결과/차트ID·시간을 기록하고 `NOT_GRADED`로 남긴다. 실행이 끝나도 종료코드2로 독립 채점 필요를 표시한다. 설정 실패는 실행 자체의 실패다.

기대 설명/수치는 앞선 로컬 MySQL snapshot 기준이며 모델 입력으로 전달하지 않는다. DB/스키마/행이 바뀌면 직접 조회한 독립 정답을 새로 만든다. 특히 없는 테이블을 다른 테이블로 바꾼 답변, 필터를 잃은 차트, 이미지 없이 생성했다고 하는 답변을 성공으로 계산하지 않는다. 수치·PNG·범위·출처·원본 보존을 대조하고, 웹 평가에서는 브라우저 표시도 따로 검사한다. 이 저장 작업은 새 41개 실제 실행의 통과나 DeepEval/Spider 점수를 의미하지 않는다.

저장/재실행 도구 계약 검사:

```bash
python3 -m unittest tests.test_saved_prompt_scenarios -v
```

## 2026-10-09 실제 웹 평가와 출력 증거

같은 최종 로컬 MySQL/Ollama 코드의 실제 브라우저 첫 입력 기준으로 konlo31/31 + AI10/10 요구사항을 확인했다. 없는 테이블 안내2건은 분석 출력39건과 구분한다. 공식 DeepEval/Spider 점수나 회사 운영 GO를 뜻하지 않는다. 원문별 결과·쿼리 횟수·화면·남은 범위는 [최종 보고서](../../docs/evaluation/2026-10-09_mysql_go_prompt_final/report.md)에 있다.

실제 프롬프트를 입력한 브라우저의 메시지 DOM을 `browser_messages.json`으로 저장한 뒤, 실행별 checkpoint·plan과 함께 실제 이미지/표/본문 출력 누락을 읽기 전용으로 채점한다. 아래 명령은 새 프롬프트를 제출하거나 agent를 다시 실행하지 않는다.

```sh
python3 test_set/check_prompt_browser_evidence.py docs/evaluation/2026-10-09_mysql_go_konlo_output
python3 test_set/check_prompt_browser_evidence.py docs/evaluation/2026-10-09_mysql_go_ai10_output
```

DB 수치/PNG 내용/보존 oracle 검사와 UI 검사는 별도다. 내부 schema에 타입이 있다는 것만으로 실제 타입 답변을 통과 처리하지 않는다. Markdown 화면의 공백 축약만 허용하며 원래 제출한 prompt는 checkpoint에서 원문 그대로 비교한다.
