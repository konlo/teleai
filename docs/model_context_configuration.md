# 모델별 문맥 설정

문맥은 시스템 지침, 대화, 도구 정의/결과, 사용자 입력 및 생성 응답에 사용하는 토큰 공간이다. 데이터베이스 선택(`TELLY_DATA_BACKEND`)과 독립적이다. 코드 변경 후 서비스 프로세스를 완전히 재시작한다.

| 모델 identity | 프로필 최대 문맥 | 기본 응답 예약 |
| --- | ---: | ---: |
| Ollama `gemma4:e4b` | 131,072 | 4,096 |
| Azure 배포의 `gpt-5.6` / `gpt-5.6-sol` | 1,050,000 | 4,096 |

근거: [Google Gemma4 모델 카드](https://ai.google.dev/gemma/docs/core/model_card_4), [OpenAI GPT-5.6 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.6-sol). 로컬 설치 Gemma GGUF의 `gemma4.context_length`도 131072로 확인했다. Azure 값은 **명시한 모델의 OpenAI 문서상 최대**이며 회사 Azure 배포의 실제 허용량이나 접근 가능성을 검증한 기록이 아니다. 실제 배포가 더 작은 한도를 제공하면 설정을 낮춘다.

## 로컬 Ollama

```dotenv
OLLAMA_MODEL=gemma4:e4b
OLLAMA_CONTEXT_TOKENS=131072
OLLAMA_MAX_OUTPUT_TOKENS=4096
```

문맥 변수를 비우면 알려진 모델의 프로필을 사용한다. 모르는 모델은 이전 16384 문맥을 유지하므로 해당 모델의 지원 한도를 직접 지정한다. 모델별 프로필은 `core/analysis_agent/model_context_limits.json`에서 관리한다. 모델을 변경하면서 이전 모델의 문맥 값을 그대로 적용하지 않는다. 알려진 모델의 최대값 초과, 음수/0/잘못된 숫자, 문맥보다 큰 응답 예약은 모델 호출 전에 거절한다.

목표 해석 모델을 다르게 쓰면 `TELLY_GOAL_MODEL`과 선택적인 `TELLY_GOAL_CONTEXT_TOKENS`를 사용한다. 목표 역할의 모델이 바뀌면 새 모델 프로필 또는 기존 기본값으로 문맥도 다시 결정한다. JSON 역할은 같은 모델 문맥을 유지하되 해당 역할에 필요한 출력 예약으로 예산을 계산한다.

## Azure GPT-5.6

```dotenv
LLM_PROVIDER=azure
AZURE_OPENAI_MODEL=gpt-5.6
TELLY_AZURE_CONTEXT_TOKENS=1050000
TELLY_AZURE_MAX_OUTPUT_TOKENS=4096
TELLY_AZURE_TOKEN_PARAMETER=max_completion_tokens
```

기존 endpoint, API key, deployment, API version은 유지한다. `AZURE_OPENAI_DEPLOYMENT`는 서비스 배포 이름이고 `AZURE_OPENAI_MODEL`은 그 배포의 실제 모델 identity다. 위 설정만으로 Azure에 모델이 배포되거나 배포가 교체되지는 않는다. 기존 구형 배포가 `max_tokens`를 요구하면 기존 token parameter를 유지한다.

GPT-5.6 identity를 지정하면 문맥 변수 없이도 프로필 최대값이 적용된다. 문맥을 명시하면 지원 최대 이하로 덮어쓸 수 있다. 알려지지 않은 모델은 명시한 한도를 사용하며, identity/문맥이 모두 없는 기존 Azure 설정에는 예전의 제한된 bytes 입장 정책을 유지한다. 배포 이름만 보고 GPT-5.6이라고 가정하지 않는다. 출력 예약 기본값은 최대 출력128000과 별개이며, 한도를 크게 늘려도 실제 출력이 항상 그만큼 생성되거나 분석 정확도가 높아지는 것은 아니다.

## 입력 계측과 진단

- 프로필이 있는 Azure GPT는 설치된 `tiktoken`의 `o200k_base`로 본문·도구·호출 인자를 토큰으로 추정한다. 서버의 실제 채팅 template 토큰 수와 동일하다는 보장은 없으므로 별도 headroom을 둔다.
- Gemma의 정확한 tokenizer를 연결하지 않은 경우 UTF-8 크기를 **보수적인 토큰 상한**으로 사용한다. GPT tokenizer를 Gemma의 실제 tokenizer인 것처럼 사용하지 않는다. tokenizer를 사용할 수 없는 Azure도 이 안전 경로로 돌아간다.
- 진단의 `input_tokens_estimate`와 `input_tokens_upper_bound`를 구분한다. `payload_bytes`는 입력 크기 정보로 별도 유지한다. `context_window - output_reserved`가 입력 예산이며 JSON 역할마다 예약량은 달라질 수 있다.
- 이전 로그에는 새 토큰 필드가 없다. 이전 `입력 추정(bytes)` 기록을 실제 토큰 수로 해석하지 않는다. 입력 측정은 모델에 보내는 view만 바꾸고 원래 요청·조건·도구 receipt·저장 원본은 삭제하지 않는다.
- 문맥 크기와 요청별 모델 호출 한도(Azure50), 시간 제한, DB 결과 행 제한은 각각 독립적이다. 문맥 확대는 잘린 DB 결과를 전체 데이터로 만드는 기능이 아니다.
