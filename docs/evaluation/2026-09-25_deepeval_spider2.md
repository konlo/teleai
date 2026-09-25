# 실제 챗봇 DeepEval·Spider 2.0 평가 — 2026-09-25

판정: **로컬 단일 사용자 제한 출시도 보류**. 이 보고서는 실제 실행 중인 챗봇과 같은 Python 환경의 `GraphAnalysisRuntime`을 평가한다. 원격 Databricks 조회는 새로 실행하지 않았다. 전체 Spider 2.0 점수나 전체 자연어 성공률은 아직 측정하지 못했다.

[기계 판독 가능한 요약 결과](2026-09-25_deepeval_spider2_results.json). 평가 런타임: LangChain 1.4.0, LangGraph 1.2.11, DeepEval 4.2.3, `gemma4:e4b`와 로컬 judge `qwen3:8b`.

## 점수와 분모

| 평가 | 관측값 | 해석 |
|---|---:|---|
| 대표 fixture 10문항, 독립 수치·상태 oracle | **10/10 PASS** | 합성 로컬 데이터·현재 런타임. **9문항은 모델 호출 0회**인 결정적 경로, 모델이 실제 도구를 선택한 것은 1문항뿐이다. |
| DeepEval `ToolCorrectnessMetric` | **10/10** | 기대한 *도구 이름*의 정확 일치. 도구 인자·최종 자연어 충실도·자율 복구는 이 수치에 포함되지 않는다. |
| DeepEval GEval, 로컬 `qwen3:8b` 판정 | **6문항 평균 0.917, 기준 0.8 이상 5/6** | 5개 1.0, 이상치 답변 0.5. 선택된 답변 문항의 judge 점수이며 독립 oracle 또는 제품 출시 점수를 대체하지 않는다. |
| Spider 2.0 Lite 공식 SQLite 고정 5문항 | **SQL 제안 완료 0/5** | 135개 로컬 SQLite 과제 중 5개. 제출 SQL이 없어 공식 실행 정답률(EX)은 **산출 불가**. 공식 전체 547문항 리더보드 점수가 아니다. |
| 회귀 | **pytest 237 PASS, 참고 runner 217 PASS** | 계약·참고 테스트이며 실제 모델 자연어 성공률로 합산하지 않는다. |

## 평가 설계

- 실행 중인 localhost:8502 프로세스의 venv `.telly_runtime/v1-venv`와 `OLLAMA_MODEL=gemma4:e4b`를 사용했다. 별도 임시 venv는 LangChain/LangGraph 패치 버전이 달라 비교·실패 탐지에만 사용한다.
- 대표 10문항은 메타데이터, 집계, SQL, 피벗, histogram, 이상치, 통계, 차트로 분산했다. fixture의 독립 계산·원본 불변·승인/원격 실행 0회 등을 기존 실제 graph 평가기로 검사했다.
- Spider는 결과를 보기 전에 `local221`, `local009`, `local210`, `local358`, `local198`를 고정했다. 과제의 현재 SQLite 스키마만 읽어 agent에게 주고 gold SQL·정답 CSV는 주지 않았다. 실제 원격 실행과 자동 승인은 금지하며, SQL *제안*이 나오면 공식 평가기에 제출하도록 설계했다. 일반적인 SQL/도구 사용 지시를 덧붙였으므로 이 실행은 원문 질문 그대로의 공식 benchmark 참가가 아닌 **adapted SQL-proposal track**이다.
- `local009`가 공식적으로 지정한 `haversine_formula.md` 참고 문서를 첫 실행에서 빠뜨린 것을 검출했다. 평가 adapter가 공식 `resource/documents`에서 이를 제공하도록 고친 뒤 **같은 문항을 다시 실행**했고, 아래 표에는 교정된 실행을 사용한다. 다른 네 문항의 `external_knowledge`는 없다.
- Spider 배포물의 30개 DB 파일이 처음에는 0바이트였다. [공식 Spider2 저장소](https://github.com/xlang-ai/Spider2)의 SQLite 압축 파일을 복원한 뒤 필요한 30개 DB 모두 `PRAGMA quick_check=ok`를 확인했다. 다운로드한 압축 파일 SHA-256: `d56acf7c9d89be4bdf1f4f1281f4a03d91735f1858f6d6d0cfe0f9d562e3a94f`. 공식 scorer에 정답 CSV 자체를 넣는 배선 확인은 3/3이었다. 이는 agent 점수가 아니다.

### Spider 2.0 고정 표본의 실제 실행

| 과제 | 결과 | 관측된 중단 원인 | 시간 |
|---|---|---|---:|
| `local221` EU_soccer | SQL 없음 | `inspect_table_context` 호출 후 2회 모델 호출에서 `exhausted` | 202.493초 |
| `local009` Airlines | SQL 없음 | 공식 참고 문서 제공 후 재실행해도 `ReadTimeout` | 60.121초 |
| `local210` delivery_center | SQL 없음 | `ReadTimeout` | 60.143초 |
| `local358` log | SQL 없음 | `ReadTimeout` | 60.133초 |
| `local198` chinook | SQL 없음 | `ReadTimeout` | 60.099초 |

교정된 다섯 실행의 합계 442.989초. SQL 제안 완료율 **0/5**, 무승인 원격 실행 **0회**다. 무제안을 SQL 오답으로 자동 제출하지 않았으므로 공식 EX를 0%라고 부르지 않는다. 첫 문제의 추가 진단에서 HTTP timeout을 120초로 늘려도 첫 응답 115.665초 이후 도구 확인을 거쳐 `exhausted`로 끝나 제안이 나오지 않았다. 시간을 늘리는 것만으로 해결됐다는 근거는 없다.

## 발견한 제품 결함과 한계

- DeepEval의 도구 선택 10/10은 모델의 자율성 10/10을 뜻하지 않는다. 빠른 규칙 경로 9개가 모델 없이 처리했다.
- `L2_036` 이상치 계산은 구조화 값과 원본 계보가 맞았지만 최종 답변에 사용자가 요구한 Q1=98, Q3=1824, IQR=1726이 빠졌다. 기준선 4413과 이상치 137명은 답했다. 구조화 oracle은 PASS였고 별도 GEval judge는 0.5를 부여했다. 최종 답변의 필수 결과 필드를 검증하는 기준이 추가로 필요하다.
- GEval은 답변 충실도를 검사할 만한 6개 문항을 현재 앱 환경에서 채점했다. `L1_001`, `L1_016`, `L1_017`, `L1_036`, `L2_005`가 각 1.0, `L2_036`이 0.5다. 차트·피벗 3개는 이미지/구조 oracle을 사용했다. 통계 질문 `L2_051`의 judge는 별도 재시도에서도 120초 상한 내 판정하지 못해 **미채점**으로 남겼다. 6개 평균 0.917을 10개 전체 평균으로 확대하지 않는다.
- 모델 호출의 시간 초과와 도구 없이 종료하는 사례가 남아 있다. 임시 venv에서 동일 10문항은 9/10 PASS였고 `L1_016`은 첫 모델 응답 후 다음 응답에서 `ReadTimeout`으로 약 118초 뒤 미완료됐다. 현재 앱 venv의 같은 질문 단회는 47.116초에 PASS했다. 환경·실행별 변동을 고려하면 단회 성공은 안정성 증거가 아니다.
- Spider SQL 생성의 낮은 성공은 스키마가 없어서가 아니다. 실행 때 DB 스키마를 동적으로 읽었고, 첫 문제는 `inspect_table_context`를 호출했다. 병목은 모델의 후속 SQL 작성·시간 제한·도구 선택 경로다. 현재 수준을 Claude Code 수준의 자율 문제 해결로 평가할 수 없다.

## 실행 명령과 검증 범위

```bash
# teleai 루트. 아래 경로는 현재 평가 호스트 기준.
.telly_runtime/v1-venv/bin/python scripts/evaluate_analysis_agent.py --live-local-model --include-final-output \
  --id L1_001 --id L1_016 --id L1_017 --id L1_036 --id L1_062 --id L1_076 \
  --id L2_005 --id L2_036 --id L2_051 --id L2_061 --output /tmp/teleai_eval_20260925/exact_runtime_10.json

/Users/najongseong/git_repository/ai_agent_eval/.venv/bin/python scripts/evaluate_deepeval_real_report.py \
  --report /tmp/teleai_eval_20260925/exact_runtime_10.json \
  --output /tmp/teleai_eval_20260925/deepeval_exact_tools_10.json --skip-judge

OLLAMA_MODEL=gemma4:e4b .telly_runtime/v1-venv/bin/python scripts/evaluate_spider2_teleai.py \
  --spider-root /Users/najongseong/git_repository/ai_agent_eval/Spider2-main \
  --benchmark-instruction --id local221 --id local009 --id local210 --id local358 --id local198 \
  --output-dir /tmp/teleai_eval_20260925/spider2/exact_runtime_5
```

`/tmp/teleai_eval_20260925`의 원본 평가 로그에는 합성 fixture 답변이 들어 있다. 요약 이외의 사용자 데이터·비밀값은 이 문서나 git에 복사하지 않았다. `ai_agent_eval`의 기존 `run_eval.sh`는 mock adapter, `run_spider2_eval.sh`는 TeleAI와 분리된 SQL 생성기를 호출하므로 두 실행을 실제 챗봇 점수로 계산하지 않도록 기본 실행을 차단했다.

## 남은 출시 검증

현재 agent의 실제 Databricks 승인→정확한 1회 적재→보호 원본 보존→후속 EDA·차트까지 연결된 여정, 반복 모델 성공률·p95, 최종 자연어 충실도, 큰 데이터·동시 요청, Spider 과제 확대가 필요하다. 사용자가 Databricks UI에서 직접 실행한 10행 조회는 챗봇 내부 승인/적재 여정의 검증이 아니다. 별도 서버 운영은 사용자 지시에 따라 이번 로컬 범위에서 제외한다.
