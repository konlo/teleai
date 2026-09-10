# Telly LangChain/LangGraph 전환 설계

작성일: 2026-09-06. 상태: 설계 확정안, 전환 구현 전.
이 문서는 실행 기반 전환의 최신 기준이다. 사용자 요구가 최우선이며, 기존 데이터/승인/시각화/스킬 설계의 제품 요구는 유지한다. 과거 진단 문서는 작성 당시 코드의 기록이다. 아래 충돌 표에서 명시한 실행 방식만 대체한다.

## 1. 목표와 현재 기준선

목표는 대화가 이어지는 Databricks 분석 agent다. 프레임워크 교체 자체를 완료 기준으로 삼지 않는다.

확인한 현재 상태:
- LangChain 0.3.27, langchain-core 0.3.79, LangGraph 1.0.1 설치. app-requirements.txt는 일부 넓은 버전 범위와 langchain-openai==0.3.32를 혼용한다.
- pages/Telly.py는 현재 AnalysisSession을 사용한다. 예전 SQL/EDA AgentExecutor 및 chat_flow 코드는 남아 있지만 새 페이지의 주 실행 루프는 아니다.
- core/analysis_loop.py는 모델/도구 반복, transcript, 승인 대기를 직접 구현한다. durable checkpoint/요약/종합 완료 검증은 아직 없다.
- 새 UI의 조회 제안/취소가 DB를 호출하지 않는 테스트, 단위·통합 테스트 26개 및 기존 정적 시나리오가 통과했다.
- 현재 Ollama gemma4:e4b로 실제 두 턴 평균 40→A그룹 평균 20을 계산했다. 다른 모델, 폭넓은 다중 턴 분석, live Databricks 성공은 아직 검증하지 않았다.

## 2. 목표 구조와 소유권

Streamlit UI → AnalysisRuntime 인터페이스 → LangChain create_agent / LangGraph runtime
                                              ↕
                                  모델 ↔ Telly 분석 도구
                                              ↕
                            데이터 자산·차트 저장소 / 승인 실행 기록

기본 선택: LangChain v1 create_agent의 모델-도구 반복, HumanInTheLoopMiddleware, LangGraph checkpointer를 사용한다. 복잡한 그래프가 필요한 증거가 생기기 전에는 별도 SQL/EDA 라우팅 그래프를 추가하지 않는다.

책임을 한 곳에 둔다:
- LangGraph: 메시지/분석 상태, interrupt, 실행 재개와 checkpoint.
- 승인 실행 기록: 구체적 조회의 승인 증거, 원자적 실행 획득, DB 작업 ID, 완료 결과/불명 상태. 또 하나의 대화 대기 관리기가 아니다.
- DatasetStore/ArtifactStore: DataFrame 및 이미지 실체와 출처/조건/집계/범위 metadata.
- Telly 도구: 데이터 충분성 검사, SQL 검증, 로컬 계산, 차트 추천, 스킬 읽기.
- UI: 메시지·결과·승인 카드 표시와 사용자 결정 전달. UI가 SQL을 직접 실행하거나 임의 상태를 덮어쓰지 않는다.

create_agent가 동적 행동 선택을 맡는다. 모든 발화를 정해진 분석 JSON으로 분류하는 이전 설계로 돌아가지 않는다. 구조화된 schema는 도구 입력과 재현 가능한 결과에 사용한다.

## 3. 유지·교체·보완 매핑

| 현재 파일/기능 | 전환 결정 | 유지할 계약 |
|---|---|---|
| core/analysis_loop.py | 자체 반복을 LangGraphRuntime으로 교체 | 동일 대화에서 관찰을 읽고 후속 요청 처리 |
| core/analysis_model.py | v1-compatible BaseChatModel/provider adapter로 통합 | 기존 Ollama 모델/endpoint 유지, native tool calls 및 필요한 reasoning 상태 |
| core/analysis_approval.py | in-memory queue를 durable execution ledger로 전환 | 매번 명시적 승인, 승인된 조회만 1회 제출, 변경 시 재승인 |
| core/analysis_runtime_tools.py | session closure 의존 제거, typed tool/runtime context로 연결 | 기존 도구 기능과 결과 의미 유지 |
| core/analysis_databricks.py | 승인 실행 기록 검증 후 실행하는 backend service | bounded fetch, 읽기 전용, 실제 출처/결과 기록 |
| utils/analysis_datasets.py | 보존·확장 | 원본/집계·완전성·범위 판단, parent/result 참조 |
| utils/analysis_charts.py | 보존·확장 | 실제 데이터 PNG, 표본 표시, 미리보기와 최종 config 일치 |
| utils/analysis_skill_registry.py, analysis_skills/ | 보존 | 이름/설명 발견 후 필요한 본문만 읽기 |
| core/analysis_instructions.py | 단일 system 지침으로 유지·정리 | 한국어, 맥락 유지, 로컬 실행과 DB 승인 구분 |
| pages/Telly.py | runtime client 역할로 축소 | 승인/차트 선택 UI, 대화 연속성 |
| core/agent.py, core/chat_flow.py 및 기존 tools | legacy로 분리 후 호출 여부 확인해 정리 | 기존 회귀 재현용으로 보존, 새 runtime의 우회 경로로 등록 금지 |

기존 도구를 모두 새 agent에 등록하지 않는다. 기존 unrestricted Python REPL과 직접 DB 실행 도구는 승인/데이터 정책을 우회할 수 있다. 우선 새 로컬 SQL·검증된 차트·스킬 도구를 사용한다. 고급 Python은 별도 격리 실행 도구가 갖춰진 뒤 추가한다.

## 4. 승인과 재개의 일관된 처리

새 query_databricks 도구 호출의 의미는 '이 인자로 조회를 요청한다'다. 실제 실행 전 HITL이 항상 interrupt한다. 현재 propose_databricks_query + 수동 resume 방식은 전환 시 이 흐름으로 대체하고 동시에 켜지 않는다.

1. SQL과 바인딩 인자, source/namespace, connection identity, 범위·집계·제한을 정규화/검증해 immutable request envelope를 만든다. 모델이 제공한 source 라벨만 신뢰하지 않는다.
2. HITL interrupt와 request_id를 checkpoint에 기록한다. 이 시점에는 DB 호출이 없어야 한다.
3. UI는 이유/범위/SQL 및 기존 데이터 유지 여부를 표시한다.
4. 사용자가 승인하면 인증된 controller가 thread/interrupt/request fingerprint를 확인하고 승인 증거를 실행 기록에 쓴다. 같은 thread에 Command(resume=...)를 보낸다.
5. 실행 service는 UI가 만든 승인 기록을 확인하고 원자적으로 실행 권리를 획득한다. agent에게 approve API나 승인 증거 생성 도구를 제공하지 않는다.
6. 완료 결과는 tool result로 같은 대화에 돌아온다. 현재처럼 DB 결과를 사용자 발화 JSON으로 넣지 않는다. 승인은 사용자 event, 실제 조회 결과는 tool observation으로 분리한다.

허용 결정은 approve/reject만으로 시작한다. SQL 수정 후 바로 실행하는 edit는 사용하지 않는다. 변경 요청은 이전 제안을 폐기하고 새 내용을 보여준 뒤 재승인한다. 프로파일/목록/스키마 갱신 등 원격 읽기가 추가되면 같은 gate에 등록한다. 기본 구현이 SELECT만 지원하면 임의 SQL로 metadata 조회를 우회하지 않고 승인형 metadata 도구를 추가한다.

'현재 데이터로 보기'는 reject + 기존 범위로 분석하라는 사용자 피드백이다. 원래 요청이 충족됐다고 표시하지 않는다. 단순 상태 질문은 승인 내용을 무효화하지 않는다. 기간/필터/대상/연결 변경은 intent revision을 증가시켜 이전 승인을 무효화한다. 현재 ApprovalQueue.advance()의 모든 신규 발화 무효화 방식은 이를 따르도록 수정한다.

여러 도구 호출이 동시에 제안되면 각 원격 작업에 독립 승인과 UI 항목을 부여한다. 첫 버전에서는 원격 동시 제출을 제한한다. 프레임워크 interrupt batch의 순서를 검증하되 UI는 request_id로 사용자의 선택을 묶어 잘못된 작업에 승인이 적용되지 않게 한다.

### 재실행·장애 의미

LangGraph checkpoint/interrupt가 외부 DB exactly-once를 보장하지는 않는다. interrupt가 포함된 노드는 재개 시 처음부터 다시 실행될 수 있으므로, interrupt 이전에는 DB 실행을 두지 않는다.

실행 기록 상태: PROPOSED → APPROVED → SUBMITTING → RUNNING → COMPLETED.
추가 상태: REJECTED, INVALIDATED, FAILED, UNKNOWN.
- 동일 작업 재개 + COMPLETED: 저장된 결과 참조 반환, DB 재제출 금지.
- RUNNING: 가능한 DB 작업 ID로 상태 조회. 새 SQL 제출로 바꾸지 않는다.
- SUBMITTING 중 장애/timeout: 실제 제출 여부가 불명확하면 UNKNOWN. 자동 재시도하지 않는다.
- backend가 작업 ID/재조회 기능을 제공하지 않으면 이를 구현·검증하기 전 자동 복구를 제공한다고 주장하지 않는다. 사용자에게 상태 확인과 명시적 새 제출 선택을 제공한다.
- FAILED 후 새 조회는 새 승인. DB execute에는 일반적인 자동 retry middleware를 적용하지 않는다.
- 취소된 HTTP/UI 요청이 DB 작업 취소를 의미하지 않는다. 실제 DB 취소 여부를 확인해 표시한다.

## 5. 상태·대용량 데이터·컨텍스트

thread_id는 사용자/workspace/conversation에 귀속한다. 브라우저와 모델이 보낸 ID만으로 다른 대화 상태를 읽지 않도록 소유권을 확인한다.

Checkpoint state에는 messages, 현재 목표/확정 조건의 요약, active_dataset_id, result/card 참조, intent_revision, pending request 참조와 적용 skill version만 저장한다. DataFrame, PNG bytes, credentials, Python callback, RLock, live connection을 넣지 않는다.

자산은 별도 저장소에 보존한다. 첫 로컬 버전은 검증된 파일 기반 checkpoint와 Parquet/이미지 artifact 저장소를 사용할 수 있다. 다중 사용자 배포는 트랜잭션을 지원하는 checkpoint DB와 접근 제어된 자산 저장소로 전환한다. 실제 saver 패키지/버전은 호환성 단계에서 고정한다.

자산 기록에는 owner, schema version, content version, parent, source, predicate, grain, coverage, query/snapshot, 경로/크기를 저장한다. 메모리 예산에 따라 내려놓더라도 필요한 파일 참조를 유지한다. 파일을 사용할 수 없으면 자산 만료로 처리하며 Databricks를 몰래 재조회하지 않는다. 재승인 없이 복구 가능한 것은 보유한 로컬 자산뿐이다.

모델 context는 현재 관련 자산 metadata와 짧은 도구 관찰 위주로 구성한다. 전체 DataFrame/모든 과거 SQL/모든 스킬 본문을 반복 주입하지 않는다. 모델 token budget에 맞게 최근 대화와 유효한 조건/결정/결과 참조를 보존하며 요약한다. tool call/result 쌍과 대기 interrupt를 깨뜨리는 단순 메시지 개수 삭제는 사용하지 않는다. 요약 전후 동일한 후속 질문으로 의미 보존을 검사한다.

## 6. 데이터 재사용·이미지·스킬의 충돌 방지

데이터 재사용:
- REUSE/DERIVE_LOCAL은 로컬 실행. QUERY_SOURCE는 실행 승인이 아니라 조회 제안 필요 상태다.
- NEED_SCOPE는 사용자 의미 확인, QUERY_SOURCE는 외부 실행 승인이다. 둘을 섞어 로컬 계산에도 승인받지 않는다.
- 단순 조건 비교만 지원하는 기존 assess_reuse의 한계를 유지 표시한다. OR/NOT을 AND로 축소하지 않는다.
- local_analysis_sql 역시 데이터 요구/coverage 검사에 연결한다. SELECT가 실행된다는 이유만으로 전체 모집단 정답을 보장하지 않는다.
- 신규 SQL의 출처/집계/coverage는 코드에서 판정한다. 판정 불가면 unknown을 유지한다.

시각화:
- recommend_chart_images는 원격 조회 없이 동작하며 tool result에는 이미지 본문 대신 card/artifact ID를 반환한다.
- UI의 차트 선택은 정상 사용자 이벤트로 같은 thread에 남는다. 재조회 승인으로 사용하지 않는다.
- dataset/config/renderer version 및 sampling seed를 카드에 묶는다. stale 카드를 새 데이터에 암묵 적용하지 않는다.
- 기존 선택 이미지 그대로 확대하고, 조건 변경은 새 결과/새 카드로 만든다.

스킬:
- SKILL.md 파일과 registry 형식은 유지한다. read_analysis_skill이 LangChain tool로 바뀔 뿐이다.
- 스킬은 분석 절차이지 승인·권한의 근거가 아니다. 읽기 실패/미선택 시에도 DB gate는 유지된다.
- 스킬의 domain schema는 외부 TableContext/문서에서 읽는다. LangChain 전환 때문에 데이터별 컬럼 규칙을 system prompt에 복제하지 않는다.

## 7. 의존성·모델 호환 전략

운영 .venv를 즉시 업그레이드하지 않는다. 별도 환경에서 Python/Streamlit/LangChain v1/LangGraph/checkpointer/provider 패키지의 호환 조합을 확인하고 재현 가능한 lock을 만든다. 버전 번호를 현재 설치 상태와 맞는지 검증하지 않은 채 문서 예제만 복사하지 않는다.

기존 langchain-openai==0.3.32 등 core 0.3 의존 패키지를 v1과 그대로 혼용하지 않는다. langchain-community/experimental의 legacy imports도 조사한다. 필요한 구형 코드만 별도 legacy 환경 또는 명시적인 호환 패키지로 격리하고 새 페이지가 우연히 import하지 않게 한다.

모델은 기존 설정을 유지한다. Ollama는 v1-compatible ChatOllama 또는 검증된 BaseChatModel adapter 중 native tools와 reasoning round-trip을 통과하는 것을 사용한다. 현재 raw HTTP adapter를 검증 없이 폐기하지 않는다. Google/Azure는 같은 contract tests 및 실제 사용 가능한 환경에서 검증한다. 모델이나 자격 증명을 자동 변경하지 않는다.

모델 연결, 모델 선택, 런타임 전환을 같은 실험에서 한꺼번에 바꾸지 않는다. 동일 모델/데이터/질문으로 현행 루프와 새 루프를 비교해 변경 효과를 분리한다.

## 8. 단계별 전환과 롤백

M0. 기준선: 현재 코드/의존성/26개 테스트/실제 두 턴 결과를 보존한다. user 변경 및 미커밋 파일을 덮어쓰지 않는다.
M1. 도구 경계: 데이터·스킬·차트·SQL 도구를 현재 AnalysisSession closure에서 분리한다. AnalysisRuntime 인터페이스(submit, inspect, respond, cancel, events)를 정하고 동일 기능 테스트를 양쪽 구현에 적용한다.
M2. 격리 환경: v1 의존성 lock과 모델 어댑터 contract를 통과시킨다. 제품 기본값은 바꾸지 않는다.
M3. 새 runtime: create_agent, typed messages/state, persistent checkpointer, context budget를 연결한다. DataFrame은 저장소 ID로 접근한다.
M4. 승인: HITL과 durable execution ledger를 연결하고 restart/중복 resume/UNKNOWN 상태를 검증한다. 이 단계 전 새 runtime에 실 DB 실행 권한을 연결하지 않는다.
M5. UI: 기존 승인 카드/차트/데이터 목록을 runtime events에 연결한다. 예전 ApprovalQueue와 HITL이 같은 요청에 두 번 묻거나 실행하지 않도록 단일 writer를 강제한다.
M6. 검증 후 전환: 새 대화부터 새 runtime을 기본값으로 한다. 기존 대화는 생성 당시 runtime_version으로 고정한다. 라이브 승인형 Databricks 테스트는 사용자 승인 이후에만 수행한다.
M7. 안정화: 전체 수용 기준 후 사용되지 않는 자체 loop/legacy 경로를 정리한다. 이전 테스트는 폐기하지 않고 해당 계약에 이식한다.

롤백: 배포별 환경/lock/runtime version을 함께 되돌린다. 진행 중 DB 작업을 다른 runtime에서 재개하지 않는다. 새 runtime 장애 시 기존 실행 중 세션을 임의로 구형 루프로 넘기지 않고 상태를 보존한다. 기존 기능에 되돌리더라도 승인 gate는 항상 유지하며 실패 시 조회를 차단한다.

기존 세션 이전은 idle 상태에서만 별도 지원한다. user/assistant/tool 메시지 ID를 검증·정규화하고 자산을 저장한 뒤 import한다. 이전 구현이 사용자 JSON으로 넣은 승인 결과는 명시적 execution event로 정리한다. 대기 승인·진행 중 조회는 원래 runtime에서 끝내거나 취소/상태 확인 후 새 요청으로 재승인한다. 다른 런타임으로 승인권을 복사하지 않는다.

## 9. 회귀/완료 검증 표

| 기준 | 필수 증거 |
|---|---|
| 기존 동작 보존 | 기존 26개 테스트 계약 + legacy static suite의 의미 있는 케이스 이식/통과 |
| 모델 ↔ 도구 | 같은 모델에서 actual tool-call/result round-trip, 잘못된 인자/길이 초과/오류 회복 |
| 다중 턴 | 평균→A그룹→기간 비교→중앙값, 지시어·조건 수정·주제 전환의 SQL/결과값 검사 |
| 승인 | 제안/거절/무응답/단순 화면 재실행 DB 0회, 승인된 정확한 query 1회 |
| 변경 | SQL/기간/connection identity 변경 시 재승인, 단순 상태 질문으로 승인 소실 없음 |
| 장애 | 승인 전·후/제출 직후/결과 저장 후 강제 종료 및 checkpoint 재개, 중복 제출 없음 |
| 데이터 | A→B, 기간 확대/축소, LIMIT/표본/집계/완전한 빈 결과, OR/NOT 의미 보존 |
| 이미지 | 실제 PNG, 원본/집계 적합성, 선택 이미지/config 동일, stale/만료 카드 처리 |
| 스킬 | 필요 본문만 로딩, 이전 대화 유지, 스킬 미선택에도 승인 gate 적용 |
| 긴 대화 | context budget 적용 후 조건/미해결 질문/결과 참조 유지, 자산 메모리 예산 검증 |
| UI | 승인→분석 재개, 취소→현재 범위 대안, 차트 선택→후속 수정의 브라우저 검증 |
| live | 사용자가 승인한 Databricks 조회로 비교 가능한 SQL/통계/차트 및 후속 요청 검증 |

결과 검증은 실행 상태와 사용자 목표 충족을 구분한다. 계산 요청의 답변에는 실제 result artifact와 수치 근거가 필요하다. 단순 대화나 명확화 질문에는 불필요한 계산을 강제하지 않는다. 프레임워크 종료/green test만으로 전체 목표 달성을 선언하지 않는다.

## 10. 이전 설계와의 충돌 해소

| 이전 내용 | 최종 기준 |
|---|---|
| 명확한 요구면 원본 자동 조회 | 의미가 명확해도 Databricks 재조회는 매번 승인 |
| 자체 루프 확장 또는 Claude SDK 채택 제안 | 제품 기본 전환안은 LangChain v1/LangGraph; Claude Code는 동작 원리 참고. Claude SDK는 별도 비교 후보 |
| 별도 분류기→고정 AnalysisRequest | 주 agent가 맥락을 읽고 행동 선택; 구조화 검증은 도구 경계 |
| ApprovalQueue와 UI가 실행/재개 | graph interrupt와 인증 controller, 실행 기록으로 역할 분리 |
| 승인 결과를 user JSON에 첨부 | 사용자 결정 event와 실제 tool result 분리 |
| 매 발화 시 모든 pending 무효화 | 의미/범위 변경만 무효화, 상태 질문은 유지 |
| raw dataframe와 PNG를 session 객체에 보유 | 별도 자산 저장소, graph에는 참조만 |
| 스킬 파일이 승인 규칙 담당 | 실행 service가 승인 강제, 스킬은 방법 안내 |
| 26개 테스트/두 턴 성공=완료 | 제한된 기준선이며 위 전체 수용 기준을 충족해야 완료 |

## 공식 근거

- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents): create_agent는 LangGraph 기반 모델/도구 반복을 구성한다.
- [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop): 도구별 승인 interrupt와 checkpointer를 통한 재개. 본 설계는 approve/reject만 사용한다.
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts): 재개 시 노드 재실행에 주의해야 한다. 외부 실행 기록은 Telly의 추가 설계다.
- [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory): thread 상태와 메시지 관리. 데이터 자산 저장소/통계 검증은 별도 설계다.
- [v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1): 기존 0.3 계열과 새 API/패키지 경계의 호환성 확인 기준.

## 기존 기능의 추가 보존 계약

이전 작업 로그의 기능은 파일을 남기는 것만으로 보존됐다고 판정하지 않는다. 현재 새 페이지에서 빠졌거나 의미가 달라진 기능은 M0에서 목록화하고 새 runtime의 수용 항목으로 추적한다.

- TableContext 학습 결과, 컬럼 별칭·범주 값·비교 조건은 외부 metadata로 재사용한다. `%table training`의 원격 갱신은 승인형 작업으로 연결하며, 저장된 context를 읽는 로컬 작업과 구분한다.
- 테이블 표본(`df_table_sample`)과 분석 결과 미리보기는 별도 자산 역할을 유지한다. 테이블 선택만으로 원격 표본을 자동 갱신하던 이전 동작은 최신 승인 요구에 맞춰 갱신 제안으로 바꾼다. 표본을 전체 분석 데이터로 승격하지 않는다.
- `utils/config.py`의 기존 SQL LIMIT 설정과 새 어댑터의 fetch 상한·추천 차트 표본 크기를 서로 다른 제한으로 명시한다. 현재 상한 차이를 M0에서 기록하고 중앙 설정으로 관리한다. 사용자 SQL LIMIT을 조용히 변경하거나 fetch 상한으로 잘린 결과를 complete로 표시하지 않는다.
- 과거 범주/숫자 범위 필터, 조건 누락 차단, 그룹 분포, 지정한 X/Y 축, 차트 이미지 출력 사례를 새 도구의 결과 검증으로 이식한다. 과거 정규식 라우터 자체를 복구할 필요는 없다.
- 기존 runtime trace와 번호 기반 회귀 시나리오는 thread/request/tool-call/dataset ID로 연결한다. 새 UI에서 과거 시나리오를 실행할 수 없는 경우 통과로 간주하지 않고 대응 시나리오를 추가한다.
