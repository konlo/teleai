# Claude Code 공식 구조 조사와 Telly 비교

코드 비교는 초기 진단 당시의 구조를 대상으로 한다. 이후 자체 AnalysisSession 구현을 거쳤으며, 최신 전환 기준은 [LangChain/LangGraph 전환 설계](langchain_migration_design_2026-09-06.md)에 정리했다.

조사일: 2026-09-06. 공개 공식 문서에 근거한 구조 비교이며 비공개 내부 코드 전체를 감사한 결과는 아니다. 제품 코드 변경 및 외부 모델 실행은 수행하지 않았다.

## 공식 자료에서 확인한 골격

- Claude Code는 모델이 맥락을 모으고 도구를 실행하며 결과를 확인하는 루프를 제공한다. 작업 순서는 관찰 결과에 따라 바뀐다. [공식 개요](https://code.claude.com/docs/en/how-claude-code-works)
- Agent SDK는 Claude Code의 루프를 애플리케이션에 사용할 수 있게 한다. 시스템 지침·도구 정의·대화 이력과 함께 모델을 호출하고, tool call 결과를 다시 모델에 전달한다. 횟수/비용 제한도 제공한다. [SDK 루프](https://code.claude.com/docs/en/agent-sdk/agent-loop)
- 세션에는 사용자 메시지뿐 아니라 도구 호출과 결과가 포함된다. 후속 요청은 해당 세션을 재개해야 한다. 세션 재개가 파일 시스템 복원을 뜻하지는 않는다. [세션](https://code.claude.com/docs/en/agent-sdk/sessions)
- AskUserQuestion은 답변을 루프에 반환하는 상호작용이다. 애플리케이션이 질문을 보여주고 응답을 돌려주는 연결을 구현해야 한다. [사용자 입력](https://code.claude.com/docs/en/agent-sdk/user-input)
- Subagent는 개별 작업을 위임받고 결과를 주 대화에 반환한다. 대화가 자주 오가고 맥락을 많이 공유하는 작업은 주 대화에서 처리하는 방향을 공식 문서가 안내한다. [Subagents](https://code.claude.com/docs/en/sub-agents)
- Anthropic은 코드가 정한 경로를 따르는 workflow와 모델이 작업 진행을 동적으로 결정하는 agent를 구분한다. 이 글은 2024년 글이며 구체적인 현재 SDK 정보는 위 최신 문서를 기준으로 삼았다. [설계 구분](https://www.anthropic.com/engineering/building-effective-agents)

## Telly 코드와의 차이 — 로컬 코드에 근거한 판단

| 영역 | Telly의 현재 상태 | 변경할 골격 |
|---|---|---|
| 사용자 발화 진입 | 키워드/controlled planner/router에서 경로 결정 | 같은 분석 세션에 사용자 발화를 추가 |
| 모델의 컨텍스트 | planner/router에는 현재 질문만, SQL/EDA에는 분리 이력 | 주 agent에 최근 대화·과거 도구 결과·필요한 분석 artifact 참조 전달 |
| 행동 선택 | 차트 전용 스키마 또는 SQL→EDA 고정 연결 | 탐색, 조회, 계산, 차트, 질문, 답변을 agent가 선택 |
| 결과 관찰 | controlled 경로는 렌더 후 기술 메시지로 종료 | 표/통계/차트 결과와 오류를 agent 루프로 반환 |
| 후속 수정 | 이전 분석과 신규 질문을 연결하는 공통 처리 부족 | 같은 세션에서 기존 결과·조건을 참고해 다음 행동 결정 |
| 명확화 | 질문 출력 후 종료, 응답 병합 대기 상태 없음 | 질문과 사용자 답을 동일 대화/실행에 연결 |
| 데이터 관리 | 전역 df_A가 현재 결과 역할 | dataset/result ID와 출처·조건·집계 정의를 보관 |
| 검증 | 일부 정규식 해석/조건 검사, 플롯 생성 확인 | 도구 입력·결과 검증과 실제 다중 턴 정답 평가 |

근거 위치:
- `core/agent.py:54`: 국소 SQL/EDA AgentExecutor는 실제 도구 반복을 수행한다.
- `pages/Telly.py:64`: SQL/EDA history 분리와 EDA history 삭제.
- `utils/chatbot_plan.py:1422`: 현재 질문 중심 시각화 planner.
- `core/chat_flow.py:402`: clarification 표시 후 return.
- `core/chat_flow.py:1428`: SQL 성공 및 flag 기반 auto_eda_pending/rerun.
- `test_scenario.py:4179`: CSV E2E에 llm=None, 실제 모델의 다중 턴 이해 평가는 아니다.

## 앞선 제안의 수정

유지할 제안은 하나의 주 분석 agent, 연속된 대화, 도구 실행 결과의 환류, 공통 데이터 출처/결과 참조다.

수정할 제안은 '별도 대화 분류기 → 고정 AnalysisRequest → 실행기'를 모든 요청의 필수 입구로 두는 방식이다. 또 하나의 workflow가 되어 '왜 그런가', '이전과 비교', '그 조건 말고' 같은 발화를 고정 틀에 맞추게 할 수 있다. 대화 해석은 주 agent가 맥락을 보고 수행하고, 구조화된 입력은 실행 도구 경계에서 검증한다. 복잡한 작업의 계획도 관찰에 따라 바꿀 수 있어야 한다.

명시적인 목표/조건 상태 객체가 agent의 보편적인 필수 구성요소라고 단정한 앞선 표현도 좁혀야 한다. 공식 문서가 확인해 주는 기반은 세션 transcript와 context 관리다. Telly에 조건/지표/결과 참조를 별도로 저장하자는 것은 데이터 분석의 재현성과 검증을 위한 우리 설계 제안이다.

## 권장 최소 골격 — Telly에 대한 설계 제안

1. UI는 메시지와 결과를 표시하고 conversation ID를 해당 agent session ID에 연결한다.
2. 주 분석 agent가 그 세션의 대화와 필요 컨텍스트를 읽는다.
3. 도구는 처음에는 데이터 목록/메타데이터 읽기, SQL 조회, Python 분석, 결과/차트 읽기 및 사용자 질문 정도로 제한한다.
4. 각 도구는 result ID, 실제 컬럼, 적용 SQL/조건, 행 수, 집계 여부, 경고 및 제한된 크기의 결과를 반환한다. 전체 데이터는 모델 메시지가 아니라 결과 저장소에 둔다.
5. agent는 결과를 보고 다음 도구를 호출하거나 답한다. 서버는 조회/실행 한도와 실패 상태를 제어한다.
6. 다음 발화도 같은 세션으로 전달한다. 결과 저장소의 데이터 수명은 세션 transcript와 별도로 관리한다.

후속 전환 설계에서는 기존 모델과 데이터 도구를 유지하면서 LangChain v1 create_agent/LangGraph로 실행 기반을 바꾸는 안을 기본으로 정했다. Claude Agent SDK는 별도 비교 후보이며 이번 전환의 의존성이 아니다. 프레임워크 교체만으로 데이터 도구·컨텍스트 문제가 해결되지는 않는다.

## 평가 게이트

같은 데이터로 '지난달 오류 많은 장비 → 그중 A 모델 → 이전 달과 비교 → 평균 대신 중앙값'을 하나의 세션에서 실행한다. 각 턴의 대상 집합, 기간, 필터, 통계값과 이전 result 참조를 정답과 비교한다. 되묻기와 답변 재개도 별도 시나리오로 검증한다. 정규식 controlled 경로, 같은 모델의 단일 agent loop, 가능하면 Claude SDK를 같은 조건에서 비교해야 구조와 모델의 영향을 분리할 수 있다. 현재 모델별 우열/실패율은 측정하지 않았다.
