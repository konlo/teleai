# 코드 책임 분리 및 개발 규칙 점검 — 2026-10-03

## 판정

**외곽 계층은 분리되어 있지만 agent의 핵심 제어 로직은 과도하게 집중되어 있다. 책임 분리 원칙은 일부 문서·계약에 있으며, 파일/함수 집중과 의존성 방향을 지속적으로 강제하는 규칙은 없다.**

이번 작업은 구조 점검이다. 제품 로직·DB·실행 서버·AGENTS.md·CI는 변경하지 않았다. 현재 미커밋 작업을 포함한 working tree를 대상으로 측정했고 HEAD와 일부 비교했다. 기존 테스트 통과는 유지보수 구조가 적절하다는 판정과 구분한다.

## 측정 근거

- 제품 패키지 `core`, `ui`, `utils`, `pages`, `modules`, `app_io`, `app` 및 `main.py`: Python 126파일, 27,918줄. 테스트/평가 스크립트/migration 파일은 이 합계에서 제외했다. AST 문법 파싱 실패 0개.
- 현재 진입점 `main.py`/`pages/Telly.py`/`ui/analysis_page.py`에서 정적으로 도달 가능한 내부 모듈: 79파일, 17,536줄. 함수 안의 선택적 import까지 포함한 보수적인 도달 집합이며 모든 모듈이 한 번에 실행된다는 뜻은 아니다. runpy 진입점은 명시적으로 포함했다.
- `recovery.py` 한 파일이 이 도달 집합 줄 수의 약 26%를 차지한다. 가장 긴 두 메서드가 이 파일의 약 62%다.
- 함수 길이는 AST 시작/끝의 물리적 줄 수다. 주석·공백·중첩 함수·선언적 schema를 포함한다. 순환복잡도 점수나 성능 측정값으로 해석하지 않는다.

| 현재 실행 경로의 파일 | 줄 수 | 집중된 책임 | 판정 |
|---|---:|---|---|
| `core/analysis_agent/recovery.py` | 4,560 | 요청 해석, 맥락 연결, 상태 초기화, 도구 관찰/근거 수용, 범위 검증, 계획 선택, 예산·복구, 완료 판단 | 최우선 분리 대상 |
| `core/analysis_runtime_tools.py` | 993 | 도구 등록/schema, 데이터·메타데이터·SQL·차트 실행 wrapper, 히스토그램 계획 | 기능별 factory와 실행 service 분리 필요 |
| `core/analysis_agent/runtime.py` | 740 | graph 조립, 영속 자산/장부 초기화, remote tool, 요청·재개·결과/선택 관리 | composition과 요청 controller를 분리할 후보 |
| `core/analysis_agent/intent_scope.py` | 708 | 자연어 범위 해석과 SQL AST의 실행 조건 비교 | 요청 해석과 실행 검증을 다른 책임으로 분리할 후보 |
| `utils/analysis_charts.py` | 693 | 추천, 차트 종류별 렌더링, COUNT 분포·이중축 표시 | 차트 renderer별 분리 후보 |
| `ui/analysis_page.py` | 257 | 런타임 호출·대화·결과 표시 | 핵심 문제가 아님; UI helper 분리도 이미 진행됨 |
| `core/analysis_agent/completion.py` | 133 | 공통 완료 계약 registry·근거 확인·render 호출 | 좋은 공통 경계; 확장 기준점으로 유지 |
| `core/analysis_agent/backends.py` | 64 | DB adapter 선택·설정/저장 경계 | 좋은 분리; 다른 DB로 자동 fallback하지 않음 |

구형 `core/chat_flow.py` 1,506줄, `utils/chatbot_plan.py` 1,846줄, `core/tools.py` 1,309줄도 저장소에 남아 있다. 현재 지원 진입점의 정적 도달 집합에는 없으며 참고/구형 회귀 경로와 구분해야 한다. 잔존한다는 이유만으로 현재 웹이 구형 agent를 실행한다고 판단하지 않는다.

## 주요 구조 문제

### 1. recovery 이름보다 훨씬 넓은 역할

- `RecoveryMiddleware._state`, 190행에서 시작, 1,714줄: 원문 해석/새 요청 초기화/조건 상속/각 capability 활성화와 도구 receipt 수용을 같은 메서드에서 처리한다.
- `_next_local`, 2,618행에서 시작, 1,115줄: 표 미리보기, 고유값, schema 탐색, 히스토그램, 최신행, 숫자 변환, 피벗/시계열/통계 등의 실행 우선순위를 거대한 분기로 결정한다.
- `_proposed_scope_valid`, 4,220행에서 시작, 227줄: 기능별 호출 검증까지 중앙 분기에 추가된다.
- 현재 HEAD 대비 recovery는 4,037→4,560줄(+523줄), 도구 registry는 917→993줄(+76줄)이다. 작은 기능 모듈을 추가해도 중앙 요청/관찰/계획/검증 분기를 계속 함께 편집해야 하므로 구조적 집중은 계속 늘어난다.

### 2. 파일을 나눠도 공유 상태 결합이 남음

- `RecoveryState`(160행)는 `recovery: NotRequired[dict]`로 선언되어 내부 필드 계약이 타입에 드러나지 않는다.
- 이 파일에서 `current[...]`, `get/pop/setdefault`, `update`로 접근하는 문자열 키가 최소 153개다. 실제 전체 schema나 동시에 존재하는 키 수가 아니라 정적 literal 접근 목록이다.
- `chart_grouping`은 같은 dict를 수정하고, `conversation_context.suspend_analysis`는 완료 계약 키를 None으로 지운다. 앞선 row-preview 검사에서 이 전환을 고려하지 않은 `.get()`이 실제 회귀를 일으켰다.
- 따라서 거대한 함수를 여러 파일로 옮기고 모두에게 `current`와 RecoveryMiddleware 전체를 넘기는 방식은 충분하지 않다. 필드 소유권과 상태 변경 경계를 정해야 한다.

### 3. 일부 의존성의 방향이 왕복함

AST에서 발견된 내부 import 왕복 세 묶음:

- `completion` ↔ `completion_renderers`: renderer가 registry의 active_contracts를 다시 조회한다.
- `analysis_datasets` ↔ `analysis_pivot`: 저장 데이터 digest가 피벗 모듈에 들어 있고 피벗은 저장소를 사용한다.
- `analysis_group_summary` ↔ `analysis_group_streaming`: streaming reducer가 summary 모듈의 private 검증 함수를 가져온다.

일부 역참조는 함수 안 import라 현재 초기화 오류를 발생시킨다는 증거는 없다. 다만 하위 service가 상위 feature를 다시 참조하고 있어 독립 분석·교체를 어렵게 한다. digest와 그룹 metric 검증을 공통 하위 모듈로 옮기고 renderer에는 필요한 완료 맥락을 인자로 주는 방식이 적합하다.

## 현재 규칙은 어디까지 있나

| 확인한 근거 | 있는 규칙/검증 | 빠진 부분 |
|---|---|---|
| `AGENTS.md` | 프로젝트 로그·회귀/출시 기준 적용 | 현재 주요 구현 안내가 구형 agent/chat_flow 중심; 파일/함수 책임 기준 없음 |
| `docs/langchain_migration_design_2026-09-06.md` 17행 이후 | UI→Runtime→도구/자산 책임, UI 직접 SQL 금지, 구형 도구 우회 등록 금지 | 초기에 작성된 전환 문서; 현재 recovery 내부 책임 분해와 import 계층의 강제 검사 없음 |
| 프로젝트 manager/logger skill | 데이터 context 외부화, generic 코드, 구조적 복구·회귀·작업 기록 | 모듈·함수 크기, 상태 변경 소유권, import 방향의 구체 기준 없음 |
| `core/analysis_tool_contract.py` | provider 독립 ToolDefinition/Context 및 공통 observation envelope | capability의 요청/계획/근거 수용/상태 patch까지 통합된 분리 계약은 없음 |
| `core/analysis_agent/completion.py` | 완료·표시·누락 근거 복구의 단일 registry | 요청 해석/계획/관찰 검증은 recovery 중앙 분기에 남음 |
| `.github/workflows/agent-release-gate.yml` | migration/application/agentic/reference 검사와 compileall | 파일·함수 집중도, 순환 import, 계층 역참조, 구형 경로 신규 import 방지 검사 없음 |
| 기존 boundary tests | 세션 독립 도구, 데이터 충분성·세션 격리, DB adapter 분리 | 전체 코드 의존성 아키텍처를 자동 검사하는 테스트는 아님 |

저장소 파일 검색에서 pyproject/ruff/import-linter/radon/pre-commit 관련 설정은 발견되지 않았다. 이 사실은 개별 개발자의 IDE/전역 설정이 없음을 뜻하지 않는다. 기존 일부 경계 테스트는 이번에 21 PASS/27 subtests(14.50초)로 확인했다. 새 제품 테스트나 전체 691개 회귀는 이 정적 점검에서 다시 실행하지 않았다.

## 권고할 개발 규칙 — 제안이며 아직 적용하지 않음

1. **크기와 책임을 함께 관리**: 새 논리 모듈은 400줄 안팎을 목표로 하고 600줄 초과는 분해 또는 명시적인 예외 사유를 요구한다. 논리 함수 80줄 초과는 검토, 150줄 초과는 분해/예외 대상으로 삼는다. 선언적 schema·대형 fixture는 별도 분류하며 행 수만 맞추기 위해 코드를 다른 파일로 옮기지 않는다.
2. **기존 대형 파일의 증가를 통제**: 현재 파일별 baseline을 고정한다. 핵심 대형 파일에 기능 분기를 계속 추가하는 변경은 기존 책임 추출을 함께 수행하거나 일시 예외·담당 작업을 기록한다. 기존 부채 때문에 전체 CI를 당장 실패시키지 않는다.
3. **capability 단위 소유권**: 요청 결합, 다음 ToolProposal, observation 검증/EvidencePatch, 결과 표시를 기능별 handler로 모은다. 중앙 loop는 우선순위·중복 방지·실행 정책·예산·공통 완료 호출을 담당한다. 매 기능마다 중앙 파일 네 군데를 편집하는 구조를 줄인다.
4. **상태 계약과 변경 주체**: Request/Scope/Plan/Evidence/ExecutionBudget/ConversationAnchor를 typed 계약으로 구분한다. handler는 허용된 결과 patch를 반환하고 공통 reducer가 변경을 적용한다. 기존 checkpoint 필드와 호환되게 단계적으로 도입하고 None/기본값 규칙을 명시한다.
5. **한 방향 의존성**: UI→Runtime→capability/실행 service→공통 계약/저장소/DB adapter 방향을 문서화한다. 검증기·저장소가 UI/runtime/상위 renderer에 역참조하지 않도록 한다. 상호 참조가 필요한 정보는 인자로 전달한다.
6. **등록과 실행 분리**: 도구 registry는 기능별 factory를 합성한다. SQL 계획·데이터 실행·차트 생성 구현을 등록 함수 안에 계속 추가하지 않는다. 이름/schema/실행 wrapper·capability 도구 목록이 서로 어긋나지 않는 검사를 둔다.
7. **자동 검사**: AST 기반 파일/함수 baseline 검사, 허용 import 방향·새 순환 방지·구형 경로 신규 import 금지, 상태 변경 경계와 capability 등록 일관성을 CI에 추가한다. schema/doc/테스트 파일 제외 및 예외 관리 기준을 같이 둔다.

## 권장 분리 순서 및 보존할 계약

1. 현재 실행 경로로 AGENTS.md를 정정하고, 위 규칙/아키텍처 baseline 및 자동 검사를 먼저 마련한다.
2. recovery의 요청 결합과 도구 관찰 수용을 각각 분리한다. 기존 state 키·완료/검증 순서·tool 우선순위는 유지하고 typed facade를 먼저 도입한다.
3. `_next_local`을 metadata/데이터 재사용·로딩/EDA/시각화 handler로 분리한다. 분기 우선순위와 정책 gate는 명시적인 registry로 유지한다.
4. 도구 factory를 메타데이터/데이터/통계/차트 그룹으로 분리하고 왕복 의존의 공통 기능을 하위 모듈로 이동한다.
5. runtime의 graph 조립과 remote 실행/controller 책임, intent 해석과 SQL 조건 검증을 분리한다. 구형 코드는 현재 실행 경로와 대체 회귀를 확인한 뒤 별도 정리 작업으로 처리한다.

이 순서는 하나의 설계/평가 계획 아래 진행하되 책임 단위로 검증 가능한 변경을 만드는 것이다. 지금의 모든 미커밋 수정과 대규모 파일 이동을 한꺼번에 섞지 않는다. LangChain/LangGraph agent loop를 다른 프레임워크로 교체할 필요도 없다.

분리 전후에는 동일 질문의 출처·조건·SQL 제출 횟수·완료 상태·선택 dataset·snapshot·원본 digest·이미지/표 artifact를 비교한다. 정상 탐색/로딩/후속 EDA뿐 아니라 실패→대체 도구→재개, DB 거절/불명 제출, 요약·재시작, 데이터 전환을 포함한다. 파일이 작아졌다는 이유로 agent 품질 향상을 선언하지 않는다.

근거: `metrics.json`, `boundary_validation.json`. 이번 점검에서 실행한 제품 경계 검사 21 PASS/27 subtests, 정적 파싱 126파일 성공. 실모델·DB·브라우저 재평가는 제품 동작을 변경하지 않아 수행하지 않았다.
