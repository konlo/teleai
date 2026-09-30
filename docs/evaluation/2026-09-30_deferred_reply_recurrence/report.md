# 결과 도착 후 안내 문구 재발 수정 — 2026-09-30

이전 수정은 불완전했다. 일반 문자열 최종 답변과 실행 receipt 기반 완료는 검증했지만, 구조화된 텍스트 블록·중간 도구 안내·구버전 실행·receipt 없는 과거 표시 경로를 빠뜨렸다. 보내준 문구를 외부 fixture에 그대로 추가해 누락을 재현하고 수정했다.

## 재현 및 원인

[수정 전 검사](initial_failure.txt)에서 다음4개 실패를 확인했다.

1. `deferred_execution_claim`이 문자열만 검사했다. `[{type: text, text: ...}]`로 온 같은 문구는 통과했고, production graph에 “승인했어”를 넣으면 실제 실행0회인데 `answered`로 끝났다.
2. AIMessage가 tool_calls를 함께 포함하면 최종 답변 검증과 별개로 안내문이 transcript에 저장돼 화면에 노출됐다.
3. LangChain 구버전에서 사용하는 `AnalysisSession`은 tool_calls 없는 응답을 바로 answered로 처리했다. 승인 후 실제 실행1회가 끝났는데도 모델의 미래 안내가 성공 응답이 됐다.
4. 과거 정정 로직은 완료 receipt가 있을 때만 실제 결과로 복구했다. receipt 없는 과거 안내는 표시 단계에서 계속 노출될 수 있었다.

사용자 발생 대화 자체는 아직 확인하지 못했다. 로컬 영속대화38개의 transcript에서 전달한 문구·식별자·관련 문구 검색 결과0이었다. 사용자에게 localhost8502인지 다른PC/서버인지 확인 요청했다. **위4개는 실제 코드 재현된 결함이며 사용자 환경의 단일 원인으로 단정하지 않는다.** 최초 로컬 앱 PID15225는 06:53 시작한 프로세스였다.

## 변경

- 문자열과 text/output_text 블록을 같은 검증으로 처리한다. 내부 reasoning/이미지 블록은 답변 텍스트로 취급하지 않는다.
- transcript 저장과 읽기 양쪽에 표시 검증을 적용한다. 도구 호출의 잘못된 안내 내용만 숨기고 tool_calls는 유지한다. 근거 없는 최종 약속은 “자동 알림은 예약되지 않았고 실제 결과/상태 확인이 필요하다”는 blocked 안내로 바꾼다.
- 원문은 rejected_transcript에 보존한다. graph의 실제 실행·receipt는 수정하지 않으며 표시 정정 때문에 모델/SQL을 재호출하지 않는다. 완료 receipt가 있으면 기존 로직으로 실제 결과를 표시한다.
- 구버전 loop도 동일 검사와 최대2회 재계획 뒤 blocked 처리한다. 정상 결과로 고치는 모델은 계속 진행할 수 있고, 이미 끝난 승인 실행을 다시 호출하지 않는다. 구버전 과거 UI 표시에도 같은 보호를 적용한다.
- 특정 테이블명이나 깨진 식별자를 production 코드에 넣지 않았다. 표의 실명은 임의로 복원하지 않는다.

## 검증

- 관련 **27 PASS**. 전달 문구의 문자열/구조화 블록, tool narration, 과거 기록, 구버전 승인 후 복구/지속실패, 실제 결과 목록, 빈 결과, 거절, 불명확한 제출, 재시작/중복실행, 일반 설명 정상 통과 포함.
- 실제 진입점 `main.py` AppTest: 거짓 안내+tool call → 자동조회 대역1회 → 실제 반환된 목록 표시, 거짓 안내 미표시. 모델/DB는 대역이며 실환경 SQL 성공으로 계산하지 않는다.
- 제품 전체 **475/475 PASS (55.617초)**, migration **136/136 PASS (18.061초)**. compile/diff 검사 PASS.
- 첫 수정 중 누락된 AIMessage import가 focused test에서 실패했고 바로 보완했다. 최초 결함과 수정 도중 실패 및 최종 성공 로그를 분리 보존했다.
- 실행중 대화가 없음을 lock으로 확인 후 실제 localhost8502 앱 재시작, **PID28426/health200**. 저장된 합성 과거 문구를 실제 브라우저에서 열어 정정 안내가 표시되고 “도착하면/승인해주셔서”가 표시되지 않는 것을 확인했다. [앱 적용](app_restart.json), [화면 검증](ui_probe.json).

이번 실제 브라우저 점검은 합성 과거 transcript 표시 검증이며 모델0/SQL0이다. Databricks 공급자400 가용성 문제는 별도 미해결 상태다. 이번 수정을 실환경 데이터 조회 성공 또는 범용 GO로 확대하지 않는다.

## 적용 범위

현재 작업 브랜치 `codex/agentic-analysis-rc-2026-09-14` 및 로컬8502 앱에 반영했다. 다른PC/서버에서 사용 중이라면 해당 배포의 브랜치·commit과 프로세스 재시작 여부를 확인해야 한다. 사용자 실제 발생 주소/대화가 확인되면 동일 실행 경로인지 대조할 수 있다.

## 실제 사용 서버 확인 후 추가 진단

사용자는 재발 환경이 코드를 업데이트한 실제 사용 서버라고 확인했다. 앞선 로컬 검증은 그 서버의 정상 동작 근거가 아니다. 원격 main을 새로 fetch한 결과 `6f5edac`(PR #70)이며 최근 수정 `59355b6`은 아직 main에 포함되지 않았다. 서버가 어느 branch/revision을 사용하는지는 미확인이다. main 갱신만 했다면 수정이 빠져 있을 수 있으나 이를 실제 원인으로 단정하지 않는다.

서비스가 사용하는 Python 환경에서 다음 진단을 실행할 수 있다. 아래 `python`은 실제 Streamlit 서비스의 Python 경로로 대체한다.

```sh
python scripts/diagnose_checkout.py --expected-revision 59355b6
```

진단 스크립트가 아직 없는 checkout에서는 우선 다음 결과로 서버의 revision과 분기 조건을 확인한다. 환경변수나 `.env` 내용을 공유할 필요는 없다.

```sh
git log -1 --format='%H %s'
git branch --show-current
python -c 'import sys; from importlib.metadata import version; print(sys.executable); print(version("langchain")); print(version("streamlit"))'
```

스크립트는 원격 호출·데이터 조회·환경변수 출력 없이 Git/패키지 버전과 관련 파일 hash를 출력한다. `expected_revision_in_history=null`은 commit을 확인할 수 없음이며 미적용 확정이 아니다. `true`여도 파일이 수정되었거나 서버 프로세스에 이전 코드가 남아 있을 수 있다. LangChain 1 이상은 `ui/analysis_page.py`, 이전 버전의 Telly 페이지는 `ui/legacy_telly.py`로 분기한다. 서비스 실행 경로·재시작 여부·문제 대화의 로그를 추가로 대조해야 한다. 실제 서버 접속 정보가 없어 이 검증은 미완료다.
