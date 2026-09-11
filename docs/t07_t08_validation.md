# T07–T08 구현 및 검증 결과

2026-09-07 기준. 시제품의 graph runtime을 `core/analysis_agent/`로 옮기고 `ui/analysis_page.py`를 새 환경의 기본 화면으로 연결했습니다. `migration/`의 기존 import/실행 경로는 호환 wrapper로 유지합니다.

## 구현

- `runtime.py`: LangChain create_agent의 모델→도구→관찰 반복, 영속 checkpoint, 중단/재개, 진행 표시, 후속 요청 처리.
- `approvals.py`: 정확한 SQL·연결에 묶인 개별 승인, 원자적 제출 권리, 완료 receipt 재사용. 변경 요청은 기존 승인 무효화, 상태 질문은 승인 유지. 불명 제출은 자동 재실행하지 않습니다.
- `databricks.py`: 제품의 실제 데이터 로딩 도구. 승인 후에만 연결/실행합니다. OpenSession 단계 실패는 미제출 실패로 구분합니다.
- `assets.py`: scope별 SQLite 및 Parquet/PNG, 지연 로딩과 제한된 유지 캐시. 자산 누락으로 Databricks를 자동 호출하지 않습니다.
- `memory.py`: 모델 문맥 요약과 별도 원본 transcript 보존. 요약이 승인을 부여할 수 없습니다.
- 로컬 분석 도구: 결과의 coverage와 predicate 정보를 확인합니다. 단순 원본 조회의 AND 조건은 추적하며 복잡한 변환은 보수적으로 불명 처리합니다.
- 화면: 저장 대화/차트 복원, 통계 기반 실제 이미지 추천·선택, SQL 승인 카드, 스킬 읽기, 저장 TableContext 로딩, 사용자 표시용 ‘결과 N’ 이름.

## 검증

| 검증 | 결과 |
|---|---|
| 기존 unittest | 32개 통과 |
| 새 runtime/memory/approval/Streamlit 계약 | 18개 통과 |
| 기존 static suite | exit 0, 시각화 10/10 |
| 격리 환경 pip check | 통과 |
| 실제 로컬 모델 4턴·매 턴 runtime 재열기 | 40 → 40 → 20 → 4, SQL/답변 일치 |
| 실제 모델 요약 후 후속 계산 | 기간·그룹 유지, 중앙값 20, DB 호출 0회 |

실제 모델 증거: [4턴](v1_conversation_evidence.json), [요약](production_memory_evidence.json). 요약 평가에서 화면 메시지 26개를 보존하고 모델 메시지는 4개로 축약했습니다. 4턴은 각각 약 163/81/49/73초, 요약 평가 전체는 약 217초였습니다. 기능 통과를 속도 개선으로 해석하지 않습니다.

계약 테스트에는 거절 시 DB 미호출, 승인 후 단일 제출, 프로세스 강제 종료 후 불명 제출 차단, 연결 변경, 승인 상태 질문, 요청 변경, 요약/복원, 소유자 격리, 차트 선택 후 정상 종료 및 기본 main.py AppTest가 포함됩니다. 대역 모델 검사는 자연어 품질 평가와 구분합니다.

## Databricks 결과

사용자 설명에 따라 개발용 조회 승인 대기를 해제하고 합성 SELECT smoke를 실행했지만 OpenSession 단계에서 실패했습니다. 해당 시점의 불명 receipt는 역사적 증거로 보존하며 자동 재실행하지 않았습니다. [실행 기록](v1_databricks_smoke.json).

후속 연결 전용 진단은 **RequestError / HTTP 403 / OpenSession**, SELECT 실행 0회입니다. [진단 기록](databricks_connection_diagnostic.json). 이후 알려진 OpenSession 실패를 미제출 실패로 구분하도록 수정하고 회귀 검사했습니다. 실제 Databricks 데이터의 통계·차트 성공은 아직 검증되지 않았습니다. 연결 설정·토큰·warehouse 접근 권한을 확인해야 합니다.

## 실행 및 남은 항목

`python3 scripts/run_telly.py`로 기본 agent를 실행합니다. root 요구사항은 `requirements-agent.txt`를 사용합니다. 기존 `.venv`와 legacy 화면은 보존했습니다. 구형 환경에서 main.py를 실행하면 기존 화면으로 돌아가며 새 승인권/세션을 구형 runtime으로 복사하지 않습니다.

현재는 localhost 데스크톱 실행입니다. 다중 사용자 인증, 실사용 대용량 peak memory, 자산 정리, 로컬 파생 결과 crash replay 중복 제거 및 응답 지연 개선은 남아 있습니다. 유지 캐시 제한이 SQL/Parquet 실행 중 전체 메모리를 제한하지는 않습니다.

최종 브라우저 확인: 기본 main.py 실행에서 저장된 7행 결과·대화·차트 추천 3개·선택 PNG가 복원되었고, 미완료 오류 없이 후속 입력 화면을 확인했습니다.
