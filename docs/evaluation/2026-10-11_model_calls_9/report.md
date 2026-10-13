# 모델 호출 9/9 진단

## 계약 거절의 정확한 조건

- 9번째 goal_interpreter는 응답 수신 성공이다. 계약 거절은 7·8번째 population_audit에서 발생했다. 9번째 이후의 population_audit는 호출 한도 때문에 실행 전에 차단됐다.
- population_requirements.py:113~116: resolved인 restriction_review clause의 컬럼에 대응하는 conditions/any_conditions predicate가 없고, 해당 컬럼이 removed_columns에도 없고, change가 clear도 아니면 거절한다. 오류: Quoted restriction has no matching population predicate.
- 로그의 audit change는 modify다. 선언한 제한 조건과 실제 실행 조건의 연결이 없었다는 것은 확인된다. raw restriction_review/조건 JSON을 진단 로그에 저장하지 않아 정확히 어떤 quote/column을 선언했는지는 확인할 수 없다. 출력 컬럼 또는 전체 데이터 범위를 잘못 제한 조건으로 선언했을 가능성은 가설이다.
- 전체 원본 범위와 출력 컬럼을 행 필터와 구분해야 한다. 검증 삭제나 호출 한도 증가만으로 의미 불일치가 해결되지 않는다.

- 실행 ID: `19f8ef3e56bd403e991cceaf96bf0bff`
- 요청 ID: `5b2430c1c3d8480aaf1995779d2bf508`
- 현재 입력: `전체 데이타로 해줘`
- 연결된 원래 요청: `전체 데이타에 대해서 longitude 의 histogram을 그려줘`
- 모델: Ollama gemma4:e4b
- 모델 응답 성공 9회 / provider 실패 0회 / 네트워크 재시도 0회
- 실행 전체 85.210초. 보조 호출 한도9=전체10-예약1. 다음 population_audit 호출은 실행 전에 차단.

| 순서 | 역할 | 작업 | 소요(초) | 관찰 |
|---|---|---|---:|---|
| 1 | `clarification_memory` | 확인 답변을 원래 요청에 연결 | 4.118 | 모델 응답 수신 |
| 2 | `subject_identity` | 테이블·컬럼 식별자 구분 | 2.633 | 모델 응답 수신 |
| 3 | `task_selection` | 수행 작업 선택 | 12.524 | 모델 응답 수신 |
| 4 | `population_basis` | 전체 원본/표시 결과 등 분석 범위 판단 | 9.445 | 모델 응답 수신 |
| 5 | `goal_interpreter` | 실행 목표 계약 생성 | 13.687 | 모델 응답 수신 |
| 6 | `goal_interpreter` | 목표 의미 재검토 | 19.144 | 모델 응답 수신 |
| 7 | `population_audit` | 모집단·조건 독립 검토 | 7.621 | 모델 응답 수신; restriction_review_incomplete로 계약 거절 |
| 8 | `population_audit` | 모집단·조건 수정 재검토 | 5.805 | 모델 응답 수신; restriction_review_incomplete로 계약 거절 |
| 9 | `goal_interpreter` | 목표 계약 수정 재생성 | 10.099 | 모델 응답 수신; 다음 모집단 검토는 호출 한도 차단 |

## 실패 경계

- population_basis는 unfiltered_source를 선택했으나 population_audit는 change=modify와 비어 있는 predicate_shapes를 반환한 경로가 검증에서 거절됐다. 로그의 오류는 Quoted restriction has no matching population predicate. 전체 범위 해석과 조건 감사가 일관되게 연결되지 않은 해석/검증 문제이다.
- 9개 모두 auxiliary 역할이다. 이번 실행에서 agent 행동 모델·SQL 생성/DB 조회·차트 실행 단계는 진입하지 못했다. calls=9는 DB 조회 9회 또는 대화 누적 호출 9회를 뜻하지 않는다.
- 모델 원문 응답/비공개 추론은 진단 로그에 저장하지 않는다. 역할·시간·오류 코드·ledger로 확인한 내용만 기록했다. 응답 수신 성공은 의미 검증 통과와 구분한다.
