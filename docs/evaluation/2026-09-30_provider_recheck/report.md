# Databricks HTTP400 재진단 — 2026-09-30

판정: **실행 계층 BLOCKED. 짧은 일시 장애로 확정할 수 없으며 원인 점검이 필요하다.** 최초 06:18 KST 모델 오류부터 08:15 KST까지 같은 오류가 반복됐다. 재시도 가능 분류는 회복이 보장된다는 뜻이 아니다.

## 실제 관측

| 확인 경로 | 결과 | 의미 |
| --- | --- | --- |
| warehouse 상태 API | HTTP200, STOPPED, serverless 2X-Small | 이 API에서 인증·워크스페이스 접근 가능. STOPPED 자체는 자동 정지일 수 있어 장애 원인으로 단정하지 않음 |
| 선택한 Foundation Model endpoint 정보 | HTTP200, READY, NOT_UPDATING | 설정된 모델이 존재한다. READY metadata가 실제 추론 가능함을 보장하지 않음 |
| agent를 거치지 않는 최소 모델 요청 | HTTP400 BAD_REQUEST | 도구 schema·분석 프롬프트 없이도 실패 |
| 제품 모델 adapter의 최소 요청 | HTTP400 BadRequestError | 실제 제품 경로에서도 동일. 시도1회, 재시도0 |
| SELECT 1 연결 | OpenSession HTTP400 RequestError | SQL 제출 전에 연결 실패. 테이블/schema/분석 SQL이 원인이 아님. 사용자 테이블 읽기0 |
| Chrome 로그인 | 만료된 세션 복원, Free Edition 확인 | 브라우저 로그인과 API 실행 권한/리소스 가용성은 별개 |
| 콘솔 추가 점검 | 자동 브라우저 제어 미지원 안내 | 이를 우회하지 않았으며 사용자에게 한도 안내 문구 확인을 요청함 |

최소 요청에서 관측한 공급자 메시지:
- 모델: `Cannot create or query foundation model endpoints, please try again later.`
- SQL: `Cannot create the resource, please try again later.`

[직접 API·SQL](probes.json), [모델 상태](serving_status.json), [제품 adapter 재검사](production_probe.json).

## 원인에 대한 판단 범위

- 현재 400은 agent가 잘못 만든 분석 SQL이나 tool schema 때문에 발생한 것으로 볼 근거가 없다. agent 이전의 최소 요청에서도 동일하게 재현된다.
- 모델과 SQL 실행이 모두 차단되므로 workspace 실행 리소스/사용량/계정 정책 또는 공급자 장애를 점검해야 한다. 현재 응답은 세부 원인을 알려주지 않아 **한도 초과 확정이 아니다**.
- Free Edition은 실제 UI에서 확인했다. [공식 제한 문서](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)에 따르면 사용 한도를 넘으면 그날 남은 시간, 심한 경우 그달 남은 기간 compute가 중단될 수 있다. **이 workspace의 한도 소진 여부·리셋 시각은 미확인**이다.
- Verify identity 버튼도 보였으나, 공식 문서는 이를 일부 추가 기능의 확인 절차로 설명한다. 해당 버튼이 이번 SQL/추론400의 원인이라는 증거는 없다. 인증/결제/플랜을 임의 변경하지 않았다.
- 상태 API와 모델 정보 API가200이므로 토큰 재발급이 필요하다는 증거는 없다. 공급자 전체 장애 여부도 확인되지 않았다.

## 이번 보강과 검증

1. `scripts/probe_databricks_availability.py`: 비용이 큰 live 평가 전 warehouse 상태/제품 모델 최소 호출/SELECT1을 각각 점검한다. 원본 테이블을 읽지 않고 자동 재시도하지 않는다. 실제 모델·SQL 모두 성공해야 AVAILABLE/exit0이며 그 외는 BLOCKED/exit1이다. AVAILABLE은 연결 확인일 뿐 agent GO를 뜻하지 않는다.
2. 공급자400 사용자 안내에 일시 장애와 사용 한도·계정 제한을 확정할 수 없음을 명시했다. 기존 좁은 오류 분류와 최대2회 재시도, 원본 보존·재개 정책은 유지한다.
3. 상태200/READY를 성공으로 오인하지 않기, OpenSession 실패 후 SQL 미실행, 가짜SELECT결과 거부, 인증 오류 구분, 로그 비밀정보 제외와 기존 복구/재개를 검증했다.
4. 관련19개 unittest PASS. 최초 pytest 실행은 해당 런타임에 pytest가 없어 실패했고, 프로젝트 CI와 동일한 unittest로 실행했다. 실패 시도도 `recovery_tests.txt`에 보존한다.

재검사 명령(서비스 복구/사용량 확인 뒤 1회 실행):

```sh
.telly_runtime/v1-venv/bin/python scripts/probe_databricks_availability.py --output /tmp/telly-provider-check.json
```

## 다음 단계

- Databricks 화면의 실제 사용량/한도/계정 제한 안내 확인. 한도 소진이면 리셋 후 재검사하거나 적절한 별도 평가 환경을 준비해야 한다. 알려지지 않은 리셋 시각을 추측하지 않는다.
- 사전 점검 성공 후 interactive Spider 고정 사례 및 실제 latest-row SQL/oracle 평가를 이어간다. 현재 반복 실행하면 같은 외부 차단을 재현할 뿐이므로 추가 대량 평가는 실행하지 않았다.
- agent 잔여4묶음(일반 최신행 EDA, 고급 SQL, 대규모 전체 여정, 독립 oracle98)은 여전히 미완료. 이번 외부 차단과 기존 agent 품질 부족을 분리하며 범용 NO-GO를 유지한다.

최종 제품 전체 **467/467 PASS (55.303초)**. 실제 공급자 점검은 **BLOCKED/exit1**이며 테스트 통과로 이를 덮어쓰지 않았다.
