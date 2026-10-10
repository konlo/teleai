# 입력 문맥·전체 산점도·후속 요청 수정

2026-10-04. 실제 검증 환경: localhost:8504, MySQL, Ollama gemma4:e4b. `.env`의 기본 Databricks 선택은 변경하지 않았다. 이번 수정은 아래 검증 범위에 대한 완료이며 범용 agent GO 판정은 아니다.

## 원인과 수정

1. 기존 전체 산점도 실패는 SQL 제출 전 발생했다. Ollama 입력 16,083/16,310토큰이 문맥 16,384를 거의 채웠다. prefill에 54~58초가 걸렸고, 재시도 응답은 301토큰에서 문맥 끝으로 잘렸다. 종전 문자수 검사는 제공하는 도구 schema와 출력 예약 공간을 포함하지 않았다.
2. `model_context.py`는 기본 지침/catalog을 간결하게 제공하고, 단계에 필요한 도구를 노출하며 검색한 등록 도구를 다음 호출에 추가한다. 실행기 권한·전체 transcript·현재 요청의 tool-call/observation 쌍은 유지한다. 모델 view만 압축한다. 분석 호출은 도구 정의·메시지·시스템 지침의 UTF-8 바이트와 template 여유를 보수적 예산 단위로 측정하고, Ollama 출력 4,096토큰을 예약한다. 이 측정값은 실제 토큰 수가 아니며 로그에 구분한다. 실제 usage/prefill도 별도 기록한다. 사용자 지정 지침을 기본 지침으로 바꾸지 않는다.
3. `source_scatter.py`는 명시적인 전체 범위·단일 출처·두 수치 축·무필터 기본 scatter에서 최신 관측 schema로 정확한 좌표별 COUNT SQL을 만든다. 중복 좌표만 빈도로 압축하며 표본이나 binning을 사용하지 않는다. 동일 문장의 완료 receipt, 동일 연결/출처/스냅샷/결과 메타데이터, 완전성·고유 좌표·양의 정수 빈도·PNG를 검증한다. 빈도는 색상 범례로 표시하고 NULL 축 제외를 설명한다. 원본/기존 선택은 유지한다.
4. 일반 원격 결과 한도 100,000행은 유지한다. 정확한 전용 좌표 SQL만 `TELLY_MAX_SCATTER_COORDINATES=250000`을 적용하며 양 adapter의 batch fetch와 잘림 판정도 같은 한도를 사용한다. byte/disk 한도도 적용한다. 잘린 결과는 전체 차트로 완료하지 않는다. 완전한 보유 raw snapshot을 먼저 사용하며, 저장된 좌표/검증된 동일 PNG도 재사용한다. 사용자 지정 스타일/회귀선 등은 이 기본 controller 경로에서 제외한다.
5. 실사용 중 추가로 확인한 `row 수` 표현은 전체 행 수 intent로 인정하되, COUNT(column)/DISTINCT/count+1/그룹 건수를 COUNT(*) 결과로 오인하지 않는다. 과거 완료된 SQL receipt를 재검증하여 재조회 없이 복구한다. `column들을`, `column를`의 한국어 조사를 테이블 목록 분기가 잘못 처리한 부분도 수정했다.
6. opt-in 실제 MySQL 검사에서 새로 관측한 관리 테이블은 조건을 만족하는 최신행이 0개였다. 비어 있는 histogram bin을 합계 불일치 예외로 처리하지 않고 `latest_empty`를 반환하도록 보강했다. 동적인 DB 검사에서 NULL/동률/0행을 정상 정책 결과로 구분하며, 이 경우를 차트 성공으로 계산하지 않는다.

## 실제 증거

| 검증 | 결과 |
|---|---|
| 별도 SQL oracle | 유효 원본 750,000행, 정확한 좌표 132,613개 |
| 웹 전체 scatter | 9.919초, SQL 1회, 분석 LLM 0회, 실제 PNG/색상 범례 생성, oracle 일치 |
| 서버 재시작 후 웹 재요청 | 저장된 좌표 재사용, 추가 SQL/LLM 0회, 9.571초에 PNG 재생성 |
| 최종 동일 PNG 재사용 | 실제 저장소의 격리 복사본으로 재시작 검증. 별도 `cached_chart_restart.json`에 측정값; 추가 SQL/LLM/차트 asset 생성 없음 |
| 원본 보존 | 오늘 초기 기존 17 assets의 metadata/payload SHA-256 모두 동일. 반복 직전 모든 assets와 분석 기준 선택도 동일 |
| 실제 모델 행 수 조회 | 입력 3,331토큰, 출력 278토큰, prefill 11.718초, 모델 22.166초, 정상 종료/tool-call 1회. **산점도와 다른 요청**이므로 같은 프롬프트의 속도 비교로 해석하지 않는다 |
| 해당 COUNT DB 실행 | `tianchi_ssd_shared300k_raw` 실제 182,673,242행. SQL은 83.871초에 완료. 이 DB 지연은 이전 산점도의 SQL 제출 전 LLM timeout과 다르다 |
| 완료 COUNT 재개 | 이미 받은 결과를 0.009초에 표시, 새 SQL/LLM 없음 |
| `table column를보여줘` 웹 | 18개 컬럼, 0.281초, SQL/LLM 0회 |

최초 입력 예산 보호가 보수적으로 과도한 메뉴를 차단한 중간 실패, 실제 COUNT를 받은 후 완료하지 못한 중간 실패도 `web_evidence.json`에 남겼다. 중간 오류를 성공으로 지우지 않았다. 수정 후 각각 완료 결과를 실제 화면에서 확인했다.

## 자동 검사와 범위

- 전체 `tests migration`: **716 PASS / 0 FAIL / 4 SKIP**, 496 subtests, 105.84초. 원자료: `regression.txt`. 마지막 PNG 재사용 보강 전의 전체 검사이며, 이후 관련 source scatter/chart journey/model context 14 PASS/10 subtests를 별도로 확인했다.
- MySQL opt-in 포함 신규 관련 검사: 16 PASS/4 subtests. 전체 검사의 MySQL 4 SKIP은 별도 실제 DB 검사로 보완했다.
- metadata/row count/context/remote completion 관련: 30 PASS/17 subtests.
- 양 dialect의 source query 및 scope/한도, 잘림 거절, NULL/중복 oracle, 원본/선택 보존, 재시작/failed model checkpoint 복구, 출력 공간/현재 요청·도구 쌍 보존, 도구 검색에 의한 capability 추가, 잘못된 COUNT 거절을 검증했다.
- Databricks 전체 source scatter의 새 live warehouse 실행은 이번에 재검증하지 않았다. MySQL 실측과 Databricks 문법/adapter 회귀를 운영 host 성공으로 확대하지 않는다.

## 남은 범위

필터/여러 출처/사용자 지정 시각화, 좌표 25만 개를 넘는 전체 산점도의 다른 실행 전략, 요약/semantic 보조 호출의 문맥 예산, 엄격한 전체 wall-clock deadline, 새로운 자유계획·held-out/Spider 평가 및 회사 Databricks 운영 smoke는 미완료다. 대형 COUNT의 정확성 비용을 추정값으로 바꾸지 않았다. 커밋/push하지 않았으며 8504 서버는 현재 코드로 유지한다.
