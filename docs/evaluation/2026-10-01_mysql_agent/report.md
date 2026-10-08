# 로컬 MySQL 분석 agent 연결 및 실행

2026-10-01. `TELLY_DATA_BACKEND=mysql`을 지정하면 동일한 LangGraph 분석 runtime에 로컬 MySQL 읽기 전용 실행기를 연결한다. 기본값은 `databricks`이고 기존 Databricks 실행 경로를 유지한다. MySQL 옵션 파일은 `TELLY_MYSQL_OPTION_FILE`(기본 `.telly_runtime/mysql_eval/reader.cnf`), DB는 `TELLY_MYSQL_DATABASE`(기본 `teleai_default`)로 선택한다. 비밀번호나 원본 데이터는 Git에 넣지 않는다.

MySQL 모드는 실행 시 `information_schema.columns`에서 실제 테이블·컬럼을 읽는다. SQL은 MySQL dialect의 SELECT AST·출처 일치·선택한 DB 범위로 검증하고, 결과는 기존 승인 장부와 제한된 배치 저장소를 통과한다. 서버 SELECT 실행 시간은 세션당 최대 120초이며 클라이언트 읽기 시간도 제한한다. 0행 스키마 조회는 실제 타입의 빈 결과로 저장하며, 행 한도 초과는 `truncated`로 기록한다. `mysql.user` 같은 외부 DB와 쓰기 SQL은 차단했다. UI는 MySQL 데이터 소스를 표시하고, 저장 경로를 연결 지문별 `mysql_eval/<fingerprint>`로 분리해 Databricks 체크포인트/데이터와 혼용하지 않는다. 공유 agent 도구 이름 `query_databricks`는 기존 체크포인트 호환을 위해 유지하지만 MySQL 모드에서는 MySQL 읽기 전용 실행기를 호출한다.

실제 MySQL 평가 DB에서 확인한 결과:

| 확인 | 결과 |
| --- | --- |
| 동적 메타데이터 탐색 | 5개 테이블 확인 |
| 자연어 `teleai_default에는 어떤 테이블이 있지?` | MySQL `information_schema.tables`의 실제 5개 목록 반환 |
| 자연어 `bank_loan에는 어떤 컬럼이 있어?` | 관측된 18개 컬럼 반환 |
| 자연어 `bank_loan age의 histogram을 보여줘` | MySQL 집계 78행, 빈도 합계 750,000, PNG 차트 11,321바이트 저장 |
| 화면의 `error_test` 데이터 불러오기 | 9행·5열의 실제 MySQL 결과 저장 |
| 안전장치 | DELETE, 다른 DB 접근 거절; 2행 한도 결과는 `truncated` |
| Streamlit | 별도 `127.0.0.1:8504` 서버 실행, `/_stcore/health` 응답 `ok` |

검증: `TELLY_TEST_MYSQL=1 ... pytest tests/test_mysql_backend_live.py` 2개 PASS. MySQL 통합과 기존 분석 경계/선택/스키마/히스토그램을 함께 실행한 37개 PASS, 15개 subtest PASS. Databricks 과거 승인·완료 테스트 두 파일은 현재 브랜치에서 18 FAIL/13 PASS다. 변경 전 커밋에서 동일 두 파일은 20 FAIL/11 PASS였으므로 이번 변경만의 새 회귀로 분류하지 않으며, 오래된 기대 계약 정합화는 남아 있다.

실행 명령:

```bash
python -m pip install -r requirements-mysql-eval.txt
TELLY_DATA_BACKEND=mysql .telly_runtime/v1-venv/bin/python scripts/run_telly.py --port 8504 --server.headless=true
```

이 호스트에서는 두 번째 줄로 서버를 기동했고, 기존 `8501` Databricks 세션은 유지했다. 이 연결은 MySQL 환경에서 기본 테이블 탐색·스키마·집계·히스토그램 경로가 동작함을 보인 것이다. Databricks 전용 고급 metadata/SQL 계획 도구의 MySQL 동등성, 실제 모델의 폭넓은 대화 평가와 MySQL→Databricks 교차 검증은 아직 완료되지 않았다. MySQL 성공을 범용 출시 GO로 해석하지 않는다.
