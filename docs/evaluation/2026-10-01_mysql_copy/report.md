# Databricks `default` → 로컬 MySQL 평가 DB 이관

실행일: 2026-10-01 (Asia/Seoul). 대상은 이 개발 호스트의 MySQL 8.4.11, 데이터베이스 `teleai_default`이다. 원본은 Databricks `workspace.default`이며, 사용자 요청의 `titianic`은 실제 테이블명 `titanic`으로 확인해 이관했다.

| 원본/대상 테이블 | 행 수 | 컬럼 수 | 검증 |
| --- | ---: | ---: | --- |
| `bank_loan` | 750,000 | 18 | 일치 |
| `error_test` | 9 | 5 | 일치 |
| `ncr_ride` | 150,000 | 21 | 일치 |
| `stormtrooper` | 9,524,806 | 13 | 일치 |
| `titanic` | 891 | 12 | 일치 |
| **합계** | **10,425,706** | | |

이관기는 Databricks Arrow 커서에서 20,000행씩 읽고 MySQL에 최대 1,000행씩 배치 삽입했다. 각 테이블은 임시 테이블에 적재한 뒤, 원본·대상의 **전체 행**에 대해 순서 독립적인 128비트 BLAKE2b 다중집합 지문(합 및 XOR)과 행 수를 비교했다. 원본의 이관 전후 `COUNT(*)`도 비교하고, 모두 일치한 뒤에만 최종 이름으로 게시했다. 별도의 MySQL 읽기 전용 연결에서 게시된 테이블의 정확한 `COUNT(*)`, 컬럼명·순서·매핑 타입·NULL 허용을 재확인했다. 남은 임시 테이블은 0개다. 상세 원본/대상 스키마와 지문은 Git에 포함되지 않는 `.telly_runtime/mysql_eval/migration_report.json`에 있다.

MySQL은 Homebrew `mysql@8.4` 서비스로 시작되고 `127.0.0.1:3306`에서만 수신한다. `teleai_eval` 계정은 `teleai_default`에 `SELECT` 권한만 갖는다. 관리자·읽기 전용 접속 설정은 Git에서 제외되는 `.telly_runtime/mysql_eval/root.cnf`, `reader.cnf`에 0600 권한으로 저장했다. `local_infile`은 꺼져 있고 서버 `secure_file_priv`는 `NULL`이다. 데이터 파일은 이 호스트의 Homebrew MySQL 데이터 디렉터리에 있으며 Git에 포함되지 않는다.

로컬 확인 명령:

```bash
brew services list | rg mysql@8.4
/opt/homebrew/opt/mysql@8.4/bin/mysql --defaults-extra-file=.telly_runtime/mysql_eval/reader.cnf -D teleai_default -e 'SHOW TABLES; SELECT COUNT(*) FROM bank_loan;'
```

이관 의존 패키지는 `requirements-mysql-eval.txt`에 분리했다. 이관기는 `python -m scripts.copy_databricks_default_to_mysql --tables <table>`로 실행한다. **이미 존재하는 최종 테이블은 덮어쓰지 않도록** 설계했으므로 재실행 전에 별도 재이관 절차가 필요하다.

이 결과는 이관 시점의 로컬 평가 복사본이다. Databricks에서 같은 행 수를 유지한 채 데이터가 수정되는 경우까지 고정된 트랜잭션 snapshot으로 보증하지는 않는다. 문자열은 `LONGTEXT`/`utf8mb4_bin`, timestamp는 UTC 값의 `DATETIME(6)`으로 보존해 값 비교를 통과했으나, DB별 collation·SQL 함수·시간대 의미는 별도로 평가해야 한다. 현재 **제품 chatbot의 실제 조회 경로는 여전히 Databricks**다. MySQL용 backend adapter와 agent 통합 평가는 후속 작업이며, 이번 이관만으로 출시 GO 판정을 갱신하지 않는다.
