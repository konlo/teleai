# 스키마 타입 후속 질문의 출처 오류

## 판정과 원래 실패

사용자의 현재 대화 대상은 `teleai_default.alibaba_ssd`다. `table schema에 data type이 있지 않아 ?`의 올바른 동작은 해당 테이블의 스키마 타입을 확인하는 것이다. 그러나 run `d3c41b0967f14d3a81290eb4956f2f7d`는 47.374초 동안 모델2회와 inspect_dataset1회를 수행해 과거에 선택된 bank_loan 집계 프레임의 age/__frequency int64를 답했다. 원본 테이블 타입 답변이 아니며 잘못된 출처의 완료였다. DB 오류/입력 한도 오류는 아니다.

원인 세 가지가 겹쳤다.

- 타입 목적 판별이 column/필드 표현을 요구해 table schema 질문을 놓쳤다. `type이`처럼 영어에 한국어 조사가 붙으면 Unicode 단어 경계도 매칭되지 않았다.
- 직전 의미 질문은 자유 답변 계약이어서 inspect_table_context의 실제 alibaba_ssd 관찰이 confirmed_analysis에 갱신되지 않았다. 과거 bank_loan 확정 맥락과 선택된 집계 결과가 남았다.
- 자유 답변으로 처리된 요청은 스키마 증거 완료 계약을 요구하지 않았다. 모델이 과거 선택 결과를 inspect_dataset으로 읽고 그럴듯하게 답하면 출처 오류를 완료로 인정했다.

## 수정

schema_questions.py에서 테이블/스키마 타입 질문을 column 단어 없이도 인식한다. 영어에 한국어 조사가 붙는 표현과 생략형 질문도 처리한다. 계산/차트의 정상 목적 분기는 유지한다.

실행된 inspect_table_context의 도구 인자와 ready 관찰이 일치할 때만 schema_subject를 저장한다. 대화의 스키마 대상과 선택된 분석 dataset ID를 분리한다. 옛 checkpoint는 매칭된 tool call/result에서만 대상을 복원하며 assistant의 자유 문장은 증거로 사용하지 않는다. 프로세스 재시작 후에도 대상이 유지된다. 명시적 UI 선택 또는 새 완료 분석은 과거 schema_subject보다 우선한다.

타입 질문은 기존 metadata 완료 계약을 통해 대상 테이블의 컬럼별 타입 증거를 요구한다. 다른 테이블의 schema inspection/선택 dataset 조회는 실행 전 범위 검사로 거절한다. 집계 프레임만으로 테이블 스키마 완료를 채울 수 없다. 모델 입력에서도 보유 분석 기준과 대화의 확인된 스키마 대상을 구분했다.

DB SQL 타입과 DataFrame storage dtype도 분리했다. 실제 DB 관측 메타데이터가 fresh이고 로딩 결과 이후 관측됐으며 컬럼 이름이 정확히 일치할 때만 database_dtype를 제공한다. 오래된 학습 프로필/타입 추정/불일치 스키마로 SQL 타입을 만들지 않는다. SQL varchar와 pandas object의 표현 차이는 스키마 변경 증거로 안내하지 않는다. 원본·선택·영속 payload는 수정하지 않는다.

## 실제 웹/DB 검증

기존 사용자 대화에서 다음을 실행했다.

| 요청 | 실행 ID | 시간 | 결과 |
|---|---|---:|---|
| 원래 table schema에 data type이 있지 않아 ? | 681ad54707bf4044875379975b451d11 | 0.522초 | alibaba_ssd107컬럼, 모델0/분석SQL0 |
| 이 테이블 컬럼별 data type 보여줘 | 4eb6d9e287894365b106f3ec93727a96 | 0.538초 | 같은 테이블 유지, 모델0/분석SQL0 |
| 최종 타입 보강 후 서버 재시작·원래 질문 반복 | 9999bc1c5a7b4ff7946dd0e62891571d | 0.512초 | 같은 테이블 유지, SQL/Frame 표현 차이의 변경 오안내 없음 |

여기서 분석SQL0은 원본 데이터 재로딩/집계 쿼리0을 뜻한다. MySQL context loader는 정보 스키마를 읽으며 메타데이터 SQL0이라고 주장하지 않는다. 독립 정보 스키마 SELECT의107개 이름/타입이 답변의107개와 모두 일치했다(oracle.json). 현재 로컬 MySQL에서 ds/disk_id/__source_file은 varchar, __source_row_number는 int다. VARCHAR 타입만으로 시간 의미가 없다고 단정하지 않으며 날짜 의미 검증은 별도 목표다.

기존31assets metadata/payload SHA256과 선택이 모두 동일하고 신규asset0이다(before.json/validation.json). 작성 중이던 입력도 보존·복원했다. 화면은 final_web.png, 실행/확정 대상 근거는 web_execution.json에 저장했다.

## 자동 검증과 한계

- 전체 pytest tests migration: **732 PASS/0FAIL/4SKIP/500subtests,106.17초**. SKIP은 환경 opt-in이며 이번 실웹 검증과 구분한다.
- 마지막 소규모 보강(타입 체계 차이를 drift로 안내하지 않기/새 완료 분석의 대상 우선): 관련 **52 PASS/17subtests,12.15초**. 마지막 전체 이후 변경의 검증 범위를 구분한다.
- 회귀는 양 backend metadata 계약·semantic→타입→재시작→UI 선택·명시 출처 변경, 과거 선택 집계 보존, legacy tool 증거와 위조 assistant 문장, 잘못된 출처 실행 차단, DB 타입 vs object/오래된·불일치 metadata를 포함한다.
- 초기 신규 검사3FAIL(영어+한국어 조사 경계/fixture snapshot 누락), 그다음1FAIL(컬럼별 생략형)도 확인하고 수정했다. 최초 실패를 최종 통과로 감추지 않는다.

테스트 개수를 범용 agent GO로 환산하지 않는다. 시간 컬럼 후보의 근거 없는 의미 추정, 일반 자유 답변의 증거 검증, 넓은 스키마 페이지 탐색·보조 모델 예산과 운영 Databricks 실검증은 추가 범위다. 현재 MySQL8504 개발 서버에서 테스트를 계속할 수 있다. .env 기본 Databricks는 변경하지 않았다. commit/push하지 않았다.
