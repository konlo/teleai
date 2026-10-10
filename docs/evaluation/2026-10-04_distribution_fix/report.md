# 분포 요청·맥락·입력 예산 복구 검증

## 결과

실제 8504/MySQL 대화에서 `n_1 column의 데이타 분포를 보여줘`가 전체 값별 빈도 PNG로 완료됐다. 후속 `y축을 10000으로해줘`도 같은 저장 집계를 사용해 실제 Y축 0~10,000 PNG로 완료됐다. 기존 28개 asset의 metadata/payload SHA256 및 선택 상태는 모두 동일하다. 최종 코드의 `pytest tests migration -q`는 725 PASS, 0 FAIL, 4 SKIP, 500 subtests(106.42초)다. SKIP은 별도 환경을 요구하는 opt-in 테스트이며 이번 전체 회귀에서 실행하지 않았다. 이번 수정은 해당 분포·후속 축 변경 여정의 검증이며 범용 agent/Databricks 운영 GO 판정이 아니다.

## 원인과 수정

1. `column`이라는 명사가 분포를 보여달라는 목적보다 먼저 처리돼 컬럼 조회로 분류됐다. 일반 분포 요청을 chart intent로 인식하되 표로 출력/개념 설명은 제외한다.
2. 직전 `alibaba_ssd` 10행 미리보기와 확인된 분석 scope의 `bank_loan` 출처가 달랐다. 저장된 미리보기의 dataset/source/snapshot/컬럼 증거를 검증해 대화 출처를 연결한다. 기존 선택을 변경하지 않으며 다른 테이블의 필터를 이식하지 않는다. 불명확한 조건은 확인 대상으로 유지한다.
3. 여러 과거 dataset의 넓은 스키마 및 확인 증거가 모델 입력에 중복됐다. 관련 컬럼만 제공하고 무관한 결과는 identity로 축소한다. 넓은 미리보기는 모델 뷰만 5행/기본 8필드로 투영하고 저장 결과·scope·receipt·도구 call ID는 보존한다. 검증된 분석 맥락이 있으면 별도 모델 요약 대신 예산 검사에서 이전 턴을 압축한다. 일반 설명의 요약 동작은 유지한다.
4. 입력 검사는 UTF-8 byte 기반 보수 단위다. 출력 4,096/문맥 16,384 설정은 변경하지 않았다. 실제 tokenizer 측정이라고 주장하지 않으며 현재 요청/사용자 조건/custom instruction을 잘라 호출하지 않는다.
5. 복구된 분포 SQL의 최초 대형 스캔도 실패했다. MySQL 3024/1317은 서버가 명시한 종료로 기록하고 errno/error_code를 진단에 남긴다. 연결 오류로 실행 결과가 불확실한 경우는 계속 unknown으로 유지한다. 개발 환경 쿼리 제한은 1~600초의 명시적 설정으로 분리했다(기본 120). Databricks adapter 및 `.env`의 기본 backend는 변경하지 않았다.
6. 새 후속 축 변경 요청이 모델 예산 검사에 걸린 것도 추가 확인했다. 검증된 직전 COUNT 빈도 차트의 단순 Y축 상한 변경은 실제 데이터셋·분석 범위를 이어서 등록된 렌더링 도구로 수행한다. 빈도를 다시 세거나 DB를 다시 읽지 않는다. 요청 범위가 실제 PNG `y_limits`와 일치해야 완료하며, 상한보다 큰 막대가 잘려 보인다는 설명을 표시한다. 현재 이 직접 경로는 그룹 없는 값별 COUNT 분포에 한정된다.

## 실제 데이터·실행 근거

- 출처: `teleai_default.alibaba_ssd`, `n_1`은 실제 VARCHAR(64) NOT NULL. 임의 수치 변환 대신 범주별 COUNT(*) 막대 분포를 사용했다.
- SQL: `SELECT n_1, COUNT(*) AS __frequency FROM teleai_default.alibaba_ssd WHERE n_1 IS NOT NULL GROUP BY n_1`.
- 결과 28행, 값 고유성 확인, 빈도 합계 **182,673,242**. 원본 1.8억 행을 클라이언트에 펼치지 않았다. NULL 제외 정책이며 빈 문자열 85,020,499건과 문자열 `\\N`은 실제 범주로 유지했다. `100`/`100.0`도 원래 문자열대로 별도 범주다.
- 저장 데이터셋: `05f10df7-ff49-4c63-b633-a550ef2013b5`.
- 원래 분포 차트: `1142ee20-3c4c-4a89-98da-352299cd55dc`, 웹 완료 run `94770d24e7f14c8d82259fe055916a71`(7.998초). 이미 저장된 SQL receipt를 재사용해 렌더링했으며 이 재개에는 모델·SQL 0회다.
- 같은 실제 대화를 격리 복사한 새 runtime에서 동일 분포 반복: answered(0.464초), 모델·SQL 0회, 같은 차트 ID 재사용(`cached_repeat.json`). 원래 대화에는 이 반복을 추가하지 않았다.
- 실제 사용자 Y축 요청 오류 `f92d6702f5e8`: 압축 후에도 payload 12,153 byte + headroom 1,216 > 입력 예산 12,288로 모델 HTTP 전에 차단. 마지막 수정 후 실제 웹 재개 run `9efdb72d119f428eb884629612e148fc`: answered(8.137초), render_histogram 1회, 모델·SQL 0회. 차트 `e41a2126-58b0-4088-b8d5-c634cc9870c4`, y_limits=[0,10000], 실제 빈도 불변. 별도 restart 격리 검증도 통과(`axis_restart.json`).

## 실패 기록과 제한

첫 SQL run `b963f5a0e3b247fbb67b973f9cbe8240`는 약 120초 후 DatabaseError로 실패했다. 당시 adapter는 errno를 기록하지 않아 이 실패를 3024라고 확정할 수 없다. 실제 실행 connection ID 246이 종료된 것을 확인한 뒤 unknown ledger를 실패로 정리한 **1회 수동 조정**을 `terminal_reconciliation.json`에 남겼다. 성공 결과를 만들어 넣거나 자동 재조회하지 않았다. 새 종료 분류는 명시적 errno만 사용한다.

이후 SQL run `8d97dd24c03847339f1bcafea2482b50`은 완전한 28행을 저장했으나 후속 단계가 완료되지 않아 저장 receipt로 재개했다. 시작/종료 로그의 wall-clock 차이는 925.008초다. 호스트 시간/일시 정지 영향 여부는 확인하지 못했으므로 이를 DB CPU 실행 시간이나 600초 wall-clock 보장으로 해석하지 않는다. 약 61GB 테이블 최초 집계의 스캔 비용, 엄격한 요청 전체 deadline, 범용 복합 EDA/held-out 및 실제 Databricks 검증은 남은 과제다.

## 회귀·화면·보존 증거

- `regression_final.txt`: 마지막 기능 변경 후 전체 회귀 725 PASS/4 SKIP/500 subtests. 폰트 경고 등을 포함한 원문 로그를 보존했다.
- `axis_tests.txt`: 관련 14 PASS/4 subtests. 재시작·빈도 동일성·축 범위·원본/선택 보존 검증.
- `regression_prior.txt`, `recheck.txt`: 앞선 724 PASS 전체 검사와 17 PASS 관련 재검사. 그 중 일회 AppTest 초기 로딩 제한 실패는 별도 `recheck_initial.txt`에 보존하고 재검사 성공과 구분했다.
- `regression_initial.txt`: 최초 전체 검사에서 발견한 5개 회귀 실패를 보존했다. 표 요청 intent 및 일반 요약 유지 조건을 수정한 후 전체 재검사했다.
- `before.json`, `validation.json`: 기존 28개 asset 해시/선택 불변.
- `count_validation.json`, `completed_receipt.json`: 실제 집계·저장 receipt.
- `final_web.png`: 실제 웹에서 완료된 Y축 수정 차트와 원본 빈도·잘림 설명 표시.
- 개발 서버는 8504/MySQL(기본 SQL 설정은 Databricks)로 유지했다. 커밋/push하지 않았다.
