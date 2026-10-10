# Databricks konlo 시나리오 #1 실제 브라우저 평가

- 환경: Databricks / 로컬 Ollama qwen3:8b / 8504 / 제품 e29c5bb.
- 데이터 backend는 서버 프로세스의 TELLY_DATA_BACKEND=databricks로 전환했다. 기존 .env는 변경하지 않았다.
- 고정된 원문31개를 같은 새 대화에 순차 입력31회. 동일 요청 재제출이나 수동 상태/조건 보정 없음.
- 결과: 1 PASS / 30 FAIL, 요청 통과율3.23%. 범용 운영 판정 NO-GO.
- 이는 해당 사용자 여정의 점수이며 DeepEval/Spider 공식 점수는 아니다.

## 확인한 사실

실제 Databricks 목록 조회는 성공했다. 기본 schema에는5개 테이블이 있으며 에이전트는 catalog 전체의37개 목록을 표시했다. 독립 SQL에서 bank_loan 750,000행, education=primary AND age30..40 27,439행, primary/secondary AND age30..40 208,699행을 확인했다. 전체 산점도 고유좌표132,613개의 빈도합750,000도 확보했다.

두 번째 요청부터 모두 ModelContextBudgetExceeded로 모델 호출 전에 중단됐다. 최초 실패의 run_id=b77a75c7e4314cb38ca6530efe2197ce, 오류ID=2cfbd4661091, elapsed=0.021초, SQL실행0회. 모델 응답 오류나 Databricks 응답 실패로 계산하지 않는다.

모델 입력은 시스템6,661바이트·메시지7,142바이트, 전체직렬화14,711바이트와추가여유640단위다. 출력2,048을예약한16384문맥의 입력한도14,336단위를 넘는다. **UTF8바이트를 사용한 보수적 추정이며 실제 토큰 측정값이 아니다.**

목록 조회 이후 의도 판독 입력이 테이블32개를 포함한다. 독립 재구성에서 테이블 목록 부분만4,667바이트였다. 관련 없는 information_schema 테이블과 각 테이블의 반복적 columns/column_count/more_columns_tool 필드를 넘기는 구조가 크기를 늘렸다. 일반 모델 budget middleware의 도구 관찰 축약은 의도 판독의 HumanMessage JSON을 줄이지 못한다. 실패한 의도가 checkpoint에 확정되기 전이라 마지막 완료목표가 첫 목록 조회로 남고, 다음 요청에서도 동일한 큰 catalog 입력이 반복된다.

## 해석과 다음 수정 범위

30개가 서로 다른 SQL/시각화 구현 오류라는 뜻은 아니다. 공통 의도 판독 입력 한도 때문에 이후 기능을 평가할 수 없었다. 관련 테이블의 스키마와 현재 목표·조건을 보존하면서 이름 목록/반복 안내/불필요한 스키마를 모델용 뷰에서 축약하고, 호출 전 JSON 입력 압축과 실패 요청의 맥락 보존을 함께 검증해야 한다. 입력한도 증가만으로 해결했다고 판단하지 않는다.

현재 Databricks에는 시나리오의 MySQL 전용 출처 _tianchi_ssd_shared300k_import, tianchi_ssd_shared300k_raw 및 teleai_default namespace가 없다. 원문은 바꾸지 않았고 해당 요청은 모두 선행 입력차단에서 끝났으므로 출처 부재 대응의 성공/실패는 평가하지 않았다. 후속 검증에서는 정확한 부재 안내와 다른 출처 무단대체 금지를 별도로 판단해야 한다.

이번 실행에서는 기능 코드 수정이나 예산 우회·재제출을 하지 않아 실패 baseline을 보존했다. 회사 Windows/Azure API 및 전체 운영 검증은 범위 밖이다. 첫 목록 자산1개와 선택 상태는 후속30요청 이후에도 변경되지 않았다.

## 요청별 결과

| 번호 | 요청 | 결과 | 원인 |
|---:|---|---|---|
| 1 | 지금 볼 수있는 table를 보여줘 | PASS | 실제 테이블 목록과 독립 DB 대조 |
| 2 | bank_loan 컬럼들을 보여줘 | FAIL | ModelContextBudgetExceeded |
| 3 | age 의 histogram 그려줘 | FAIL | ModelContextBudgetExceeded |
| 4 | education은 어떤 값들로 되어 있지 ? | FAIL | ModelContextBudgetExceeded |
| 5 | histogram을 그려줘 | FAIL | ModelContextBudgetExceeded |
| 6 | education이 primary 이고 age가 30~40인 사람들을 시각화 해줘 | FAIL | ModelContextBudgetExceeded |
| 7 | education이 primary과 secondary 이고 age가 30~40인 사람들을 시각화 해줘 | FAIL | ModelContextBudgetExceeded |
| 8 | 같은 조건을 유지해서 histogram을 다시 보여줘 | FAIL | ModelContextBudgetExceeded |
| 9 | legend를 넣어서 primary과 secondary 에 따라서 색을 좀 넣어줘, | FAIL | ModelContextBudgetExceeded |
| 10 | 색 구별이 안 되고 있어 | FAIL | ModelContextBudgetExceeded |
| 11 | education별 색상 구분과 legend를 넣어줘 | FAIL | ModelContextBudgetExceeded |
| 12 | 하나의 stack으로 legend를 포함해서 다시 그려줘 | FAIL | ModelContextBudgetExceeded |
| 13 | stackbin으로 다시 그려줘 | FAIL | ModelContextBudgetExceeded |
| 14 | 이거 legend를 넣어줘 | FAIL | ModelContextBudgetExceeded |
| 15 | 테이블의 column 다시 보여줘 | FAIL | ModelContextBudgetExceeded |
| 16 | 데이타 10개 row만 추출해서 table로 보여줘 | FAIL | ModelContextBudgetExceeded |
| 17 | age를 x축으로 하고 balance를 Y축으로 하는 scatter plot를 그려줘 | FAIL | ModelContextBudgetExceeded |
| 18 | age와 balance의 관계를 그려줘봐 | FAIL | ModelContextBudgetExceeded |
| 19 | 전체 데이타에 대해서 age와 balance scatter plot으로 그려줘 | FAIL | ModelContextBudgetExceeded |
| 20 | age와 balance 의 scatter plot으로 그려줘 | FAIL | ModelContextBudgetExceeded |
| 21 | bank_loan 데이터를 10개 row로 보여줘 | FAIL | ModelContextBudgetExceeded |
| 22 | 방금 보여준 10행 데이터의 age를 x축, balance를 y축으로 scatter plot을 그려줘 | FAIL | ModelContextBudgetExceeded |
| 23 | 전체 데이타를 이용해서 age와 balance의 관계를 다시 scatter plot으로 그려줘 | FAIL | ModelContextBudgetExceeded |
| 24 | 내가 볼 수 있는 table 다시 한번 보여줘 | FAIL | ModelContextBudgetExceeded |
| 25 | _tianchi_ssd_shared300k_import  row 수가 몇개야 ? | FAIL | ModelContextBudgetExceeded |
| 26 | table 다시 한번 보여줘 | FAIL | ModelContextBudgetExceeded |
| 27 | tianchi_ssd_shared300k_raw 이 table row 수가 몇개야 ? | FAIL | ModelContextBudgetExceeded |
| 28 | 이 테이블의 column들을 보여줘 | FAIL | ModelContextBudgetExceeded |
| 29 | teleai_default.bank_loan 전체 데이타를 이용해서 age와 balance의 관계를 다시 scatter plot으로 그려줘 | FAIL | ModelContextBudgetExceeded |
| 30 | table column를보여줘 | FAIL | ModelContextBudgetExceeded |
| 31 | teleai_default.bank_loan 전체 데이터의 age와 balance scatter plot을 다시 그려줘 | FAIL | ModelContextBudgetExceeded |

실행별 진단: NN_receipt.json, 첫 목록 독립검증:01_check.json, 현재 DB 정답:oracles.json, 입력구성:input_diagnosis.json, 최종화면:31_final.png. 일부 초기UI 스냅샷은 렌더링 시점 이전 오류표시가 남아 있어 판정은 transcript 요청ID와 고유 run_id의 terminal event를 대조했다.31개의 고유 요청과 실행 완료 기록을 검증했다.
