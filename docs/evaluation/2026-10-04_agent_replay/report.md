# 2026-10-04 data agent 실행 및 어제 여정 재검증

8504 서버가 꺼져 연결 오류가 표시돼 MySQL 평가 모드로 다시 시작했다. 기존 conversation 0728dee0-d555-4aa2-853b-1101679346da 및 저장 asset을 복원했다. MySQL 서비스와 Ollama API는 가동 상태였다. 기본 Databricks 설정은 변경하지 않았다.

## 실제 브라우저 결과

| 요청 | 결과 | 실행 시간 | 분석용 추가 SQL | 모델 본호출 |
|---|---|---:|---:|---:|
| 테이블의 column 다시 보여줘 | 실제 컬럼18개 | 0.352초 | 0 | 0 |
| bank_loan 데이터를10개 row로 보여줘 | 10행18열 표 | 0.259초 | 0 | 0 |
| age x축/balance y축 scatter | 10개 좌표 이미지 | 8.607초 | 0 | 0 |
| 전체 데이터 age/balance scatter | 미완료·ReadTimeout | 207.789초 | 0 | 2 |
| 실패 뒤 방금 보여준10행 scatter | 이미지 및 새 요청 처리 | 0.317초 | 0 | 0 |

연결 preflight/현재 schema 탐색의 SQL은 위 분석 도구 제출 횟수와 별도다. 성공한 일반 여정들은 저장 데이터 및 결정적 도구를 사용한 것으로, 모델의 자유 계획 성공률로 계산하지 않는다. 이번은 실제 웹 실행 검증이며 제품 코드 변경이나 전체 pytest 재실행은 하지 않았다. 어제704PASS/4SKIP 결과와 이번 결과는 구분한다.

## 전체 요청 실패 근거

실행fa670b17b2574a6e99942069d4266cdb/요청11c7ee62-92e3-46ce-b191-5f03532dee53/오류c3a097b5d105. 대화 요약31.884초 성공 후 모델이102.093초에 render_chart_spec을 선택했으나, dataset_id에 로딩 asset ID 대신 테이블 이름 `teleai_default.bank_loan`을 전달했다. agent가 dataset_not_loaded로 거절했다. 추가 요약13.497초 뒤 복구 모델 호출이60.022초 ReadTimeout으로 중단됐다. SQL 제출/전체 출처 로딩/새 전체 차트는0이고 로딩 계획은None이었다.

원인에 모델의 잘못된 도구 인자와 전체 출처 계획/복구 부재가 함께 포함된다. 이번은 Databricks SQL 실패가 아니다. 전체207.789초로180초 turn SLO를 넘긴 것도 기록했다. HTTP read timeout은 스트리밍 전체 wall-clock deadline과 다르며 현재 SLO가 진행 중 호출을 정확히 중단하는지는 후속 보강 필요하다.

실패 뒤 다른 10행 요청을 입력했을 때 이전 요청을 cancelled로 종료하고 마지막 확정 맥락에서 성공했다. 원래17개 asset의 metadata/payload SHA256가 모두 같고 선택도 같다. 데이터8개 유지, 차트9개→11개(오늘 정상 산점도2개 추가), 최신 recovery complete. [실행 근거](web.json), [사용 가능한 현재 화면](final_web.png), [전체 요청 실패 화면](full_scatter_failure.png).

## 남은 일

전체 출처의 schema/두 축 투영/행수·byte/coverage/로딩 한도 계획과 미적재 데이터 복구를 구현·평가해야 한다. 모델 지연 및 엄격한 요청 전체 deadline도 함께 확인해야 한다. 서버는 실행 상태로 남겼고 일반 테스트는 이어갈 수 있다. 전체 데이터 산점도 또는 범용 agent GO 완료로 보고하지 않는다. 회사 Databricks 접속/데이터 검증과 push는 수행하지 않았다.
