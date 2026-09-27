# 실제 Databricks 적재와 agent 수치 EDA 복구

2026-09-27. 사용자는 이번 개발·검증 SQL을 추가 질문 없이 실행하도록 승인했다. 준비했던 `SELECT * FROM workspace.default.ncr_ride LIMIT 100000`을 1회 실행했다. 원본 100,000행·21컬럼이 파일로 저장됐다. 기존 사용자 대화의 데이터는 변경하지 않았다. 제품 전체 사용자의 승인 기본값을 바꾼 것은 아니다.

## 실제 실패와 공통 수정

1. **성공한 SQL을 실패로 오인**: 날짜가 포함된 preview가 LangChain에서 Python repr로 출력돼 JSON 해석에 실패했다. ledger는 completed였지만 graph는 미완료였다. [최초 기록](warehouse_actual.json)은 selected dataset만 조회한 당시 harness 결함으로 rows=null도 기록했다. 이는 적재 실패가 아니다. 타입별 JSON 정규화와 최초 실행/receipt 재생의 동일 계약으로 수정했다. 원본 Date/Decimal 값을 바꾸지 않는다.
2. **성공 결과 복구**: 현재 요청의 tool-call ID·source·query·reason·저장 dataset metadata가 일치하는 completed receipt만 복원한다. Python repr을 eval하지 않고 SQL을 재실행하지 않는다. uncertain 상태에는 적용하지 않는다. 재시작 회귀에서 모델/SQL 추가 실행 없이 로딩을 완료했다.
3. **숫자 문자열**: 실제 테이블의 수치 지표가 문자열이고 `null` 문자열도 포함됐다. 새 `prepare_numeric_dataset` 도구는 필요한 1~8개 컬럼만 수치 projection으로 만들고 모든 행·parent·snapshot·coverage를 보존한다. unknown 값을 삭제하지 않고 미확정으로 반환한다. 명시한 결측 문자열만 NULL 처리하고 건수를 보고한다. 큰 정수의 Float64 정밀도 손실과 무한대를 거절한다. 동일 변환은 재사용한다. SQL의 일반 결과 20,000행 상한으로 원본 100,000행을 잘라내지 않는다.
4. **도구 결과와 남은 목표 연결**: 실모델 첫 실행은 변환과 평균에는 성공했지만 차트를 원격 계획으로 돌려 10회 한도로 실패했다. 변환 도구의 실제 lineage와 요청 컬럼을 검증해 같은 요청의 남은 평균·히스토그램을 로컬 도구로 이어가도록 보강했다. 변환 결과 자체를 분석 완료로 처리하지 않는다. 명시되지 않은 결측 문자열 정책은 실행 전 차단한다.
5. **따옴표 컬럼의 후속 필터**: 공백이 있는 backtick 컬럼이 조건과 평균의 대상에 함께 나오면 계산 대상을 놓쳤다. 인용된 실제 컬럼명도 연산에 결합하도록 수정했다. 다른 schema fixture로 회귀 검증했다.

## 독립 검증

[저장된 실제 데이터 EDA](warehouse_eda_verified.json): 전체 100,000행을 유지하며 `Avg VTAT`의 `null` 문자열 7,043개를 명시적으로 결측 처리했다. 유효 92,957개 평균 **8.46354013145856**, 8.3 이상 조건 평균 **11.619873749302139**가 독립 Pandas 계산과 일치했다. 조건 해제 평균·PNG와 재시작 선택/원본 hash 보존도 통과했다. 이 경로의 도구 선택은 검증기가 지정했으며 LLM 자율성 점수가 아니다. 전체 11.915초, process peak RSS 387,678,208 bytes(약370MiB), 추가 warehouse SQL 0회다. 최초 [필터 실패 기록](warehouse_eda_recovered.json)도 보존했다.

[실모델 최초 실패](live_numeric_repair.json)와 [수정 후 검증](live_numeric_repair_verified.json)을 구분했다. 새 격리 대화에는 실제 저장 원본의 복사본만 제공했고 기존 변환 결과나 과거 대화를 제공하지 않았다. 실제 Databricks serving 모델이 `prepare_numeric_dataset`을 선택했다. 그 뒤 graph가 남은 평균과 히스토그램을 실행했다. **모델 1회, 11.562초, 정답 평균과 PNG 1개, 원본 불변, warehouse SQL 0회**다. 결측 문자열 기준은 평가 요청에 명시했으므로 모호한 결측 의미를 agent가 스스로 해결했다는 증거는 아니다. 이 실행 이후 답변에 변환/결측 건수 표시를 추가했고 회귀 테스트로 확인했다.

## 검증 범위와 판정

- application 378/378, migration 136/136, reference/Level3 217/217(58 figures), compile/diff PASS.
- 새 계약 9개: 타입별 JSON, ledger 첫 응답/재생, receipt 재시작 복구·불일치 차단, 90,000행 변환/보존/재사용, 비수치·정밀도·무한대, 인용 컬럼 필터/복귀, 모델 1회 후 두 목표 자동 완료.
- 해당 실제 10만 행 적재·명시적 수치 EDA 여정은 통과했다. LIMIT 표본이며 실제 100만 행 warehouse 적재나 동시 세션 부하를 검증한 것은 아니다.
- **범용 agent 정식 GO는 보류**: 앞선 Spider 고정 5문항의 역할별 JOIN/JSON OR 과잉 차단, 임의 연도, DISTINCT·중앙값 단계 누락을 이 변경이 해결하지 않는다. 103개 미채점 oracle, DeepEval judge calibration, 복합 분석 확대도 남는다. SQL 승인은 더 이상 이번 검증의 blocker가 아니다.

최신 코드로 Streamlit을 재시작한 뒤 [실제 웹 smoke](web_smoke.json)도 통과했다. 기존 대화의 10,000행 원본에서 age 평균 40.931과 히스토그램이 표시됐고 model 0회, 원본 SHA256 유지, 기존 completed SQL 1회 그대로였다.
