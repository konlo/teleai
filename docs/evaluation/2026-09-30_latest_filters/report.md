# 최신행 선택 전후 필터와 히스토그램 후속 검증

2026-09-30. 사용자가 명시한 필터 적용 단계가 있어도 실행하지 못했던 공백을 수정했다. **범용 GO는 아직 NO-GO**이며 Databricks HTTP400을 해결했다는 의미는 아니다.

## 변경

- `conditions`와 `filter_stage`를 도구 입력·실행 전 검사·결과 계보·완료 검증·저장 결과 재사용에 연결했다. 원본을 보존하며 로컬 pandas/배치 DuckDB와 원격 집계 SQL에 동일한 순서를 적용한다.
- 명시한 하나의 단계에서 최대16개 AND 조건(eq/ne/gt/ge/lt/le/in)을 지원한다. 스키마와 비교값의 자료형을 검증하고 SQL AST로 값을 인코딩한다. 단계를 생략하거나 상충되게 지정하면 확인하며 임의로 선택하지 않는다. OR·다단계 혼합·일반 부정 조건까지 해결한 것은 아니다.
- 단계 확인 응답, 재시작 후 단계 변경, 숫자 히스토그램 구간 변경에서 키·정렬·분포·조건과 출처를 보존한다. 다른 선택 데이터에서는 이전 요청을 자동 결합하지 않는다.
- 부동소수점 필터는 SQL의 NaN을 결측으로 제외해 pandas와 동일하게 처리한다.
- 선택 후 필터로 동률을 숨기지 않는다. 조건/단계가 다른 결과의 실행 receipt를 재사용하지 않는다. 대용량 입력 행 수를 검증한 후 결과를 발행한다.
- 후속 검증에서 **수치1개일 때 로컬 히스토그램 실패**를 발견해 수정했다. 실제 렌더러가 반환한 구간 경계·빈도를 차트 명세에 보존한다. 빈 입력은 계속 거절한다.

## 증거

|검증|결과|
|---|---|
|최초 실패|[baseline](baseline.txt): 명시적 전/후 요청4종 실패, 단계 답변 후 연결 실패, 도구 필터 인자 미지원|
|중간 실패|[intermediate](intermediate.txt): 신규 선택 인자의 빈 기본값 schema 불일치와 원격 wrapper 인자 누락을 발견·수정|
|숫자 후속 추가 실패|[numeric_followup](numeric_followup.txt): 로컬 차트 실제 구간/빈도 명세 누락 및 수치1개 실패. [수정 후 통과](numeric_followup_final.txt)|
|제품 전체|[498 PASS / 60.252초](application_final.txt). 이후 NaN 일치 처리·독립 검사1개 추가는 [관련23개 PASS](float_filter_final.txt)로 확인; 최종 전체는 CI에서 재검증|
|migration|[136 PASS / 18.957초](migration_final.txt)|
|reference/agentic|[217 PASS, 58그림](reference_final.txt). 독립98문항의 실모델 재채점이 아님|
|대용량 전체 경로|[102만행](million_rows.json): 필터 선택 전 20,000키(new10,000+old10,000), 선택 후 10,000키(new10,000). 외부 fixture 독립 정답과 일치|
|자원·보존|대용량 원본 전체 projection/get 차단 하에 성공. frame cache0, 원본 파일 hash·선택·계보·재시작 후 차트 보존. 프로세스 peak RSS486,850,560 bytes. SQL 엔진128MB 제한은 Python 전체 RSS 제한이 아님|
|실물 이미지|[선택 전](before.png), [선택 후](after.png), [수치1개 히스토그램](single_value_histogram.png)을 실제 이미지로 확인|

합성 대용량 경로는 production graph의 결정적 도구 계획이며 실모델 호출0·원격 쿼리0이다. 첫 실행7.981초, 다음 실행0.181초는 한 프로세스의 측정으로, 워밍업/캐시 조건이 달라 속도 비교나 운영 SLA로 해석하지 않는다. 원격 SQL 의미는 합성 데이터에서 SQLGlot 변환 후 DuckDB로 검증했다. 실제 Databricks 실행 검증이 아니다.

재현:

```sh
python -m unittest tests.test_latest_filters
python scripts/evaluate_latest_filters.py --output /tmp/latest-filters/result.json
```

## 외부 장애와 남은 작업

[19:00 KST 점검](provider_final.json): Warehouse 조회200/STOPPED, 모델400, OpenSession400. 일시 장애/사용량 제한/계정 제한의 정확한 원인은 아직 미확인이다. 자동 브라우저 제어 차단을 우회하지 않았으며 새 토큰 생성이나 계정 설정 변경도 하지 않았다.

남은 기능은 최신행 추가 통계·비균등 구간·혼합 단계/조건, 복합 SQL 의미 및 실제 복구, 대용량 혼합 raw/집계 여정, 독립98개 정답 비교기와 실제 모델 평가다. 사용자 서버는 외부 접근·원본 로그 반출 없이 진단 상태로 확인해야 한다. 이번 변경을 사용자 서버에 배포한 것은 아니다.
