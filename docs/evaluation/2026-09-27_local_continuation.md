# 모델 실패 후 남은 로컬 작업 자동 수행

## 변경과 원인

모델 추론 재시도가 모두 실패하면, 기존 graph는 완료된 차트를 보존하면서도 사용자 `resume()`를 요구했다. 이제 pending node가 모델이고 오류가 일시적 연결/timeout 오류일 때, 확정된 요청·스키마·범위로 실행 가능한 로컬 도구를 한 번의 복구 전환으로 이어간다. 동일 run ID와 전체 소요 시간을 유지하고 원래 오류 ID도 기록한다.

복구 모드는 checkpoint에 저장한다. 이후 단계도 검증된 로컬 도구만 실행하며, 모든 완료 조건을 충족해야 answered를 반환한다. 도구 예산을 초기화하지 않고, 실패한 로컬 계획을 재실행하지 않는다. 승인 대기·원격 제출 불명 상태, 인증/프로그래밍 오류, 검증된 계획 부재에는 자동 전환하지 않는다. 원격 SQL, 의미 재해석, 요약 모델로 우회하지 않는다. 현재 자동 복구 도구 범위는 local_analysis_sql / detect_outliers / summarize_groups / render_chart_spec이다.

실제 모델 검증에서 추가로 발견한 문제: `prepare_histogram(dataset_id=원본)`은 파생된 빈도 데이터로 차트를 만들지만, 완료 검증이 입력 원본 ID와 차트 ID를 비교하고 있었다. 생성 결과의 `loaded_dataset`을 기준으로 검증하도록 수정했다. 기존 PNG·출처·범위·신선도·빈도 검증은 유지한다. 특정 테이블/컬럼 예외는 추가하지 않았다.

## 증거

- [첫 실행 실패](2026-09-27_local_continuation_live.json), [원인 수집 실행 실패](2026-09-27_local_continuation_diagnosis.json): 실제 Databricks 모델이 prepare_histogram을 선택한 뒤 timeout 3회 주입. 차트의 입력/출력 ID 혼동으로 미완료. 실패 기록 보존.
- [수정 후 실행](2026-09-27_local_continuation_verified.json): PASS. 실제 모델 1회가 차트 도구/인자를 선택했다. 이후 APITimeoutError 3회를 주입했고 agent는 모델을 추가 호출하지 않고 로컬 SQL로 평균 9.0을 계산했다. PNG 1개, 원본 digest 불변, 원격 SQL 0회, 13.913초.
- 실제 모델을 포함하되 4행 synthetic fixture와 장애 주입을 사용했다. 실제 공급자 장애 지속 시간·대규모 Databricks 로딩·브라우저 장애 주입 검증은 아니다. 최초 planning은 테스트 목적상 비활성화하고 장애 복구 단계에서 production planner를 사용했다.
- 이전 실모델 종합 7/8여정 및 실제 APITimeoutError 실패는 그대로 유지한다. 이번 단일 복구 성공을 종합 점수나 공식 DeepEval/Spider 점수로 합산하지 않는다.

## 회귀 검증

`tests/test_local_continuation.py` 11건: 차트 후 평균, 여러 미완료 목표, 필터 보존, 인증/코드 오류 제외, 불명 원격 제출 제외, 계획 부재, 로컬 도구 추가 실패, 재시작과 새 요청 상태 격리, 도구 예산, 실제 모델이 생성한 명시적 원본 ID 호출.

- Application: 369/369 PASS.
- Migration: 136/136 PASS.
- Reference + Level 3: 217/217 PASS, 이미지 58개.
- compileall / git diff --check PASS.

## 판정

지원되는 보유 데이터 분석의 자동 복구 기능을 개선했다. 범용 자율 분석 정식 출시 NO-GO는 유지한다. 복합 조건·다중 출처·고급 계획, 독립 정답 103건, judge calibration, 승인형 대규모 적재/RSS, 장기 공급자 장애, 격리 코드 실행 검증은 남는다. 별도 서버 배포는 사용자 요청에 따라 제외한다.

## 실제 웹 후속 검증과 추가 수정

[웹 증거](2026-09-27_local_continuation_web.json): 이미 승인·보관된 10,000행에서 “이전 age 조건은 적용하지 말고” 평균+히스토그램을 요청했다. 최초 실행은 이전 age < 60을 잘못 상속해 30.576초 후 scope mismatch로 중단했다. scope parser가 “말고”를 단순 후속 참조로 처리했기 때문이다.

명시적인 필터 해제 명령을 runtime schema의 컬럼/alias에 연결하도록 수정했다. 지정한 컬럼의 조건만 제거하고 다른 조건은 유지한다. 전역 해제와 새 조건, “평균 말고 중앙값”의 필터 유지, “해제하지 말고 유지”의 부정 지시를 회귀 검사한다. 임의의 복잡한 자연어 해제가 모두 지원된다는 의미는 아니다. 세 개의 scope 계약과 하나의 실제 graph 후속 여정을 추가했다.

동일 웹 요청 재실행: answered, 평균 40.931(독립 Parquet 계산 일치), 실제 PNG 1개/유효값 10,000개/20 bins, 0.383초, 모델 호출 0회. 원본 SHA256 불변, 승인 장부 completed 1건 유지, 신규 SQL 0회. 최초 실패도 증거에 포함했다. PID 87703의 loopback 앱 health=ok, Databricks 모델 선택과 기존 대화를 복원했다.

## 원격 검증

코드 commit `3dd30a5892309e7d6c290089c3e428d9561b5b63`을 push했다. [GitHub Actions 36310768258](https://github.com/konlo/teleai/actions/runs/36310768258)이 2분 46초에 migration/application/agentic/reference/compile 전 단계를 통과했다. Draft PR #68은 갱신했으며 병합하지 않았다.
