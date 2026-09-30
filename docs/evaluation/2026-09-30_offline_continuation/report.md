# 공급자 차단 중 대용량 EDA 개선 — 2026-09-30

계속 진행 가능한 오프라인 구현·검증을 수행했다. Databricks는 13:30 KST 재점검에서도 모델400 및 SQL OpenSession400으로 차단됐다. [최소 사전점검](provider_probe.json). 공급자 가용성과 agent 기능의 품질은 별도로 평가하며 범용 NO-GO를 유지한다.

## 해결한 문제

이상치 몇 행을 추출하는 도구가 임계값 계산 후 원본의 모든 컬럼·행을 pandas로 복원했다. 30,000행의 fixture에서 정답이3행뿐이어도 전체 읽기 예산 때문에 거부됐다. [수정 전 재현](initial_failure.txt).

영속 데이터는 임계값 계산에 대상 수치 컬럼만 읽고, 행 추출은 최대1,024행 Arrow 배치를 읽어 필터링한 뒤 staging Parquet에 저장하도록 변경했다. 전체 읽기 cache를 채우지 않는다. 원본·snapshot·부모 계보는 유지한다. 출력 행 수가 탐지 근거와 정확히 일치해야 발행하며, 기존 결과 byte/컬럼/디스크 용량 제한을 적용한다. 중간 실패 때 staging 파일은 제거되고 원본 선택은 유지된다.

임계값 계산은 아직 **대상 숫자 컬럼 전체를 메모리에 읽는다**. 모든 연산이 상수 메모리라는 뜻은 아니다. 모든 컬럼의 동시 복원과 두 번째 전체 프레임 할당을 제거한 개선이다. 메모리 내 비영속 store는 기존 방식과 예산 검사를 유지한다.

## 검증

- 관련27개 검사 PASS. 정상 IQR/quantile/zscore/MAD와 agent의 이상치 후속 통계 경로 포함.
- 외부 fixture의 정답3행과 실제 행·값 비교, 원본 파일 불변, 선택 유지, digest 일치 및 재시작 검증.
- 첫 batch 후 읽기 오류/조기종료/결과 byte 한도 초과: 새 dataset 발행0, staging 잔류0. 복구 후 다중 batch 정상 전체 cohort/digest 확인.
- 기존 전체 읽기 예산 검사에서 조인·복합SQL의 거부는 유지한다. 영속 이상치 cohort만 실제 전체 읽기 금지 상태에서 성공하도록 기대 결과를 갱신했다. [회귀](focused_final.txt).
- 합성 **1,000,000행→100행**, 독립적인 행번호/값 oracle와 일치. 최대batch1,024행/977batch/cache0, 원본 및 재시작 보존. 추출0.416초, 전체 프로세스 peak RSS245,907,456bytes. 시간은 fixture 적재·사후검증 제외, RSS는 fixture 생성 포함이다. [실행 결과](million_row_outliers.json).
- 이번 scale 검증은 production 도구에 합성 보유 Parquet를 제공한 것이다. 실제 LLM0/원격SQL0이며 실모델 자연어 성공률·Databricks 네트워크 처리량으로 해석하지 않는다.

재현:

```sh
.telly_runtime/v1-venv/bin/python scripts/evaluate_outlier_streaming.py --output /tmp/telly-outlier-scale.json
.telly_runtime/v1-venv/bin/python -m unittest tests.test_outlier_streaming tests.test_analysis_outliers tests.test_analysis_full_read_budget tests.test_data_preservation_acceptance
```

## 남은 작업

미완료4묶음은 유지한다. 대용량 도구 중 이상치 cohort의 전체 복원을 제거했지만, 조인 전체 복원/키 집계 메모리와 수치 임계값 전체컬럼 읽기 등은 별도 한계다. 일반 최신행 EDA, 고급 SQL, 대용량 혼합 전체 여정, 독립 oracle98도 남아 있다. 공급자가 복구되면 최소probe 확인 후 실제 모델/SQL 평가를 재개한다. 이번 결과를 실환경 GO로 바꾸지 않는다.

최종 제품 전체 **469/469 PASS (55.308초)**, compile/diff 검사 PASS.
