# Telly Chatbot Benchmark Test Suite

데이터 분석 AI 챗봇 **Telly**를 평가하기 위한 **200개 질문·참조 Python 코드**와 **17개 production agentic recovery 계약**입니다.

> Level 1·2와 `execution_report.md`의 PASS는 준비된 참조 Python 코드가 예외 없이 실행되었다는 뜻입니다. 실제 에이전트 성공률이 아닙니다. Level 3는 production `GraphAnalysisRuntime`의 고장 주입·복구 계약이며 자연어 이해 정확도는 별도 실제 에이전트 평가로 확인합니다.

## 실제 에이전트 평가

`scripts/evaluate_analysis_agent.py`는 원래 질문을 운영 `GraphAnalysisRuntime.submit()`에 입력하고, 별도로 실행한 참조 코드의 수치·집계 결과 또는 히스토그램 분포와 실제 도구 결과를 비교합니다. 정답 코드와 정답 변수는 모델에 제공하지 않습니다. 참조 정의 파일의 의미를 변경하지 않고 `tests/fixtures/analysis_agent_grading.json`에 명시한 20개 문항을 채점할 수 있습니다. 나머지 180개 문항은 `UNGRADED`이며 성공으로 합산하지 않습니다. 채점 지원 확대는 실제 모델 통과를 뜻하지 않습니다. 새 9문항은 별도 실행 결과가 있어야 성공률에 포함할 수 있습니다.

외부 `data_context/*.json`에는 이 README의 전체 테이블 스키마에 이미 명시된 별칭만 저장합니다. 실행 시 CSV 타입과 최대 10개 저카디널리티 값 분포를 함께 제공합니다. 질문별 `synonym_mapping`이나 참조 코드·정답을 모델에 전달하지 않습니다. 따라서 기존 별칭 없는 평가와는 입력 문맥이 달라졌으며, 새 실행 보고서에는 `reference_context.alias_source`와 실제 주입 문맥이 남습니다.

```sh
# 목록/채점 범위 확인: 모델 및 Databricks 호출 없음
.telly_runtime/v1-venv/bin/python scripts/evaluate_analysis_agent.py --list
.telly_runtime/v1-venv/bin/python scripts/evaluate_analysis_agent.py --all

# 실제 로컬 Ollama 모델, 원래 자연어 질문, 실제 로컬 SQL/PNG 실행
.telly_runtime/v1-venv/bin/python scripts/evaluate_analysis_agent.py --live-local-model --id L1_016 --id L1_076

# 하네스 자체의 실패 감지 검사: 대역 모델이므로 자연어 품질 평가와 구별
.telly_runtime/v1-venv/bin/python -m unittest discover -s tests -p test_actual_agent_evaluation.py -v
```

각 문항은 별도 임시 대화에 해당 CSV fixture 전체를 완전성·출처·스냅샷과 함께 로딩합니다. Databricks 실행기는 차단되어 있으며 승인 대기는 `NOT_COMPLETE`로 남습니다. 원격 DB 장애 복구, 브라우저 출력, 여러 턴의 맥락 연결은 별도 여정 검증 대상입니다. 기본 실행은 `NOT_RUN`/`UNGRADED` 범위 보고서만 만듭니다. `--live-local-model`을 명시한 경우에만 `.env`의 localhost Ollama 설정을 사용합니다.

수치 채점은 최종 `local_analysis_sql` 결과의 데이터와 fixture 계보를 비교하며 모델의 문장만으로 통과시키지 않습니다. 히스토그램은 실제 Matplotlib에 전달된 값/가중치 분포, 그 Figure로부터 저장된 PNG의 해시, 컬럼과 출처를 검사합니다. 원래 자연어 질문이 bin 개수를 지정하지 않아 bin 수는 기록만 하고 스타일·bin 개수의 일치를 요구하지 않습니다. 최종 설명문이 도구 결과를 충실하게 전달하는지는 이 채점 범위에 포함되지 않습니다.

기본 결과 파일은 `docs/actual_agent_evaluation.json`이며 선택 문항, 지원 범위, 실패/미채점, 도구·SQL, 원격 실행 수, 지연시간을 기록합니다. 대화 원문·모델 추론을 보고서에 덤프하지 않으며, 참조 fixture의 집계 정답과 비교값만 남깁니다.

## 개요

| 항목 | 내용 |
|------|------|
| **총 질문 수** | 200개 (Level 1: 100, Level 2: 100) |
| **데이터셋** | `bank_loan` (2,000행 × 18컬럼), `titanic` (891행 × 12컬럼) |
| **정답 형식** | Python 코드 + 실행 결과 (stdout / 차트 생성 여부) |
| **참조 코드 실행 기록** | 200/200 예외 없음 (차트 생성: Level 1 25개, Level 2 33개). 에이전트 평가 아님. |
| **Agentic 계약 실행 기록** | 17/17 (`A3_001` ~ `A3_017`). 복구·승인·저장·UI 경계의 결정적 고장 주입 테스트. |

---

## 디렉토리 구조

```
test_set/
├── README.md                   # 이 문서
├── generate_data.py            # 재현 가능한 데이터셋 생성기 (seed=42)
├── build_and_run_all.py        # 200개 전체 실행 및 JSON 벤치마크 저장
├── run_test_set.py             # CLI 테스트 러너 (필터링, 개별 실행 지원)
├── benchmark_level1.json       # Level 1 실행 결과 저장 파일
├── benchmark_level2.json       # Level 2 실행 결과 저장 파일
├── execution_report.md         # 최신 실행 보고서
├── data/
│   ├── bank_loan.csv           # 은행 텔레마케팅 데이터 (2,000행)
│   └── titanic.csv             # 타이타닉 승객 데이터 (891행)
├── level1/
│   ├── __init__.py
│   ├── definitions_part1.py    # L1_001 ~ L1_025: 테이블 스키마 + 동의어 기본
│   ├── definitions_part2.py    # L1_026 ~ L1_050: 동의어 심화 + 단순 필터/집계
│   ├── definitions_part3.py    # L1_051 ~ L1_075: 그룹 통계 + 피벗 테이블
│   └── definitions_part4.py    # L1_076 ~ L1_100: 단일 시각화 차트
├── level2/
    ├── __init__.py
    ├── definitions_part1.py    # L2_001 ~ L2_025: 다중 조건 필터 + 복합 피벗
    ├── definitions_part2.py    # L2_026 ~ L2_050: 이상치 감지 + 고급 집계
    ├── definitions_part3.py    # L2_051 ~ L2_075: 통계 가설 검정 + 이중축 시각화
│   └── definitions_part4.py    # L2_076 ~ L2_100: 2x2 대시보드 + 방어적 예외 처리
└── level3/
    └── definitions_part5.py    # A3_001 ~ A3_017: production agentic recovery 계약 레지스트리
```

---

## 데이터 스키마

### `workspace.default.bank_loan` (bank_loan.csv)

| 컬럼 | 타입 | 설명 | 동의어 (한국어) |
|------|------|------|----------------|
| `id` | int | 고객 ID | 아이디 |
| `age` | int | 고객 나이 | 나이, 연령 |
| `job` | str | 직업 | 직업, 직종, 일 |
| `marital` | str | 결혼 여부 (married/single/divorced) | 결혼, 혼인, 배우자 |
| `education` | str | 학력 (primary/secondary/tertiary/unknown) | 학력, 교육, 학교 |
| `default` | str | 채무 불이행 여부 (yes/no) | 연체, 불이행, 디폴트 |
| `balance` | int | 계좌 잔액 | 잔액, 잔고, 예금액, 통장 잔액 |
| `housing` | str | 주택 담보 대출 여부 (yes/no) | 주택 대출, 집 대출, 부동산 대출 |
| `loan` | str | 개인 신용 대출 여부 (yes/no) | 개인 대출, 신용 대출 |
| `contact` | str | 통신 수단 (cellular/telephone/unknown) | 연락 방법, 통신 수단 |
| `day` | int | 마지막 접촉 일 (월중) | 접촉 날짜 |
| `month` | str | 마지막 접촉 월 | 월, 달 |
| `duration` | int | 통화 지속 시간 (초) | 상담 시간, 통화 시간, 상담 길이 |
| `campaign` | int | 이번 캠페인 접촉 횟수 | 접촉 횟수, 캠페인 횟수 |
| `pdays` | int | 이전 캠페인 후 경과 일 수 (-1=없음) | 경과일 |
| `previous` | int | 이전 캠페인 접촉 횟수 | 이전 접촉 횟수 |
| `poutcome` | str | 이전 캠페인 결과 (success/failure/other/unknown) | 이전 결과, 이전 마케팅 결과 |
| `y` | str | 정기예금 가입 여부 (yes/no) | 가입 여부, 정기예금, 예금 가입 |

### `workspace.default.titanic` (titanic.csv)

| 컬럼 | 타입 | 설명 | 동의어 (한국어) |
|------|------|------|----------------|
| `PassengerId` | int | 승객 ID | 승객 번호 |
| `Survived` | int | 생존 여부 (1=생존, 0=사망) | 생존율, 생존 여부 |
| `Pclass` | int | 객실 등급 (1/2/3) | 등급, 좌석 등급, 객실 등급 |
| `Name` | str | 승객 이름 | 이름 |
| `Sex` | str | 성별 (male/female) | 성별, 남성, 여성 |
| `Age` | float | 나이 (결측치 있음) | 나이, 연령 |
| `SibSp` | int | 동반 형제자매/배우자 수 | 형제자매, 배우자 |
| `Parch` | int | 동반 부모/자녀 수 | 부모, 자녀 |
| `Ticket` | str | 티켓 번호 | 티켓 |
| `Fare` | float | 운임료 | 요금, 운임, 승선료 |
| `Cabin` | str | 객실 번호 (결측치 많음) | 객실 번호, 객실 |
| `Embarked` | str | 승선 항구 (C/Q/S) | 승선 항구, 출발지 |

---

## 테스트 유형별 분류

### Level 1 (기본 ~ 중급)

| 유형 | 개수 | 범위 | 설명 |
|------|------|------|------|
| `schema` | 15 | L1_001 ~ L1_015 | 테이블 스키마 조회, dtype, 결측치, distinct 값 |
| `synonym` | 15 | L1_016 ~ L1_030 | 한국어 동의어로 컬럼 찾기 |
| `table` | 45 | L1_031 ~ L1_075 | 단순 필터, 집계, 그룹 통계, 피벗 테이블 |
| `chart` | 25 | L1_076 ~ L1_100 | 단일 시각화 (막대, 원형, 히스토그램, 산점도 등) |

### Level 2 (고급 ~ 전문가)

| 유형 | 개수 | 범위 | 설명 |
|------|------|------|------|
| `table` | 30 | L2_001 ~ L2_050 | 다중 조건 필터, 복합 피벗, 이상치 감지 |
| `table` | 10 | L2_051 ~ L2_060 | 통계 가설 검정 (t-test, Chi-square, ANOVA) |
| `chart` | 30 | L2_061 ~ L2_090 | 이중축, 2x2 대시보드, 버블차트, 로렌츠 곡선 |
| `schema` | 3 | L2_091, L2_095, L2_099 | 비존재 컬럼/테이블 방어 처리 |
| `table` | 7 | L2_092 ~ L2_100 | 방어적 예외 처리 및 데이터 품질 파이프라인 |

---

## 동의어 매핑 딕셔너리

Telly 챗봇은 아래 한국어 동의어를 올바른 컬럼명으로 해석해야 합니다.

```python
SYNONYM_MAP = {
    # bank_loan 컬럼
    "나이": "age", "연령": "age",
    "직업": "job", "직종": "job", "일": "job",
    "결혼": "marital", "혼인 여부": "marital",
    "학력": "education", "교육 수준": "education",
    "연체": "default", "불이행": "default",
    "잔액": "balance", "잔고": "balance", "예금액": "balance", "통장 잔액": "balance",
    "주택 대출": "housing", "집 대출": "housing",
    "개인 대출": "loan", "신용 대출": "loan",
    "통화 시간": "duration", "상담 시간": "duration",
    "접촉 횟수": "campaign",
    "이전 결과": "poutcome",
    "가입 여부": "y", "예금 가입": "y", "정기예금": "y",
    # titanic 컬럼
    "생존율": "Survived", "생존 여부": "Survived",
    "객실 등급": "Pclass", "등급": "Pclass",
    "성별": "Sex",
    "요금": "Fare", "운임": "Fare",
    "객실 번호": "Cabin", "덱": "Cabin",
    "승선 항구": "Embarked",
}
```

---

## 사용 방법

### 환경 설정

```bash
cd /Users/najongseong/git_repository/teleai
source .venv/bin/activate   # 또는: .venv/bin/python 직접 사용
```

### 전체 200개 실행 (벤치마크 JSON 저장)

```bash
.venv/bin/python test_set/build_and_run_all.py
```

출력 파일:
- `test_set/benchmark_level1.json`
- `test_set/benchmark_level2.json`
- `test_set/execution_report.md`

### CLI 러너 활용

```bash
# 참조 200개와 agentic 계약 17개 모두 실행 (요약만)
.venv/bin/python test_set/run_test_set.py --all --quiet

# Level 1만 실행
.venv/bin/python test_set/run_test_set.py --level 1

# Level 2만 실행
.venv/bin/python test_set/run_test_set.py --level 2

# Level 3 production agentic recovery 계약만 실행
.venv/bin/python test_set/run_test_set.py --level 3

# 특정 ID 실행 (상세 출력)
.venv/bin/python test_set/run_test_set.py --id L1_042
.venv/bin/python test_set/run_test_set.py --id L2_077
.venv/bin/python test_set/run_test_set.py --id A3_017

# 특정 ID의 질문 + 코드 조회 (실행 안 함)
.venv/bin/python test_set/run_test_set.py --show L2_082

# 유형 필터 (schema / synonym / table / chart)
.venv/bin/python test_set/run_test_set.py --level 1 --type chart
.venv/bin/python test_set/run_test_set.py --all --type schema

# 카테고리 키워드 필터
.venv/bin/python test_set/run_test_set.py --all --category "시각화"

# 모든 테스트 ID 목록 조회
.venv/bin/python test_set/run_test_set.py --list
```

---

## 벤치마크 JSON 구조

```json
[
  {
    "id": "L1_042",
    "category": "단순 집계 및 요약",
    "type": "table",
    "difficulty": "Intermediate",
    "prompt": "직업별 평균 잔액을 높은 순으로 정렬해서 보여줘",
    "target_table": "bank_loan",
    "expected_output_type": "table",
    "status": "PASS",
    "stdout": "직업별 평균 잔액...",
    "has_figure": false,
    "error": null
  }
]
```

---

## 평가 루브릭 (4점 척도)

다른 에이전트가 이 테스트 셋으로 Telly를 평가할 때 아래 기준을 사용합니다.

| 점수 | 기준 |
|------|------|
| **4 (완벽)** | 동의어 해석 정확, 올바른 컬럼 사용, 실행 가능한 코드, 예상 출력 일치 |
| **3 (양호)** | 정답 컬럼 사용, 코드 실행 성공, 출력 형식 소소한 차이 허용 |
| **2 (부분)** | 컬럼 해석 오류 또는 코드 일부 실행 실패, 부분적 답변 |
| **1 (불량)** | 잘못된 컬럼 사용, 코드 실행 불가, 전혀 무관한 답변 |
| **0 (무응답)** | 응답 없거나 오류 메시지만 반환 |

### 최소 합격 기준 (Release Criteria)

- Level 1 평균 점수 ≥ 3.5 / 4.0
- Level 2 평균 점수 ≥ 3.0 / 4.0
- `schema` 유형 PASS율 ≥ 90%
- `synonym` 유형 PASS율 ≥ 85%
- `chart` 유형 figure 생성 성공율 ≥ 80%

---

## 데이터 재생성

```bash
# 동일한 데이터를 재생성 (seed=42 고정)
.venv/bin/python test_set/generate_data.py
```

---

## 주의사항

> ⚠️ 이 `test_set/` 폴더 내의 파일들은 검증 전용으로, 기존 프로젝트 소스 코드(`core/`, `pages/`, `utils/`)를 수정하지 않습니다.
