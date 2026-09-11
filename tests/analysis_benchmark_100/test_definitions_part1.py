"""100 Test Definitions for Data Analysis Agent Benchmark Suite.
Defines questions, python code, categories, output types, and validation rules for TEST_001 ~ TEST_100.
"""

def get_all_test_specs():
    specs = []
    
    # -------------------------------------------------------------
    # Category 1: 기초 데이터 탐색 및 요약 통계 (Basic EDA & Summary Statistics)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_001",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "고객 데이터의 전체 행과 열 개수, 메모리 사용량을 확인해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_001: 행과 열 개수 및 메모리 사용량 확인
print(f"Shape: {df.shape}")
mem_kb = df.memory_usage(deep=True).sum() / 1024
print(f"Memory Usage: {mem_kb:.2f} KB")
print("Data Types:")
print(df.dtypes.value_counts())
"""
    })

    specs.append({
        "id": "TEST_002",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "각 컬럼별 결측치(Null) 개수와 전체 대비 결측 비율(%)을 내림차순으로 보여줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_002: 결측치 현황 및 비율 산출
null_count = df.isnull().sum()
null_ratio = (null_count / len(df)) * 100
null_df = pd.DataFrame({"null_count": null_count, "null_ratio_pct": null_ratio.round(2)})
result = null_df[null_df["null_count"] > 0].sort_values(by="null_count", ascending=False)
print("Columns with missing values:")
print(result)
"""
    })

    specs.append({
        "id": "TEST_003",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "수치형 컬럼들에 대한 기술통계 요약표(평균, 표준편차, 사분위수 등)를 소수점 둘째 자리까지 생성해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_003: 수치형 컬럼 기술통계 요약
num_desc = df.describe().round(2)
print("Descriptive Statistics for Numeric Columns:")
print(num_desc)
"""
    })

    specs.append({
        "id": "TEST_004",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "범주형 컬럼들의 고유값(Unique) 개수를 내림차순으로 확인해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_004: 범주형 컬럼 고유값 개수
cat_cols = df.select_dtypes(include=['object']).columns
unique_counts = df[cat_cols].nunique().sort_values(ascending=False)
print("Unique value counts in categorical columns:")
print(unique_counts)
"""
    })

    specs.append({
        "id": "TEST_005",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "고객 연령(age)과 잔액(balance)의 최솟값, 최댓값, 중앙값(50%) 및 25%, 75% 분위수를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_005: 주요 분위수 산출
quantiles = df[["age", "balance"]].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
quantiles.index = ["Min (0%)", "Q1 (25%)", "Median (50%)", "Q3 (75%)", "Max (100%)"]
print(quantiles.round(2))
"""
    })

    specs.append({
        "id": "TEST_006",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "결측치가 존재하는 컬럼만 식별하고, 해당 컬럼의 유효값 평균과 결측 행 수를 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_006: 결측치 존재 컬럼의 통계 확인
cols_with_na = df.columns[df.isna().any()].tolist()
for col in cols_with_na:
    n_na = df[col].isna().sum()
    mean_valid = df[col].dropna().mean()
    print(f"컬럼 '{col}': 결측치 수 = {n_na}, 유효값 평균 = {mean_valid:.2f}")
"""
    })

    specs.append({
        "id": "TEST_007",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "전체 고객 중 이탈 고객(churn=1)과 유지 고객(churn=0)의 인원수와 백분율을 표로 요약해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_007: 이탈 고객 요약
churn_counts = df['churn'].value_counts()
churn_pct = (df['churn'].value_counts(normalize=True) * 100).round(2)
summary = pd.DataFrame({"고객수": churn_counts, "비율(%)": churn_pct})
summary.index = ["유지 (0)", "이탈 (1)"]
print(summary)
"""
    })

    specs.append({
        "id": "TEST_008",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "서비스 등급(service_tier)별 고객 수와 전체 대비 비중(%)을 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_008: 서비스 티어별 고객 비중
tier_counts = df['service_tier'].value_counts()
tier_pct = (df['service_tier'].value_counts(normalize=True) * 100).round(2)
tier_summary = pd.DataFrame({"고객수": tier_counts, "비율(%)": tier_pct})
print(tier_summary)
"""
    })

    specs.append({
        "id": "TEST_009",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "고객 연령(age)의 분포를 확인하기 위해 히스토그램과 KDE 곡선을 화면에 그려줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_009: 연령 분포 히스토그램 및 KDE
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.histplot(df['age'], kde=True, bins=25, color='steelblue')
plt.title('고객 연령(Age) 분포')
plt.xlabel('연령')
plt.ylabel('인원수')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_010",
        "category": "기초 데이터 탐색 및 요약 통계",
        "difficulty": "Basic",
        "question": "고객 계좌 잔액(balance)의 분포 형태와 이상치 유무를 확인하기 위해 박스플롯을 그려줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_010: 잔액 박스플롯 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 3))
sns.boxplot(x=df['balance'], color='lightseagreen')
plt.title('고객 계좌 잔액(Balance) 박스플롯')
plt.xlabel('잔액 ($)')
plt.tight_layout()
plt.show()
"""
    })

    # -------------------------------------------------------------
    # Category 2: 조건부 필터링 및 서브셋 추출 (Filtering & Subsetting)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_011",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Basic",
        "question": "연령이 30세 이상 39세 이하인 30대 고객들의 인원수와 평균 잔액을 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_011: 30대 고객 필터링 및 집계
sub_30s = df[(df['age'] >= 30) & (df['age'] <= 39)]
print(f"30대 고객 수: {len(sub_30s)}명")
print(f"30대 평균 잔액: {sub_30s['balance'].mean():.2f} 달러")
"""
    })

    specs.append({
        "id": "TEST_012",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Basic",
        "question": "잔액(balance)이 0원 미만인 마이너스 통장 고객들의 인원수와 평균 연령, 주요 직업 상위 3개를 조회해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_012: 마이너스 잔액 고객 서브셋
neg_balance = df[df['balance'] < 0]
print(f"마이너스 잔액 고객 수: {len(neg_balance)}명")
print(f"평균 연령: {neg_balance['age'].mean():.1f}세")
print("주요 직업 상위 3개:")
print(neg_balance['job'].value_counts().head(3))
"""
    })

    specs.append({
        "id": "TEST_013",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Basic",
        "question": "주택대출(housing_loan='yes')과 신용대출(personal_loan='yes')을 둘 다 보유한 다중 대출 고객들을 추출하고 전체 대비 비중을 구해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_013: 주택 및 신용대출 다중 보유자
multi_loan = df[(df['housing_loan'] == 'yes') & (df['personal_loan'] == 'yes')]
ratio = (len(multi_loan) / len(df)) * 100
print(f"다중 대출 고객 수: {len(multi_loan)}명")
print(f"전체 고객 대비 비중: {ratio:.2f}%")
"""
    })

    specs.append({
        "id": "TEST_014",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Basic",
        "question": "직업이 'management'이거나 'entrepreneur'인 고위 경영/사업가 고객 그룹의 평균 연소득을 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_014: 경영/사업가 직군 고객 연소득
mgmt_ent = df[df['job'].isin(['management', 'entrepreneur'])]
print(f"해당 고객 수: {len(mgmt_ent)}명")
print(f"평균 연소득: {mgmt_ent['annual_income'].mean():.2f} 달러")
"""
    })

    specs.append({
        "id": "TEST_015",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Intermediate",
        "question": "신용점수(credit_score)가 750점 이상이고 연소득이 70,000 이상인 최우수 고객 서브셋을 필터링하고 인원수를 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_015: 고신용/고소득 최우수 고객 필터링
vip_filter = (df['credit_score'] >= 750) & (df['annual_income'] >= 70000)
vip_df = df[vip_filter]
print(f"조건 만족 고객 수: {len(vip_df)}명")
print(f"이들의 평균 잔액: {vip_df['balance'].mean():.2f} 달러")
"""
    })

    specs.append({
        "id": "TEST_016",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Basic",
        "question": "접속 기기(device_type)가 'iOS' 또는 'Android'인 모바일 사용자 고객들만 필터링하여 총 몇 명인지 확인해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_016: 모바일 사용자 필터링
mobile_users = df[df['device_type'].isin(['iOS', 'Android'])]
print(f"모바일 사용자 수: {len(mobile_users)}명 (전체의 {len(mobile_users)/len(df)*100:.1f}%)")
print(mobile_users['device_type'].value_counts())
"""
    })

    specs.append({
        "id": "TEST_017",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Basic",
        "question": "만족도 점수(satisfaction_score)가 입력되지 않은(결측치) 고객들의 평균 연소득과 평균 신용점수를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_017: 만족도 결측 고객 집계
missing_sat = df[df['satisfaction_score'].isna()]
print(f"만족도 결측 고객 수: {len(missing_sat)}명")
print(f"평균 연소득: {missing_sat['annual_income'].mean():.2f} 달러")
print(f"평균 신용점수: {missing_sat['credit_score'].mean():.2f} 점")
"""
    })

    specs.append({
        "id": "TEST_018",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Intermediate",
        "question": "미혼('single')이면서 주택대출이 없는('no') 20대(20~29세) 고객 서브셋을 추출하고 인원수를 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_018: 미혼 & 무대출 20대 고객
cond = (df['marital'] == 'single') & (df['housing_loan'] == 'no') & (df['age'].between(20, 29))
subset = df[cond]
print(f"미혼 & 주택대출 무 20대 고객 수: {len(subset)}명")
print(f"평균 잔액: {subset['balance'].mean():.2f} 달러")
"""
    })

    specs.append({
        "id": "TEST_019",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Intermediate",
        "question": "잔액 기준 상위 5% 초과인 고액 자산가들의 기준 잔액(Cutoff)과 이들의 평균 연소득을 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_019: 상위 5% 고액 자산가
cutoff_95 = df['balance'].quantile(0.95)
top_5_pct = df[df['balance'] > cutoff_95]
print(f"상위 5% 기준 잔액: {cutoff_95:.2f} 달러")
print(f"고액 자산가 인원수: {len(top_5_pct)}명")
print(f"고액 자산가 평균 연소득: {top_5_pct['annual_income'].mean():.2f} 달러")
"""
    })

    specs.append({
        "id": "TEST_020",
        "category": "조건부 필터링 및 서브셋 추출",
        "difficulty": "Intermediate",
        "question": "40세 이상이면서 잔액이 10,000 달러 이상인 비이탈(churn=0) 고객들의 직업별 인원수 분포를 막대 차트로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_020: 특정 복합 조건 만족 고객의 직업별 막대 차트
import matplotlib.pyplot as plt

cond = (df['age'] >= 40) & (df['balance'] >= 10000) & (df['churn'] == 0)
sub = df[cond]
counts = sub['job'].value_counts()

plt.figure(figsize=(9, 4))
counts.plot(kind='bar', color='coral')
plt.title('40세 이상 & 잔액 $10,000 이상 비이탈 고객의 직업 분포')
plt.xlabel('직업')
plt.ylabel('고객 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    })

    return specs
