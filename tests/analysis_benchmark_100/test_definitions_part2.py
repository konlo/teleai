"""100 Test Definitions for Data Analysis Agent Benchmark Suite - Part 2 (Q021 ~ Q040).
"""

def get_part2_specs():
    specs = []

    # -------------------------------------------------------------
    # Category 3: 그룹 집계 및 피벗 분석 (Grouping, Aggregation & Pivot Tables)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_021",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Basic",
        "question": "직업(job)별 고객 수와 평균 잔액을 계산하고, 평균 잔액이 높은 순으로 정렬해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_021: 직업별 고객수 및 평균 잔액 정렬
job_agg = df.groupby('job').agg(고객수=('customer_id', 'count'), 평균잔액=('balance', 'mean'))
result = job_agg.sort_values(by='평균잔액', ascending=False).round(2)
print(result)
"""
    })

    specs.append({
        "id": "TEST_022",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Basic",
        "question": "교육수준(education)별 연소득의 평균, 중앙값, 표준편차를 한 번에 다중 집계해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_022: 교육수준별 연소득 다중 집계
edu_income = df.groupby('education')['annual_income'].agg(['mean', 'median', 'std']).round(2)
print(edu_income)
"""
    })

    specs.append({
        "id": "TEST_023",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Basic",
        "question": "서비스 등급(service_tier)별로 고객 수, 평균 신용점수, 총 잔액 합계를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_023: 서비스 티어별 다중 집계
tier_metrics = df.groupby('service_tier').agg(
    고객수=('customer_id', 'count'),
    평균신용점수=('credit_score', 'mean'),
    총잔액합계=('balance', 'sum')
).round(2)
print(tier_metrics)
"""
    })

    specs.append({
        "id": "TEST_024",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "직업(job)과 결혼 상태(marital)에 따른 고객 수를 피벗 테이블(Pivot Table)로 집계해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_024: 직업 x 결혼상태 피벗 테이블
pivot_tbl = df.pivot_table(index='job', columns='marital', values='customer_id', aggfunc='count', fill_value=0)
print(pivot_tbl)
"""
    })

    specs.append({
        "id": "TEST_025",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "교육수준과 주택대출 보유 여부에 따른 평균 잔액 피벗 테이블을 생성하고, 행과 열의 전체 총계(margins=True)를 포함해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_025: 피벗 테이블 총계(margin) 산출
pivot_margin = df.pivot_table(index='education', columns='housing_loan', values='balance', aggfunc='mean', margins=True).round(2)
print(pivot_margin)
"""
    })

    specs.append({
        "id": "TEST_026",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "서비스 등급(service_tier)별 이탈률(churn 평균)을 계산하고 막대 차트로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_026: 티어별 이탈률 막대 차트
import matplotlib.pyplot as plt

churn_by_tier = (df.groupby('service_tier')['churn'].mean() * 100).round(2)
plt.figure(figsize=(7, 4))
churn_by_tier.plot(kind='bar', color='salmon')
plt.title('서비스 티어별 고객 이탈률(%)')
plt.xlabel('서비스 티어')
plt.ylabel('이탈률 (%)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print(churn_by_tier)
"""
    })

    specs.append({
        "id": "TEST_027",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "직업군별로 잔액(balance)이 가장 높은 고객(Top 1)을 추출해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_027: 직업군별 최고 잔액 고객 추출
top1_per_job = df.sort_values('balance', ascending=False).groupby('job').head(1)[['job', 'customer_id', 'age', 'balance']]
result = top1_per_job.sort_values('balance', ascending=False).reset_index(drop=True)
print(result)
"""
    })

    specs.append({
        "id": "TEST_028",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "연령대(20대 이하, 30대, 40대, 50대 이상) 구간을 만들고, 연령대별 평균 신용점수와 고객 수를 표로 요약해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_028: 연령대별 신용점수 및 고객수 요약
bins = [0, 29, 39, 49, 120]
labels = ['20대 이하', '30대', '40대', '50대 이상']
age_group = pd.cut(df['age'], bins=bins, labels=labels)
age_summary = df.groupby(age_group).agg(고객수=('customer_id', 'count'), 평균신용점수=('credit_score', 'mean')).round(2)
print(age_summary)
"""
    })

    specs.append({
        "id": "TEST_029",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "결혼 상태(marital)별 주택대출(housing_loan) 보유 비율을 교차표(crosstab) 백분율로 산출해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_029: 교차표 정규화 백분율
ct = pd.crosstab(df['marital'], df['housing_loan'], normalize='index') * 100
print("결혼 상태별 주택대출 보유 비율 (%):")
print(ct.round(2))
"""
    })

    specs.append({
        "id": "TEST_030",
        "category": "그룹 집계 및 피벗 분석",
        "difficulty": "Intermediate",
        "question": "직업별 평균 연소득을 내림차순 정렬하여 수평 막대 그래프(Horizontal Bar Chart)로 그려줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_030: 직업별 평균 연소득 수평 막대 그래프
import matplotlib.pyplot as plt

job_inc = df.groupby('job')['annual_income'].mean().sort_values(ascending=True)
plt.figure(figsize=(8, 5))
job_inc.plot(kind='barh', color='mediumpurple')
plt.title('직업별 평균 연소득 (내림차순 정렬)')
plt.xlabel('평균 연소득 ($)')
plt.ylabel('직업')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    # -------------------------------------------------------------
    # Category 4: 시계열 및 트렌드 분석 (Time-Series & Trend Analysis)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_031",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Basic",
        "question": "고객 가입일(signup_date)을 기준으로 연도별 신규 가입자 수를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_031: 가입 연도별 신규 고객 수
years = pd.to_datetime(df['signup_date']).dt.year
yearly_signups = years.value_counts().sort_index()
print("연도별 신규 가입 고객 수:")
print(yearly_signups)
"""
    })

    specs.append({
        "id": "TEST_032",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Intermediate",
        "question": "2023년 월별 신규 가입자 수 추이를 계산하고 꺾은선 그래프(Line Plot)로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_032: 2023년 월별 가입자 수 추이 꺾은선 그래프
import matplotlib.pyplot as plt

df_dates = pd.to_datetime(df['signup_date'])
df_2023 = df[df_dates.dt.year == 2023].copy()
df_2023['month'] = pd.to_datetime(df_2023['signup_date']).dt.month
monthly_signups = df_2023['month'].value_counts().sort_index()

plt.figure(figsize=(8, 4))
plt.plot(monthly_signups.index, monthly_signups.values, marker='o', color='teal', linewidth=2)
plt.title('2023년 월별 신규 가입 고객 수 추이')
plt.xlabel('월 (Month)')
plt.ylabel('가입자 수')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
print(monthly_signups)
"""
    })

    specs.append({
        "id": "TEST_033",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Intermediate",
        "question": "거래 내역(df_tx)의 일별 총 거래금액을 집계하고, 7일 이동평균(Rolling Mean)을 계산해 상위 10일을 출력해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_033: 일별 거래액 및 7일 이동평균
df_tx['date'] = pd.to_datetime(df_tx['tx_datetime']).dt.date
daily_sum = df_tx.groupby('date')['amount'].sum().reset_index()
daily_sum = daily_sum.sort_values('date')
daily_sum['rolling_7d_mean'] = daily_sum['amount'].rolling(window=7, min_periods=1).mean().round(2)
print("일별 거래액 및 7일 이동평균 (상위 10일):")
print(daily_sum.head(10))
"""
    })

    specs.append({
        "id": "TEST_034",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Basic",
        "question": "거래 일시(tx_datetime)를 기준으로 요일(Day of Week)별 총 거래 건수와 거래 총액을 집계해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_034: 요일별 거래 건수 및 거래액 집계
tx_dt = pd.to_datetime(df_tx['tx_datetime'])
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_tx_copy = df_tx.copy()
df_tx_copy['day_of_week'] = tx_dt.dt.day_name()
dow_agg = df_tx_copy.groupby('day_of_week').agg(거래건수=('tx_id', 'count'), 거래총액=('amount', 'sum')).reindex(day_names).round(2)
print(dow_agg)
"""
    })

    specs.append({
        "id": "TEST_035",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Intermediate",
        "question": "거래 발생 시간대(Hour of Day, 0시~23시)별 거래 건수를 집계하고 막대 차트로 시각화해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_035: 시간대별 거래 건수 막대 차트
import matplotlib.pyplot as plt

hours = pd.to_datetime(df_tx['tx_datetime']).dt.hour
hourly_counts = hours.value_counts().sort_index()

plt.figure(figsize=(9, 4))
hourly_counts.plot(kind='bar', color='dodgerblue')
plt.title('시간대별 거래 발생 빈도 (0시~23시)')
plt.xlabel('시간 (Hour)')
plt.ylabel('거래 건수')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_036",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Advanced",
        "question": "2024년 월별 총 거래금액의 전월 대비 증감율(MoM Growth Rate, %)을 계산해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_036: 월별 거래액 및 MoM 증감율
df_tx_copy = df_tx.copy()
df_tx_copy['month'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.to_period('M')
monthly_tx = df_tx_copy.groupby('month')['amount'].sum().reset_index()
monthly_tx['MoM_Growth_pct'] = (monthly_tx['amount'].pct_change() * 100).round(2)
print("2024년 월별 거래액 및 전월 대비 증감율(MoM):")
print(monthly_tx)
"""
    })

    specs.append({
        "id": "TEST_037",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Advanced",
        "question": "거래 유형(tx_type)별 월별 총 거래금액 추이를 멀티 라인 그래프로 시각화해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_037: 거래 유형별 월별 금액 추이 멀티 라인
import matplotlib.pyplot as plt

df_tx_copy = df_tx.copy()
df_tx_copy['month'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.month
type_monthly = df_tx_copy.pivot_table(index='month', columns='tx_type', values='amount', aggfunc='sum')

plt.figure(figsize=(9, 4.5))
for col in type_monthly.columns:
    plt.plot(type_monthly.index, type_monthly[col], marker='o', label=col)
plt.title('거래 유형(tx_type)별 월별 총 거래금액 추이')
plt.xlabel('월 (Month)')
plt.ylabel('총 거래금액 ($)')
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_038",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Basic",
        "question": "2024년 분기(Q1, Q2, Q3, Q4)별 총 거래금액과 총 발생 수수료(fee)를 집계해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_038: 분기별 거래액 및 수수료 집계
quarters = pd.to_datetime(df_tx['tx_datetime']).dt.to_period('Q')
quarterly = df_tx.groupby(quarters).agg(총거래금액=('amount', 'sum'), 총수수료=('fee', 'sum')).round(2)
print("분기별 거래 및 수수료 실적:")
print(quarterly)
"""
    })

    specs.append({
        "id": "TEST_039",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Advanced",
        "question": "고객별 첫 거래일시와 마지막 거래일시를 계산하여 평균 활동 기간(일수)을 산출해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "text",
        "python_code": """# TEST_039: 고객별 거래 활동 기간 계산
tx_dt = pd.to_datetime(df_tx['tx_datetime'])
df_tx_copy = df_tx.copy()
df_tx_copy['datetime'] = tx_dt
cust_span = df_tx_copy.groupby('customer_id')['datetime'].agg(first_tx='min', last_tx='max')
cust_span['active_days'] = (cust_span['last_tx'] - cust_span['first_tx']).dt.total_seconds() / 86400
avg_days = cust_span['active_days'].mean()
print(f"거래 고객 수: {len(cust_span)}명")
print(f"고객별 평균 첫-마지막 거래 간격: {avg_days:.1f}일")
print(f"활동 기간 중앙값: {cust_span['active_days'].median():.1f}일")
"""
    })

    specs.append({
        "id": "TEST_040",
        "category": "시계열 및 트렌드 분석",
        "difficulty": "Intermediate",
        "question": "2024년 일별 거래금액의 누적합(Cumulative Sum)을 계산하고 누적 성장 곡선을 시각화해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_040: 거래금액 일별 누적 성장 곡선
import matplotlib.pyplot as plt

df_tx_copy = df_tx.copy()
df_tx_copy['date'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.date
daily_totals = df_tx_copy.groupby('date')['amount'].sum().sort_index()
cumsum_totals = daily_totals.cumsum()

plt.figure(figsize=(9, 4))
plt.plot(cumsum_totals.index, cumsum_totals.values, color='indigo', linewidth=2)
plt.title('2024년 일별 거래금액 누적 성장 곡선 (Cumulative Sum)')
plt.xlabel('일자')
plt.ylabel('누적 거래금액 ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print(f"연간 총 누적 거래액: {cumsum_totals.iloc[-1]:,.2f} 달러")
"""
    })

    return specs
