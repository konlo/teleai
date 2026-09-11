"""100 Test Definitions for Data Analysis Agent Benchmark Suite - Part 3 (Q041 ~ Q060).
"""

def get_part3_specs():
    specs = []

    # -------------------------------------------------------------
    # Category 5: 범주형 데이터 및 빈도 분석 (Categorical & Frequency Analysis)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_041",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Basic",
        "question": "고객의 직업군별 빈도수와 전체 대비 백분율(%)을 계산한 도수분포표를 만들어줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_041: 직업별 도수분포표
freq = df['job'].value_counts()
pct = (df['job'].value_counts(normalize=True) * 100).round(2)
freq_table = pd.DataFrame({'빈도수': freq, '백분율(%)': pct})
print(freq_table)
"""
    })

    specs.append({
        "id": "TEST_042",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Intermediate",
        "question": "거래 채널(channel)별 총 거래금액의 파레토 누적 비중(%)을 계산해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_042: 채널별 거래액 파레토 누적 비중
channel_amt = df_tx.groupby('channel')['amount'].sum().sort_values(ascending=False).reset_index()
channel_amt['비중(%)'] = (channel_amt['amount'] / channel_amt['amount'].sum() * 100).round(2)
channel_amt['누적비중(%)'] = channel_amt['비중(%)'].cumsum().round(2)
print("거래 채널별 파레토 분석:")
print(channel_amt)
"""
    })

    specs.append({
        "id": "TEST_043",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Basic",
        "question": "고객 연령을 20대 이하, 30대, 40대, 50대, 60대 이상으로 구간화(pd.cut)하여 각 연령대별 고객 수를 세어줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_043: 연령대 구간화 빈도표
bins = [0, 29, 39, 49, 59, 120]
labels = ['20대 이하', '30대', '40대', '50대', '60대 이상']
age_cuts = pd.cut(df['age'], bins=bins, labels=labels)
print(age_cuts.value_counts().sort_index())
"""
    })

    specs.append({
        "id": "TEST_044",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Intermediate",
        "question": "연소득(annual_income)을 4분위수 기준(pd.qcut, Q1~Q4)으로 4단계 등급을 부여하고 등급별 고객 수를 확인해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_044: 연소득 4분위 등급화
income_q = pd.qcut(df['annual_income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
q_counts = income_q.value_counts().sort_index()
print("연소득 분위 등급별 인원수:")
print(q_counts)
"""
    })

    specs.append({
        "id": "TEST_045",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Basic",
        "question": "만족도 점수(1점~5점)의 응답 분포를 파이 차트(Pie Chart)로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_045: 만족도 파이 차트
import matplotlib.pyplot as plt

sat_counts = df['satisfaction_score'].dropna().value_counts().sort_index()
plt.figure(figsize=(6, 6))
plt.pie(sat_counts, labels=[f'{int(k)}점' for k in sat_counts.index], autopct='%1.1f%%', startangle=140)
plt.title('고객 만족도 점수(1~5점) 응답 비율')
plt.tight_layout()
plt.show()
print(sat_counts)
"""
    })

    specs.append({
        "id": "TEST_046",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Intermediate",
        "question": "고객의 접속 기기(device_type) 비율을 도넛 차트(Donut Chart) 형태로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_046: 기기 비율 도넛 차트
import matplotlib.pyplot as plt

device_counts = df['device_type'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(device_counts, labels=device_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.8)
center_circle = plt.Circle((0,0), 0.60, fc='white')
plt.gca().add_artist(center_circle)
plt.title('접속 기기(Device Type) 비중 (도넛 차트)')
plt.tight_layout()
plt.show()
print(device_counts)
"""
    })

    specs.append({
        "id": "TEST_047",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Basic",
        "question": "교육수준(education)에서 'unknown' 값을 제외하거나 최빈값(mode)으로 대체했을 때의 최빈값을 확인해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_047: 결측 범주 제외 최빈값 도출
valid_edu = df[df['education'] != 'unknown']['education']
mode_val = valid_edu.mode()[0]
print(f"'unknown' 제외 최빈 교육수준: {mode_val}")
print("유효 교육수준 빈도:")
print(valid_edu.value_counts())
"""
    })

    specs.append({
        "id": "TEST_048",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Basic",
        "question": "직업군 중 전체 비중이 5% 미만인 소수 직업군을 식별해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_048: 소수 범주(5% 미만) 식별
job_ratios = (df['job'].value_counts(normalize=True) * 100).round(2)
minority_jobs = job_ratios[job_ratios < 5.0]
print("5% 미만 소수 직업군 목록 및 비율(%):")
print(minority_jobs)
"""
    })

    specs.append({
        "id": "TEST_049",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Basic",
        "question": "거래 유형(tx_type)별 실패/거절(is_declined=1) 건수와 거절율(%)을 계산해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_049: 거래 유형별 거절율 산출
tx_declined = df_tx.groupby('tx_type').agg(
    총거래건수=('tx_id', 'count'),
    거절건수=('is_declined', 'sum'),
    거절율_pct=('is_declined', lambda x: round(x.mean() * 100, 2))
)
print(tx_declined)
"""
    })

    specs.append({
        "id": "TEST_050",
        "category": "범주형 데이터 및 빈도 분석",
        "difficulty": "Intermediate",
        "question": "서비스 티어(service_tier)와 접속 기기(device_type) 간의 빈도 교차표를 히트맵(Heatmap)으로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_050: 티어 x 기기 히트맵
import matplotlib.pyplot as plt
import seaborn as sns

ct = pd.crosstab(df['service_tier'], df['device_type'])
plt.figure(figsize=(7, 4))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('서비스 티어 x 접속 기기 교차 빈도 히트맵')
plt.xlabel('접속 기기')
plt.ylabel('서비스 티어')
plt.tight_layout()
plt.show()
print(ct)
"""
    })

    # -------------------------------------------------------------
    # Category 6: 상관관계 및 변수 간 연관성 분석 (Correlation & Relationships)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_051",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Basic",
        "question": "주요 수치형 변수(age, annual_income, balance, credit_score, campaign_contact_count) 간의 피어슨 상관계수 행렬을 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_051: 수치형 변수 피어슨 상관계수 행렬
cols = ['age', 'annual_income', 'balance', 'credit_score', 'campaign_contact_count']
corr_matrix = df[cols].corr().round(3)
print("피어슨 상관계수 행렬:")
print(corr_matrix)
"""
    })

    specs.append({
        "id": "TEST_052",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Intermediate",
        "question": "수치형 변수들 간의 상관계수 행렬을 히트맵(Heatmap)으로 시각화하고 상관계수 값을 표기해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_052: 상관계수 히트맵
import matplotlib.pyplot as plt
import seaborn as sns

cols = ['age', 'annual_income', 'balance', 'credit_score', 'campaign_contact_count']
corr = df[cols].corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('수치형 변수 간 상관계수 히트맵')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_053",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Basic",
        "question": "연소득(annual_income)과 잔액(balance) 간의 스피어만 순위상관계수를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_053: 소득-잔액 스피어만 순위상관계수
spearman_corr = df['annual_income'].corr(df['balance'], method='spearman')
pearson_corr = df['annual_income'].corr(df['balance'], method='pearson')
print(f"스피어만 순위상관계수: {spearman_corr:.4f}")
print(f"피어슨 선형상관계수: {pearson_corr:.4f}")
"""
    })

    specs.append({
        "id": "TEST_054",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Basic",
        "question": "신용점수(credit_score)와 고객 이탈(churn) 간의 상관계수를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_054: 신용점수와 이탈 여부 간 상관성
corr_val = df['credit_score'].corr(df['churn'])
print(f"신용점수와 이탈(churn) 간 상관계수: {corr_val:.4f}")
print(f"이탈 여부별 평균 신용점수:")
print(df.groupby('churn')['credit_score'].mean().round(2))
"""
    })

    specs.append({
        "id": "TEST_055",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Intermediate",
        "question": "연소득과 잔액의 관계를 보여주는 산점도(Scatter Plot)를 그리고 회귀 추세선을 추가해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_055: 연소득 vs 잔액 산점도 및 추세선
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='annual_income', y='balance', scatter_kws={'alpha':0.3, 'color':'blue'}, line_kws={'color':'red'})
plt.title('연소득 vs 계좌 잔액 관계 (산점도 및 회귀선)')
plt.xlabel('연소득 ($)')
plt.ylabel('잔액 ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_056",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Intermediate",
        "question": "신용점수와 연소득의 산점도를 이탈 여부(churn)에 따라 색상(hue)을 다르게 하여 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_056: 신용점수 vs 연소득 (이탈 여부 구분 산점도)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='credit_score', y='annual_income', hue='churn', alpha=0.6, palette=['gray', 'red'])
plt.title('신용점수 vs 연소득 (이탈여부별)')
plt.xlabel('신용점수')
plt.ylabel('연소득 ($)')
plt.legend(title='이탈(Churn)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_057",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Basic",
        "question": "고객 연령(age)과 캠페인 접촉 횟수(campaign_contact_count) 간의 공분산(Covariance)을 구해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_057: 연령-캠페인 접촉 횟수 공분산
cov_val = df['age'].cov(df['campaign_contact_count'])
print(f"연령과 캠페인 접촉 횟수 간 공분산: {cov_val:.4f}")
"""
    })

    specs.append({
        "id": "TEST_058",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Intermediate",
        "question": "잔액(balance)과 가장 상관관계가 높은(양 또는 음) 상위 3개 수치형 변수를 순서대로 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_058: 잔액과의 상관계수 상위 변수
num_df = df.select_dtypes(include=['number'])
corrs_with_balance = num_df.corr()['balance'].drop('balance')
top3_corr = corrs_with_balance.abs().sort_values(ascending=False).head(3)
result = pd.DataFrame({'변수': top3_corr.index, '절대값_상관계수': top3_corr.values.round(4), '실제_상관계수': corrs_with_balance[top3_corr.index].values.round(4)})
print(result)
"""
    })

    specs.append({
        "id": "TEST_059",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Intermediate",
        "question": "서비스 티어(service_tier)별로 연소득과 잔액 간의 상관계수가 어떻게 다른지 그룹별로 비교해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_059: 티어별 소득-잔액 상관계수
tier_corr = df.groupby('service_tier').apply(lambda g: g['annual_income'].corr(g['balance'])).round(4)
print("서비스 티어별 소득-잔액 상관계수:")
print(tier_corr)
"""
    })

    specs.append({
        "id": "TEST_060",
        "category": "상관관계 및 변수 간 연관성 분석",
        "difficulty": "Intermediate",
        "question": "주택대출 보유 여부(housing_loan)에 따라 신용점수와 잔액의 분포를 1행 2열 서브플롯 산점도로 비교해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_060: 주택대출 여부별 서브플롯 산점도
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, hl, color in zip(axes, ['yes', 'no'], ['purple', 'teal']):
    sub = df[df['housing_loan'] == hl]
    ax.scatter(sub['credit_score'], sub['balance'], alpha=0.4, color=color)
    ax.set_title(f'주택대출: {hl} (n={len(sub)})')
    ax.set_xlabel('신용점수')
    ax.grid(True, linestyle='--', alpha=0.5)
axes[0].set_ylabel('잔액 ($)')
plt.suptitle('주택대출 보유 여부별 신용점수 vs 잔액 비교')
plt.tight_layout()
plt.show()
"""
    })

    return specs
