"""100 Test Definitions for Data Analysis Agent Benchmark Suite - Part 4 (Q061 ~ Q080).
"""

def get_part4_specs():
    specs = []

    # -------------------------------------------------------------
    # Category 7: 이상치 탐지 및 분포 통계 (Outliers & Distribution Diagnostics)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_061",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Basic",
        "question": "잔액(balance)과 연소득(annual_income) 컬럼의 왜도(Skewness)와 첨도(Kurtosis)를 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_061: 왜도 및 첨도 통계량 계산
skewness = df[['balance', 'annual_income']].skew().round(3)
kurtosis = df[['balance', 'annual_income']].kurt().round(3)
shape_stats = pd.DataFrame({'왜도(Skewness)': skewness, '첨도(Kurtosis)': kurtosis})
print("분포 비대칭성 및 꼬리 두께 통계:")
print(shape_stats)
"""
    })

    specs.append({
        "id": "TEST_062",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "IQR 방식을 사용하여 잔액(balance) 컬럼의 Q1, Q3, IQR, 상한선(Upper Fence)을 구하고 상한선을 넘는 이상치 개수를 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_062: IQR 기반 잔액 이상치 탐지
q1 = df['balance'].quantile(0.25)
q3 = df['balance'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr
outliers = df[(df['balance'] > upper_fence) | (df['balance'] < lower_fence)]
print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"상한선(Upper Fence): {upper_fence:.2f}, 하한선(Lower Fence): {lower_fence:.2f}")
print(f"이상치(Outlier) 고객 수: {len(outliers)}명 (상한 초과: {(df['balance'] > upper_fence).sum()}명)")
"""
    })

    specs.append({
        "id": "TEST_063",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "IQR 기준 잔액 상위 이상치 고객들의 평균 연소득과 주요 직업 상위 3개를 조회해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_063: 잔액 이상치 고객 프로필 분석
q1 = df['balance'].quantile(0.25)
q3 = df['balance'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
outliers = df[df['balance'] > upper_fence]
print(f"이상치 고객 수: {len(outliers)}명")
print(f"평균 연소득: {outliers['annual_income'].mean():.2f} 달러")
print("주요 직업 TOP 3:")
print(outliers['job'].value_counts().head(3))
"""
    })

    specs.append({
        "id": "TEST_064",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "Z-Score 절대값이 3을 초과하는 연소득(annual_income) 극단값 고객들을 찾고 인원수를 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_064: Z-Score 기준 극단값 탐지
mean_inc = df['annual_income'].mean()
std_inc = df['annual_income'].std()
z_scores = (df['annual_income'] - mean_inc) / std_inc
outliers_z = df[z_scores.abs() > 3]
print(f"연소득 평균: {mean_inc:.2f}, 표준편차: {std_inc:.2f}")
print(f"|Z| > 3 극단치 고객 수: {len(outliers_z)}명")
if len(outliers_z) > 0:
    print(outliers_z[['customer_id', 'job', 'annual_income']])
"""
    })

    specs.append({
        "id": "TEST_065",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "거래 내역(df_tx)의 거래금액(amount)에서 IQR 기준 상위 이상치 거래 건수와 최고 거래금액을 확인해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "text",
        "python_code": """# TEST_065: 거래금액 IQR 이상치 건수
q1 = df_tx['amount'].quantile(0.25)
q3 = df_tx['amount'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
high_tx = df_tx[df_tx['amount'] > upper_bound]
print(f"금액 IQR 상한선: {upper_bound:.2f} 달러")
print(f"고액 이상치 거래 건수: {len(high_tx)}건 (전체의 {len(high_tx)/len(df_tx)*100:.2f}%)")
print(f"최고 거래 금액: {df_tx['amount'].max():.2f} 달러")
"""
    })

    specs.append({
        "id": "TEST_066",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "직업(job)별 잔액(balance) 분포의 이상치를 한눈에 비교할 수 있는 박스플롯(Boxplot)을 그려줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_066: 직업별 잔액 분포 박스플롯
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='job', y='balance', palette='Set3')
plt.title('직업별 고객 계좌 잔액 분포 및 이상치')
plt.xlabel('직업')
plt.ylabel('잔액 ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_067",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "잔액(balance) 컬럼에 대해 상위 1% 값으로 상한을 대체하는 윈저라이징(Capping)을 적용하기 전과 후의 최댓값 및 평균을 비교해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_067: 잔액 윈저라이징 상한 캡핑
cap_val = df['balance'].quantile(0.99)
capped_bal = df['balance'].clip(upper=cap_val)
comp = pd.DataFrame({
    '구분': ['적용 전 (Original)', '적용 후 (Capped 99%)'],
    '최댓값': [df['balance'].max(), capped_bal.max()],
    '평균값': [round(df['balance'].mean(), 2), round(capped_bal.mean(), 2)],
    '표준편차': [round(df['balance'].std(), 2), round(capped_bal.std(), 2)]
})
print(comp)
"""
    })

    specs.append({
        "id": "TEST_068",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Intermediate",
        "question": "거래금액(amount)의 원본 분포와 로그 변환(np.log1p) 후의 왜도를 비교하고 변환 후 히스토그램을 그려줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_068: 로그 변환 전후 왜도 및 히스토그램
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

log_amount = np.log1p(df_tx['amount'])
print(f"변환 전 왜도: {df_tx['amount'].skew():.3f}")
print(f"로그 변환(log1p) 후 왜도: {log_amount.skew():.3f}")

plt.figure(figsize=(8, 4))
sns.histplot(log_amount, kde=True, bins=30, color='darkcyan')
plt.title('거래금액 로그 변환(log1p) 후 분포 (왜도 개선)')
plt.xlabel('log1p(Amount)')
plt.ylabel('빈도')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_069",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Basic",
        "question": "고객 연령(age) 데이터에 정상 범위(18세~100세)를 벗어난 비정상 데이터가 존재하는지 무결성을 검증해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_069: 연령 데이터 무결성 검증
invalid_age = df[(df['age'] < 18) | (df['age'] > 100)]
print(f"최소 연령: {df['age'].min()}세, 최대 연령: {df['age'].max()}세")
if len(invalid_age) == 0:
    print("검증 결과: 모든 연령 데이터가 정상 범위(18세~100세) 내에 있습니다 (무결성 통과).")
else:
    print(f"경고: 비정상 연령 데이터 {len(invalid_age)}건 발견!")
"""
    })

    specs.append({
        "id": "TEST_070",
        "category": "이상치 탐지 및 분포 통계",
        "difficulty": "Basic",
        "question": "신용점수(credit_score)의 분포를 정규분포 적합선(KDE)과 함께 히스토그램으로 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_070: 신용점수 정규분포 적합 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
sns.histplot(df['credit_score'], kde=True, bins=25, color='forestgreen')
plt.title('고객 신용점수(Credit Score) 분포')
plt.xlabel('신용점수')
plt.ylabel('고객 수')
plt.axvline(df['credit_score'].mean(), color='red', linestyle='--', label=f"평균: {df['credit_score'].mean():.1f}")
plt.legend()
plt.tight_layout()
plt.show()
"""
    })

    # -------------------------------------------------------------
    # Category 8: 비즈니스 KPI 및 세그먼트 분석 (Business KPIs & Segmentation)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_071",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Intermediate",
        "question": "전체 평균 잔액 대비 각 직업군의 잔액 비율 지수(Index = 직업평균/전체평균 * 100)를 산출해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_071: 직업군별 잔액 지수(Index) 산출
total_mean = df['balance'].mean()
job_mean = df.groupby('job')['balance'].mean()
balance_idx = ((job_mean / total_mean) * 100).round(1)
idx_df = pd.DataFrame({'평균잔액': job_mean.round(2), '잔액지수(Index)': balance_idx}).sort_values('잔액지수(Index)', ascending=False)
print(f"전체 평균 잔액: {total_mean:.2f} 달러")
print(idx_df)
"""
    })

    specs.append({
        "id": "TEST_072",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Basic",
        "question": "전체 고객의 이탈율(Churn Rate, %)과 성별(gender)별 이탈율을 비교해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_072: 전체 및 성별 이탈율 산출
overall_churn = df['churn'].mean() * 100
gender_churn = (df.groupby('gender')['churn'].mean() * 100).round(2)
print(f"전체 고객 이탈율: {overall_churn:.2f}%")
print("성별 이탈율:")
print(gender_churn)
"""
    })

    specs.append({
        "id": "TEST_073",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Intermediate",
        "question": "고객별 총 거래 건수, 총 거래 금액, 1회당 평균 거래금액(AOV)을 계산하고 상위 5명을 출력해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_073: 고객별 거래 빈도 및 객단가(AOV)
cust_tx = df_tx.groupby('customer_id').agg(
    거래건수=('tx_id', 'count'),
    총거래액=('amount', 'sum'),
    평균거래액_AOV=('amount', 'mean')
).round(2)
top5 = cust_tx.sort_values('총거래액', ascending=False).head(5)
print("총 거래금액 상위 5명 고객 지표:")
print(top5)
"""
    })

    specs.append({
        "id": "TEST_074",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Advanced",
        "question": "거래 내역을 바탕으로 고객별 최근 거래일 경과일수(Recency), 거래 빈도(Frequency), 총 거래금액(Monetary) RFM 기초 테이블을 생성해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_074: RFM 기초 지표 생성
df_tx_copy = df_tx.copy()
df_tx_copy['date'] = pd.to_datetime(df_tx_copy['tx_datetime'])
snapshot_date = df_tx_copy['date'].max() + pd.Timedelta(days=1)
rfm = df_tx_copy.groupby('customer_id').agg(
    Recency=('date', lambda d: (snapshot_date - d.max()).days),
    Frequency=('tx_id', 'count'),
    Monetary=('amount', 'sum')
).round(2)
print("RFM 기초 지표 요약 (상위 5건):")
print(rfm.head(5))
"""
    })

    specs.append({
        "id": "TEST_075",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Advanced",
        "question": "RFM 지표를 바탕으로 상위 20%를 'VIP', 하위 20%를 'At-Risk', 나머지를 'Standard'로 세그먼트 분류하고 세그먼트별 고객 수를 집계해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_075: RFM 세그먼트 분류
df_tx_copy = df_tx.copy()
df_tx_copy['date'] = pd.to_datetime(df_tx_copy['tx_datetime'])
snapshot = df_tx_copy['date'].max() + pd.Timedelta(days=1)
rfm = df_tx_copy.groupby('customer_id').agg(
    Recency=('date', lambda d: (snapshot - d.max()).days),
    Frequency=('tx_id', 'count'),
    Monetary=('amount', 'sum')
)
m_p80 = rfm['Monetary'].quantile(0.80)
m_p20 = rfm['Monetary'].quantile(0.20)
def classify_rfm(row):
    if row['Monetary'] >= m_p80:
        return 'VIP'
    elif row['Monetary'] <= m_p20:
        return 'At-Risk'
    else:
        return 'Standard'
rfm['Segment'] = rfm.apply(classify_rfm, axis=1)
seg_counts = rfm['Segment'].value_counts()
print("RFM 세그먼트별 고객 분포:")
print(seg_counts)
"""
    })

    specs.append({
        "id": "TEST_076",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Intermediate",
        "question": "서비스 티어(service_tier)별 고객 1인당 평균 거래 기여액(ARPU)을 계산해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_076: 서비스 티어별 ARPU 산출
merged = df.merge(df_tx.groupby('customer_id')['amount'].sum().reset_index(), on='customer_id', how='left')
merged['amount'] = merged['amount'].fillna(0)
arpu_df = merged.groupby('service_tier').agg(
    총고객수=('customer_id', 'count'),
    총매출액=('amount', 'sum'),
    ARPU=('amount', 'mean')
).round(2)
print("서비스 티어별 ARPU (1인당 평균 거래액):")
print(arpu_df.sort_values('ARPU', ascending=False))
"""
    })

    specs.append({
        "id": "TEST_077",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Intermediate",
        "question": "캠페인 접촉 횟수(1~2회, 3~4회, 5회 이상) 구간별 고객 이탈율을 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "table",
        "python_code": """# TEST_077: 캠페인 접촉 횟수별 이탈율 분석
bins = [0, 2, 4, 100]
labels = ['1~2회 (낮음)', '3~4회 (중간)', '5회 이상 (높음)']
df_copy = df.copy()
df_copy['contact_grp'] = pd.cut(df_copy['campaign_contact_count'], bins=bins, labels=labels)
churn_by_contact = df_copy.groupby('contact_grp').agg(
    고객수=('customer_id', 'count'),
    이탈율_pct=('churn', lambda x: round(x.mean() * 100, 2))
)
print(churn_by_contact)
"""
    })

    specs.append({
        "id": "TEST_078",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Intermediate",
        "question": "신용점수가 580점 미만이고 잔액이 마이너스(< 0)인 고위험(High-Risk) 고객의 수와 전체 대비 비율을 산출해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_078: 고위험 고객군 식별
high_risk = df[(df['credit_score'] < 580) & (df['balance'] < 0)]
pct = (len(high_risk) / len(df)) * 100
print(f"고위험(신용<580 & 잔액<0) 고객 수: {len(high_risk)}명")
print(f"전체 대비 비율: {pct:.2f}%")
"""
    })

    specs.append({
        "id": "TEST_079",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Intermediate",
        "question": "고객들의 가입일(signup_date)부터 마지막 활동일(last_active_date)까지의 서비스 이용 기간(일수) 평균과 중앙값을 계산해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_079: 고객 서비스 이용 기간 통계
signup = pd.to_datetime(df['signup_date'])
last_act = pd.to_datetime(df['last_active_date'])
tenure_days = (last_act - signup).dt.days
print(f"평균 서비스 이용 일수: {tenure_days.mean():.1f}일")
print(f"이용 일수 중앙값: {tenure_days.median():.1f}일")
print(f"최소: {tenure_days.min()}일, 최대: {tenure_days.max()}일")
"""
    })

    specs.append({
        "id": "TEST_080",
        "category": "비즈니스 KPI 및 세그먼트 분석",
        "difficulty": "Advanced",
        "question": "서비스 티어별 평균 거래금액과 평균 잔액을 나란히 비교하는 그룹 막대 차트(Grouped Bar Chart)를 시각화해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_080: 서비스 티어별 거래액 및 잔액 그룹 막대 차트
import matplotlib.pyplot as plt

cust_spend = df_tx.groupby('customer_id')['amount'].sum().reset_index(name='total_tx')
m = df.merge(cust_spend, on='customer_id', how='left')
m['total_tx'] = m['total_tx'].fillna(0)
tier_comp = m.groupby('service_tier')[['balance', 'total_tx']].mean().round(2)

tier_comp.plot(kind='bar', figsize=(8, 4.5), color=['royalblue', 'orange'])
plt.title('서비스 티어별 평균 계좌 잔액 vs 평균 총 거래액 비교')
plt.xlabel('서비스 티어')
plt.ylabel('금액 ($)')
plt.xticks(rotation=0)
plt.legend(['평균 잔액', '평균 총거래액'])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print(tier_comp)
"""
    })

    return specs
