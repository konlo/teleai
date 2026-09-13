"""Level 2 Definitions Part 2 (L2_026 ~ L2_050):
- 📋 Multi-Dimensional Pivot Tables Continued (10 items: L2_026 ~ L2_035)
- 🔎 Outlier Detection (IQR / Z-score) & Condition-based Analysis (15 items: L2_036 ~ L2_050)
"""

def get_level2_part2():
    specs = []
    
    # -------------------------------------------------------------
    # 📋 Multi-Dimensional Pivot Tables Continued (L2_026 ~ L2_035)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_026",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "연령대(20대, 30대, 40대, 50대 이상)와 주택대출(housing) 여부에 따른 평균 잔액 피벗 테이블을 생성해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_026: 연령대 x 주택대출 평균 잔액 피벗
bins = [0, 29, 39, 49, 120]
labels = ['20대 이하', '30대', '40대', '50대 이상']
df_copy = df_bank.copy()
df_copy['age_group'] = pd.cut(df_copy['age'], bins=bins, labels=labels)
pivot_age_h = df_copy.pivot_table(index='age_group', columns='housing', values='balance', aggfunc='mean', observed=False).round(2)
print("연령대 x 주택대출 평균 잔액 피벗:")
print(pivot_age_h)
"""
    })

    specs.append({
        "id": "L2_027",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업(job)과 정기예금 가입 여부(y)별 평균 통화 상담 시간(duration) 피벗 테이블을 margins=True와 함께 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_027: 직업 x 가입여부 평균 상담시간 피벗
pivot_jy_dur = df_bank.pivot_table(index='job', columns='y', values='duration', aggfunc='mean', margins=True, margins_name='전체평균').round(1)
print("직업 x 가입여부 평균 상담시간(초) 피벗:")
print(pivot_jy_dur)
"""
    })

    specs.append({
        "id": "L2_028",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 등급(Pclass)과 탑승지(Embarked)별 평균 운임 요금(Fare) 피벗 테이블을 소수점 둘째 자리까지 출력해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_028: Pclass x Embarked 평균 요금 피벗
pivot_pe_fare = df_titanic.pivot_table(index='Pclass', columns='Embarked', values='Fare', aggfunc='mean').round(2)
print("타이타닉 등급 x 탑승지 평균 운임 피벗:")
print(pivot_pe_fare)
"""
    })

    specs.append({
        "id": "L2_029",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "교육 수준(education)과 혼인 상태(marital)별 정기예금 가입 성공률(%) 교차표를 백분율로 산출해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_029: 교육수준 x 혼인상태 가입 성공률 교차표
crosstab_em = (df_bank.groupby(['education', 'marital'])['y'].apply(lambda s: (s == 'yes').mean() * 100)).unstack().round(2)
print("교육수준 x 혼인상태 예금 가입 성공률(%):")
print(crosstab_em)
"""
    })

    specs.append({
        "id": "L2_030",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업군별로 전체 고객 중 잔액이 음수(< 0)인 고객의 비율(%)과 평균 음수 잔액을 표로 정리해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_030: 직업군별 마이너스 잔액 비율 및 평균 음수액
def neg_stats(group):
    neg_mask = group['balance'] < 0
    neg_ratio = neg_mask.mean() * 100
    neg_mean = group.loc[neg_mask, 'balance'].mean() if neg_mask.sum() > 0 else 0.0
    return pd.Series({'마이너스비율(%)': round(neg_ratio, 2), '평균음수잔액': round(neg_mean, 2)})

job_neg = df_bank.groupby('job').apply(neg_stats, include_groups=False).reset_index()
print("직업별 마이너스 잔액 고객 비율 및 평균 음수잔액:")
print(job_neg.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_031",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "월(month)과 주택대출(housing) 여부별 고객 수 피벗 테이블을 만들고 월 순서대로 정렬해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_031: 월별 주택대출 피벗 테이블 정렬
months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
pivot_mh = df_bank.pivot_table(index='month', columns='housing', values='id', aggfunc='count', fill_value=0)
pivot_mh = pivot_mh.reindex([m for m in months_order if m in pivot_mh.index])
print("월별 주택대출 유무 고객수 피벗:")
print(pivot_mh)
"""
    })

    specs.append({
        "id": "L2_032",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 탑승 등급(Pclass)과 성별(Sex)별 평균 연령(Age)과 승객 수를 동시에 집계한 피벗 테이블을 출력해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_032: 타이타닉 등급 x 성별 평균나이 및 인원수
pivot_ps_multi = df_titanic.pivot_table(index='Pclass', columns='Sex', values='Age', aggfunc=['count', 'mean']).round(1)
print("타이타닉 등급 x 성별 승객수 및 평균연령 피벗:")
print(pivot_ps_multi)
"""
    })

    specs.append({
        "id": "L2_033",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업별로 캠페인 접촉 횟수(campaign)의 평균, 중앙값, 최댓값을 요약해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_033: 직업별 접촉 횟수 다중 집계
job_camp = df_bank.groupby('job')['campaign'].agg(['mean', 'median', 'max']).round(2).reset_index()
job_camp.columns = ['직업', '평균접촉수', '중앙값', '최대접촉수']
print("직업별 접촉 횟수 통계 요약:")
print(job_camp.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_034",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "이전 마케팅 결과(poutcome)와 이번 가입 결과(y) 간의 교차 빈도표를 작성하고 전체 대비 백분율(normalize=True)을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_034: poutcome x y 교차 빈도 및 정규화 백분율
ct_pct = (pd.crosstab(df_bank['poutcome'], df_bank['y'], normalize=True) * 100).round(2)
print("이전 결과 x 이번 가입 전체 대비 비율 (%):")
print(ct_pct)
"""
    })

    specs.append({
        "id": "L2_035",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업(job)별로 잔액(balance)이 0 이상인 정상 고객들만의 평균 잔액과 고객 수를 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_035: 직업별 잔액 >= 0 정상 고객 통계
non_neg = df_bank[df_bank['balance'] >= 0]
pos_job_bal = non_neg.groupby('job').agg(정상고객수=('id', 'count'), 평균정상잔액=('balance', 'mean')).round(2).reset_index()
print("직업별 잔액 0 이상 고객 평균 잔액:")
print(pos_job_bal.to_string(index=False))
"""
    })

    # -------------------------------------------------------------
    # 🔎 Outlier Detection (IQR / Z-score) & Analysis (L2_036 ~ L2_050)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_036",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "계좌 잔액(balance)의 IQR(Q3 - Q1)을 계산하고 상한 이상치 기준선(Q3 + 1.5*IQR)을 넘는 이상치 고객 수를 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_036: 잔액 IQR 이상치 기준선 및 건수
q1 = df_bank['balance'].quantile(0.25)
q3 = df_bank['balance'].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
outliers = df_bank[df_bank['balance'] > upper_fence]
print(f"Q1: ${q1:,.1f}, Q3: ${q3:,.1f}, IQR: ${iqr:,.1f}")
print(f"상한 이상치 기준선(Upper Fence): ${upper_fence:,.2f}")
print(f"이상치 고객 수: {len(outliers)}명 (전체의 {len(outliers)/len(df_bank)*100:.2f}%)")
"""
    })

    specs.append({
        "id": "L2_037",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "잔액(balance)의 IQR 상한 이상치 고객들의 평균 연령과 주요 직업 TOP 3를 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_037: 잔액 이상치 고객 특성 분석
q1 = df_bank['balance'].quantile(0.25)
q3 = df_bank['balance'].quantile(0.75)
upper_fence = q3 + 1.5 * (q3 - q1)
outliers = df_bank[df_bank['balance'] > upper_fence]
top3 = outliers['job'].value_counts().head(3).reset_index()
top3.columns = ['직업', '이상치고객수']
print(f"잔액 이상치 고객 평균 나이: {outliers['age'].mean():.1f}세")
print("이상치 고객 주요 직업 TOP 3:")
print(top3.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_038",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "잔액 이상치 고객(상한 1.5*IQR 초과)을 제외한 일반 고객들의 직업별 평균 잔액을 계산하고 이상치 포함 전/후 평균을 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_038: 이상치 제외 전/후 평균 잔액 비교
q1 = df_bank['balance'].quantile(0.25)
q3 = df_bank['balance'].quantile(0.75)
upper_fence = q3 + 1.5 * (q3 - q1)
filtered_df = df_bank[df_bank['balance'] <= upper_fence]

comp = pd.DataFrame({
    '직업': sorted(df_bank['job'].unique()),
    '전체평균': df_bank.groupby('job')['balance'].mean().round(1).values,
    '이상치제외평균': filtered_df.groupby('job')['balance'].mean().round(1).values
})
print("이상치 제외 전/후 직업별 평균 잔액 비교:")
print(comp.head(5).to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_039",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "통화 상담 시간(duration)의 Z-Score가 3을 초과하는 극단치 통화 건수와 최장 통화 시간을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_039: 상담시간 Z-score 극단치 탐지
mean_d = df_bank['duration'].mean()
std_d = df_bank['duration'].std()
z_d = (df_bank['duration'] - mean_d) / std_d
extreme_calls = df_bank[z_d > 3]
print(f"상담시간 평균: {mean_d:.1f}초, 표준편차: {std_d:.1f}초")
print(f"Z-score > 3 극단치 통화 건수: {len(extreme_calls)}건")
print(f"최장 통화 시간: {df_bank['duration'].max()}초 ({df_bank['duration'].max()/60:.1f}분)")
"""
    })

    specs.append({
        "id": "L2_040",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 요금(Fare)에서 IQR 기준 상한 이상치(Q3 + 1.5*IQR)를 초과하는 고액 요금 승객 수와 이들의 생존율을 구해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_040: 타이타닉 요금 IQR 이상치 분석
q1 = df_titanic['Fare'].quantile(0.25)
q3 = df_titanic['Fare'].quantile(0.75)
fence = q3 + 1.5 * (q3 - q1)
high_fares = df_titanic[df_titanic['Fare'] > fence]
print(f"요금 상한 기준선: ${fence:.2f}")
print(f"이상치 고액 요금 승객 수: {len(high_fares)}명")
print(f"고액 승객 생존율: {high_fares['Survived'].mean()*100:.2f}%")
"""
    })

    specs.append({
        "id": "L2_041",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "잔액(balance)의 왜도(Skewness)와 첨도(Kurtosis)를 계산하고, 로그 변환(np.log1p) 적용 후 왜도 변화를 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_041: 잔액 왜도/첨도 및 로그 변환 비교
orig_skew = df_bank['balance'].skew()
orig_kurt = df_bank['balance'].kurt()
# 양수 잔액에 대해서만 log1p 적용
pos_bal = df_bank[df_bank['balance'] > 0]['balance']
log_skew = np.log1p(pos_bal).skew()
print(f"원본 잔액 왜도(Skewness): {orig_skew:.3f}, 첨도(Kurtosis): {orig_kurt:.3f}")
print(f"양수 잔액 로그 변환 후 왜도: {log_skew:.3f} (왜도 대폭 개선)")
"""
    })

    specs.append({
        "id": "L2_042",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "직업(job)별 잔액(balance)의 이상치 분포를 한눈에 비교할 수 있도록 박스플롯(Boxplot)을 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_042: 직업별 잔액 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
df_bank.boxplot(column='balance', by='job', figsize=(10, 5), rot=45)
plt.title('직업별 계좌 잔액 분포 및 이상치')
plt.suptitle('')
plt.xlabel('직업')
plt.ylabel('잔액 ($)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_043",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "잔액(balance)의 상위 1% 값으로 상한 캡핑(Winsorization)을 적용하기 전과 후의 평균 및 표준편차를 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_043: 잔액 99% 윈저라이징 캡핑
cap_99 = df_bank['balance'].quantile(0.99)
capped = df_bank['balance'].clip(upper=cap_99)
comp = pd.DataFrame({
    '구분': ['적용 전 (Original)', '적용 후 (Capped 99%)'],
    '최댓값': [df_bank['balance'].max(), capped.max()],
    '평균값': [round(df_bank['balance'].mean(), 2), round(capped.mean(), 2)],
    '표준편차': [round(df_bank['balance'].std(), 2), round(capped.std(), 2)]
})
print(comp.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_044",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 요금(Fare)의 왜도를 구하고 log1p 변환 후의 왜도 개선 폭을 출력해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_044: 타이타닉 요금 왜도 및 로그 변환
f_orig_skew = df_titanic['Fare'].skew()
f_log_skew = np.log1p(df_titanic['Fare']).skew()
print(f"타이타닉 원본 요금 왜도: {f_orig_skew:.3f}")
print(f"log1p 변환 후 요금 왜도: {f_log_skew:.3f}")
"""
    })

    specs.append({
        "id": "L2_045",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 등급(Pclass)별 요금(Fare)의 이상치 유무를 비교하는 박스플롯을 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_045: 타이타닉 등급별 요금 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4.5))
df_titanic.boxplot(column='Fare', by='Pclass', figsize=(7, 4.5))
plt.title('탑승 등급(Pclass)별 요금(Fare) 분포 및 이상치')
plt.suptitle('')
plt.xlabel('탑승 등급')
plt.ylabel('요금 ($)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_046",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "캠페인 접촉 횟수(campaign)가 10회를 초과하는 이상 과다 접촉 고객들의 인원수와 이들의 가입률을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_046: 10회 초과 과다 접촉 고객 분석
over_contacted = df_bank[df_bank['campaign'] > 10]
rate = (over_contacted['y'] == 'yes').mean() * 100 if len(over_contacted) > 0 else 0.0
print(f"10회 초과 접촉 고객 수: {len(over_contacted)}명")
print(f"이들의 가입 성공률: {rate:.2f}%")
"""
    })

    specs.append({
        "id": "L2_047",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "상담 시간(duration)과 캠페인 접촉 횟수(campaign) 모두에서 상위 5% 이상치를 기록한 극단 피로 고객군 수를 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_047: 시간 & 횟수 동시 상위 5% 이상치 고객
d_95 = df_bank['duration'].quantile(0.95)
c_95 = df_bank['campaign'].quantile(0.95)
cond = (df_bank['duration'] > d_95) & (df_bank['campaign'] > c_95)
fatigued = df_bank[cond]
print(f"상담시간 95% 기준: {d_95:.1f}초, 접촉횟수 95% 기준: {c_95:.1f}회")
print(f"두 조건 동시 초과 극단 피로 고객 수: {len(fatigued)}명")
"""
    })

    specs.append({
        "id": "L2_048",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "나이(age) 데이터에서 정상 범위를 벗어난 이상치가 있는지 최솟값, 최댓값 및 3-시그마 범위를 점검해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_048: 나이 데이터 3-시그마 무결성 점검
mean_a = df_bank['age'].mean()
std_a = df_bank['age'].std()
low_3s, high_3s = mean_a - 3*std_a, mean_a + 3*std_a
out_3s = df_bank[(df_bank['age'] < low_3s) | (df_bank['age'] > high_3s)]
print(f"나이 범위: 최소 {df_bank['age'].min()}세 ~ 최대 {df_bank['age'].max()}세")
print(f"3-시그마 범위: {low_3s:.1f}세 ~ {high_3s:.1f}세")
print(f"3-시그마 초과 고객 수: {len(out_3s)}명")
"""
    })

    specs.append({
        "id": "L2_049",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "마이너스 잔액(< 0) 고객들의 잔액 분포 최댓값(0에 가장 가까운 값), 최솟값, 중앙값을 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_049: 마이너스 잔액 요약 통계
neg_bal = df_bank[df_bank['balance'] < 0]['balance']
print(f"마이너스 잔액 고객 수: {len(neg_bal)}명")
print(f"최소값: ${neg_bal.min():,.0f}, 중앙값: ${neg_bal.median():,.0f}, 최대값: ${neg_bal.max():,.0f}")
"""
    })

    specs.append({
        "id": "L2_050",
        "category": "이상치 탐지 및 조건부 분석",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "예금 가입 성공(y='yes') 고객과 실패(y='no') 고객의 상담 시간(duration) 이상치를 비교하는 박스플롯을 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_050: 가입 여부별 상담 시간 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 4))
df_bank.boxplot(column='duration', by='y', figsize=(7, 4))
plt.title('예금 가입 여부(y)별 상담 시간(초) 비교')
plt.suptitle('')
plt.xlabel('예금 가입 여부')
plt.ylabel('상담 시간 (초)')
plt.tight_layout()
plt.show()
"""
    })

    return specs
