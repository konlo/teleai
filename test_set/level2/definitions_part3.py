"""Level 2 Definitions Part 3 (L2_051 ~ L2_075):
- 🧪 Statistical Hypothesis Testing & Inference (10 items: L2_051 ~ L2_060)
- 📈 Marketing Campaign Conversion & Dual-Axis Visualizations (15 items: L2_061 ~ L2_075)
"""

def get_level2_part3():
    specs = []
    
    # -------------------------------------------------------------
    # 🧪 Statistical Hypothesis Testing & Inference (L2_051 ~ L2_060)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_051",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "정기예금 가입자(y='yes')와 미가입자(y='no') 간의 계좌 잔액(balance) 차이에 대해 독립표본 t-검정(Two-Sample t-test)을 수행하고 p-value를 확인해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_051: 가입 여부별 잔액 독립표본 t-검정
from scipy import stats

bal_yes = df_bank[df_bank['y'] == 'yes']['balance']
bal_no = df_bank[df_bank['y'] == 'no']['balance']
t_stat, p_val = stats.ttest_ind(bal_yes, bal_no, equal_var=False)
print(f"가입자 평균 잔액: ${bal_yes.mean():,.2f}, 미가입자: ${bal_no.mean():,.2f}")
print(f"t-통계량: {t_stat:.4f}, p-value: {p_val:.4e}")
print(f"결론: {'통계적으로 유의미한 차이 있음 (p < 0.05)' if p_val < 0.05 else '통계적 유의미성 없음'}")
"""
    })

    specs.append({
        "id": "L2_052",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "정기예금 가입자(y='yes')와 미가입자(y='no') 간의 통화 상담 시간(duration) 차이에 대해 독립표본 t-검정을 수행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_052: 가입 여부별 상담 시간 t-검정
from scipy import stats

dur_yes = df_bank[df_bank['y'] == 'yes']['duration']
dur_no = df_bank[df_bank['y'] == 'no']['duration']
t_stat, p_val = stats.ttest_ind(dur_yes, dur_no, equal_var=False)
print(f"가입자 평균 상담시간: {dur_yes.mean():.1f}초, 미가입자: {dur_no.mean():.1f}초")
print(f"t-통계량: {t_stat:.4f}, p-value: {p_val:.4e}")
"""
    })

    specs.append({
        "id": "L2_053",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "주택 대출(housing) 여부와 정기예금 가입(y) 간의 독립성 검정을 위해 카이제곱 검정(Chi-Square Test)을 수행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_053: 주택대출 x 예금가입 카이제곱 검정
from scipy import stats

ct = pd.crosstab(df_bank['housing'], df_bank['y'])
chi2, p_val, dof, _ = stats.chi2_contingency(ct)
print("교차표 (Contingency Table):")
print(ct)
print(f"카이제곱 통계량: {chi2:.4f}, 자유도: {dof}, p-value: {p_val:.4e}")
print(f"결론: {'두 변수 간 통계적 연관성 있음 (p < 0.05)' if p_val < 0.05 else '독립적임'}")
"""
    })

    specs.append({
        "id": "L2_054",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 등급(Pclass)과 생존 여부(Survived) 간의 카이제곱 독립성 검정을 수행해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_054: 타이타닉 등급 x 생존 카이제곱 검정
from scipy import stats

ct_ps = pd.crosstab(df_titanic['Pclass'], df_titanic['Survived'])
chi2, p_val, dof, _ = stats.chi2_contingency(ct_ps)
print("등급 x 생존 교차표:")
print(ct_ps)
print(f"카이제곱 통계량: {chi2:.4f}, p-value: {p_val:.4e}")
"""
    })

    specs.append({
        "id": "L2_055",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "교육 수준(education) 그룹 간 고객 잔액(balance)의 차이가 유의미한지 일원분산분석(One-way ANOVA)을 수행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_055: 교육수준별 잔액 ANOVA 검정
from scipy import stats

groups = [group['balance'].values for _, group in df_bank.groupby('education')]
f_stat, p_val = stats.f_oneway(*groups)
print(f"F-통계량: {f_stat:.4f}, p-value: {p_val:.4e}")
print(f"결론: {'교육수준 간 잔액 차이 유의미함 (p < 0.05)' if p_val < 0.05 else '차이 유의미하지 않음'}")
"""
    })

    specs.append({
        "id": "L2_056",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "혼인 상태(marital) 그룹 간 고객 연령(age)의 차이에 대해 일원분산분석(One-way ANOVA)을 수행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_056: 혼인상태별 연령 ANOVA 검정
from scipy import stats

groups_age = [group['age'].values for _, group in df_bank.groupby('marital')]
f_stat, p_val = stats.f_oneway(*groups_age)
print(f"혼인상태 그룹별 평균 연령:")
print(df_bank.groupby('marital')['age'].mean().round(1))
print(f"ANOVA F-통계량: {f_stat:.4f}, p-value: {p_val:.4e}")
"""
    })

    specs.append({
        "id": "L2_057",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "잔액(balance) 데이터의 비정규성을 감안하여, 가입 여부(y)별 잔액 차이에 대해 비모수 검정인 맨-휘트니 U 검정(Mann-Whitney U Test)을 수행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_057: 가입 여부별 잔액 맨-휘트니 U 검정
from scipy import stats

bal_yes = df_bank[df_bank['y'] == 'yes']['balance']
bal_no = df_bank[df_bank['y'] == 'no']['balance']
u_stat, p_val = stats.mannwhitneyu(bal_yes, bal_no)
print(f"Mann-Whitney U 통계량: {u_stat:,.1f}, p-value: {p_val:.4e}")
"""
    })

    specs.append({
        "id": "L2_058",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 생존자(1)와 사망자(0) 간의 지불 요금(Fare) 차이에 대해 독립표본 t-검정을 수행해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_058: 타이타닉 생존 여부별 요금 t-검정
from scipy import stats

fare_surv = df_titanic[df_titanic['Survived'] == 1]['Fare'].dropna()
fare_dead = df_titanic[df_titanic['Survived'] == 0]['Fare'].dropna()
t_stat, p_val = stats.ttest_ind(fare_surv, fare_dead, equal_var=False)
print(f"생존자 평균 요금: ${fare_surv.mean():.2f}, 사망자 평균 요금: ${fare_dead.mean():.2f}")
print(f"t-통계량: {t_stat:.4f}, p-value: {p_val:.4e}")
"""
    })

    specs.append({
        "id": "L2_059",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "전체 고객 잔액(balance)의 모평균에 대한 95% 신뢰구간(Confidence Interval)을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_059: 모평균 95% 신뢰구간 계산
from scipy import stats

mean_b = df_bank['balance'].mean()
sem_b = stats.sem(df_bank['balance'])
ci = stats.t.interval(0.95, len(df_bank)-1, loc=mean_b, scale=sem_b)
print(f"표본 평균 잔액: ${mean_b:,.2f}")
print(f"모평균 95% 신뢰구간: [${ci[0]:,.2f}, ${ci[1]:,.2f}]")
"""
    })

    specs.append({
        "id": "L2_060",
        "category": "통계적 가설 검정 및 추론",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "신용불량(default) 여부와 개인 대출(loan) 보유 여부 간의 카이제곱 검정을 수행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_060: default x loan 카이제곱 검정
from scipy import stats

ct_dl = pd.crosstab(df_bank['default'], df_bank['loan'])
chi2, p_val, dof, _ = stats.chi2_contingency(ct_dl)
print("신용불량 x 개인대출 교차표:")
print(ct_dl)
print(f"카이제곱 통계량: {chi2:.4f}, p-value: {p_val:.4e}")
"""
    })

    # -------------------------------------------------------------
    # 📈 Marketing Campaign Conversion & Dual-Axis (L2_061 ~ L2_075)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_061",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "월(month)별 총 접촉 건수(막대)와 예금 가입 전환율(꺾은선)을 하나의 차트에 이중 축(Dual Y-axis)으로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_061: 월별 접촉 건수(막대) & 가입률(선) 이중 축 차트
import matplotlib.pyplot as plt

months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
m_stats = df_bank.groupby('month').agg(
    총접촉=('id', 'count'),
    가입률=('y', lambda s: (s == 'yes').mean() * 100)
).reindex([m for m in months_order if m in df_bank['month'].values])

fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax2 = ax1.twinx()

ax1.bar(m_stats.index, m_stats['총접촉'], color='cornflowerblue', alpha=0.6, label='접촉 건수')
ax2.plot(m_stats.index, m_stats['가입률'], color='crimson', marker='o', linewidth=2, label='가입 전환율 (%)')

ax1.set_xlabel('접촉 월')
ax1.set_ylabel('총 접촉 건수 (건)', color='navy')
ax2.set_ylabel('가입 전환율 (%)', color='crimson')
plt.title('월별 마케팅 접촉 건수 및 예금 전환율 (이중 축 추이)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_062",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "캠페인 접촉 횟수(campaign, 1~10회)별 고객 수(막대)와 가입 성공률(선)을 이중 축으로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_062: 접촉 횟수(1~10회)별 고객수 및 가입률 이중 축
import matplotlib.pyplot as plt

camp_sub = df_bank[df_bank['campaign'] <= 10]
camp_stats = camp_sub.groupby('campaign').agg(
    고객수=('id', 'count'),
    가입률=('y', lambda s: (s == 'yes').mean() * 100)
)

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()

ax1.bar(camp_stats.index, camp_stats['고객수'], color='lightgray', alpha=0.8)
ax2.plot(camp_stats.index, camp_stats['가입률'], color='darkorange', marker='s', linewidth=2)

ax1.set_xlabel('캠페인 접촉 횟수 (회)')
ax1.set_ylabel('고객 수', color='black')
ax2.set_ylabel('가입 성공률 (%)', color='darkorange')
plt.title('접촉 횟수 증가에 따른 가입률 한계 감소 (이중 축)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_063",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "상담 시간(duration)을 0~100초, 100~200초, ..., 600초 이상 구간으로 나누고 구간별 가입 성공률(%)을 막대 차트로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_063: 상담 시간 구간별 가입 성공률 막대차트
import matplotlib.pyplot as plt

bins = [0, 100, 200, 300, 400, 500, 600, 10000]
labels = ['~100초', '100~200', '200~300', '300~400', '400~500', '500~600', '600초+']
df_copy = df_bank.copy()
df_copy['dur_group'] = pd.cut(df_copy['duration'], bins=bins, labels=labels)
conv_by_dur = (df_copy.groupby('dur_group', observed=False)['y'].apply(lambda s: (s == 'yes').mean() * 100)).round(2)

plt.figure(figsize=(8, 4))
conv_by_dur.plot(kind='bar', color='seagreen')
plt.title('상담 시간 구간별 정기예금 가입 전환율(%)')
plt.xlabel('상담 시간 구간')
plt.ylabel('가입 전환율 (%)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_064",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "직업별 고객 수(막대)와 예금 가입 성공률(선)을 이중 축 그래프로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_064: 직업별 고객수 & 가입률 이중 축
import matplotlib.pyplot as plt

job_stats = df_bank.groupby('job').agg(
    고객수=('id', 'count'),
    가입률=('y', lambda s: (s == 'yes').mean() * 100)
).sort_values('고객수', ascending=False)

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax2 = ax1.twinx()

ax1.bar(job_stats.index, job_stats['고객수'], color='thistle', alpha=0.7)
ax2.plot(job_stats.index, job_stats['가입률'], color='darkmagenta', marker='o', linewidth=2)

ax1.set_xlabel('직업')
ax1.set_ylabel('고객 수', color='purple')
ax2.set_ylabel('가입 성공률 (%)', color='darkmagenta')
ax1.set_xticklabels(job_stats.index, rotation=45)
plt.title('직업별 규모 및 마케팅 가입 성공률 비교')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_065",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 등급(Pclass)별 탑승객 수(막대)와 생존율(선)을 이중 축 그래프로 표현해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_065: 타이타닉 Pclass 승객수 & 생존율 이중 축
import matplotlib.pyplot as plt

pc_stats = df_titanic.groupby('Pclass').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', lambda s: s.mean() * 100)
)

fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

ax1.bar(pc_stats.index, pc_stats['승객수'], color='skyblue', alpha=0.7)
ax2.plot(pc_stats.index, pc_stats['생존율'], color='red', marker='D', linewidth=2)

ax1.set_xlabel('탑승 등급 (Pclass)')
ax1.set_ylabel('승객 수', color='navy')
ax2.set_ylabel('생존율 (%)', color='red')
ax1.set_xticks([1, 2, 3])
plt.title('탑승 등급별 승객 규모 및 생존율')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_066",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "마케팅 접촉일(day)별 고객 수(막대)와 가입 성공률(선)의 일별 이중 축 추이를 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_066: 일자(day)별 접촉수 및 가입률 이중 축
import matplotlib.pyplot as plt

day_stats = df_bank.groupby('day').agg(
    접촉수=('id', 'count'),
    가입률=('y', lambda s: (s == 'yes').mean() * 100)
).sort_index()

fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

ax1.bar(day_stats.index, day_stats['접촉수'], color='lightsteelblue', alpha=0.6)
ax2.plot(day_stats.index, day_stats['가입률'], color='crimson', marker='.', linewidth=1.5)

ax1.set_xlabel('일자 (1~31일)')
ax1.set_ylabel('접촉 수', color='navy')
ax2.set_ylabel('가입률 (%)', color='crimson')
plt.title('월중 일자별 접촉 건수 및 가입 성공률')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_067",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "직업별 평균 잔액 집계 데이터는 히스토그램이 아닌 막대 차트로 올바르게 표현해야 하는 시각화 규칙을 준수하여 막대 차트로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_067: 집계 데이터 막대차트 규칙 준수
import matplotlib.pyplot as plt

# 사전 집계된 데이터
agg_data = df_bank.groupby('job')['balance'].mean().sort_values(ascending=False)

plt.figure(figsize=(9, 4.5))
agg_data.plot(kind='bar', color='royalblue')
plt.title('직업별 평균 계좌 잔액 (집계 데이터 막대 차트 표출)')
plt.xlabel('직업')
plt.ylabel('평균 잔액 ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_068",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "월별 총 접촉 건수의 누적합(Cumulative Sum) 곡선을 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_068: 월별 접촉 건수 누적 성장 곡선
import matplotlib.pyplot as plt

months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_counts = df_bank['month'].value_counts().reindex([m for m in months_order if m in df_bank['month'].values])
cumsum_months = month_counts.cumsum()

plt.figure(figsize=(8, 4))
plt.plot(cumsum_months.index, cumsum_months.values, marker='o', color='indigo', linewidth=2)
plt.title('월별 마케팅 접촉 건수 누적 곡선 (Cumulative Sum)')
plt.xlabel('월')
plt.ylabel('누적 접촉 건수')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_069",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "주택 대출(housing)과 개인 대출(loan) 4개 조합별 예금 가입률(%)을 막대 차트로 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_069: 대출 조합 4개 그룹별 가입률 비교
import matplotlib.pyplot as plt

df_copy = df_bank.copy()
df_copy['loan_combo'] = 'Housing:' + df_copy['housing'] + ' / Loan:' + df_copy['loan']
combo_rate = (df_copy.groupby('loan_combo')['y'].apply(lambda s: (s == 'yes').mean() * 100)).round(2)

plt.figure(figsize=(8, 4))
combo_rate.plot(kind='bar', color='darkcyan')
plt.title('대출 보유 조합별 정기예금 가입 전환율(%)')
plt.xlabel('대출 조합')
plt.ylabel('가입 전환율 (%)')
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_070",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "연령대별(20대 이하, 30대, 40대, 50대 이상) 예금 가입 성공률(%)을 꺾은선 차트로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_070: 연령대별 가입 성공률 선그래프
import matplotlib.pyplot as plt

bins = [0, 29, 39, 49, 120]
labels = ['20대 이하', '30대', '40대', '50대 이상']
df_copy = df_bank.copy()
df_copy['age_grp'] = pd.cut(df_copy['age'], bins=bins, labels=labels)
age_rate = (df_copy.groupby('age_grp', observed=False)['y'].apply(lambda s: (s == 'yes').mean() * 100)).round(2)

plt.figure(figsize=(7, 4))
plt.plot(age_rate.index, age_rate.values, marker='s', color='darkred', linewidth=2)
plt.title('연령대별 정기예금 가입 성공률 추이')
plt.xlabel('연령대')
plt.ylabel('가입 성공률 (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_071",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "혼인 상태(marital)별로 주택 대출 보유율과 개인 대출 보유율을 나란히 그룹 막대 차트로 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_071: 혼인상태별 대출 유형 보유율 그룹 막대
import matplotlib.pyplot as plt

m_loans = df_bank.groupby('marital').agg(
    주택대출보유율=('housing', lambda s: (s == 'yes').mean() * 100),
    개인대출보유율=('loan', lambda s: (s == 'yes').mean() * 100)
).round(2)

m_loans.plot(kind='bar', figsize=(7, 4.5), color=['royalblue', 'salmon'])
plt.title('혼인 상태별 주택대출 vs 개인대출 보유율 비교')
plt.xlabel('혼인 상태')
plt.ylabel('보유율 (%)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_072",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 나이(Age)를 10세 단위로 구간화하고 각 연령 구간별 생존율(%)을 막대 차트로 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_072: 타이타닉 10세 단위 생존율 막대차트
import matplotlib.pyplot as plt

t_clean = df_titanic.dropna(subset=['Age']).copy()
t_clean['age_decade'] = (t_clean['Age'] // 10 * 10).astype(int)
surv_by_decade = (t_clean.groupby('age_decade')['Survived'].mean() * 100).round(2)

plt.figure(figsize=(8, 4))
surv_by_decade.plot(kind='bar', color='cadetblue')
plt.title('타이타닉 10세 단위 연령대별 생존율(%)')
plt.xlabel('연령대 (대)')
plt.ylabel('생존율 (%)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_073",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "이전 마케팅 결과(poutcome)별 예금 가입 성공률(%)을 수평 막대 그래프로 표현해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_073: poutcome별 가입률 수평 막대
import matplotlib.pyplot as plt

pout_rate = (df_bank.groupby('poutcome')['y'].apply(lambda s: (s == 'yes').mean() * 100)).sort_values(ascending=True)

plt.figure(figsize=(7, 4))
pout_rate.plot(kind='barh', color='orange')
plt.title('이전 마케팅 결과(Poutcome)별 이번 가입 성공률(%)')
plt.xlabel('가입 성공률 (%)')
plt.ylabel('이전 결과')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_074",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "상담 시간(duration)과 나이(age)의 산점도에 예금 가입 여부(y='yes' vs 'no')로 색상을 구분하여 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_074: 나이 vs 상담시간 가입여부 색상 구분 산점도
import matplotlib.pyplot as plt

yes_mask = df_bank['y'] == 'yes'
plt.figure(figsize=(8, 4.5))
plt.scatter(df_bank.loc[~yes_mask, 'age'], df_bank.loc[~yes_mask, 'duration'], alpha=0.3, color='gray', label='미가입 (no)')
plt.scatter(df_bank.loc[yes_mask, 'age'], df_bank.loc[yes_mask, 'duration'], alpha=0.7, color='crimson', label='가입 (yes)')
plt.title('나이 vs 상담 시간 (가입 성공 여부 구분 산점도)')
plt.xlabel('나이')
plt.ylabel('상담 시간 (초)')
plt.legend()
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_075",
        "category": "캠페인 전환율 및 이중 축 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "통신 수단(contact)별 고객 수와 평균 잔액을 나란히 비교하는 막대 차트를 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L2_075: 통신수단별 고객수 & 평균잔액 비교
import matplotlib.pyplot as plt

c_stats = df_bank.groupby('contact').agg(고객수=('id', 'count'), 평균잔액=('balance', 'mean')).round(2)
c_stats.plot(kind='bar', subplots=True, figsize=(8, 5), layout=(1, 2), color=['teal', 'coral'], legend=False)
plt.suptitle('통신 수단별 고객수 및 평균 잔액 비교')
plt.tight_layout()
plt.show()
"""
    })

    return specs
