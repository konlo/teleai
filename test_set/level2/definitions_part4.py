"""Level 2 Definitions Part 4 (L2_076 ~ L2_100):
- 📊 Multi-Panel Dashboards & Advanced Visualizations (15 items: L2_076 ~ L2_090)
- 🛡️ Defensive Edge Cases & Schema Boundary Handling (10 items: L2_091 ~ L2_100)
"""

def get_level2_part4():
    specs = []
    
    # -------------------------------------------------------------
    # 📊 Multi-Panel Dashboards & Advanced Visualizations (L2_076 ~ L2_090)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_076",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "고객 인구통계학적 특성(나이 분포, 상위 5대 직업, 학력 분포, 결혼 여부)을 2x2 서브플롯 대시보드로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"나이": "age", "직업": "job", "학력": "education", "결혼": "marital"},
        "expected_output_type": "chart",
        "python_code": """# L2_076: 2x2 고객 인구통계학적 대시보드
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 나이 분포 (히스토그램)
axes[0, 0].hist(df_bank['age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('고객 나이 분포')
axes[0, 0].set_xlabel('나이')
axes[0, 0].set_ylabel('고객 수')

# 2. 상위 5대 직업 (수평 막대)
top_jobs = df_bank['job'].value_counts().head(5)
axes[0, 1].barh(top_jobs.index[::-1], top_jobs.values[::-1], color='coral', edgecolor='black')
axes[0, 1].set_title('상위 5대 직업')
axes[0, 1].set_xlabel('고객 수')

# 3. 학력 분포 (막대)
edu_counts = df_bank['education'].value_counts()
axes[1, 0].bar(edu_counts.index, edu_counts.values, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('학력 수준 분포')
axes[1, 0].set_xlabel('학력')
axes[1, 0].set_ylabel('고객 수')
axes[1, 0].tick_params(axis='x', rotation=30)

# 4. 결혼 여부 (원형 차트)
marital_counts = df_bank['marital'].value_counts()
axes[1, 1].pie(marital_counts.values, labels=marital_counts.index, autopct='%1.1f%%', startangle=140)
axes[1, 1].set_title('결혼 여부 비율')

plt.suptitle('고객 인구통계 대시보드 (Demographics Dashboard)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_077",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 생존 관련 핵심 지표(객실 등급별 생존율, 성별 생존율, 승선 항구별 탑승객 수, 생존 여부별 나이 분포)를 2x2 서브플롯 대시보드로 시각화해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"객실 등급": "Pclass", "성별": "Sex", "승선 항구": "Embarked", "생존 여부": "Survived", "나이": "Age"},
        "expected_output_type": "chart",
        "python_code": """# L2_077: 타이타닉 2x2 생존 분석 대시보드
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 객실 등급별 생존율
pclass_surv = df_titanic.groupby('Pclass')['Survived'].mean() * 100
axes[0, 0].bar([f'{c}등급' for c in pclass_surv.index], pclass_surv.values, color='cornflowerblue', edgecolor='black')
axes[0, 0].set_title('객실 등급(Pclass)별 생존율 (%)')
axes[0, 0].set_ylabel('생존율 (%)')

# 2. 성별 생존율
sex_surv = df_titanic.groupby('Sex')['Survived'].mean() * 100
axes[0, 1].bar(sex_surv.index, sex_surv.values, color=['salmon', 'steelblue'], edgecolor='black')
axes[0, 1].set_title('성별(Sex) 생존율 (%)')
axes[0, 1].set_ylabel('생존율 (%)')

# 3. 승선 항구별 승객 수
emb_counts = df_titanic['Embarked'].value_counts()
axes[1, 0].bar(emb_counts.index, emb_counts.values, color='mediumpurple', edgecolor='black')
axes[1, 0].set_title('승선 항구(Embarked)별 탑승 승객 수')
axes[1, 0].set_xlabel('항구')
axes[1, 0].set_ylabel('승객 수')

# 4. 생존 여부별 나이 분포
surv_age = df_titanic[df_titanic['Survived'] == 1]['Age'].dropna()
dead_age = df_titanic[df_titanic['Survived'] == 0]['Age'].dropna()
axes[1, 1].hist([surv_age, dead_age], bins=15, label=['생존', '사망'], color=['lightgreen', 'indianred'], alpha=0.7)
axes[1, 1].set_title('생존 여부별 나이 분포')
axes[1, 1].set_xlabel('나이')
axes[1, 1].legend()

plt.suptitle('타이타닉 승객 생존 분석 종합 대시보드', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_078",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "주택 대출(housing), 개인 대출(loan), 연체 여부(default), 정기예금 가입 여부(y)에 따른 고객 잔액(balance) 분포를 2x2 박스플롯 대시보드로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"주택 대출": "housing", "개인 대출": "loan", "연체 여부": "default", "잔액": "balance"},
        "expected_output_type": "chart",
        "python_code": """# L2_078: 대출 및 재무상태별 잔액 2x2 박스플롯 대시보드
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 주택 대출 유무별 잔액
df_bank.boxplot(column='balance', by='housing', ax=axes[0, 0], showfliers=False)
axes[0, 0].set_title('주택 대출(housing) 유무별 잔액')
axes[0, 0].set_xlabel('주택 대출')
axes[0, 0].set_ylabel('계좌 잔액 ($)')

# 2. 개인 대출 유무별 잔액
df_bank.boxplot(column='balance', by='loan', ax=axes[0, 1], showfliers=False)
axes[0, 1].set_title('개인 신용대출(loan) 유무별 잔액')
axes[0, 1].set_xlabel('개인 대출')
axes[0, 1].set_ylabel('계좌 잔액 ($)')

# 3. 채무 불이행/연체 유무별 잔액
df_bank.boxplot(column='balance', by='default', ax=axes[1, 0], showfliers=False)
axes[1, 0].set_title('채무 불이행(default) 유무별 잔액')
axes[1, 0].set_xlabel('연체 여부')
axes[1, 0].set_ylabel('계좌 잔액 ($)')

# 4. 정기예금 가입 여부별 잔액
df_bank.boxplot(column='balance', by='y', ax=axes[1, 1], showfliers=False)
axes[1, 1].set_title('정기예금 가입(y) 여부별 잔액')
axes[1, 1].set_xlabel('예금 가입')
axes[1, 1].set_ylabel('계좌 잔액 ($)')

plt.suptitle('고객 대출 및 금융 상태별 계좌 잔액 분포 비교', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_079",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "직업군(job)별 정기예금 가입(y) 비율을 100% 누적 가로 막대 차트(100% Stacked Bar Chart)로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"직업군": "job", "가입": "y"},
        "expected_output_type": "chart",
        "python_code": """# L2_079: 직업군별 예금 가입 100% 누적 막대 차트
import matplotlib.pyplot as plt
import pandas as pd

ct = pd.crosstab(df_bank['job'], df_bank['y'], normalize='index') * 100
ct_sorted = ct.sort_values(by='yes')

ax = ct_sorted.plot(kind='barh', stacked=True, figsize=(10, 6), color=['#d9534f', '#5cb85c'], edgecolor='black')
plt.title('직업군별 정기예금 가입/미가입 비율 (100% 누적)', fontsize=12)
plt.xlabel('비율 (%)')
plt.ylabel('직업 (Job)')
plt.legend(['미가입 (no)', '가입 (yes)'], loc='lower right')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_080",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "객실 등급(Pclass)과 성별(Sex)의 조합별 생존율을 100% 누적 막대 차트로 시각화해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"객실 등급": "Pclass", "성별": "Sex", "생존율": "Survived"},
        "expected_output_type": "chart",
        "python_code": """# L2_080: Pclass와 Sex 조합별 생존율 100% 누적 막대 차트
import matplotlib.pyplot as plt
import pandas as pd

df_temp = df_titanic.copy()
df_temp['Group'] = df_temp['Pclass'].astype(str) + '등석_' + df_temp['Sex']
ct = pd.crosstab(df_temp['Group'], df_temp['Survived'], normalize='index') * 100

ax = ct.plot(kind='bar', stacked=True, figsize=(9, 5), color=['#e74c3c', '#2ecc71'], edgecolor='black')
plt.title('객실 등급 및 성별 집단별 생존/사망 비율 (100% 누적)')
plt.xlabel('등급 및 성별 집단')
plt.ylabel('비율 (%)')
plt.xticks(rotation=45)
plt.legend(['사망 (0)', '생존 (1)'], loc='upper right')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_081",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "고객 나이(age)와 계좌 잔액(balance)의 관계를 산점도로 그리고, 각 축의 상단과 우측에 히스토그램을 배치한 조인트 플롯 형태로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"나이": "age", "계좌 잔액": "balance"},
        "expected_output_type": "chart",
        "python_code": """# L2_081: 나이와 잔액의 조인트 산점도 및 분포 플롯
import matplotlib.pyplot as plt

# 99번째 백분위수 이하 데이터로 가시성 확보
sub_df = df_bank[(df_bank['balance'] >= 0) & (df_bank['balance'] <= df_bank['balance'].quantile(0.98))]

fig = plt.figure(figsize=(9, 8))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

# 메인 산점도
ax_main = fig.add_subplot(grid[1:4, 0:3])
ax_main.scatter(sub_df['age'], sub_df['balance'], alpha=0.3, color='teal', s=15)
ax_main.set_xlabel('고객 나이 (Age)')
ax_main.set_ylabel('계좌 잔액 ($)')

# 상단 나이 히스토그램
ax_xDist = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
ax_xDist.hist(sub_df['age'], bins=25, color='teal', alpha=0.6, edgecolor='white')
ax_xDist.axis('off')
ax_xDist.set_title('고객 나이 vs 잔액 분포 조인트 플롯')

# 우측 잔액 히스토그램
ax_yDist = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
ax_yDist.hist(sub_df['balance'], bins=25, orientation='horizontal', color='teal', alpha=0.6, edgecolor='white')
ax_yDist.axis('off')

plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_082",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "은행 데이터의 모든 수치형 컬럼(age, balance, day, duration, campaign, pdays, previous) 간 상관계수 행렬을 계산하고 수치가 표시된 히트맵(Correlation Heatmap)을 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"수치형 컬럼": "numeric_columns", "상관계수": "correlation"},
        "expected_output_type": "chart",
        "python_code": """# L2_082: 은행 데이터 수치형 변수 상관계수 히트맵
import matplotlib.pyplot as plt
import numpy as np

num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
corr = df_bank[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 7))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)

ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha='left')
ax.set_yticklabels(num_cols)
plt.title('은행 데이터 수치형 변수 간 상관계수 행렬', pad=30, fontsize=12)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_083",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 데이터의 수치형 변수(Survived, Pclass, Age, SibSp, Parch, Fare) 간 상관계수 히트맵을 생성해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"수치형 변수": "numeric_variables", "상관계수": "correlation"},
        "expected_output_type": "chart",
        "python_code": """# L2_083: 타이타닉 수치형 변수 상관관계 히트맵
import matplotlib.pyplot as plt

cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr = df_titanic[cols].corr()

fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.matshow(corr, cmap='Blues', vmin=-1, vmax=1)
fig.colorbar(cax)

for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', 
                color='white' if abs(corr.iloc[i, j]) > 0.4 else 'black')

ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=45, ha='left')
ax.set_yticklabels(cols)
plt.title('타이타닉 수치형 변수 간 상관계수 행렬', pad=30)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_084",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "이전 마케팅 결과(poutcome) 유형별로 상담 통화 시간(duration)의 분포를 박스플롯으로 나란히 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"이전 마케팅 결과": "poutcome", "상담 통화 시간": "duration"},
        "expected_output_type": "chart",
        "python_code": """# L2_084: 이전 마케팅 결과별 상담 시간 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
df_bank.boxplot(column='duration', by='poutcome', showfliers=False, grid=True)
plt.title('이전 캠페인 결과(poutcome)별 상담 통화 시간(duration) 분포')
plt.suptitle('')
plt.xlabel('이전 캠페인 결과')
plt.ylabel('통화 시간 (초)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_085",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객의 등급(Pclass)별 요금(Fare) 분포를 생존 여부(Survived)별로 비교하는 박스플롯을 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": {"등급": "Pclass", "요금": "Fare", "생존 여부": "Survived"},
        "expected_output_type": "chart",
        "python_code": """# L2_085: Pclass 및 생존 여부별 운임료 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
df_titanic.boxplot(column='Fare', by=['Pclass', 'Survived'], showfliers=False)
plt.title('객실 등급 및 생존 여부별 요금(Fare) 분포')
plt.suptitle('')
plt.xlabel('(객실 등급, 생존 여부: 0=사망, 1=생존)')
plt.ylabel('요금 (£)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_086",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "고객 계좌 잔액(balance)의 로렌츠 곡선(Lorenz Curve)을 그리고 지니계수(Gini Coefficient)를 계산하여 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"계좌 잔액": "balance"},
        "expected_output_type": "chart",
        "python_code": """# L2_086: 계좌 잔액 로렌츠 곡선 및 지니계수 계산
import matplotlib.pyplot as plt
import numpy as np

# 양수 잔액 기준 로렌츠 곡선
pos_balances = np.sort(df_bank[df_bank['balance'] > 0]['balance'].values)
n = len(pos_balances)
cum_wealth = np.cumsum(pos_balances) / np.sum(pos_balances)
cum_pop = np.arange(1, n + 1) / n

# 지니계수 계산 (사다리꼴 공식)
gini = 1 - 2 * np.trapz(cum_wealth, cum_pop)

plt.figure(figsize=(7, 6))
plt.plot(cum_pop, cum_wealth, label=f'잔액 로렌츠 곡선 (Gini={gini:.3f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='완전 균등 분배선')
plt.fill_between(cum_pop, cum_pop, cum_wealth, color='orange', alpha=0.2)
plt.title('고객 계좌 잔액의 불평등도 (로렌츠 곡선)')
plt.xlabel('누적 고객 비율')
plt.ylabel('누적 잔액 비율')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print(f"계산된 잔액 지니계수: {gini:.4f}")
"""
    })

    specs.append({
        "id": "L2_087",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "X축을 고객 나이(age), Y축을 잔액(balance), 점의 크기를 상담 시간(duration), 점의 색상을 예금 가입 여부(y)로 나타낸 버블 차트(Bubble Chart)를 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"나이": "age", "잔액": "balance", "상담 시간": "duration", "예금 가입 여부": "y"},
        "expected_output_type": "chart",
        "python_code": """# L2_087: 다변량 버블 차트 (Age, Balance, Duration, Subscription)
import matplotlib.pyplot as plt

sub = df_bank[(df_bank['balance'] >= 0) & (df_bank['balance'] <= 25000)].sample(300, random_state=42)

plt.figure(figsize=(9, 6))
colors = sub['y'].map({'yes': 'crimson', 'no': 'dodgerblue'})
sizes = (sub['duration'] / sub['duration'].max()) * 200 + 10

scatter = plt.scatter(sub['age'], sub['balance'], s=sizes, c=colors, alpha=0.5, edgecolors='black', linewidth=0.5)
plt.title('나이 vs 잔액 버블 차트 (크기=상담시간, 빨강=가입, 파랑=미가입)')
plt.xlabel('고객 나이 (Age)')
plt.ylabel('계좌 잔액 ($)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_088",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객의 동반 가족 수(SibSp + Parch)에 따른 생존율 변화를 신뢰구간 또는 추세선과 함께 선 그래프로 시각화해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"동반 가족 수": "FamilySize", "생존율": "Survived"},
        "expected_output_type": "chart",
        "python_code": """# L2_088: 동반 가족 수에 따른 생존율 선 그래프
import matplotlib.pyplot as plt

df_temp = df_titanic.copy()
df_temp['FamilySize'] = df_temp['SibSp'] + df_temp['Parch']
fam_stats = df_temp.groupby('FamilySize').agg(
    생존율=('Survived', 'mean'),
    승객수=('PassengerId', 'count')
).reset_index()

fig, ax1 = plt.subplots(figsize=(8, 5))

# 승객수 막대
ax1.bar(fam_stats['FamilySize'], fam_stats['승객수'], color='lightgray', alpha=0.6, label='승객 수')
ax1.set_xlabel('동반 가족 수 (Family Size)')
ax1.set_ylabel('승객 수', color='gray')

# 생존율 선 그래프
ax2 = ax1.twinx()
ax2.plot(fam_stats['FamilySize'], fam_stats['생존율'] * 100, color='blue', marker='o', lw=2, label='생존율 (%)')
ax2.set_ylabel('생존율 (%)', color='blue')
ax2.set_ylim(0, 100)

plt.title('동반 가족 수(SibSp + Parch)에 따른 생존율 추이')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_089",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "월별(month) 접촉 고객 수와 예금 가입 전환율을 2열 서브플롯(막대 차트 & 선 그래프)으로 비교 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"월별": "month", "가입 전환율": "y"},
        "expected_output_type": "chart",
        "python_code": """# L2_089: 월별 접촉 고객수 및 전환율 2열 서브플롯
import matplotlib.pyplot as plt

months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
m_stats = df_bank.groupby('month').agg(
    고객수=('id', 'count'),
    가입수=('y', lambda s: (s == 'yes').sum())
)
m_stats['전환율'] = (m_stats['가입수'] / m_stats['고객수']) * 100
m_stats = m_stats.reindex([m for m in months_order if m in m_stats.index])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. 월별 접촉 고객 수 막대
ax1.bar(m_stats.index, m_stats['고객수'], color='royalblue', edgecolor='black')
ax1.set_title('월별 접촉 고객 수')
ax1.set_xlabel('월')
ax1.set_ylabel('고객 수')

# 2. 월별 전환율 선 그래프
ax2.plot(m_stats.index, m_stats['전환율'], color='crimson', marker='s', lw=2)
ax2.set_title('월별 정기예금 가입 전환율 (%)')
ax2.set_xlabel('월')
ax2.set_ylabel('전환율 (%)')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('월별 마케팅 캠페인 성과 지표 대시보드', fontsize=13)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L2_090",
        "category": "다중 패널 대시보드 및 고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객의 객실 번호(Cabin)의 첫 글자에서 덱(Deck)을 추출하고, 덱별 승객 수와 생존율을 수평 막대 차트로 나란히 비교해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"객실 번호": "Cabin", "덱": "Deck", "생존율": "Survived"},
        "expected_output_type": "chart",
        "python_code": """# L2_090: 타이타닉 덱(Deck)별 승객수 및 생존율 수평 막대
import matplotlib.pyplot as plt

df_deck = df_titanic.dropna(subset=['Cabin']).copy()
df_deck['Deck'] = df_deck['Cabin'].str[0]
# 상위 주요 덱 기준 집계
deck_stats = df_deck.groupby('Deck').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', 'mean')
).sort_index(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.barh(deck_stats.index, deck_stats['승객수'], color='darkcyan', edgecolor='black')
ax1.set_title('덱(Deck)별 승객 수')
ax1.set_xlabel('승객 수')

ax2.barh(deck_stats.index, deck_stats['생존율'] * 100, color='mediumseagreen', edgecolor='black')
ax2.set_title('덱(Deck)별 생존율 (%)')
ax2.set_xlabel('생존율 (%)')

plt.suptitle('선실 데크(Deck) 추출 및 생존율 비교 분석')
plt.tight_layout()
plt.show()
"""
    })

    # -------------------------------------------------------------
    # 🛡️ Defensive Edge Cases & Schema Boundary Handling (L2_091 ~ L2_100)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_091",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "schema",
        "difficulty": "Advanced",
        "prompt": "고객 데이터에서 '연봉(salary)' 또는 '소득(income)' 컬럼이 있는지 확인하고, 없을 경우 가장 유사한 금융 컬럼을 찾아 안내해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"연봉": "salary", "소득": "income", "대체 컬럼": "balance"},
        "expected_output_type": "text",
        "python_code": """# L2_091: 비존재 컬럼 방어 및 유사 컬럼 제안
requested_cols = ['salary', 'income']
existing_cols = df_bank.columns.tolist()

found = [c for c in requested_cols if c in existing_cols]
if not found:
    print(f"경고: 요청하신 컬럼 {requested_cols}은(는) 테이블에 존재하지 않습니다.")
    alternatives = [c for c in ['balance', 'loan', 'housing', 'default'] if c in existing_cols]
    print(f"안내: 대체 가능한 고객 재무/자산 관련 컬럼: {alternatives}")
    print("계좌 잔액('balance') 컬럼을 기반으로 재무 상태를 분석할 수 있습니다.")
    print(df_bank[['id', 'balance']].head(3))
else:
    print(f"존재 컬럼: {found}")
"""
    })

    specs.append({
        "id": "L2_092",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "나이가 120세 초과인 고객을 필터링했을 때, 결과 데이터프레임이 빈 경우 ZeroDivisionError나 크래시 없이 안전하게 처리해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"나이": "age"},
        "expected_output_type": "text",
        "python_code": """# L2_092: 필터 결과가 빈 데이터프레임(Empty DF)일 때 안전 처리
filtered_df = df_bank[df_bank['age'] > 120]

if filtered_df.empty:
    print("알림: 조건(나이 > 120)에 부합하는 고객 레코드가 0건입니다.")
    avg_bal = 0.0
    print(f"안전하게 집계한 평균 잔액: ${avg_bal:,.2f}")
else:
    avg_bal = filtered_df['balance'].mean()
    print(f"해당 고객 수: {len(filtered_df)}명, 평균 잔액: ${avg_bal:,.2f}")
"""
    })

    specs.append({
        "id": "L2_093",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 요금(Fare)이 0원이거나 결측된 승객을 식별하고, 0으로 나누기(ZeroDivisionError) 오류 없이 승객당 요금 배분 안전 검증을 수행해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"요금": "Fare"},
        "expected_output_type": "text",
        "python_code": """# L2_093: Fare=0 및 결측치 방어적 처리
import numpy as np

zero_fare_count = (df_titanic['Fare'] == 0).sum()
null_fare_count = df_titanic['Fare'].isna().sum()

print(f"요금이 0원인 승객 수: {zero_fare_count}명")
print(f"요금 결측 승객 수: {null_fare_count}명")

family_size = df_titanic['SibSp'] + df_titanic['Parch'] + 1
fare_per_person = df_titanic['Fare'].fillna(0) / np.maximum(family_size, 1)

print(f"인당 평균 운임(0원 승객 포함 방어 계산): £{fare_per_person.mean():.2f}")
print(f"인당 운임 상위 3명:")
print(fare_per_person.nlargest(3))
"""
    })

    specs.append({
        "id": "L2_094",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 원본 데이터프레임을 훼손(mutation)하지 않고, 나이(Age) 결측치를 Pclass와 Sex별 중앙값으로 대체하는 안전한 전처리 파이프라인을 작성해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"나이": "Age", "객실 등급": "Pclass", "성별": "Sex"},
        "expected_output_type": "text",
        "python_code": """# L2_094: 원본 보존 불변 전처리 및 그룹 중앙값 결측치 보정
original_nulls = df_titanic['Age'].isna().sum()

df_cleaned = df_titanic.copy()
median_ages = df_cleaned.groupby(['Pclass', 'Sex'])['Age'].transform('median')
df_cleaned['Age_Imputed'] = df_cleaned['Age'].fillna(median_ages)

print(f"원본 Age 결측치 수: {original_nulls}개 (원본 불변 검증 완료: {df_titanic['Age'].isna().sum() == original_nulls})")
print(f"보정 후 Age_Imputed 결측치 수: {df_cleaned['Age_Imputed'].isna().sum()}개")
print(f"보정 전 평균 나이: {df_titanic['Age'].mean():.2f}세 -> 보정 후: {df_cleaned['Age_Imputed'].mean():.2f}세")
"""
    })

    specs.append({
        "id": "L2_095",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "schema",
        "difficulty": "Advanced",
        "prompt": "은행 데이터(bank_loan)와 타이타닉(titanic) 데이터의 스키마를 비교하여 공통 컬럼이 있는지 확인하고 교차 분석 가능 여부를 진단해줘.",
        "target_table": "both",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_095: 두 독립 테이블 간 스키마 중복 및 조인 가능성 진단
bank_cols = set(df_bank.columns)
titanic_cols = set(df_titanic.columns)

common_cols = bank_cols.intersection(titanic_cols)
print(f"공통 컬럼 목록: {list(common_cols)}")

if not common_cols:
    print("진단: 두 테이블은 공통 키(Primary/Foreign Key)가 없으므로 직접 조인(JOIN)할 수 없는 독립 데이터셋입니다.")
    print("개별 테이블 단위 질의 또는 공통 인구통계 통계치(나이 등) 비교 분석만 가능합니다.")
else:
    print(f"조인 가능한 공통 컬럼 존재: {common_cols}")
"""
    })

    specs.append({
        "id": "L2_096",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "계좌 잔액(balance)의 극단치 왜곡을 방지하기 위해 상하위 1% 윈저화(Winsorization/Clipping)를 적용하고, 원본 평균과 보정 평균을 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"계좌 잔액": "balance", "윈저화": "clipping"},
        "expected_output_type": "text",
        "python_code": """# L2_096: 잔액 상하위 1% 윈저화(Clipping) 및 통계 왜곡 보정
lower_bound = df_bank['balance'].quantile(0.01)
upper_bound = df_bank['balance'].quantile(0.99)

clipped_balance = df_bank['balance'].clip(lower=lower_bound, upper=upper_bound)

print(f"하위 1% 경계값: ${lower_bound:,.2f}")
print(f"상위 1% 경계값: ${upper_bound:,.2f}")
print(f"원본 잔액 평균: ${df_bank['balance'].mean():,.2f} (최소: ${df_bank['balance'].min():,.2f}, 최대: ${df_bank['balance'].max():,.2f})")
print(f"윈저화 보정 잔액 평균: ${clipped_balance.mean():,.2f} (최소: ${clipped_balance.min():,.2f}, 최대: ${clipped_balance.max():,.2f})")
"""
    })

    specs.append({
        "id": "L2_097",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "목표 변수 'y'(yes/no)를 이진 변수(1/0)로 변환할 때, 알 수 없는 결측값이나 제3의 범주가 들어와도 에러 없이 처리하는 방어적 매핑 코드를 작성해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"목표 변수": "y"},
        "expected_output_type": "text",
        "python_code": """# L2_097: 미지의 범주/결측치 대응 방어적 이진 변환 매핑
import numpy as np
import pandas as pd

def safe_binary_encode(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().lower()
    if val_str in ['yes', 'y', 'true', '1']:
        return 1
    elif val_str in ['no', 'n', 'false', '0']:
        return 0
    return np.nan

encoded_series = df_bank['y'].apply(safe_binary_encode)
print(f"변환 완료 값 분포:\\n{encoded_series.value_counts(dropna=False)}")
print(f"가입률(1의 비율): {encoded_series.mean() * 100:.2f}%")
"""
    })

    specs.append({
        "id": "L2_098",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업(job)과 교육(education)의 소수 그룹에서 표본 수가 1개 이하이거나 분산이 0인 경우 발생할 수 있는 표준편차(std) 결측치를 0으로 안전하게 보정해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"직업": "job", "교육": "education", "표준편차": "std"},
        "expected_output_type": "text",
        "python_code": """# L2_098: 소수 그룹 분산/표준편차 결측치 안전 보정
grp_stats = df_bank.groupby(['job', 'education'])['balance'].agg(
    표본수='count',
    평균='mean',
    표준편차='std'
)

null_std_count = grp_stats['표준편차'].isna().sum()
grp_stats_safe = grp_stats.fillna({'표준편차': 0.0}).round(2)

print(f"표본 수 1개로 인해 표준편차가 NaN이었던 그룹 수: {null_std_count}개")
print("보정 완료 상위 5개 그룹 통계:")
print(grp_stats_safe.head(5))
"""
    })

    specs.append({
        "id": "L2_099",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "schema",
        "difficulty": "Advanced",
        "prompt": "존재하지 않는 가상의 테이블 'workspace.default.credit_card'에 대한 쿼리 요청이 들어왔을 때, 시스템 등록 테이블 목록을 확인하고 적절한 예외 메시지를 반환해줘.",
        "target_table": "system",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_099: 비존재 테이블 요청 방어 및 유효 테이블 목록 안내
available_tables = ['workspace.default.bank_loan', 'workspace.default.titanic']
requested_table = 'workspace.default.credit_card'

if requested_table not in available_tables:
    print(f"오류: '{requested_table}' 테이블을 찾을 수 없습니다.")
    print("현재 분석 가능한 유효 테이블 목록:")
    for t in available_tables:
        print(f"  - {t}")
    print("안내: 신용 및 대출 관련 데이터는 'workspace.default.bank_loan'을 참조하시기 바랍니다.")
else:
    print(f"테이블 정상 로드 완료: {requested_table}")
"""
    })

    specs.append({
        "id": "L2_100",
        "category": "예외 대응 및 스키마 경계 검증",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "입력 데이터에 결측치, 음수 잔액, 비정상치가 혼재되어 있을 때 4단계 검증(스키마 확인 -> 결측치 감지 -> 음수 플래그 -> 요약 통계 산출)을 수행하는 안전한 파이프라인을 실행해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L2_100: 다단계 방어적 데이터 품질 검증 및 안전 파이프라인
print("=== [1단계] 필수 스키마 유효성 검증 ===")
required_cols = ['age', 'balance', 'job', 'y']
missing_cols = [c for c in required_cols if c not in df_bank.columns]
assert len(missing_cols) == 0, f"누락 컬럼: {missing_cols}"
print("-> 필수 스키마 검증 통과 (모든 컬럼 존재)")

print("\\n=== [2단계] 결측치 현황 검증 ===")
null_counts = df_bank[required_cols].isna().sum()
print(f"-> 결측치 수: {null_counts.to_dict()}")

print("\\n=== [3단계] 음수 계좌 잔액(마이너스 통장) 감지 ===")
negative_bal_count = (df_bank['balance'] < 0).sum()
neg_ratio = (negative_bal_count / len(df_bank)) * 100
print(f"-> 마이너스 잔액 고객 수: {negative_bal_count}명 ({neg_ratio:.2f}%)")

print("\\n=== [4단계] 극단치 방어적 요약 통계 ===")
safe_summary = {
    '총_고객수': len(df_bank),
    '정상_잔액_고객수': (df_bank['balance'] >= 0).sum(),
    '양수_잔액_중앙값': df_bank[df_bank['balance'] >= 0]['balance'].median(),
    '예금_가입자수': (df_bank['y'] == 'yes').sum()
}
for k, v in safe_summary.items():
    print(f"  {k}: {v:,.0f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
print("파이프라인 정상 완료: 데이터 분석을 안전하게 수행할 준비가 되었습니다.")
"""
    })

    return specs
