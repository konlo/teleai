"""Level 1 Definitions Part 4 (L1_076 ~ L1_100):
- 📊 Single Chart Visualizations (25 items: L1_076 ~ L1_100)
"""

def get_level1_part4():
    specs = []
    
    # -------------------------------------------------------------
    # 📊 Single Chart Visualizations (L1_076 ~ L1_100)
    # -------------------------------------------------------------
    specs.append({
        "id": "L1_076",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "고객들의 연령(age) 분포를 확인하기 위해 히스토그램을 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_076: 연령 히스토그램
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(df_bank['age'], bins=20, color='steelblue', edgecolor='black')
plt.title('고객 연령(Age) 분포')
plt.xlabel('연령')
plt.ylabel('인원수')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_077",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "고객 계좌 잔액(balance)의 분포를 박스플롯(Boxplot)으로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_077: 잔액 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
plt.boxplot(df_bank['balance'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('고객 계좌 잔액(Balance) 박스플롯')
plt.xlabel('잔액 ($)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_078",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "직업(job)별 고객 수를 막대 차트(Bar Chart)로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_078: 직업별 고객수 막대그래프
import matplotlib.pyplot as plt

job_counts = df_bank['job'].value_counts()
plt.figure(figsize=(9, 4))
job_counts.plot(kind='bar', color='coral')
plt.title('직업(job)별 고객 수')
plt.xlabel('직업')
plt.ylabel('고객 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_079",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "혼인 상태(marital)별 고객 비중을 파이 차트(Pie Chart)로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_079: 혼인상태 파이차트
import matplotlib.pyplot as plt

m_counts = df_bank['marital'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(m_counts, labels=m_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('혼인 상태(Marital)별 고객 비중')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_080",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "교육 수준(education)별 고객 수를 막대 차트로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_080: 교육수준별 고객수 막대차트
import matplotlib.pyplot as plt

edu_counts = df_bank['education'].value_counts()
plt.figure(figsize=(7, 4))
edu_counts.plot(kind='bar', color='mediumpurple')
plt.title('교육 수준(Education)별 고객 수')
plt.xlabel('교육 수준')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_081",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "주택 대출(housing) 보유 여부('yes' vs 'no')를 막대 차트로 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_081: 주택대출 여부 막대차트
import matplotlib.pyplot as plt

h_counts = df_bank['housing'].value_counts()
plt.figure(figsize=(5, 4))
h_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('주택 대출(Housing Loan) 보유 여부')
plt.xlabel('보유 여부')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_082",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "개인 대출(loan) 보유 여부('yes' vs 'no')를 막대 차트로 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_082: 개인대출 여부 막대차트
import matplotlib.pyplot as plt

l_counts = df_bank['loan'].value_counts()
plt.figure(figsize=(5, 4))
l_counts.plot(kind='bar', color=['lightgreen', 'crimson'])
plt.title('개인 대출(Personal Loan) 보유 여부')
plt.xlabel('보유 여부')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_083",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "통화 상담 시간(duration)의 분포를 히스토그램으로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_083: 상담시간 히스토그램
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(df_bank['duration'], bins=25, color='teal', edgecolor='black')
plt.title('통화 상담 시간(Duration) 분포')
plt.xlabel('상담 시간 (초)')
plt.ylabel('빈도')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_084",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "접촉 월(month)별 고객 접촉 건수 추이를 막대 차트로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_084: 월별 접촉 건수 막대차트
import matplotlib.pyplot as plt

months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_counts = df_bank['month'].value_counts().reindex(months_order).dropna()

plt.figure(figsize=(9, 4))
month_counts.plot(kind='bar', color='dodgerblue')
plt.title('월(Month)별 마케팅 접촉 건수')
plt.xlabel('월')
plt.ylabel('접촉 건수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_085",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "정기예금 가입 여부(y)별 인원수를 막대 차트로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_085: 타깃 y 막대그래프
import matplotlib.pyplot as plt

y_counts = df_bank['y'].value_counts()
plt.figure(figsize=(5, 4))
y_counts.plot(kind='bar', color=['slategray', 'gold'])
plt.title('정기예금 가입 여부(y) 고객 수')
plt.xlabel('가입 여부 (y)')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_086",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "나이(age)와 잔액(balance) 간의 관계를 확인하기 위해 산점도(Scatter Plot)를 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_086: 나이 vs 잔액 산점도
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.scatter(df_bank['age'], df_bank['balance'], alpha=0.3, color='royalblue')
plt.title('고객 나이(Age) vs 계좌 잔액(Balance)')
plt.xlabel('나이 (세)')
plt.ylabel('잔액 ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_087",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "상담 시간(duration)과 접촉 횟수(campaign) 간의 산점도를 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_087: 상담시간 vs 접촉횟수 산점도
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.scatter(df_bank['duration'], df_bank['campaign'], alpha=0.3, color='forestgreen')
plt.title('상담 시간(Duration) vs 접촉 횟수(Campaign)')
plt.xlabel('상담 시간 (초)')
plt.ylabel('접촉 횟수')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_088",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "직업별 평균 잔액을 내림차순 정렬하여 수평 막대 차트(Horizontal Bar)로 그려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_088: 직업별 평균 잔액 수평 막대
import matplotlib.pyplot as plt

job_bal = df_bank.groupby('job')['balance'].mean().sort_values(ascending=True)
plt.figure(figsize=(8, 5))
job_bal.plot(kind='barh', color='darkorange')
plt.title('직업별 평균 계좌 잔액')
plt.xlabel('평균 잔액 ($)')
plt.ylabel('직업')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_089",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 탑승 등급(Pclass)별 승객 수를 막대 차트로 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_089: 타이타닉 Pclass 막대차트
import matplotlib.pyplot as plt

pc_counts = df_titanic['Pclass'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
pc_counts.plot(kind='bar', color='mediumseagreen')
plt.title('타이타닉 탑승 등급(Pclass)별 승객 수')
plt.xlabel('탑승 등급 (1=1등실, 2=2등실, 3=3등실)')
plt.ylabel('승객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_090",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 성별(Sex) 분포를 파이 차트로 시각화해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_090: 타이타닉 성별 파이차트
import matplotlib.pyplot as plt

sex_counts = df_titanic['Sex'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', colors=['cornflowerblue', 'lightpink'], startangle=90)
plt.title('타이타닉 승객 성별(Sex) 비중')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_091",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 생존자(1)와 사망자(0) 수를 막대 차트로 비교해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_091: 생존/사망 막대차트
import matplotlib.pyplot as plt

surv_counts = df_titanic['Survived'].value_counts().sort_index()
plt.figure(figsize=(5, 4))
surv_counts.plot(kind='bar', color=['gray', 'salmon'])
plt.title('타이타닉 생존(1) vs 사망(0) 승객 수')
plt.xlabel('생존 여부')
plt.ylabel('승객 수')
plt.xticks([0, 1], ['사망 (0)', '생존 (1)'], rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_092",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 요금(Fare)의 분포를 박스플롯으로 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_092: 타이타닉 요금 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
plt.boxplot(df_titanic['Fare'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='khaki'))
plt.title('타이타닉 탑승 요금(Fare) 박스플롯')
plt.xlabel('요금 ($)')
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_093",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 나이(Age)의 분포를 히스토그램으로 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_093: 타이타닉 나이 히스토그램
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(df_titanic['Age'].dropna(), bins=20, color='mediumpurple', edgecolor='black')
plt.title('타이타닉 승객 연령(Age) 분포')
plt.xlabel('나이')
plt.ylabel('승객 수')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_094",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 탑승지(Embarked)별 승객 수를 막대 차트로 시각화해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_094: 타이타닉 탑승지 막대차트
import matplotlib.pyplot as plt

emb_counts = df_titanic['Embarked'].value_counts()
plt.figure(figsize=(6, 4))
emb_counts.plot(kind='bar', color='darkcyan')
plt.title('탑승지(Embarked)별 승객 수 (S=Southampton, C=Cherbourg, Q=Queenstown)')
plt.xlabel('탑승지')
plt.ylabel('승객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_095",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "교육 수준(education)별 평균 연령을 막대 차트로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_095: 교육수준별 평균 연령 막대차트
import matplotlib.pyplot as plt

edu_age = df_bank.groupby('education')['age'].mean()
plt.figure(figsize=(7, 4))
edu_age.plot(kind='bar', color='rosybrown')
plt.title('교육 수준별 평균 연령')
plt.xlabel('교육 수준')
plt.ylabel('평균 연령 (세)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_096",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "통신 수단(contact)별 고객 수를 막대 차트로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_096: 통신수단별 고객수 막대차트
import matplotlib.pyplot as plt

c_counts = df_bank['contact'].value_counts()
plt.figure(figsize=(6, 4))
c_counts.plot(kind='bar', color='slateblue')
plt.title('통신 수단(Contact)별 고객 수')
plt.xlabel('통신 수단')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_097",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "이전 마케팅 결과(poutcome) 종류별 고객 수를 막대 차트로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_097: poutcome별 고객수 막대차트
import matplotlib.pyplot as plt

p_counts = df_bank['poutcome'].value_counts()
plt.figure(figsize=(6, 4))
p_counts.plot(kind='bar', color='chocolate')
plt.title('이전 마케팅 결과(Poutcome)별 고객 수')
plt.xlabel('결과')
plt.ylabel('고객 수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_098",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "접촉일(day, 1~31일)별 접촉 건수를 꺾은선 그래프(Line Plot)로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_098: 일자별 접촉 건수 선그래프
import matplotlib.pyplot as plt

day_counts = df_bank['day'].value_counts().sort_index()
plt.figure(figsize=(9, 4))
plt.plot(day_counts.index, day_counts.values, marker='o', color='navy')
plt.title('월중 접촉 일자(Day)별 마케팅 건수')
plt.xlabel('일자 (1~31일)')
plt.ylabel('접촉 건수')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_099",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "타이타닉 요금(Fare)과 나이(Age) 간의 산점도를 그려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_099: 타이타닉 나이 vs 요금 산점도
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.scatter(df_titanic['Age'], df_titanic['Fare'], alpha=0.4, color='crimson')
plt.title('타이타닉 승객 나이(Age) vs 요금(Fare)')
plt.xlabel('나이')
plt.ylabel('요금 ($)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "L1_100",
        "category": "단일 차트 시각화",
        "type": "chart",
        "difficulty": "Basic",
        "prompt": "캠페인 접촉 횟수(campaign)의 분포를 박스플롯으로 시각화해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "chart",
        "python_code": """# L1_100: 접촉 횟수 박스플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
plt.boxplot(df_bank['campaign'], vert=False, patch_artist=True, boxprops=dict(facecolor='wheat'))
plt.title('캠페인 접촉 횟수(Campaign) 박스플롯')
plt.xlabel('접촉 횟수')
plt.tight_layout()
plt.show()
"""
    })

    return specs
