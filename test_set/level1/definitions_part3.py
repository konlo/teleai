"""Level 1 Definitions Part 3 (L1_051 ~ L1_075):
- 📋 Single Group Statistics, Aggregations & Pivot Tables (25 items: L1_051 ~ L1_075)
"""

def get_level1_part3():
    specs = []
    
    # -------------------------------------------------------------
    # 📋 Single Group Statistics & Aggregations (L1_051 ~ L1_075)
    # -------------------------------------------------------------
    specs.append({
        "id": "L1_051",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "직업(job)별 고객 수와 평균 잔액(balance)을 계산하고 평균 잔액 내림차순으로 정렬해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_051: 직업별 평균 잔액 정렬
job_agg = df_bank.groupby('job').agg(고객수=('id', 'count'), 평균잔액=('balance', 'mean'))
job_agg = job_agg.sort_values('평균잔액', ascending=False).round(2).reset_index()
print("직업별 고객수 및 평균 잔액 (내림차순):")
print(job_agg.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_052",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "교육 수준(education)별 평균 잔액과 중앙값 잔액을 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_052: 교육수준별 평균 및 중앙값 잔액
edu_stats = df_bank.groupby('education')['balance'].agg(['mean', 'median']).round(2).reset_index()
edu_stats.columns = ['교육수준', '평균잔액', '중앙값잔액']
print("교육 수준별 잔액 통계:")
print(edu_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_053",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "결혼 상태(marital)별 평균 연령과 고객 수를 표로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_053: 결혼 상태별 평균 연령
m_stats = df_bank.groupby('marital').agg(고객수=('id', 'count'), 평균연령=('age', 'mean')).round(1).reset_index()
print("결혼 상태별 평균 연령:")
print(m_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_054",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "주택 대출(housing) 보유 여부에 따른 평균 잔액 차이를 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_054: 주택대출 여부별 평균 잔액
h_stats = df_bank.groupby('housing')['balance'].agg(고객수='count', 평균잔액='mean').round(2).reset_index()
print("주택 대출 유무별 잔액 비교:")
print(h_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_055",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "개인 대출(loan) 보유 여부에 따른 평균 잔액 차이를 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_055: 개인대출 여부별 평균 잔액
l_stats = df_bank.groupby('loan')['balance'].agg(고객수='count', 평균잔액='mean').round(2).reset_index()
print("개인 대출 유무별 잔액 비교:")
print(l_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_056",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "월(month)별 고객 접촉 건수와 평균 상담 시간(duration)을 집계해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_056: 월별 접촉 건수 및 상담시간
month_stats = df_bank.groupby('month').agg(접촉수=('id', 'count'), 평균시간=('duration', 'mean')).round(1).reset_index()
print("월별 접촉 건수 및 평균 통화시간:")
print(month_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_057",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "접속 수단(contact)별 예금 가입률(y='yes' 비율)을 백분율로 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_057: 통신수단별 예금 가입률
contact_y = (df_bank.groupby('contact')['y'].apply(lambda s: (s == 'yes').mean() * 100)).round(2).reset_index()
contact_y.columns = ['접속수단', '예금가입률(%)']
print("통신수단별 가입률:")
print(contact_y.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_058",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "이전 마케팅 결과(poutcome)별 고객 수와 이번 예금 가입률(%)을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_058: poutcome별 가입률
poutcome_stats = df_bank.groupby('poutcome').agg(
    고객수=('id', 'count'),
    가입률=('y', lambda s: round((s == 'yes').mean() * 100, 2))
).reset_index()
print("이전 마케팅 결과별 성과:")
print(poutcome_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_059",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 탑승 등급(Pclass)별 승객 수와 생존율(%)을 표로 보여줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_059: Pclass별 생존율
pclass_surv = df_titanic.groupby('Pclass').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', lambda s: round(s.mean() * 100, 2))
).reset_index()
print("탑승 등급별 승객수 및 생존율:")
print(pclass_surv.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_060",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 성별(Sex)별 승객 수와 생존율(%)을 표로 보여줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_060: 성별 생존율
sex_surv = df_titanic.groupby('Sex').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', lambda s: round(s.mean() * 100, 2))
).reset_index()
print("성별 승객수 및 생존율:")
print(sex_surv.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_061",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 탑승지(Embarked)별 승객 수와 평균 요금(Fare)을 계산해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_061: 탑승지별 요금 통계
emb_stats = df_titanic.groupby('Embarked').agg(승객수=('PassengerId', 'count'), 평균요금=('Fare', 'mean')).round(2).reset_index()
print("탑승지별 승객수 및 평균 요금:")
print(emb_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_062",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "직업(job)과 혼인상태(marital)에 따른 고객 수를 피벗 테이블(pivot table)로 만들어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_062: 직업 x 혼인상태 피벗 테이블
pivot_jm = df_bank.pivot_table(index='job', columns='marital', values='id', aggfunc='count', fill_value=0)
print("직업 x 혼인상태 고객수 피벗:")
print(pivot_jm)
"""
    })

    specs.append({
        "id": "L1_063",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "교육수준(education)과 주택대출(housing) 여부별 평균 잔액 피벗 테이블을 만들어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_063: 교육수준 x 주택대출 평균 잔액 피벗
pivot_eh = df_bank.pivot_table(index='education', columns='housing', values='balance', aggfunc='mean').round(2)
print("교육수준 x 주택대출 평균 잔액 피벗:")
print(pivot_eh)
"""
    })

    specs.append({
        "id": "L1_064",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 등급(Pclass)과 성별(Sex)에 따른 생존율 피벗 테이블을 백분율로 출력해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_064: 등급 x 성별 생존율 피벗
pivot_ps = (df_titanic.pivot_table(index='Pclass', columns='Sex', values='Survived', aggfunc='mean') * 100).round(2)
print("등급 x 성별 생존율(%) 피벗:")
print(pivot_ps)
"""
    })

    specs.append({
        "id": "L1_065",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "연령을 10대, 20대, 30대, 40대, 50대, 60대 이상으로 구간화하여 연령대별 고객 수를 요약해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_065: 연령대 구간화 빈도 집계
bins = [0, 19, 29, 39, 49, 59, 120]
labels = ['10대', '20대', '30대', '40대', '50대', '60대 이상']
age_group = pd.cut(df_bank['age'], bins=bins, labels=labels)
age_summary = age_group.value_counts().sort_index().reset_index()
age_summary.columns = ['연령대', '고객수']
print("연령대별 고객 수:")
print(age_summary.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_066",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "연령대별(20대 이하, 30대, 40대, 50대 이상) 평균 잔액을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_066: 연령대별 평균 잔액
bins = [0, 29, 39, 49, 120]
labels = ['20대 이하', '30대', '40대', '50대 이상']
df_bank_copy = df_bank.copy()
df_bank_copy['age_group'] = pd.cut(df_bank_copy['age'], bins=bins, labels=labels)
age_bal = df_bank_copy.groupby('age_group', observed=False)['balance'].agg(['count', 'mean']).round(2).reset_index()
age_bal.columns = ['연령대', '고객수', '평균잔액']
print("연령대별 평균 잔액:")
print(age_bal.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_067",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "접촉 횟수(campaign)별 고객 수와 해당 그룹의 예금 가입 성공률(%)을 상위 5회까지 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_067: 캠페인 접촉 횟수별 가입률
camp_stats = df_bank.groupby('campaign').agg(
    고객수=('id', 'count'),
    가입률=('y', lambda s: round((s == 'yes').mean() * 100, 2))
).head(5).reset_index()
print("캠페인 접촉 횟수별(1~5회) 가입률:")
print(camp_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_068",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "예금 가입 여부(y)별 평균 통화 상담 시간(duration)을 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_068: 가입 여부별 상담 시간 비교
y_dur = df_bank.groupby('y')['duration'].agg(고객수='count', 평균상담시간='mean', 중앙값상담시간='median').round(1).reset_index()
print("예금 가입 여부별 상담 시간:")
print(y_dur.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_069",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "직업별로 예금에 가입한 고객(y='yes')이 가장 많은 상위 3개 직업을 찾아줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_069: 직업별 가입자 수 상위 3개
job_yes = df_bank[df_bank['y'] == 'yes']['job'].value_counts().head(3).reset_index()
job_yes.columns = ['직업', '가입고객수']
print("예금 가입자 수 상위 3개 직업:")
print(job_yes.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_070",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "신용불량(default) 여부별 고객 수와 평균 잔액을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_070: 신용불량 여부별 잔액 비교
def_stats = df_bank.groupby('default')['balance'].agg(고객수='count', 평균잔액='mean').round(2).reset_index()
print("신용불량(default) 유무별 잔액 통계:")
print(def_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_071",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "직업(job)별 주택 대출(housing) 보유율(%)을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_071: 직업별 주택대출 보유율
job_h = (df_bank.groupby('job')['housing'].apply(lambda s: (s == 'yes').mean() * 100)).round(2).reset_index()
job_h.columns = ['직업', '주택대출보유율(%)']
print(job_h.sort_values('주택대출보유율(%)', ascending=False).to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_072",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "직업(job)별 개인 신용 대출(loan) 보유율(%)을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_072: 직업별 개인대출 보유율
job_l = (df_bank.groupby('job')['loan'].apply(lambda s: (s == 'yes').mean() * 100)).round(2).reset_index()
job_l.columns = ['직업', '개인대출보유율(%)']
print(job_l.sort_values('개인대출보유율(%)', ascending=False).to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_073",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 형제/자매/배우자 동반자 수(SibSp)별 승객 수와 생존율을 보여줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_073: 동반자수(SibSp)별 생존율
sib_stats = df_titanic.groupby('SibSp').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', lambda s: round(s.mean() * 100, 2))
).reset_index()
print("동반자수(SibSp)별 생존율:")
print(sib_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_074",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 부모/자녀 동반자 수(Parch)별 승객 수와 생존율을 보여줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_074: 부모/자녀수(Parch)별 생존율
parch_stats = df_titanic.groupby('Parch').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', lambda s: round(s.mean() * 100, 2))
).reset_index()
print("부모/자녀수(Parch)별 생존율:")
print(parch_stats.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_075",
        "category": "단일 그룹 집계 및 요약표",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "월별로 마케팅 전화를 가장 많이 건 날짜(day) 상위 3일을 요약해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_075: 월별 최다 접촉 일자 TOP 3
day_counts = df_bank['day'].value_counts().head(3).reset_index()
day_counts.columns = ['접촉일(day)', '접촉건수']
print("최다 접촉 일자 상위 3일:")
print(day_counts.to_string(index=False))
"""
    })

    return specs
