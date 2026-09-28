"""Level 1 Definitions Part 2 (L1_026 ~ L1_050):
- 🧠 Synonym/Alias-based Basic Column Lookups Continued (5 items: L1_026 ~ L1_030)
- 📋 Simple Filtering & Text/Table Summaries (20 items: L1_031 ~ L1_050)
"""

def get_level1_part2():
    specs = []
    
    # -------------------------------------------------------------
    # 🧠 Synonym/Alias Queries Continued (L1_026 ~ L1_030)
    # -------------------------------------------------------------
    specs.append({
        "id": "L1_026",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 중 '1등실 탑승객'이 몇 명인지 세어줘.",
        "target_table": "titanic",
        "synonym_mapping": {"1등실 탑승객": "Pclass==1"},
        "expected_output_type": "text",
        "python_code": """# L1_026: '1등실' -> Pclass=1 매핑
pclass1_cnt = (df_titanic['Pclass'] == 1).sum()
print(f"1등실(Pclass=1) 탑승객 수: {pclass1_cnt}명 (전체의 {pclass1_cnt/len(df_titanic)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L1_027",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들 중 신용불량으로 '파산/연체 이력'이 있는 사람이 몇 명이야?",
        "target_table": "bank_loan",
        "synonym_mapping": {"파산/연체 이력": "default=='yes'"},
        "expected_output_type": "text",
        "python_code": """# L1_027: '파산/연체 이력' -> default='yes' 매핑
default_cnt = (df_bank['default'] == 'yes').sum()
print(f"연체/신용불량 이력 보유 고객 수(default=yes): {default_cnt}명")
"""
    })

    specs.append({
        "id": "L1_028",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "이번 마케팅 행사에서 고객들에게 전화를 건 '통화 횟수'의 평균을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"통화 횟수": "campaign"},
        "expected_output_type": "text",
        "python_code": """# L1_028: '통화 횟수' -> campaign 매핑
avg_camp = df_bank['campaign'].mean()
print(f"고객당 평균 접촉/통화 횟수(campaign): {avg_camp:.2f}회")
"""
    })

    specs.append({
        "id": "L1_029",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들의 주요 '접속 통신 수단' 종류별 빈도를 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"접속 통신 수단": "contact"},
        "expected_output_type": "table",
        "python_code": """# L1_029: '접속 통신 수단' -> contact 매핑
contact_counts = df_bank['contact'].value_counts().reset_index()
contact_counts.columns = ['통신 수단(contact)', '고객수']
print(contact_counts.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_030",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "이전 마케팅에서 '성공' 판정을 받은 사람이 몇 명인지 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"이전 마케팅 성공": "poutcome=='success'"},
        "expected_output_type": "text",
        "python_code": """# L1_030: '이전 마케팅 성공' -> poutcome='success' 매핑
succ_cnt = (df_bank['poutcome'] == 'success').sum()
print(f"이전 마케팅 성공 고객 수(poutcome=success): {succ_cnt}명")
"""
    })

    # -------------------------------------------------------------
    # 📋 Simple Filtering & Text/Table Summaries (L1_031 ~ L1_050)
    # -------------------------------------------------------------
    specs.append({
        "id": "L1_031",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "나이가 30세 이상 39세 이하인 30대 고객들의 인원수와 이들의 평균 잔액을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_031: 30대 고객 필터링
sub_30s = df_bank[(df_bank['age'] >= 30) & (df_bank['age'] <= 39)]
print(f"30대 고객 수: {len(sub_30s)}명")
print(f"30대 평균 잔액: ${sub_30s['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L1_032",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "계좌 잔액이 0원 미만인 마이너스 잔액 고객들의 인원수를 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_032: 마이너스 잔액 필터링
neg_bal = df_bank[df_bank['balance'] < 0]
print(f"마이너스 잔액 고객 수: {len(neg_bal)}명")
print(f"마이너스 고객 평균 잔액: ${neg_bal['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L1_033",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "직업(job)이 'technician'인 고객들의 주택 대출(housing) 보유 현황을 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_033: technician 직업군 주택대출 현황
tech_housing = df_bank[df_bank['job'] == 'technician']['housing'].value_counts().reset_index()
tech_housing.columns = ['housing 대출여부', '고객수']
print("기술직(technician) 주택대출 보유 현황:")
print(tech_housing.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_034",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "결혼 상태가 'single'인 미혼 고객들의 평균 연령과 평균 잔액을 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_034: single 미혼 고객 지표
singles = df_bank[df_bank['marital'] == 'single']
print(f"미혼(single) 고객 수: {len(singles)}명")
print(f"평균 연령: {singles['age'].mean():.1f}세")
print(f"평균 잔액: ${singles['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L1_035",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "교육 수준이 'tertiary'(대졸 이상)인 고객들의 직업 분포를 상위 5개 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_035: tertiary 교육 수준 직업 분포
tertiary_jobs = df_bank[df_bank['education'] == 'tertiary']['job'].value_counts().head(5).reset_index()
tertiary_jobs.columns = ['직업(job)', '인원수']
print("고학력자(tertiary) 주요 직업 TOP 5:")
print(tertiary_jobs.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_036",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "상담 시간(duration)이 500초를 초과한 고객들의 예금 가입(y='yes') 비율을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_036: 통화 시간 500초 초과 고객 가입률
long_calls = df_bank[df_bank['duration'] > 500]
conv_rate = (long_calls['y'] == 'yes').mean() * 100
print(f"500초 초과 통화 고객 수: {len(long_calls)}명")
print(f"이들의 예금 가입률: {conv_rate:.2f}%")
"""
    })

    specs.append({
        "id": "L1_037",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "주택 대출(housing)과 개인 대출(loan)을 둘 다 안 받은 무대출 고객들의 인원수를 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_037: 무대출(housing=no & loan=no) 고객
no_loans = df_bank[(df_bank['housing'] == 'no') & (df_bank['loan'] == 'no')]
print(f"무대출 고객 수: {len(no_loans)}명 (전체의 {len(no_loans)/len(df_bank)*100:.1f}%)")
print(f"무대출 고객 평균 잔액: ${no_loans['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L1_038",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 중 여성(female) 승객들의 생존율(%)을 계산해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_038: 여성 승객 생존율
females = df_titanic[df_titanic['Sex'] == 'female']
fem_surv_rate = females['Survived'].mean() * 100
print(f"여성 승객 수: {len(females)}명")
print(f"여성 생존율: {fem_surv_rate:.2f}%")
"""
    })

    specs.append({
        "id": "L1_039",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 중 남성(male) 승객들의 생존율(%)을 계산해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_039: 남성 승객 생존율
males = df_titanic[df_titanic['Sex'] == 'male']
male_surv_rate = males['Survived'].mean() * 100
print(f"남성 승객 수: {len(males)}명")
print(f"남성 생존율: {male_surv_rate:.2f}%")
"""
    })

    specs.append({
        "id": "L1_040",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "은퇴자(retired) 고객들의 평균 연령과 평균 잔액을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_040: 은퇴자(retired) 지표
retired = df_bank[df_bank['job'] == 'retired']
print(f"은퇴 고객 수: {len(retired)}명")
print(f"은퇴자 평균 연령: {retired['age'].mean():.1f}세")
print(f"은퇴자 평균 잔액: ${retired['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L1_041",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "학생(student) 고객들의 총 인원수와 이들의 예금 가입(y='yes') 건수를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_041: 학생(student) 고객 지표
students = df_bank[df_bank['job'] == 'student']
st_yes = (students['y'] == 'yes').sum()
print(f"학생 고객 수: {len(students)}명, 예금 가입 성공: {st_yes}명")
"""
    })

    specs.append({
        "id": "L1_042",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "5월(may)에 접촉한 고객들의 인원수와 전체 대비 비중을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_042: 5월 접촉 고객 수
may_cust = df_bank[df_bank['month'] == 'may']
print(f"5월 접촉 고객 수: {len(may_cust)}명 (전체의 {len(may_cust)/len(df_bank)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L1_043",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "이전 캠페인에서 접촉한 적이 없는(pdays == -1) 신규 대상 고객 수를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_043: 신규 접촉(pdays=-1) 고객 수
new_contacts = df_bank[df_bank['pdays'] == -1]
print(f"이전 마케팅 미접촉 신규 고객 수: {len(new_contacts)}명 (전체의 {len(new_contacts)/len(df_bank)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L1_044",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "이번 마케팅에서 1회만 접촉(campaign == 1)한 고객들의 인원수를 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_044: 단 1회 접촉 고객
single_contact = df_bank[df_bank['campaign'] == 1]
print(f"1회 접촉 고객 수: {len(single_contact)}명 (전체의 {len(single_contact)/len(df_bank)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L1_045",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉에서 요금(Fare)을 100달러 이상 낸 고액 승객들의 생존율을 구해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_045: 고액 요금 승객 생존율
high_fare = df_titanic[df_titanic['Fare'] >= 100]
print(f"요금 100달러 이상 승객 수: {len(high_fare)}명")
print(f"고액 승객 생존율: {high_fare['Survived'].mean()*100:.2f}%")
"""
    })

    specs.append({
        "id": "L1_046",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객 중 10세 미만 어린이(Age < 10)의 생존율을 구해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_046: 어린이 승객 생존율
children = df_titanic[df_titanic['Age'] < 10]
print(f"10세 미만 어린이 승객 수: {len(children)}명")
print(f"어린이 생존율: {children['Survived'].mean()*100:.2f}%")
"""
    })

    specs.append({
        "id": "L1_047",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "연령이 60세 이상인 시니어 고객들의 직업 분포를 표로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_047: 60세 이상 시니어 고객 직업 분포
seniors = df_bank[df_bank['age'] >= 60]
senior_jobs = seniors['job'].value_counts().reset_index()
senior_jobs.columns = ['직업', '고객수']
print("60세 이상 고객의 직업 분포:")
print(senior_jobs.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_048",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "잔액이 10,000 달러 이상인 고자산 고객들의 직업 TOP 3를 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_048: 고자산($10,000+) 직업 상위 3개
wealthy = df_bank[df_bank['balance'] >= 10000]
top3_jobs = wealthy['job'].value_counts().head(3).reset_index()
top3_jobs.columns = ['직업', '인원수']
print("잔액 1만달러 이상 고객 직업 상위 3개:")
print(top3_jobs.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_049",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "계좌 잔액 상위 5명의 고객 ID, 나이, 직업, 잔액을 표로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_049: 잔액 상위 5명
top5_bal = df_bank.sort_values('balance', ascending=False)[['id', 'age', 'job', 'balance']].head(5)
print("잔액 상위 5명 고객:")
print(top5_bal.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_050",
        "category": "단순 필터링 및 텍스트/표 답변",
        "type": "table",
        "difficulty": "Basic",
        "prompt": "상담 시간(duration)이 가장 길었던 상위 5건의 통화 기록을 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_050: 통화시간 상위 5건
top5_dur = df_bank.sort_values('duration', ascending=False)[['id', 'job', 'duration', 'y']].head(5)
print("상담 시간 최장 5건:")
print(top5_dur.to_string(index=False))
"""
    })

    return specs
