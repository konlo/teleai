"""Level 2 Definitions Part 1 (L2_001 ~ L2_025):
- 🧠 Complex Multi-Condition Filters with Synonyms (20 items: L2_001 ~ L2_020)
- 📋 Multi-Dimensional Pivot Tables & Subtotals (5 items: L2_021 ~ L2_025)
"""

def get_level2_part1():
    specs = []
    
    # -------------------------------------------------------------
    # 🧠 Complex Multi-Condition Filters with Synonyms (L2_001 ~ L2_020)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_001",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "30대 기혼자('married') 중 '집 대출'이 있고 '예금 잔고'가 1,000달러 이상인 고객들의 인원수와 예금 가입률(y='yes' %)을 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"집 대출": "housing=='yes'", "예금 잔고": "balance >= 1000", "30대": "age.between(30, 39)"},
        "expected_output_type": "text",
        "python_code": """# L2_001: 30대 + 기혼 + 집대출 + 잔고 1000 이상
cond = (df_bank['age'].between(30, 39)) & (df_bank['marital'] == 'married') & (df_bank['housing'] == 'yes') & (df_bank['balance'] >= 1000)
sub = df_bank[cond]
sub_rate = (sub['y'] == 'yes').mean() * 100 if len(sub) > 0 else 0.0
print(f"조건 만족 고객 수: {len(sub)}명")
print(f"이들의 예금 가입 성공률(y=yes): {sub_rate:.2f}%")
"""
    })

    specs.append({
        "id": "L2_002",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "직업이 'management' 또는 'technician'이면서 '신용 대출'이 없고 '예금 잔액'이 5,000달러를 넘는 고객들의 평균 연령을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"신용 대출": "loan=='no'", "예금 잔액": "balance > 5000"},
        "expected_output_type": "text",
        "python_code": """# L2_002: 특정직업 + 무대출 + 잔액 5000 초과
cond = (df_bank['job'].isin(['management', 'technician'])) & (df_bank['loan'] == 'no') & (df_bank['balance'] > 5000)
sub = df_bank[cond]
print(f"조건 만족 인원수: {len(sub)}명")
print(f"평균 연령: {sub['age'].mean():.1f}세")
"""
    })

    specs.append({
        "id": "L2_003",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'상담 시간'이 400초 이상이고 '통화 횟수'가 3회 이하인 고객들의 예금 가입 성공 건수와 실패 건수를 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"상담 시간": "duration >= 400", "통화 횟수": "campaign <= 3"},
        "expected_output_type": "table",
        "python_code": """# L2_003: 통화시간 >= 400 & 캠페인 <= 3
cond = (df_bank['duration'] >= 400) & (df_bank['campaign'] <= 3)
sub = df_bank[cond]
y_counts = sub['y'].value_counts().reset_index()
y_counts.columns = ['가입여부(y)', '고객수']
print(f"조건 만족 고객 수: {len(sub)}명")
print(y_counts.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_004",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'나이'가 50세 이상이면서 '집 대출'과 '신용 대출'이 모두 없는 고객들의 평균 '통장 잔액'을 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"나이": "age >= 50", "집 대출": "housing=='no'", "신용 대출": "loan=='no'", "통장 잔액": "balance"},
        "expected_output_type": "text",
        "python_code": """# L2_004: 50세 이상 & 무대출 고객 평균 잔액
cond = (df_bank['age'] >= 50) & (df_bank['housing'] == 'no') & (df_bank['loan'] == 'no')
sub = df_bank[cond]
print(f"50세 이상 무대출 고객 수: {len(sub)}명")
print(f"평균 통장 잔액: ${sub['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L2_005",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'최종 학력'이 'tertiary'이고 미혼('single')인 20대(20~29세) 고객 중 실제로 예금에 '가입한 사람'들의 수를 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"최종 학력": "education=='tertiary'", "가입한 사람": "y=='yes'"},
        "expected_output_type": "text",
        "python_code": """# L2_005: 대졸 + 미혼 + 20대 + 가입완료
cond = (df_bank['education'] == 'tertiary') & (df_bank['marital'] == 'single') & (df_bank['age'].between(20, 29))
sub = df_bank[cond]
sub_yes = (sub['y'] == 'yes').sum()
print(f"조건 만족 20대 고객 수: {len(sub)}명, 예금 가입자 수: {sub_yes}명 (가입률: {sub_yes/len(sub)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L2_006",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'마이너스 통장'(잔액 < 0) 고객 중 '신용 대출'을 보유하고 있는 고객들의 직업 분포를 상위 3개 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"마이너스 통장": "balance < 0", "신용 대출": "loan=='yes'"},
        "expected_output_type": "table",
        "python_code": """# L2_006: 마이너스 잔액 & 개인대출 보유자
cond = (df_bank['balance'] < 0) & (df_bank['loan'] == 'yes')
sub = df_bank[cond]
top_jobs = sub['job'].value_counts().head(3).reset_index()
top_jobs.columns = ['직업', '고객수']
print(f"마이너스 잔액 & 대출 보유 고객: {len(sub)}명")
print(top_jobs.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_007",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "타이타닉에서 '1등실 또는 2등실'에 탑승한 '여성 승객'들의 생존율을 등급별로 비교해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"1등실 또는 2등실": "Pclass.isin([1, 2])", "여성 승객": "Sex=='female'"},
        "expected_output_type": "table",
        "python_code": """# L2_007: 1/2등실 여성 생존율 비교
fem_high = df_titanic[(df_titanic['Sex'] == 'female') & (df_titanic['Pclass'].isin([1, 2]))]
surv_comp = fem_high.groupby('Pclass').agg(승객수=('PassengerId', 'count'), 생존율=('Survived', lambda s: round(s.mean()*100, 2))).reset_index()
print("1/2등실 여성 생존율 비교:")
print(surv_comp.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_008",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "타이타닉에서 '3등실 승객' 중 '20세 미만 미성년자'의 생존자 수와 사망자 수를 세어줘.",
        "target_table": "titanic",
        "synonym_mapping": {"3등실": "Pclass==3", "20세 미만": "Age < 20"},
        "expected_output_type": "table",
        "python_code": """# L2_008: 3등실 20세 미만 생존/사망
p3_young = df_titanic[(df_titanic['Pclass'] == 3) & (df_titanic['Age'] < 20)]
surv_counts = p3_young['Survived'].value_counts().reset_index()
surv_counts.columns = ['생존여부(1=생존, 0=사망)', '승객수']
print(f"3등실 20세 미만 승객 ({len(p3_young)}명) 생존 현황:")
print(surv_counts.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_009",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'이전 마케팅에서 성공'(poutcome='success')했던 고객들 중 이번 캠페인에서 '가입한 사람'과 '미가입한 사람'의 평균 상담 시간을 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"이전 마케팅 성공": "poutcome=='success'", "가입한 사람": "y=='yes'"},
        "expected_output_type": "table",
        "python_code": """# L2_009: poutcome=success 고객의 이번 가입여부별 통화시간
succ_cust = df_bank[df_bank['poutcome'] == 'success']
dur_comp = succ_cust.groupby('y')['duration'].agg(고객수='count', 평균상담시간='mean').round(1).reset_index()
print("이전 성공 고객의 이번 가입 여부별 상담 시간:")
print(dur_comp.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_010",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'연체/신용불량 이력'(default='yes')이 있으면서 '집 대출'이나 '신용 대출' 중 하나라도 있는 고위험 다중 채무 고객 수를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"연체/신용불량": "default=='yes'", "집 대출/신용 대출": "(housing=='yes') | (loan=='yes')"},
        "expected_output_type": "text",
        "python_code": """# L2_010: 신용불량 & 대출보유 고위험 고객
cond = (df_bank['default'] == 'yes') & ((df_bank['housing'] == 'yes') | (df_bank['loan'] == 'yes'))
high_risk = df_bank[cond]
print(f"고위험 다중 채무 고객 수: {len(high_risk)}명 (전체의 {len(high_risk)/len(df_bank)*100:.2f}%)")
print(f"이들의 평균 잔액: ${high_risk['balance'].mean():,.2f}")
"""
    })

    specs.append({
        "id": "L2_011",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'통화 상담 시간'이 10분(600초) 이상인 장기 통화 고객들의 직업별 인원수와 예금 가입률(%)을 상위 5개 직업으로 정리해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"통화 상담 시간 10분": "duration >= 600"},
        "expected_output_type": "table",
        "python_code": """# L2_011: 10분 이상 장기 통화 고객 직업별 분석
long_calls = df_bank[df_bank['duration'] >= 600]
job_analysis = long_calls.groupby('job').agg(
    고객수=('id', 'count'),
    가입률=('y', lambda s: round((s == 'yes').mean() * 100, 2))
).sort_values('고객수', ascending=False).head(5).reset_index()
print("10분 이상 상담 고객 상위 5개 직업 실적:")
print(job_analysis.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_012",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'통장 잔고'가 10,000달러를 넘는 부유층 고객 중 '나이'가 40대인 고객들의 주택 대출 보유율을 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"통장 잔고": "balance > 10000", "나이 40대": "age.between(40, 49)"},
        "expected_output_type": "text",
        "python_code": """# L2_012: 40대 고액 자산가 주택대출 비율
cond = (df_bank['balance'] > 10000) & (df_bank['age'].between(40, 49))
sub = df_bank[cond]
h_rate = (sub['housing'] == 'yes').mean() * 100 if len(sub) > 0 else 0.0
print(f"40대 고자산 고객 수: {len(sub)}명, 주택 대출 보유율: {h_rate:.1f}%")
"""
    })

    specs.append({
        "id": "L2_013",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'가입 여부'(y)가 'yes'인 성공 고객들의 평균 '상담 시간'과 '평균 잔고'를 가입 실패 고객과 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"가입 여부": "y", "상담 시간": "duration", "평균 잔고": "balance"},
        "expected_output_type": "table",
        "python_code": """# L2_013: 예금 가입 성공 vs 실패 고객 핵심 지표 비교
comp = df_bank.groupby('y').agg(고객수=('id', 'count'), 평균상담시간=('duration', 'mean'), 평균잔고=('balance', 'mean')).round(2).reset_index()
comp['가입여부'] = comp['y'].map({'yes': '가입 성공', 'no': '가입 실패'})
print(comp[['가입여부', '고객수', '평균상담시간', '평균잔고']].to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_014",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "타이타닉에서 가족 동반 없이 '혼자 탑승한 승객'(SibSp=0 & Parch=0)들의 생존율을 가족 동반 승객과 비교해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"혼자 탑승한 승객": "SibSp==0 & Parch==0"},
        "expected_output_type": "table",
        "python_code": """# L2_014: 나홀로 승객 vs 가족 동반 승객 생존율
df_t_copy = df_titanic.copy()
df_t_copy['is_alone'] = ((df_t_copy['SibSp'] == 0) & (df_t_copy['Parch'] == 0)).map({True: '나홀로 탑승', False: '가족 동반'})
alone_comp = df_t_copy.groupby('is_alone').agg(
    승객수=('PassengerId', 'count'),
    생존율=('Survived', lambda s: round(s.mean() * 100, 2))
).reset_index()
print(alone_comp.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_015",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'은퇴자'(retired)이면서 '예금 잔액'이 0원 이하인 위험 고객들의 인원수와 평균 연령을 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"은퇴자": "job=='retired'", "예금 잔액 0원 이하": "balance <= 0"},
        "expected_output_type": "text",
        "python_code": """# L2_015: 무잔액/마이너스 은퇴자
cond = (df_bank['job'] == 'retired') & (df_bank['balance'] <= 0)
sub = df_bank[cond]
print(f"잔액 0원 이하 은퇴자 수: {len(sub)}명")
if len(sub) > 0:
    print(f"평균 연령: {sub['age'].mean():.1f}세")
"""
    })

    specs.append({
        "id": "L2_016",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'주택 자금 대출'과 '개인 신용 대출'을 모두 가지고 있는 다중 채무자들의 '최종 학력'별 인원수를 표로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"주택 자금 대출": "housing=='yes'", "개인 신용 대출": "loan=='yes'", "최종 학력": "education"},
        "expected_output_type": "table",
        "python_code": """# L2_016: 다중 채무자 학력 분포
multi_loans = df_bank[(df_bank['housing'] == 'yes') & (df_bank['loan'] == 'yes')]
edu_dist = multi_loans['education'].value_counts().reset_index()
edu_dist.columns = ['최종학력', '인원수']
print(f"다중 채무 고객 수: {len(multi_loans)}명")
print(edu_dist.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_017",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "봄 시즌인 3, 4, 5월('mar', 'apr', 'may')에 전화를 건 고객들 중 '예금 가입'에 성공한 고객 비율을 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"봄 시즌": "month.isin(['mar', 'apr', 'may'])", "예금 가입": "y=='yes'"},
        "expected_output_type": "text",
        "python_code": """# L2_017: 봄 시즌 가입 성공률
spring_cust = df_bank[df_bank['month'].isin(['mar', 'apr', 'may'])]
spring_rate = (spring_cust['y'] == 'yes').mean() * 100
print(f"봄 시즌 접촉 고객: {len(spring_cust)}명, 가입률: {spring_rate:.2f}%")
"""
    })

    specs.append({
        "id": "L2_018",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "이번 마케팅에서 5회 이상 전화를 건 '과다 접촉 고객'(campaign >= 5)의 예금 가입률을 1~2회 접촉 고객과 비교해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"과다 접촉 고객": "campaign >= 5"},
        "expected_output_type": "table",
        "python_code": """# L2_018: 과다 접촉 고객(5회+) vs 소수 접촉(1~2회) 가입률 비교
df_copy = df_bank.copy()
df_copy['contact_tier'] = np.where(df_copy['campaign'] >= 5, '5회 이상 (과다)', np.where(df_copy['campaign'] <= 2, '1~2회 (적정)', '3~4회'))
comp = df_copy.groupby('contact_tier').agg(
    고객수=('id', 'count'),
    가입률=('y', lambda s: round((s == 'yes').mean() * 100, 2))
).reset_index()
print(comp.to_string(index=False))
"""
    })

    specs.append({
        "id": "L2_019",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "타이타닉에서 '티켓 요금'이 50달러 이상이면서 '사망한 승객'들의 성별과 탑승 등급 분포를 확인해줘.",
        "target_table": "titanic",
        "synonym_mapping": {"티켓 요금": "Fare >= 50", "사망한 승객": "Survived==0"},
        "expected_output_type": "table",
        "python_code": """# L2_019: $50 이상 사망 승객 등급/성별 분포
cond = (df_titanic['Fare'] >= 50) & (df_titanic['Survived'] == 0)
sub = df_titanic[cond]
ct = pd.crosstab(sub['Pclass'], sub['Sex'])
print(f"50달러 이상 고액 사망 승객 수: {len(sub)}명")
print("등급 x 성별 사망자 수:")
print(ct)
"""
    })

    specs.append({
        "id": "L2_020",
        "category": "복합 조건 필터링 및 유사어 매핑",
        "type": "table",
        "difficulty": "Intermediate",
        "prompt": "'혼인 상태'가 '이혼'(divorced)이고 '신용 대출'이 있는 고객 중 '가입에 성공'(y='yes')한 사람의 인원수를 구해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"혼인 상태 이혼": "marital=='divorced'", "신용 대출": "loan=='yes'", "가입 성공": "y=='yes'"},
        "expected_output_type": "text",
        "python_code": """# L2_020: 이혼 + 신용대출 + 가입성공
cond = (df_bank['marital'] == 'divorced') & (df_bank['loan'] == 'yes')
sub = df_bank[cond]
succ = (sub['y'] == 'yes').sum()
print(f"이혼 & 개인대출 고객 수: {len(sub)}명, 가입 성공: {succ}명")
"""
    })

    # -------------------------------------------------------------
    # 📋 Multi-Dimensional Pivot Tables & Subtotals (L2_021 ~ L2_025)
    # -------------------------------------------------------------
    specs.append({
        "id": "L2_021",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업(job)과 혼인상태(marital)별 고객 수 피벗 테이블을 생성하고, 행과 열의 전체 총합계(margins=True)를 표기해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_021: 직업 x 혼인상태 피벗 (총합계 margin 포함)
pivot_jm_margin = df_bank.pivot_table(index='job', columns='marital', values='id', aggfunc='count', fill_value=0, margins=True, margins_name='전체총계')
print("직업 x 혼인상태 피벗 테이블 (행/열 총계 포함):")
print(pivot_jm_margin)
"""
    })

    specs.append({
        "id": "L2_022",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "교육수준(education)과 주택대출(housing)별 평균 잔액 피벗 테이블을 만들고, 전체 평균 총계(margins=True)를 포함해 소수점 둘째 자리까지 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_022: 교육수준 x 주택대출 평균 잔액 피벗 (margins=True)
pivot_eh_margin = df_bank.pivot_table(index='education', columns='housing', values='balance', aggfunc='mean', margins=True, margins_name='전체평균').round(2)
print("교육수준 x 주택대출 평균 잔액 (총계 포함):")
print(pivot_eh_margin)
"""
    })

    specs.append({
        "id": "L2_023",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "타이타닉 승객 등급(Pclass)과 성별(Sex)별 생존율 피벗 테이블을 margins=True 총 생존율과 함께 백분율(%)로 표기해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_023: Pclass x Sex 생존율 피벗 (margins=True)
pivot_surv_margin = (df_titanic.pivot_table(index='Pclass', columns='Sex', values='Survived', aggfunc='mean', margins=True, margins_name='전체생존율') * 100).round(2)
print("타이타닉 등급 x 성별 생존율(%) 피벗 (전체 평균 포함):")
print(pivot_surv_margin)
"""
    })

    specs.append({
        "id": "L2_024",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "직업(job)별로 주택 대출과 개인 대출의 교차 보유 고객 수 피벗 테이블(행: job, 열: [housing, loan])을 만들어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_024: 직업 x [housing, loan] 다중 컬럼 피벗
pivot_multi_loan = df_bank.pivot_table(index='job', columns=['housing', 'loan'], values='id', aggfunc='count', fill_value=0)
print("직업별 주택대출 x 신용대출 교차 고객수 피벗:")
print(pivot_multi_loan)
"""
    })

    specs.append({
        "id": "L2_025",
        "category": "다차원 피벗 및 소계 집계",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "월(month)과 통신 수단(contact)에 따른 마케팅 예금 가입 성공 건수(y='yes') 피벗 테이블을 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L2_025: 월 x 통신수단 가입 성공 건수 피벗
months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
y_yes = df_bank[df_bank['y'] == 'yes']
pivot_month_contact = y_yes.pivot_table(index='month', columns='contact', values='id', aggfunc='count', fill_value=0)
pivot_month_contact = pivot_month_contact.reindex([m for m in months_order if m in pivot_month_contact.index])
print("월 x 통신수단별 예금 가입 성공 건수:")
print(pivot_month_contact)
"""
    })

    return specs
