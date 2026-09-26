"""Level 1 Definitions Part 1 (L1_001 ~ L1_025):
- 🔍 Table Schema & Metadata Questions (15 items: L1_001 ~ L1_015)
- 🧠 Synonym/Alias-based Basic Column Lookups (10 items: L1_016 ~ L1_025)
"""

def get_level1_part1():
    specs = []
    
    # -------------------------------------------------------------
    # 🔍 Table Schema & Metadata Questions (L1_001 ~ L1_015)
    # -------------------------------------------------------------
    specs.append({
        "id": "L1_001",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에 어떤 컬럼들이 있는지 전체 목록과 총 컬럼 개수를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_001: 컬럼 목록 및 개수 확인
cols = list(df_bank.columns)
print(f"bank_loan 테이블 총 컬럼 수: {len(cols)}개")
print("컬럼 목록:")
for i, col in enumerate(cols, 1):
    print(f"{i}. {col}")
"""
    })

    specs.append({
        "id": "L1_002",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블의 각 컬럼별 데이터 타입(dtype)을 정리해서 표로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_002: 컬럼별 데이터 타입 확인
dtype_df = pd.DataFrame({"Column": df_bank.columns, "Dtype": df_bank.dtypes.astype(str).values})
print("bank_loan 컬럼별 데이터 타입:")
print(dtype_df.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_003",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에서 수치형(numeric) 컬럼들만 골라서 목록을 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_003: 수치형 컬럼 추출
num_cols = df_bank.select_dtypes(include=['number']).columns.tolist()
print(f"수치형 컬럼 목록 ({len(num_cols)}개): {num_cols}")
"""
    })

    specs.append({
        "id": "L1_004",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에서 문자열/범주형(categorical) 컬럼들만 골라서 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_004: 범주형 컬럼 추출
cat_cols = df_bank.select_dtypes(include=['object']).columns.tolist()
print(f"범주형 컬럼 목록 ({len(cat_cols)}개): {cat_cols}")
"""
    })

    specs.append({
        "id": "L1_005",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블의 전체 행 수(총 레코드 수)를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_005: 총 행 수 확인
print(f"bank_loan 테이블 총 데이터 행 수: {len(df_bank):,}행")
"""
    })

    specs.append({
        "id": "L1_006",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에 결측치(Null/NaN)가 존재하는 컬럼이 있는지 확인하고 결측치 개수를 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_006: 결측치 유무 및 개수 점검
null_counts = df_bank.isnull().sum()
has_null = null_counts[null_counts > 0]
if len(has_null) == 0:
    print("bank_loan 테이블의 모든 컬럼에 결측치(Null)가 없습니다. (결측치 0건)")
else:
    print("결측치가 있는 컬럼:")
    print(has_null)
"""
    })

    specs.append({
        "id": "L1_007",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "titanic 테이블의 컬럼 이름과 각각의 데이터 타입을 확인해줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_007: titanic 테이블 컬럼 스키마
t_info = pd.DataFrame({"Column": df_titanic.columns, "Dtype": df_titanic.dtypes.astype(str).values})
print("titanic 테이블 컬럼 및 데이터 타입:")
print(t_info.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_008",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "titanic 테이블에서 결측치(Null)가 존재하는 컬럼과 그 결측 개수를 보여줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_008: titanic 결측치 점검
t_nulls = df_titanic.isnull().sum()
t_nulls = t_nulls[t_nulls > 0].reset_index()
t_nulls.columns = ["Column", "Null_Count"]
print("titanic 테이블 결측치 현황:")
print(t_nulls.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_009",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블의 범주형 컬럼들이 각각 몇 개의 고유값(distinct count)을 가지는지 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_009: 범주형 컬럼 고유값 개수
cat_cols = df_bank.select_dtypes(include=['object']).columns
nunique_df = pd.DataFrame({"Column": cat_cols, "Distinct_Count": [df_bank[c].nunique() for c in cat_cols]})
print(nunique_df.sort_values("Distinct_Count", ascending=False).to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_010",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에서 고객의 직업을 나타내는 job 컬럼에 어떤 값들이 있는지 고유값 목록을 출력해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_010: job 컬럼 고유값 목록
jobs = sorted(df_bank['job'].unique().tolist())
print(f"job 고유값 종류 ({len(jobs)}개): {jobs}")
"""
    })

    specs.append({
        "id": "L1_011",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에서 교육 수준을 나타내는 education 컬럼의 고유값들을 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_011: education 컬럼 고유값 확인
edu_vals = df_bank['education'].unique().tolist()
print(f"education 고유값 종류: {edu_vals}")
"""
    })

    specs.append({
        "id": "L1_012",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에서 대출과 관련된 컬럼들이 어떤 것들이 있는지 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_012: 대출 관련 컬럼 식별
loan_related = [col for col in df_bank.columns if any(k in col.lower() for k in ['loan', 'housing', 'default', 'balance'])]
print(f"대출 및 금융 관련 컬럼: {loan_related}")
for col in loan_related:
    print(f" - {col}: {df_bank[col].dtype}, 고유값: {list(df_bank[col].unique()[:5])}")
"""
    })

    specs.append({
        "id": "L1_013",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블에서 마케팅 캠페인 결과를 나타내는 y 컬럼의 값 종류와 의미를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_013: 타깃 y 컬럼 고유값 및 빈도
y_counts = df_bank['y'].value_counts().reset_index()
y_counts.columns = ['정기예금_가입여부(y)', '고객수']
print("타깃 컬럼 y 고유값 및 고객수:")
print(y_counts.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_014",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "titanic 테이블의 탑승 등급(Pclass) 컬럼의 고유값과 탑승지(Embarked) 컬럼의 고유값을 알려줘.",
        "target_table": "titanic",
        "synonym_mapping": None,
        "expected_output_type": "text",
        "python_code": """# L1_014: Pclass 및 Embarked 고유값
pclasses = sorted(df_titanic['Pclass'].dropna().unique().tolist())
embarked = sorted([str(x) for x in df_titanic['Embarked'].dropna().unique().tolist()])
print(f"Pclass 고유값: {pclasses}")
print(f"Embarked 고유값: {embarked}")
"""
    })

    specs.append({
        "id": "L1_015",
        "category": "테이블 스키마 탐색",
        "type": "schema",
        "difficulty": "Basic",
        "prompt": "bank_loan 테이블의 상위 3개 행을 미리보기(preview)로 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L1_015: 데이터 상위 미리보기
print("bank_loan 데이터 상위 3행:")
print(df_bank[['id', 'age', 'job', 'marital', 'balance', 'housing', 'loan', 'y']].head(3))
"""
    })

    # -------------------------------------------------------------
    # 🧠 Synonym/Alias-based Basic Column Lookups (L1_016 ~ L1_025)
    # -------------------------------------------------------------
    specs.append({
        "id": "L1_016",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들의 평균 '예금 잔고'가 얼마인지 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"예금 잔고": "balance"},
        "expected_output_type": "text",
        "python_code": """# L1_016: '예금 잔고' -> balance 매핑
avg_balance = df_bank['balance'].mean()
print(f"고객들의 평균 예금 잔고(balance): {avg_balance:,.2f} 달러")
"""
    })

    specs.append({
        "id": "L1_017",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들의 평균 '나이'와 최고령 고객의 '나이'를 알려줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"나이": "age"},
        "expected_output_type": "text",
        "python_code": """# L1_017: '나이' -> age 매핑
avg_age = df_bank['age'].mean()
max_age = df_bank['age'].max()
print(f"고객 평균 나이: {avg_age:.1f}세, 최고령 나이: {max_age}세")
"""
    })

    specs.append({
        "id": "L1_018",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들 중 '집 대출'을 가지고 있는 사람이 몇 명인지 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"집 대출": "housing"},
        "expected_output_type": "text",
        "python_code": """# L1_018: '집 대출' -> housing='yes' 매핑
housing_count = (df_bank['housing'] == 'yes').sum()
print(f"집 대출(housing loan) 보유 고객 수: {housing_count:,}명 (전체의 {housing_count/len(df_bank)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L1_019",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객 중 '신용 대출'을 받은 사람의 인원수를 확인해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"신용 대출": "loan"},
        "expected_output_type": "text",
        "python_code": """# L1_019: '신용 대출' -> loan='yes' 매핑
loan_count = (df_bank['loan'] == 'yes').sum()
print(f"신용 대출(personal loan) 보유 고객 수: {loan_count:,}명 (전체의 {loan_count/len(df_bank)*100:.1f}%)")
"""
    })

    specs.append({
        "id": "L1_020",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "마케팅 전화 '상담 시간'의 평균 초(sec)를 계산해줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"상담 시간": "duration"},
        "expected_output_type": "text",
        "python_code": """# L1_020: '상담 시간' -> duration 매핑
avg_dur = df_bank['duration'].mean()
print(f"평균 상담 시간(duration): {avg_dur:.1f}초 ({avg_dur/60:.2f}분)")
"""
    })

    specs.append({
        "id": "L1_021",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들의 '혼인 상태' 종류별로 몇 명씩 있는지 세어줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"혼인 상태": "marital"},
        "expected_output_type": "table",
        "python_code": """# L1_021: '혼인 상태' -> marital 매핑
marital_counts = df_bank['marital'].value_counts().reset_index()
marital_counts.columns = ['혼인 상태(marital)', '인원수']
print(marital_counts.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_022",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "고객들의 '최종 학력'별 인원수를 보여줘.",
        "target_table": "bank_loan",
        "synonym_mapping": {"최종 학력": "education"},
        "expected_output_type": "table",
        "python_code": """# L1_022: '최종 학력' -> education 매핑
edu_counts = df_bank['education'].value_counts().reset_index()
edu_counts.columns = ['최종 학력(education)', '인원수']
print(edu_counts.to_string(index=False))
"""
    })

    specs.append({
        "id": "L1_023",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "마케팅을 통해 실제로 예금에 '가입한 사람'이 총 몇 명이야?",
        "target_table": "bank_loan",
        "synonym_mapping": {"가입한 사람": "y=='yes'"},
        "expected_output_type": "text",
        "python_code": """# L1_023: '가입한 사람' -> y='yes' 매핑
subscribed = (df_bank['y'] == 'yes').sum()
print(f"정기예금 가입 성공 고객 수(y=yes): {subscribed:,}명 (전체의 {subscribed/len(df_bank)*100:.2f}%)")
"""
    })

    specs.append({
        "id": "L1_024",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "타이타닉에서 '살아남은 사람'이 전체 중 몇 명이야?",
        "target_table": "titanic",
        "synonym_mapping": {"살아남은 사람": "Survived==1"},
        "expected_output_type": "text",
        "python_code": """# L1_024: '살아남은 사람' -> Survived=1 매핑
survived_count = (df_titanic['Survived'] == 1).sum()
print(f"타이타닉 생존자 수(Survived=1): {survived_count:,}명 (생존율: {survived_count/len(df_titanic)*100:.2f}%)")
"""
    })

    specs.append({
        "id": "L1_025",
        "category": "유사어 기반 컬럼 탐색",
        "type": "synonym",
        "difficulty": "Basic",
        "prompt": "타이타닉 승객들이 낸 평균 '티켓 요금'이 얼마인지 알려줘.",
        "target_table": "titanic",
        "synonym_mapping": {"티켓 요금": "Fare"},
        "expected_output_type": "text",
        "python_code": """# L1_025: '티켓 요금' -> Fare 매핑
avg_fare = df_titanic['Fare'].mean()
print(f"평균 티켓 요금(Fare): ${avg_fare:.2f}")
"""
    })

    return specs
