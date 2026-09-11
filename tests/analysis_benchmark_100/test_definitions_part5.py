"""100 Test Definitions for Data Analysis Agent Benchmark Suite - Part 5 (Q081 ~ Q100).
"""

def get_part5_specs():
    specs = []

    # -------------------------------------------------------------
    # Category 9: 다중 테이블 병합 및 정합성 비교 (Multi-DataFrame Join & Mismatch)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_081",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Basic",
        "question": "고객 테이블(df)과 거래 테이블(df_tx)을 customer_id로 내부 조인(Inner Join)했을 때의 총 행 개수와 결합된 컬럼 수를 확인해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "text",
        "python_code": """# TEST_081: 고객-거래 테이블 내부 조인(Inner Join)
inner_df = df.merge(df_tx, on='customer_id', how='inner')
print(f"조인 결과 행 개수: {inner_df.shape[0]:,}행")
print(f"조인 결과 컬럼 수: {inner_df.shape[1]}개")
print(f"결합된 컬럼 목록: {list(inner_df.columns)}")
"""
    })

    specs.append({
        "id": "TEST_082",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "고객 테이블 기준으로 좌측 조인(Left Join)하여 거래 내역이 단 한 건도 없는 고객(Anti-Join)의 수와 명단을 조회해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_082: 거래 내역 없는 고객 안티 조인(Anti-Join)
left_df = df.merge(df_tx, on='customer_id', how='left')
no_tx_cust = left_df[left_df['tx_id'].isna()]
print(f"거래 내역이 없는 고객 수: {len(no_tx_cust)}명")
print(no_tx_cust[['customer_id', 'age', 'job', 'service_tier', 'balance']].head(10))
"""
    })

    specs.append({
        "id": "TEST_083",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "고객 정보와 거래 내역을 조인하여 직업(job)별 총 거래금액과 총 거래건수를 집계하고 거래금액 순으로 정렬해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_083: 직업별 거래금액 및 거래건수 조인 집계
merged = df.merge(df_tx, on='customer_id', how='inner')
job_tx = merged.groupby('job').agg(총거래액=('amount', 'sum'), 거래건수=('tx_id', 'count')).sort_values('총거래액', ascending=False)
print(job_tx.round(2))
"""
    })

    specs.append({
        "id": "TEST_084",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "거래 내역이 있는 고객과 없는 고객의 평균 연령 및 평균 신용점수를 비교해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_084: 거래 유무에 따른 고객 특성 비교
tx_cust_set = set(df_tx['customer_id'].unique())
df_copy = df.copy()
df_copy['has_transaction'] = df_copy['customer_id'].apply(lambda cid: '거래 있음' if cid in tx_cust_set else '거래 없음')
comp = df_copy.groupby('has_transaction').agg(고객수=('customer_id', 'count'), 평균연령=('age', 'mean'), 평균신용점수=('credit_score', 'mean'), 평균잔액=('balance', 'mean')).round(2)
print(comp)
"""
    })

    specs.append({
        "id": "TEST_085",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Advanced",
        "question": "각 고객이 가장 많이 이용한 주 사용 거래 채널(Primary Channel)을 구하여 고객 테이블에 병합해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_085: 주 이용 거래 채널 도출 및 병합
cust_channel = df_tx.groupby(['customer_id', 'channel']).size().reset_index(name='count')
primary_channel = cust_channel.sort_values(['customer_id', 'count'], ascending=[True, False]).drop_duplicates(subset=['customer_id'])
merged_df = df.merge(primary_channel[['customer_id', 'channel']], on='customer_id', how='left')
merged_df['channel'] = merged_df['channel'].fillna('No Transactions')
print("주 이용 채널 분포:")
print(merged_df['channel'].value_counts())
print(merged_df[['customer_id', 'job', 'channel']].head(5))
"""
    })

    specs.append({
        "id": "TEST_086",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "2024년 하반기(7월~12월) 동안 단일 거래금액이 3,000 이상인 고액 거래를 발생시킨 고객들의 ID와 직업을 중복 없이 추출해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_086: 하반기 고액 거래 발생 고객 추출
df_tx_copy = df_tx.copy()
df_tx_copy['month'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.month
high_tx = df_tx_copy[(df_tx_copy['month'] >= 7) & (df_tx_copy['amount'] >= 3000)]
high_tx_cust = high_tx[['customer_id', 'amount', 'tx_datetime']].merge(df[['customer_id', 'job', 'annual_income']], on='customer_id')
unique_cust = high_tx_cust.drop_duplicates(subset=['customer_id'])
print(f"하반기 $3,000 이상 고액 거래 고객 수: {len(unique_cust)}명")
print(unique_cust[['customer_id', 'job', 'annual_income']].head(10))
"""
    })

    specs.append({
        "id": "TEST_087",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Basic",
        "question": "고객 테이블의 customer_id 고유값 개수와 거래 테이블의 customer_id 고유값 개수 간 차이를 정합성 검증으로 확인해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "text",
        "python_code": """# TEST_087: 두 테이블 간 고객 ID 정합성 비교
cust_set = set(df['customer_id'].unique())
tx_cust_set = set(df_tx['customer_id'].unique())
diff_in_cust_only = cust_set - tx_cust_set
diff_in_tx_only = tx_cust_set - cust_set
print(f"고객 테이블 고유 ID 수: {len(cust_set)}")
print(f"거래 테이블 고유 ID 수: {len(tx_cust_set)}")
print(f"거래 없는 고객 수: {len(diff_in_cust_only)}")
print(f"고객 테이블에 없는 유령 거래 ID 수 (외래키 무결성 위반): {len(diff_in_tx_only)}")
"""
    })

    specs.append({
        "id": "TEST_088",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "거래 테이블에서 동일한 고객이 동일한 날짜에 발생시킨 거래 건수가 5건 이상인 다빈도 거래 일자를 탐지해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_088: 동일 일자 다빈도 거래 고객 탐지
df_tx_copy = df_tx.copy()
df_tx_copy['date'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.date
frequent_tx = df_tx_copy.groupby(['customer_id', 'date']).size().reset_index(name='tx_count')
heavy_days = frequent_tx[frequent_tx['tx_count'] >= 5].sort_values('tx_count', ascending=False)
print(f"동일 일자 5건 이상 거래 발생 건수: {len(heavy_days)}건")
if len(heavy_days) > 0:
    print(heavy_days.head(5))
"""
    })

    specs.append({
        "id": "TEST_089",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "서비스 티어(service_tier)별 총 거래금액을 조인 집계하여 수평 막대 차트로 시각화해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_089: 티어별 거래 총액 수평 막대 그래프
import matplotlib.pyplot as plt

m = df.merge(df_tx, on='customer_id', how='inner')
tier_tx = m.groupby('service_tier')['amount'].sum().sort_values(ascending=True)

plt.figure(figsize=(8, 4))
tier_tx.plot(kind='barh', color='darkorange')
plt.title('서비스 티어별 누적 총 거래금액')
plt.xlabel('총 거래금액 ($)')
plt.ylabel('서비스 티어')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print(tier_tx)
"""
    })

    specs.append({
        "id": "TEST_090",
        "category": "다중 테이블 병합 및 정합성 비교",
        "difficulty": "Intermediate",
        "question": "이탈 고객(churn=1)과 유지 고객(churn=0)의 1인당 평균 총 거래금액을 조인 집계하여 비교해줘.",
        "target_dfs": ["df", "df_tx"],
        "expected_output_type": "table",
        "python_code": """# TEST_090: 이탈 여부별 1인당 총 거래금액 비교
cust_amt = df_tx.groupby('customer_id')['amount'].sum().reset_index()
m = df.merge(cust_amt, on='customer_id', how='left')
m['amount'] = m['amount'].fillna(0)
churn_tx_comp = m.groupby('churn').agg(고객수=('customer_id', 'count'), 평균총거래액=('amount', 'mean'), 중앙값총거래액=('amount', 'median')).round(2)
churn_tx_comp.index = ['유지 고객 (0)', '이탈 고객 (1)']
print(churn_tx_comp)
"""
    })

    # -------------------------------------------------------------
    # Category 10: 고급 화면 시각화 및 방어적 처리 (Visualization & Defensive Edge Cases)
    # -------------------------------------------------------------
    specs.append({
        "id": "TEST_091",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Basic",
        "question": "[집계 데이터 시각화 규칙] 직업별 평균 잔액 집계표를 히스토그램이 아닌 막대 차트(Bar Chart)로 올바르게 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_091: 집계 데이터에 대한 막대 차트 시각화 규칙 준수
import matplotlib.pyplot as plt

# 사전 집계 데이터 생성
job_avg_bal = df.groupby('job')['balance'].mean().sort_values(ascending=False)

# 집계 데이터는 관측치 분포(raw)가 아니므로 히스토그램이 아닌 막대 차트 사용
plt.figure(figsize=(9, 4.5))
job_avg_bal.plot(kind='bar', color='royalblue')
plt.title('직업별 평균 잔액 (집계 데이터 막대 차트)')
plt.xlabel('직업')
plt.ylabel('평균 잔액 ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print(job_avg_bal.round(2))
"""
    })

    specs.append({
        "id": "TEST_092",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Advanced",
        "question": "[다중 서브플롯] 2x2 서브플롯 그리드를 생성하여 연령 분포, 잔액 분포, 신용점수 분포, 서비스 티어 빈도를 한 화면에 시각화해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_092: 2x2 서브플롯 대시보드 시각화
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 1. 연령 분포
sns.histplot(df['age'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('1. 연령 분포')

# 2. 잔액 분포 (박스플롯)
sns.boxplot(x=df['balance'], ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('2. 잔액 박스플롯')

# 3. 신용점수 분포
sns.histplot(df['credit_score'], kde=True, ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('3. 신용점수 분포')

# 4. 서비스 티어 빈도 막대그래프
df['service_tier'].value_counts().plot(kind='bar', ax=axes[1, 1], color='gold')
axes[1, 1].set_title('4. 서비스 티어 고객 수')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.suptitle('고객 데이터 4대 핵심 변수 종합 시각화 대시보드', fontsize=14)
plt.tight_layout()
plt.show()
"""
    })

    specs.append({
        "id": "TEST_093",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Advanced",
        "question": "[이중 축 시각화] 월별 거래 총액(막대)과 거래 건수(꺾은선)를 이중 축(Dual Y-axis)으로 결합하여 시각화해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_093: 월별 거래총액(막대) & 거래건수(선) 이중 축 시각화
import matplotlib.pyplot as plt

df_tx_copy = df_tx.copy()
df_tx_copy['month'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.month
monthly_stats = df_tx_copy.groupby('month').agg(총금액=('amount', 'sum'), 건수=('tx_id', 'count'))

fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax2 = ax1.twinx()

ax1.bar(monthly_stats.index, monthly_stats['총금액'], color='lightsteelblue', alpha=0.7, label='총 거래금액 ($)')
ax2.plot(monthly_stats.index, monthly_stats['건수'], color='crimson', marker='o', linewidth=2, label='거래 건수')

ax1.set_xlabel('월 (Month)')
ax1.set_ylabel('총 거래금액 ($)', color='navy')
ax2.set_ylabel('거래 건수', color='crimson')
ax1.set_xticks(range(1, 13))
plt.title('2024년 월별 거래총액 및 거래건수 이중 축 추이')
plt.tight_layout()
plt.show()
print(monthly_stats)
"""
    })

    specs.append({
        "id": "TEST_094",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Intermediate",
        "question": "[방어적 필터링] 연령이 100세 이상인 고객 필터링 시 결과가 빈 데이터프레임(Empty)인 경우 예외 없이 안전하게 안내 메시지를 출력해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_094: 빈 데이터프레임 방어적 처리
empty_sub = df[df['age'] >= 100]
if empty_sub.empty:
    print("안내: 조건(연령 100세 이상)을 만족하는 데이터가 존재하지 않습니다. (Empty DataFrame 감지)")
else:
    print(f"조회된 고객 수: {len(empty_sub)}명")
"""
    })

    specs.append({
        "id": "TEST_095",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Intermediate",
        "question": "[방어적 결측치 처리] 100% 결측치로만 구성된 가상 컬럼을 추가한 뒤 평균을 계산할 때 오류 없이 안전하게 처리하는 코드를 작성해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_095: 완전 결측 컬럼 안전 연산 방어
df_test = df.copy()
df_test['all_null_col'] = float('nan')

val_mean = df_test['all_null_col'].mean()
if pd.isna(val_mean):
    print("안내: 'all_null_col' 컬럼의 모든 값이 결측치(NaN)이므로 평균값이 산출되지 않습니다. 기본값 0.0으로 대체합니다.")
    val_mean = 0.0
print(f"최종 안전 처리된 평균값: {val_mean}")
"""
    })

    specs.append({
        "id": "TEST_096",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Intermediate",
        "question": "[방어적 컬럼 참조] 존재하지 않는 컬럼 'non_existent_col'을 참조하려 할 때 KeyError를 방지하고 사용 가능한 컬럼 목록을 안내해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_096: 미존재 컬럼 KeyError 방어 로직
target_col = 'non_existent_col'
if target_col in df.columns:
    print(df[target_col].head())
else:
    print(f"경고: 요청하신 컬럼 '{target_col}'은 데이터셋에 존재하지 않습니다.")
    print(f"사용 가능한 유사 수치형 컬럼 목록: {list(df.select_dtypes(include=['number']).columns[:5])}")
"""
    })

    specs.append({
        "id": "TEST_097",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Intermediate",
        "question": "[0으로 나누기 방어] 건수가 0인 집계 그룹의 비율 계산 시 ZeroDivisionError를 방지하는 안전한 비율 계산 코드를 작성해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "text",
        "python_code": """# TEST_097: ZeroDivision 방어적 비율 계산
total_count = len(df[df['age'] > 120])  # 0 rows
target_count = len(df[(df['age'] > 120) & (df['churn'] == 1)])

# 0으로 나누기 방어 로직
ratio_pct = (target_count / total_count * 100) if total_count > 0 else 0.0
print(f"분모 건수: {total_count}, 분자 건수: {target_count}")
print(f"안전하게 계산된 비율: {ratio_pct:.2f}% (ZeroDivisionError 방어 성공)")
"""
    })

    specs.append({
        "id": "TEST_098",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Intermediate",
        "question": "[시계열 순서 보장] 임의로 날짜 순서가 섞인 월별 데이터에 대해 시각화 전 반드시 날짜순 오름차순 정렬을 수행하여 꺾은선 꼬임을 방지해줘.",
        "target_dfs": ["df_tx"],
        "expected_output_type": "chart",
        "python_code": """# TEST_098: 시계열 시각화 전 오름차순 정렬 보장
import matplotlib.pyplot as plt

df_tx_copy = df_tx.copy()
df_tx_copy['month'] = pd.to_datetime(df_tx_copy['tx_datetime']).dt.month
monthly_sum = df_tx_copy.groupby('month')['amount'].sum()

# 의도적으로 인덱스를 무작위 셔플링
shuffled_data = monthly_sum.sample(frac=1.0, random_state=123)

# 꺾은선 그래프 꼬임 방지를 위한 엄격한 오름차순 정렬
sorted_data = shuffled_data.sort_index(ascending=True)

plt.figure(figsize=(8, 4))
plt.plot(sorted_data.index, sorted_data.values, marker='s', color='darkviolet', linewidth=2)
plt.title('월별 거래금액 추이 (정렬 보장으로 꺾은선 꼬임 방지)')
plt.xlabel('월 (Month)')
plt.ylabel('총 거래금액 ($)')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print("정렬된 월별 순서:", list(sorted_data.index))
"""
    })

    specs.append({
        "id": "TEST_099",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Basic",
        "question": "[시각화 품질 및 레이아웃] 그래프 생성 시 제목, 축 라벨, 그리드를 지정하고 tight_layout()을 적용하여 텍스트 겹침을 방지해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "chart",
        "python_code": """# TEST_099: 시각화 레이아웃 품질 및 가이드라인 준수
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(df['credit_score'], bins=20, color='teal', edgecolor='black', alpha=0.7)
plt.title('신용점수(Credit Score) 분포 - 표준 레이아웃 적용')
plt.xlabel('신용점수 (점)')
plt.ylabel('고객 인원수 (명)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
print(f"신용점수 평균: {df['credit_score'].mean():.1f}, 중앙값: {df['credit_score'].median():.1f}")
"""
    })

    specs.append({
        "id": "TEST_100",
        "category": "고급 화면 시각화 및 방어적 처리",
        "difficulty": "Advanced",
        "question": "[종합 대시보드 화면 표출] 총 고객수, 평균 잔액, 평균 신용점수, 전체 이탈율의 4대 핵심 지표 요약표와 잔액 분포 박스플롯을 한 번에 화면에 생성해줘.",
        "target_dfs": ["df"],
        "expected_output_type": "combined",
        "python_code": """# TEST_100: 4대 핵심 KPI 요약표 및 분포 차트 종합 생성
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 4대 KPI 요약 데이터프레임
kpi_df = pd.DataFrame({
    '지표명': ['총 고객 수', '평균 계좌 잔액', '평균 신용점수', '고객 이탈율'],
    '지표값': [
        f"{len(df):,}명",
        f"${df['balance'].mean():,.2f}",
        f"{df['credit_score'].mean():.1f}점",
        f"{df['churn'].mean()*100:.2f}%"
    ]
})
print("=== 4대 핵심 비즈니스 KPI 요약 ===")
print(kpi_df.to_string(index=False))

# 2. 종합 분포 시각화
plt.figure(figsize=(9, 3.5))
sns.boxplot(x=df['balance'], color='mediumslateblue')
plt.title('고객 계좌 잔액 분포 (종합 대시보드)')
plt.xlabel('잔액 ($)')
plt.tight_layout()
plt.show()
"""
    })

    return specs
