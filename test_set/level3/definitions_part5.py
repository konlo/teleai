# Level 3 Definitions (L3_001 – L3_100)
"""Each spec in Level 3 covers advanced visualizations, complex data transformations, multi‑table queries, and defensive edge cases.
The specs are placeholders; concrete implementations can be added later.
"""
import random

def get_level3_part5() -> list[dict]:
    specs = []
    # ---- Concrete advanced specs using medical_records and sales_transactions ----
    # 1. 의료 기록에서 진단 별 평균 비용
    specs.append({
        "id": "L3_001",
        "category": "복합 테이블 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "medical_records 테이블에서 각 diagnosis 별 평균 cost 를 계산해줘.",
        "target_table": "medical_records",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L3_001
import pandas as pd
# df_medical_records is provided by the test harness
result = df_medical_records.groupby('diagnosis')['cost'].mean().reset_index()
result.columns = ['diagnosis', 'avg_cost']
print(result.to_string(index=False))"""
    })
    # 2. 매출 트랜잭션에서 지역별 총 판매액 (figure)
    specs.append({
        "id": "L3_002",
        "category": "고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "sales_transactions 테이블에서 region 별 총 매출액을 바 차트로 보여줘.",
        "target_table": "sales_transactions",
        "synonym_mapping": None,
        "expected_output_type": "figure",
        "python_code": """# L3_002
import pandas as pd
import matplotlib.pyplot as plt
agg = df_sales_transactions.groupby('region').apply(lambda x: (x['price'] * x['qty']).sum()).reset_index(name='revenue')
fig, ax = plt.subplots(figsize=(6,4))
agg.plot.bar(x='region', y='revenue', ax=ax, legend=False)
ax.set_ylabel('Total Revenue')
ax.set_title('Revenue by Region')
plt.tight_layout()
plt.show()"""
    })
    # 3. 두 테이블을 날짜(date) 로 조인해 진단 별 매출 합계
    specs.append({
        "id": "L3_003",
        "category": "다중 도메인 동의어 매핑",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "medical_records 와 sales_transactions 를 date 로 inner join 한 뒤, diagnosis 별 총 매출액을 알려줘.",
        "target_table": "medical_records+sales_transactions",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L3_003
import pandas as pd
# Assume df_medical_records and df_sales_transactions are available
merged = pd.merge(df_medical_records, df_sales_transactions, on='date', how='inner')
merged['sales'] = merged['price'] * merged['qty']
result = merged.groupby('diagnosis')['sales'].sum().reset_index()
result.columns = ['diagnosis', 'total_sales']
print(result.to_string(index=False))"""
    })
    # 4. 판매 수량이 가장 높은 상위 5개 제품 (table)
    specs.append({
        "id": "L3_004",
        "category": "복합 테이블 분석",
        "type": "table",
        "difficulty": "Advanced",
        "prompt": "sales_transactions 테이블에서 판매 수량이 가장 많은 5개 제품과 그 수량을 보여줘.",
        "target_table": "sales_transactions",
        "synonym_mapping": None,
        "expected_output_type": "table",
        "python_code": """# L3_004
import pandas as pd
agg = df_sales_transactions.groupby('product')['qty'].sum().reset_index(name='total_qty')
top5 = agg.nlargest(5, 'total_qty')
print(top5.to_string(index=False))"""
    })
    # 5. 매출 추이를 월별 라인 차트 (figure)
    specs.append({
        "id": "L3_005",
        "category": "고급 시각화",
        "type": "chart",
        "difficulty": "Advanced",
        "prompt": "sales_transactions 테이블의 월별 매출 추이를 라인 차트로 그려줘.",
        "target_table": "sales_transactions",
        "synonym_mapping": None,
        "expected_output_type": "figure",
        "python_code": """# L3_005
import pandas as pd
import matplotlib.pyplot as plt
# Ensure date column is datetime
df_sales_transactions['date'] = pd.to_datetime(df_sales_transactions['date'])
df_sales_transactions['month'] = df_sales_transactions['date'].dt.to_period('M')
monthly = df_sales_transactions.groupby('month').apply(lambda x: (x['price']*x['qty']).sum()).reset_index(name='revenue')
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(monthly['month'].astype(str), monthly['revenue'], marker='o')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue')
ax.set_title('Monthly Revenue Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""
    })
    # ---- Generate remaining placeholder specs to reach 100 total ----
    import random
    for i in range(16, 101):
        spec_id = f"L3_{i:03d}"
        spec_type = random.choice(["chart", "table", "schema", "synonym"])
        category = {"chart": "고급 시각화", "table": "복합 테이블 분석", "schema": "다중 테이블 스키마 탐색", "synonym": "다중 도메인 동의어 매핑"}[spec_type]
        specs.append({
            "id": spec_id,
            "category": category,
            "type": spec_type,
            "difficulty": "Advanced",
            "prompt": f"[Placeholder] {spec_id} – implement {spec_type} test.",
            "target_table": "medical_records+sales_transactions",
            "synonym_mapping": None,
            "expected_output_type": "figure" if spec_type == "chart" else "text",
            "python_code": f"# {spec_id}\nprint('Executed {spec_id}')",
        })
    return specs
