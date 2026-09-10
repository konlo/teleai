"""Dataset generator for the 100 Data Analysis Agent Benchmark Suite.
Generates reproducible synthetic datasets:
1. customer_analytics.csv (1,000 rows, demographic & financial profile)
2. transaction_history.csv (5,000 rows, transaction records)
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent

def generate_datasets(output_dir: Path = BENCHMARK_DIR, seed: int = 42) -> tuple:
    np.random.seed(seed)
    n_customers = 1000
    
    # 1. Customer dataset (df)
    customer_ids = [f"CUST{i:04d}" for i in range(1, n_customers + 1)]
    ages = np.random.choice(range(19, 76), size=n_customers)
    genders = np.random.choice(["Male", "Female"], size=n_customers, p=[0.49, 0.51])
    
    jobs = np.random.choice(
        ["management", "technician", "blue-collar", "admin.", "services", 
         "retired", "self-employed", "entrepreneur", "unemployed", "student"],
        size=n_customers,
        p=[0.21, 0.18, 0.17, 0.13, 0.10, 0.06, 0.05, 0.04, 0.03, 0.03]
    )
    
    maritals = np.random.choice(["married", "single", "divorced"], size=n_customers, p=[0.60, 0.28, 0.12])
    educations = np.random.choice(["tertiary", "secondary", "primary", "unknown"], size=n_customers, p=[0.35, 0.48, 0.12, 0.05])
    
    job_income_base = {
        "management": 85000, "entrepreneur": 90000, "technician": 55000,
        "self-employed": 60000, "admin.": 48000, "services": 40000,
        "retired": 35000, "blue-collar": 42000, "unemployed": 18000, "student": 15000
    }
    annual_incomes = [
        max(12000, int(np.random.normal(job_income_base[j], 12000))) for j in jobs
    ]
    
    base_balances = np.random.exponential(scale=4500, size=n_customers) + np.random.normal(500, 1500, size=n_customers)
    neg_mask = np.random.rand(n_customers) < 0.05
    base_balances[neg_mask] = -np.random.uniform(50, 1800, size=neg_mask.sum())
    outlier_mask = np.random.rand(n_customers) < 0.02
    base_balances[outlier_mask] += np.random.uniform(25000, 60000, size=outlier_mask.sum())
    balances = np.round(base_balances, 2)
    
    credit_scores = np.clip(np.random.normal(680, 75, size=n_customers).astype(int), 350, 850)
    housing_loans = np.random.choice(["yes", "no"], size=n_customers, p=[0.52, 0.48])
    personal_loans = np.random.choice(["yes", "no"], size=n_customers, p=[0.16, 0.84])
    
    start_ts = int(pd.Timestamp("2022-01-01").timestamp())
    end_ts = int(pd.Timestamp("2024-06-30").timestamp())
    random_ts = np.random.randint(start_ts, end_ts, size=n_customers)
    signup_dates = pd.to_datetime(random_ts, unit="s").strftime("%Y-%m-%d")
    
    active_start_ts = int(pd.Timestamp("2024-07-01").timestamp())
    active_end_ts = int(pd.Timestamp("2024-12-31").timestamp())
    last_active_dates = pd.to_datetime(np.random.randint(active_start_ts, active_end_ts, size=n_customers), unit="s").strftime("%Y-%m-%d")
    
    device_types = np.random.choice(["iOS", "Android", "Web"], size=n_customers, p=[0.42, 0.45, 0.13])
    service_tiers = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], size=n_customers, p=[0.40, 0.35, 0.18, 0.07])
    
    churn_prob = np.where(credit_scores < 580, 0.35, 0.12)
    churn = (np.random.rand(n_customers) < churn_prob).astype(int)
    
    campaign_contacts = np.random.poisson(lam=2.5, size=n_customers) + 1
    
    satisfaction = np.random.choice([1, 2, 3, 4, 5], size=n_customers, p=[0.08, 0.14, 0.32, 0.30, 0.16]).astype(float)
    nan_indices = np.random.choice(range(n_customers), size=52, replace=False)
    satisfaction[nan_indices] = np.nan
    
    df_customers = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "gender": genders,
        "job": jobs,
        "marital": maritals,
        "education": educations,
        "credit_score": credit_scores,
        "annual_income": annual_incomes,
        "balance": balances,
        "housing_loan": housing_loans,
        "personal_loan": personal_loans,
        "signup_date": signup_dates,
        "last_active_date": last_active_dates,
        "device_type": device_types,
        "service_tier": service_tiers,
        "churn": churn,
        "campaign_contact_count": campaign_contacts,
        "satisfaction_score": satisfaction
    })
    
    # 2. Transaction dataset (df_tx, 5,000 rows)
    n_tx = 5000
    tx_ids = [f"TX{i:05d}" for i in range(1, n_tx + 1)]
    active_cust_pool = customer_ids[:950]
    tx_cust_ids = np.random.choice(active_cust_pool, size=n_tx)
    
    tx_start_ts = int(pd.Timestamp("2024-01-01 00:00:00").timestamp())
    tx_end_ts = int(pd.Timestamp("2024-12-31 23:59:59").timestamp())
    tx_random_ts = np.random.randint(tx_start_ts, tx_end_ts, size=n_tx)
    tx_datetimes = pd.to_datetime(tx_random_ts, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    
    tx_types = np.random.choice(["Payment", "Transfer", "Withdrawal", "Deposit"], size=n_tx, p=[0.40, 0.28, 0.20, 0.12])
    
    tx_amounts = []
    for t in tx_types:
        if t == "Payment":
            amt = np.random.gamma(shape=3.0, scale=35.0) + 5.0
        elif t == "Transfer":
            amt = np.random.exponential(scale=350.0) + 50.0
        elif t == "Withdrawal":
            amt = float(np.random.choice([20, 50, 100, 200, 300, 500], p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.1]))
        else:
            amt = np.random.exponential(scale=600.0) + 100.0
        tx_amounts.append(round(float(amt), 2))
    tx_amounts = np.array(tx_amounts)
    
    channels = np.random.choice(["Mobile App", "Online Banking", "ATM", "Branch"], size=n_tx, p=[0.55, 0.25, 0.14, 0.06])
    declined = np.random.choice([0, 1], size=n_tx, p=[0.97, 0.03])
    fees = np.where(tx_types == "Deposit", 0.0, np.round(tx_amounts * 0.015, 2))
    
    df_transactions = pd.DataFrame({
        "tx_id": tx_ids,
        "customer_id": tx_cust_ids,
        "tx_datetime": tx_datetimes,
        "tx_type": tx_types,
        "amount": tx_amounts,
        "channel": channels,
        "is_declined": declined,
        "fee": fees
    })
    
    output_dir.mkdir(parents=True, exist_ok=True)
    df_customers.to_csv(output_dir / "customer_analytics.csv", index=False)
    df_transactions.to_csv(output_dir / "transaction_history.csv", index=False)
    
    return df_customers, df_transactions

if __name__ == "__main__":
    df_c, df_t = generate_datasets()
    print(f"Generated customer_analytics.csv: {df_c.shape}")
    print(f"Generated transaction_history.csv: {df_t.shape}")
