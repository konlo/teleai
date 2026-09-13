"""Generate benchmark datasets matching project TableContext schemas:
1. test_set/data/bank_loan.csv (2,000 rows, 18 columns)
2. test_set/data/titanic.csv (891 rows, 12 columns)
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

def generate_datasets(output_dir: Path = DATA_DIR, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. bank_loan.csv (2,000 rows, matching workspace.default.bank_loan)
    n_bank = 2000
    ids = list(range(n_bank))
    ages = np.clip(np.random.normal(41, 11, n_bank).astype(int), 18, 95)
    
    jobs = np.random.choice(
        ["management", "blue-collar", "technician", "admin.", "services", 
         "retired", "self-employed", "entrepreneur", "unemployed", "housemaid", "student", "unknown"],
        size=n_bank,
        p=[0.21, 0.20, 0.17, 0.12, 0.09, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.01]
    )
    
    maritals = np.random.choice(["married", "single", "divorced"], size=n_bank, p=[0.60, 0.28, 0.12])
    educations = np.random.choice(["secondary", "tertiary", "primary", "unknown"], size=n_bank, p=[0.51, 0.30, 0.15, 0.04])
    defaults = np.random.choice(["no", "yes"], size=n_bank, p=[0.98, 0.02])
    
    # Balances: realistic bank marketing distribution with some negative balances and high-wealth outliers
    raw_bal = np.random.exponential(scale=1400, size=n_bank) - 200
    neg_idx = np.random.rand(n_bank) < 0.08
    raw_bal[neg_idx] = -np.random.uniform(50, 3000, size=neg_idx.sum())
    outlier_idx = np.random.rand(n_bank) < 0.03
    raw_bal[outlier_idx] += np.random.uniform(15000, 85000, size=outlier_idx.sum())
    balances = np.clip(np.round(raw_bal).astype(int), -8019, 102127)
    
    housings = np.random.choice(["yes", "no"], size=n_bank, p=[0.55, 0.45])
    loans = np.random.choice(["no", "yes"], size=n_bank, p=[0.84, 0.16])
    contacts = np.random.choice(["cellular", "telephone", "unknown"], size=n_bank, p=[0.65, 0.06, 0.29])
    days = np.random.randint(1, 32, size=n_bank)
    months = np.random.choice(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        size=n_bank,
        p=[0.03, 0.06, 0.01, 0.06, 0.30, 0.12, 0.15, 0.14, 0.01, 0.02, 0.08, 0.02]
    )
    durations = np.clip(np.random.exponential(scale=260, size=n_bank).astype(int), 0, 4918)
    campaigns = np.clip(np.random.geometric(p=0.45, size=n_bank), 1, 63)
    
    # pdays: -1 for 80% (never contacted before)
    pdays = np.full(n_bank, -1)
    contacted_before = np.random.rand(n_bank) < 0.20
    pdays[contacted_before] = np.random.randint(1, 871, size=contacted_before.sum())
    
    previous = np.zeros(n_bank, dtype=int)
    previous[contacted_before] = np.random.geometric(p=0.35, size=contacted_before.sum())
    
    poutcome = np.full(n_bank, "unknown", dtype=object)
    poutcome[contacted_before] = np.random.choice(["failure", "other", "success"], size=contacted_before.sum(), p=[0.60, 0.25, 0.15])
    
    # Term deposit subscription (y): higher probability if duration > 500s or poutcome == 'success'
    base_prob = 0.08
    prob_y = np.where(durations > 500, base_prob + 0.35, base_prob)
    prob_y = np.where(poutcome == "success", prob_y + 0.40, prob_y)
    y = np.where(np.random.rand(n_bank) < np.clip(prob_y, 0.02, 0.90), "yes", "no")
    
    df_bank = pd.DataFrame({
        "id": ids,
        "age": ages,
        "job": jobs,
        "marital": maritals,
        "education": educations,
        "default": defaults,
        "balance": balances,
        "housing": housings,
        "loan": loans,
        "contact": contacts,
        "day": days,
        "month": months,
        "duration": durations,
        "campaign": campaigns,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "y": y
    })
    
    # 2. titanic.csv (891 rows, matching workspace.default.titanic)
    n_tit = 891
    p_ids = list(range(1, n_tit + 1))
    pclasses = np.random.choice([1, 2, 3], size=n_tit, p=[0.24, 0.21, 0.55])
    sexes = np.random.choice(["male", "female"], size=n_tit, p=[0.65, 0.35])
    
    # Age with realistic nulls
    t_ages = np.random.normal(30, 14, size=n_tit)
    t_ages = np.clip(np.round(t_ages, 1), 0.42, 80.0)
    age_null_mask = np.random.rand(n_tit) < 0.20
    t_ages[age_null_mask] = np.nan
    
    sibsp = np.random.choice([0, 1, 2, 3, 4, 5, 8], size=n_tit, p=[0.68, 0.23, 0.04, 0.02, 0.02, 0.005, 0.005])
    parch = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_tit, p=[0.76, 0.13, 0.09, 0.01, 0.005, 0.003, 0.002])
    
    fares = []
    for pc in pclasses:
        if pc == 1:
            f = np.random.exponential(scale=65) + 25
        elif pc == 2:
            f = np.random.exponential(scale=18) + 10
        else:
            f = np.random.exponential(scale=8) + 4
        fares.append(round(min(512.33, float(f)), 2))
    fares = np.array(fares)
    
    embarked = np.random.choice(["S", "C", "Q", None], size=n_tit, p=[0.72, 0.19, 0.08, 0.01])
    
    # Survived probability higher for females and 1st class
    surv_prob = np.where(sexes == "female", 0.74, 0.19)
    surv_prob = np.where(pclasses == 1, surv_prob + 0.20, surv_prob)
    surv_prob = np.where(pclasses == 3, surv_prob - 0.15, surv_prob)
    survived = (np.random.rand(n_tit) < np.clip(surv_prob, 0.05, 0.95)).astype(int)
    
    df_titanic = pd.DataFrame({
        "PassengerId": p_ids,
        "Survived": survived,
        "Pclass": pclasses,
        "Name": [f"Passenger_{i}" for i in p_ids],
        "Sex": sexes,
        "Age": t_ages,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": [f"TCK_{i:05d}" for i in p_ids],
        "Fare": fares,
        "Cabin": [f"C{i%50}" if np.random.rand() < 0.23 else np.nan for i in p_ids],
        "Embarked": embarked
    })
    
    df_bank.to_csv(output_dir / "bank_loan.csv", index=False)
    df_titanic.to_csv(output_dir / "titanic.csv", index=False)
    
    return df_bank, df_titanic

if __name__ == "__main__":
    df_b, df_t = generate_datasets()
    print(f"Generated bank_loan.csv: {df_b.shape} in {DATA_DIR}")
    print(f"Generated titanic.csv: {df_t.shape} in {DATA_DIR}")
