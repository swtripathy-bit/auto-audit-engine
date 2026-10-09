"""
Synthetic Dataset Generator for Auto-Audit-Engine.
Generates realistic messy and collinear benchmark datasets for verification.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_messy_churn(output_path: Path, n_rows: int = 250, seed: int = 42):
    np.random.seed(seed)
    
    # 1. Base realistic features
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_rows)]
    ages = np.random.normal(loc=41, scale=12, size=n_rows).round()
    tenure_months = np.random.randint(1, 72, size=n_rows)
    monthly_charges = np.random.uniform(20.0, 115.0, size=n_rows).round(2)
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
    payment_methods = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_rows)
    signup_dates = pd.date_range(start="2020-01-01", periods=n_rows, freq="3D").strftime("%Y-%m-%d").tolist()
    
    # Target: Churn with class imbalance (~82:18)
    churn = np.random.choice([0, 1], size=n_rows, p=[0.82, 0.18])
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract_type": contracts,
        "payment_method": payment_methods,
        "signup_date": signup_dates,
        "churn": churn
    })
    
    # 2. Inject intentional anomalies for deterministic cleaning tests:
    # (a) Extreme numeric outliers in 'age' and 'monthly_charges'
    outlier_idx_age = [5, 18, 42, 88, 120]
    for idx, val in zip(outlier_idx_age, [185.0, 240.0, -15.0, 310.0, 195.0]):
        df.at[idx, "age"] = val
        
    outlier_idx_charges = [12, 65, 110]
    for idx, val in zip(outlier_idx_charges, [5800.0, 9200.0, 7500.0]):
        df.at[idx, "monthly_charges"] = val

    # (b) Missing values (NaNs) in multiple columns
    null_idx_charges = [3, 14, 27, 50, 71, 99, 140, 180]
    df.loc[null_idx_charges, "monthly_charges"] = np.nan
    
    null_idx_contract = [10, 22, 45, 80, 115]
    df.loc[null_idx_contract, "contract_type"] = np.nan
    
    # (c) High-null column (>65% missing) that must be dropped
    junk_col = [np.nan if np.random.rand() < 0.70 else "legacy_survey_note" for _ in range(n_rows)]
    df["junk_survey_notes"] = junk_col
    
    # (d) Duplicate rows (exact copies)
    duplicates = df.iloc[[1, 7, 20]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated messy dataset at {output_path} (Shape: {df.shape})")

if __name__ == "__main__":
    dest = Path(__file__).parent / "messy_churn.csv"
    generate_messy_churn(dest)
