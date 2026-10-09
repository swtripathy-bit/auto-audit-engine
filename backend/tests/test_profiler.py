"""
Unit tests for the ML-Readiness & Statistical Profiler.
"""

import numpy as np
import pandas as pd
from backend.app.core.profiler import MLReadinessProfiler
from backend.app.core.ingestion import load_dataset
from backend.app.core.cleaner import DataCleaner


def test_skewness_and_kurtosis_detection():
    np.random.seed(42)
    # Right-skewed lognormal distribution
    skewed_data = np.random.lognormal(mean=0.0, sigma=1.5, size=500)
    normal_data = np.random.normal(loc=50.0, scale=5.0, size=500)
    
    df = pd.DataFrame({
        "skewed_col": skewed_data,
        "normal_col": normal_data,
    })
    
    profiler = MLReadinessProfiler(skew_threshold=1.5)
    report = profiler.profile(df)
    
    dist = report["distribution_profile"]
    assert dist["skewed_col"]["skew_flag"] is True
    assert dist["normal_col"]["skew_flag"] is False
    assert "Yeo-Johnson" in dist["skewed_col"]["recommended_transform"]


def test_correlation_redundancy_detection():
    np.random.seed(42)
    x1 = np.linspace(1, 100, 100)
    x2 = x1 * 2.0 + np.random.normal(0, 0.5, 100)  # Near-perfect correlation
    x3 = np.random.uniform(10, 50, 100)           # Independent feature
    
    df = pd.DataFrame({"feat_1": x1, "feat_2": x2, "feat_3": x3})
    
    profiler = MLReadinessProfiler(correlation_threshold=0.85)
    report = profiler.profile(df)
    
    corr_flags = report["correlation_profile"]
    assert len(corr_flags) == 1
    assert corr_flags[0]["feature_1"] == "feat_1"
    assert corr_flags[0]["feature_2"] == "feat_2"
    assert corr_flags[0]["pearson_r"] > 0.95


def test_vif_detection():
    np.random.seed(42)
    # Create 3 features where x3 is almost an exact linear combination of x1 and x2
    x1 = np.random.normal(10, 2, 200)
    x2 = np.random.normal(20, 5, 200)
    x3 = 2.0 * x1 + 1.5 * x2 + np.random.normal(0, 0.1, 200)
    
    df = pd.DataFrame({"a": x1, "b": x2, "c": x3})
    
    profiler = MLReadinessProfiler(vif_threshold=5.0)
    report = profiler.profile(df)
    
    vif = report["vif_profile"]
    assert "c" in vif
    assert vif["c"]["risk_flag"] is True
    assert vif["c"]["vif"] > 10.0


def test_id_leakage_detection():
    df = pd.DataFrame({
        "customer_uuid": [f"ID_{i}" for i in range(100)],
        "age": np.random.randint(18, 65, 100),
    })
    
    profiler = MLReadinessProfiler()
    report = profiler.profile(df)
    
    leakage = report["id_leakage"]
    assert len(leakage) == 1
    assert leakage[0]["column"] == "customer_uuid"


def test_class_imbalance_detection():
    # 95:5 class imbalance
    target = [0] * 190 + [1] * 10
    df = pd.DataFrame({
        "feature": np.random.normal(0, 1, 200),
        "target": target
    })
    
    profiler = MLReadinessProfiler()
    report = profiler.profile(df, target_column="target")
    
    target_prof = report["target_profile"]
    assert target_prof["type"] == "classification"
    assert target_prof["imbalance_severity"] == "severe"
    assert "PR-AUC" in target_prof["recommended_metrics"]


def test_end_to_end_profiling_on_messy_churn():
    df, meta = load_dataset("sample_data/messy_churn.csv")
    cleaner = DataCleaner()
    cleaned_df, clean_report = cleaner.clean(df)
    
    profiler = MLReadinessProfiler()
    profile_report = profiler.profile(cleaned_df, target_column="churn")
    
    assert profile_report["metadata"]["total_rows"] == 250
    assert 0 <= profile_report["readiness_score"] <= 100
    assert len(profile_report["critical_risks"]) > 0
    # Customer ID should be detected as ID leakage
    id_leakage_cols = [x["column"] for x in profile_report["id_leakage"]]
    assert "customer_id" in id_leakage_cols
