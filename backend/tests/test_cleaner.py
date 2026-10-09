"""
Unit tests for Ingestion and Deterministic Data Cleaner.
"""

import io
import pytest
import numpy as np
import pandas as pd
from backend.app.core.ingestion import load_dataset, IngestionError
from backend.app.core.cleaner import DataCleaner


def test_ingestion_valid_csv():
    csv_content = b"name,age,salary\nAlice,30,70000\nBob,40,90000\nCharlie,35,80000\n"
    df, meta = load_dataset(io.BytesIO(csv_content), filename="test.csv")
    assert df.shape == (3, 3)
    assert meta["row_count"] == 3
    assert meta["column_count"] == 3
    assert "salary" in df.columns


def test_ingestion_empty_file_raises():
    with pytest.raises(IngestionError):
        load_dataset(io.BytesIO(b""), filename="empty.csv")


def test_cleaner_duplicate_pruning():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": [10, 20, 20, 30]
    })
    cleaner = DataCleaner(drop_duplicates=True)
    cleaned, report = cleaner.clean(df)
    
    assert cleaned.shape[0] == 3
    assert report["duplicates_dropped"] == 1


def test_cleaner_high_null_column_drop():
    df = pd.DataFrame({
        "good_col": [1, 2, 3, 4, 5],
        "bad_col": [np.nan, np.nan, np.nan, np.nan, 42]  # 80% null
    })
    cleaner = DataCleaner(null_drop_threshold=0.50)
    cleaned, report = cleaner.clean(df)
    
    assert "bad_col" not in cleaned.columns
    assert "good_col" in cleaned.columns
    assert len(report["columns_dropped"]) == 1
    assert report["columns_dropped"][0]["column"] == "bad_col"


def test_cleaner_imputation():
    df = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 40.0, 50.0],  # median is 30.0
        "cat": ["A", "B", "A", np.nan, "A"]       # mode is 'A'
    })
    cleaner = DataCleaner(impute_numeric_strategy="median", impute_categorical_strategy="mode")
    cleaned, report = cleaner.clean(df)
    
    assert cleaned["num"].isna().sum() == 0
    assert cleaned.at[2, "num"] == 30.0  # Median imputed
    assert cleaned["cat"].isna().sum() == 0
    assert cleaned.at[3, "cat"] == "A"   # Mode imputed
    assert report["final_null_cells"] == 0


def test_cleaner_iqr_outlier_capping():
    # 20 normal values around 50, plus 2 extreme outliers: -500 and +1000
    normal = [45, 48, 50, 52, 55, 49, 51, 50, 48, 52, 50, 49, 51, 53, 47, 48, 52, 50, 49, 51]
    values = normal + [-500, 1000]
    df = pd.DataFrame({"id": list(range(len(values))), "score": values})
    
    cleaner = DataCleaner(iqr_multiplier=1.5)
    cleaned, report = cleaner.clean(df)
    
    # Rows should NOT be dropped, only capped
    assert len(cleaned) == 22
    assert "score" in report["outliers_capped"]
    capped_info = report["outliers_capped"]["score"]
    assert capped_info["total_capped"] == 2
    
    # Outliers must be bounded
    assert cleaned["score"].min() >= capped_info["lower_bound"]
    assert cleaned["score"].max() <= capped_info["upper_bound"]


def test_cleaner_preserves_binary_columns():
    # Target column (0 and 1) should never have values clipped
    df = pd.DataFrame({"target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]})
    cleaner = DataCleaner()
    cleaned, report = cleaner.clean(df)
    
    assert "target" not in report["outliers_capped"]
    assert set(cleaned["target"].unique()) == {0, 1}
