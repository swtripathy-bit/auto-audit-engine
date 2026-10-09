"""
Deterministic Data Hygiene & Cleaning Engine.
Executes predictable, vectorized transformations:
- Exact duplicate pruning
- High-null column dropping (>50% missing)
- Type auto-casting (dates, numerics, categoricals)
- Deterministic imputation (median for numeric, mode for categorical)
- Outlier winsorization (Interquartile Range - IQR clipping)
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd


class DataCleaner:
    def __init__(
        self,
        null_drop_threshold: float = 0.50,
        iqr_multiplier: float = 1.5,
        impute_numeric_strategy: str = "median",
        impute_categorical_strategy: str = "mode",
        drop_duplicates: bool = True,
        auto_cast_types: bool = True,
    ):
        """
        Args:
            null_drop_threshold: Columns with missingness exceeding this ratio are dropped (0.0 to 1.0).
            iqr_multiplier: Multiplier for IQR outlier boundaries (standard 1.5).
            impute_numeric_strategy: 'median' or 'mean'.
            impute_categorical_strategy: 'mode' or 'constant' (fills with 'Missing').
            drop_duplicates: Whether to drop identical rows.
            auto_cast_types: Whether to attempt automatic date and numeric string casting.
        """
        self.null_drop_threshold = null_drop_threshold
        self.iqr_multiplier = iqr_multiplier
        self.impute_numeric_strategy = impute_numeric_strategy
        self.impute_categorical_strategy = impute_categorical_strategy
        self.drop_duplicates = drop_duplicates
        self.auto_cast_types = auto_cast_types

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes full deterministic data hygiene on a DataFrame.

        Returns:
            Tuple of (cleaned_df, audit_report_dict)
        """
        cleaned = df.copy()
        initial_shape = (int(df.shape[0]), int(df.shape[1]))
        initial_null_count = int(df.isna().sum().sum())

        audit_report: Dict[str, Any] = {
            "initial_shape": {
                "rows": initial_shape[0],
                "columns": initial_shape[1],
            },
            "initial_null_cells": initial_null_count,
            "duplicates_dropped": 0,
            "columns_dropped": [],
            "type_transformations": {},
            "imputations": {},
            "outliers_capped": {},
            "final_shape": {},
            "final_null_cells": 0,
        }

        # -------------------------------------------------------------
        # 1. Exact Duplicate Pruning
        # -------------------------------------------------------------
        if self.drop_duplicates:
            before_len = len(cleaned)
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            duplicates_dropped = before_len - len(cleaned)
            audit_report["duplicates_dropped"] = int(duplicates_dropped)

        # -------------------------------------------------------------
        # 2. High-Null Column Elimination
        # -------------------------------------------------------------
        total_rows = len(cleaned)
        if total_rows > 0:
            cols_to_drop: List[str] = []
            for col in cleaned.columns:
                null_pct = float(cleaned[col].isna().mean())
                if null_pct > self.null_drop_threshold:
                    cols_to_drop.append(col)
                    audit_report["columns_dropped"].append({
                        "column": col,
                        "null_ratio": round(null_pct, 4),
                        "null_percentage": f"{round(null_pct * 100, 1)}%",
                    })

            if cols_to_drop:
                cleaned = cleaned.drop(columns=cols_to_drop)

        # -------------------------------------------------------------
        # 3. Type Auto-Casting (Defensive)
        # -------------------------------------------------------------
        if self.auto_cast_types:
            for col in cleaned.columns:
                orig_dtype = str(cleaned[col].dtype)

                if orig_dtype in ("object", "string"):
                    # Check if numeric string
                    numeric_series = pd.to_numeric(cleaned[col], errors="coerce")
                    valid_numeric_ratio = float(numeric_series.notna().mean())
                    if valid_numeric_ratio > 0.85 and cleaned[col].nunique() > 1:
                        cleaned[col] = numeric_series
                        audit_report["type_transformations"][col] = {
                            "from": orig_dtype,
                            "to": str(cleaned[col].dtype),
                            "reason": "Numeric string auto-cast",
                        }
                        continue

                    # Check if date string
                    # Sample non-null values to test datetime parsing
                    non_nulls = cleaned[col].dropna()
                    if len(non_nulls) > 0 and len(non_nulls) <= 1000:
                        sample = non_nulls.iloc[: min(50, len(non_nulls))]
                    elif len(non_nulls) > 1000:
                        sample = non_nulls.sample(n=50, random_state=42)
                    else:
                        sample = pd.Series([], dtype="object")

                    if len(sample) > 0:
                        try:
                            parsed_dates = pd.to_datetime(sample, errors="coerce")
                            if parsed_dates.notna().mean() > 0.80:
                                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
                                audit_report["type_transformations"][col] = {
                                    "from": orig_dtype,
                                    "to": "datetime64[ns]",
                                    "reason": "Datetime auto-cast",
                                }
                        except Exception:
                            pass

        # -------------------------------------------------------------
        # 4. Deterministic Missing Value Imputation
        # -------------------------------------------------------------
        for col in cleaned.columns:
            null_count = int(cleaned[col].isna().sum())
            if null_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(cleaned[col]):
                if self.impute_numeric_strategy == "median":
                    fill_val = float(cleaned[col].median())
                else:
                    fill_val = float(cleaned[col].mean())

                # If all were NaN or column has no variance
                if np.isnan(fill_val):
                    fill_val = 0.0

                cleaned[col] = cleaned[col].fillna(fill_val)
                audit_report["imputations"][col] = {
                    "dtype": "numeric",
                    "strategy": self.impute_numeric_strategy,
                    "fill_value": round(fill_val, 4),
                    "imputed_cells": null_count,
                }

            elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                # For dates, forward-fill then backward-fill
                cleaned[col] = cleaned[col].ffill().bfill()
                audit_report["imputations"][col] = {
                    "dtype": "datetime",
                    "strategy": "ffill_bfill",
                    "fill_value": "adjacent_date",
                    "imputed_cells": null_count,
                }

            else:
                # Categorical / string column
                if self.impute_categorical_strategy == "mode":
                    mode_series = cleaned[col].mode(dropna=True)
                    fill_val = str(mode_series.iloc[0]) if not mode_series.empty else "Missing"
                else:
                    fill_val = "Missing"

                cleaned[col] = cleaned[col].fillna(fill_val)
                audit_report["imputations"][col] = {
                    "dtype": "categorical",
                    "strategy": self.impute_categorical_strategy,
                    "fill_value": fill_val,
                    "imputed_cells": null_count,
                }

        # -------------------------------------------------------------
        # 5. Outlier Winsorization (IQR Capping)
        # -------------------------------------------------------------
        for col in cleaned.columns:
            # Only apply to continuous numeric columns (skip booleans and low-cardinality binaries)
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                unique_vals = cleaned[col].nunique()
                if unique_vals <= 2:
                    # Binary or indicator column (e.g. churn 0/1) -> do not clip
                    continue

                q1 = float(cleaned[col].quantile(0.25))
                q3 = float(cleaned[col].quantile(0.75))
                iqr = q3 - q1

                if iqr > 0:
                    lower_bound = float(q1 - self.iqr_multiplier * iqr)
                    upper_bound = float(q3 + self.iqr_multiplier * iqr)

                    under_count = int((cleaned[col] < lower_bound).sum())
                    over_count = int((cleaned[col] > upper_bound).sum())
                    total_outliers = under_count + over_count

                    if total_outliers > 0:
                        cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                        audit_report["outliers_capped"][col] = {
                            "q1": round(q1, 4),
                            "q3": round(q3, 4),
                            "iqr": round(iqr, 4),
                            "lower_bound": round(lower_bound, 4),
                            "upper_bound": round(upper_bound, 4),
                            "below_lower_capped": under_count,
                            "above_upper_capped": over_count,
                            "total_capped": total_outliers,
                        }

        # -------------------------------------------------------------
        # 6. Final Summary
        # -------------------------------------------------------------
        audit_report["final_shape"] = {
            "rows": int(cleaned.shape[0]),
            "columns": int(cleaned.shape[1]),
        }
        audit_report["final_null_cells"] = int(cleaned.isna().sum().sum())

        return cleaned, audit_report
