"""
ML-Readiness & Statistical Profiler Engine.
Computes deep statistical diagnostics to evaluate dataset readiness before model training:
1. Fisher-Pearson Skewness & Excess Kurtosis (Distribution health)
2. Pairwise Pearson Correlation (Feature redundancy)
3. Variance Inflation Factor - VIF (Multivariate collinearity)
4. High-Cardinality ID Leakage Detection
5. Target Variable Profiling (Class imbalance & entropy)
6. Structured ML-Readiness Summary JSON
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class MLReadinessProfiler:
    def __init__(
        self,
        skew_threshold: float = 1.5,
        kurtosis_threshold: float = 3.0,
        correlation_threshold: float = 0.85,
        vif_threshold: float = 5.0,
        severe_vif_threshold: float = 10.0,
    ):
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.severe_vif_threshold = severe_vif_threshold

    def profile(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs full statistical profiling on a cleaned dataset.

        Returns:
            Structured dictionary of statistical metrics and ML readiness flags.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        feature_columns = [c for c in df.columns if c != target_column]

        # 1. Distribution Health: Skewness & Kurtosis
        distribution_profile = self._compute_distributions(numeric_df, target_column)

        # 2. Pairwise Correlation Redundancy
        correlation_profile = self._compute_correlations(numeric_df, target_column)

        # 3. Variance Inflation Factor (VIF)
        vif_profile = self._compute_vif(numeric_df, target_column)

        # 4. High-Cardinality ID Leakage
        id_leakage = self._detect_id_columns(df, target_column)

        # 5. Target Variable Diagnostics
        target_profile = self._profile_target(df, target_column) if target_column and target_column in df.columns else None

        # 6. Synthesize Key Risks & Readiness Score
        risks, readiness_score = self._compute_readiness_score(
            distribution_profile=distribution_profile,
            correlation_profile=correlation_profile,
            vif_profile=vif_profile,
            id_leakage=id_leakage,
            target_profile=target_profile,
        )

        return {
            "metadata": {
                "total_rows": int(len(df)),
                "total_features": int(len(feature_columns)),
                "numeric_features": int(len(numeric_df.columns) - (1 if target_column in numeric_df.columns else 0)),
                "categorical_features": int(len(df.select_dtypes(exclude=[np.number]).columns)),
                "target_column": target_column,
            },
            "readiness_score": readiness_score,
            "critical_risks": risks,
            "distribution_profile": distribution_profile,
            "correlation_profile": correlation_profile,
            "vif_profile": vif_profile,
            "id_leakage": id_leakage,
            "target_profile": target_profile,
        }

    def _compute_distributions(
        self,
        numeric_df: pd.DataFrame,
        target_col: Optional[str]
    ) -> Dict[str, Any]:
        distributions = {}
        for col in numeric_df.columns:
            if col == target_col:
                continue
            series = numeric_df[col].dropna()
            if len(series) < 3 or series.nunique() <= 2:
                continue

            skew_val = float(series.skew())
            kurt_val = float(series.kurtosis())  # Fisher excess kurtosis (0 for normal)

            is_severe_skew = abs(skew_val) > self.skew_threshold
            is_heavy_tail = kurt_val > self.kurtosis_threshold

            distributions[col] = {
                "skewness": round(skew_val, 3),
                "excess_kurtosis": round(kurt_val, 3),
                "skew_flag": is_severe_skew,
                "kurtosis_flag": is_heavy_tail,
                "distribution_type": (
                    "severely_skewed" if is_severe_skew
                    else "heavy_tailed" if is_heavy_tail
                    else "approximately_normal"
                ),
                "recommended_transform": (
                    "Yeo-Johnson / Log transform" if is_severe_skew
                    else "RobustScaler / quantile clipping" if is_heavy_tail
                    else "StandardScaler"
                ),
            }
        return distributions

    def _compute_correlations(
        self,
        numeric_df: pd.DataFrame,
        target_col: Optional[str]
    ) -> List[Dict[str, Any]]:
        features = [c for c in numeric_df.columns if c != target_col]
        if len(features) < 2:
            return []

        sub_df = numeric_df[features].dropna()
        corr_matrix = sub_df.corr(method="pearson").abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        flagged_pairs = []
        for col in upper_tri.columns:
            matches = upper_tri[col][upper_tri[col] > self.correlation_threshold]
            for pair_col, r_val in matches.items():
                flagged_pairs.append({
                    "feature_1": pair_col,
                    "feature_2": col,
                    "pearson_r": round(float(r_val), 3),
                    "severity": "critical" if r_val > 0.95 else "high",
                    "action": f"Redundant feature pair. Recommend dropping either '{pair_col}' or '{col}'."
                })

        return sorted(flagged_pairs, key=lambda x: x["pearson_r"], reverse=True)

    def _compute_vif(
        self,
        numeric_df: pd.DataFrame,
        target_col: Optional[str]
    ) -> Dict[str, Any]:
        features = [c for c in numeric_df.columns if c != target_col]
        if len(features) < 2:
            return {}

        sub_df = numeric_df[features].dropna()
        # Drop constant features
        stds = sub_df.std()
        valid_features = stds[stds > 1e-6].index.tolist()
        if len(valid_features) < 2:
            return {}

        X = sub_df[valid_features].copy()
        X_with_const = add_constant(X, has_constant="add")

        vif_results = {}
        for i, col in enumerate(valid_features):
            try:
                score = float(variance_inflation_factor(X_with_const.values, i + 1))
                if np.isinf(score) or np.isnan(score) or score > 999.0:
                    vif_val = 999.0
                    status = "extreme_singularity"
                else:
                    vif_val = round(score, 2)
                    status = "severe" if vif_val > self.severe_vif_threshold else "moderate" if vif_val > self.vif_threshold else "safe"

                vif_results[col] = {
                    "vif": vif_val,
                    "status": status,
                    "risk_flag": vif_val > self.vif_threshold,
                }
            except Exception:
                vif_results[col] = {"vif": None, "status": "error", "risk_flag": False}

        return vif_results

    def _detect_id_columns(self, df: pd.DataFrame, target_col: Optional[str]) -> List[Dict[str, Any]]:
        leakage = []
        n_rows = len(df)
        if n_rows < 10:
            return leakage

        for col in df.columns:
            if col == target_col:
                continue

            # Continuous floating point numbers naturally have high unique ratios; skip them
            if pd.api.types.is_float_dtype(df[col]):
                continue

            unique_ratio = df[col].nunique() / n_rows
            col_lower = str(col).lower()
            is_id_named = any(term in col_lower for term in ["id", "uuid", "guid", "key", "index"])

            if unique_ratio > 0.90 or (is_id_named and unique_ratio > 0.70):
                leakage.append({
                    "column": col,
                    "unique_ratio": round(unique_ratio, 3),
                    "reason": "High unique ratio / identifier name. High risk of overfitting or data leakage.",
                    "action": f"Exclude '{col}' from model feature matrix."
                })

        return leakage

    def _profile_target(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        target_series = df[target_col].dropna()
        unique_vals = target_series.nunique()
        total_count = len(target_series)

        # Binary or Multiclass classification target
        if unique_vals <= 20:
            val_counts = target_series.value_counts()
            distribution = {str(k): int(v) for k, v in val_counts.items()}
            proportions = {str(k): round(float(v / total_count), 4) for k, v in val_counts.items()}
            
            # Shannon entropy calculation
            entropy = -sum(p * math.log2(p) for p in proportions.values() if p > 0)
            max_entropy = math.log2(unique_vals) if unique_vals > 1 else 1.0
            normalized_entropy = round(float(entropy / max_entropy), 3) if max_entropy > 0 else 1.0

            minority_prop = min(proportions.values())
            imbalance_ratio = round(max(proportions.values()) / minority_prop, 1) if minority_prop > 0 else 999.0

            severity = "severe" if minority_prop < 0.10 else "moderate" if minority_prop < 0.25 else "balanced"

            return {
                "type": "classification",
                "unique_classes": unique_vals,
                "class_counts": distribution,
                "class_proportions": proportions,
                "minority_percentage": f"{round(minority_prop * 100, 1)}%",
                "imbalance_ratio": f"{imbalance_ratio}:1",
                "normalized_entropy": normalized_entropy,
                "imbalance_severity": severity,
                "recommended_metrics": (
                    ["PR-AUC", "F1-Macro", "Balanced Accuracy"] if severity in ("severe", "moderate")
                    else ["ROC-AUC", "Accuracy", "F1"]
                ),
                "remedy": (
                    "High class imbalance detected. Do not optimize raw Accuracy. Apply class_weight='balanced', SMOTE, or Focal Loss."
                    if severity in ("severe", "moderate") else "Balanced target distribution."
                )
            }
        else:
            # Continuous regression target
            skew_val = float(target_series.skew())
            return {
                "type": "regression",
                "mean": round(float(target_series.mean()), 3),
                "std": round(float(target_series.std()), 3),
                "min": round(float(target_series.min()), 3),
                "max": round(float(target_series.max()), 3),
                "skewness": round(skew_val, 3),
                "skew_flag": abs(skew_val) > self.skew_threshold,
                "recommended_metrics": ["RMSE", "MAE", "R2", "MAPE"],
                "remedy": "Consider log1p transform on target if skewness > 1.5." if abs(skew_val) > 1.5 else "Target distribution is well-behaved."
            }

    def _compute_readiness_score(
        self,
        distribution_profile: Dict[str, Any],
        correlation_profile: List[Dict[str, Any]],
        vif_profile: Dict[str, Any],
        id_leakage: List[Dict[str, Any]],
        target_profile: Optional[Dict[str, Any]],
    ) -> tuple[List[str], int]:
        score = 100
        risks = []

        # Deduct for high skewness
        skewed_cols = [c for c, d in distribution_profile.items() if d["skew_flag"]]
        if skewed_cols:
            penalty = min(20, len(skewed_cols) * 5)
            score -= penalty
            risks.append(f"Severe feature skewness (|skew| > {self.skew_threshold}) detected in: {', '.join(skewed_cols)}.")

        # Deduct for collinear pairs
        if correlation_profile:
            penalty = min(25, len(correlation_profile) * 8)
            score -= penalty
            pair_strs = [f"({p['feature_1']} & {p['feature_2']}: r={p['pearson_r']})" for p in correlation_profile[:3]]
            risks.append(f"High feature redundancy (|r| > {self.correlation_threshold}) in: {', '.join(pair_strs)}.")

        # Deduct for severe VIF
        severe_vif_cols = [c for c, d in vif_profile.items() if d.get("status") in ("severe", "extreme_singularity")]
        if severe_vif_cols:
            penalty = min(25, len(severe_vif_cols) * 8)
            score -= penalty
            risks.append(f"Severe multicollinearity (VIF > {self.severe_vif_threshold}) detected in: {', '.join(severe_vif_cols)}.")

        # Deduct for ID leakage
        if id_leakage:
            score -= 10
            id_cols = [item["column"] for item in id_leakage]
            risks.append(f"Potential ID leakage / high cardinality features: {', '.join(id_cols)}.")

        # Deduct for target imbalance
        if target_profile and target_profile.get("type") == "classification":
            if target_profile.get("imbalance_severity") == "severe":
                score -= 15
                risks.append(f"Critical target class imbalance ({target_profile['imbalance_ratio']}). Raw accuracy will be misleading.")
            elif target_profile.get("imbalance_severity") == "moderate":
                score -= 8
                risks.append(f"Moderate target class imbalance ({target_profile['imbalance_ratio']}).")

        final_score = max(10, min(100, score))
        return risks, final_score
