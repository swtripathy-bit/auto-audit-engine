"""
Unit tests for the LLM Insight Agent.
"""

from backend.app.core.ingestion import load_dataset
from backend.app.core.cleaner import DataCleaner
from backend.app.core.profiler import MLReadinessProfiler
from backend.app.core.insight_agent import InsightAgent


def test_insight_agent_fallback_generation():
    cleaning_report = {
        "initial_shape": {"rows": 100, "columns": 5},
        "final_shape": {"rows": 98, "columns": 5},
        "duplicates_dropped": 2,
        "columns_dropped": [],
        "imputations": {"col_a": {"imputed_cells": 3}},
        "outliers_capped": {"col_b": {"total_capped": 4}},
    }
    profile_report = {
        "readiness_score": 85,
        "critical_risks": ["Moderate target class imbalance (4:1)."],
        "distribution_profile": {},
        "correlation_profile": [],
        "vif_profile": {},
        "id_leakage": [{"column": "user_id"}],
        "target_profile": {
            "type": "classification",
            "imbalance_ratio": "4:1",
            "minority_percentage": "20.0%",
            "recommended_metrics": ["PR-AUC", "F1-Macro"]
        }
    }

    # Test with no API key -> should use fallback without throwing exceptions
    agent = InsightAgent(api_key="")
    result = agent.generate_audit_narrative(cleaning_report, profile_report)

    assert result["status"] == "fallback_heuristic"
    assert result["health_grade"] == "B"
    assert "user_id" in result["narrative_markdown"]
    assert "PR-AUC" in result["narrative_markdown"]


def test_insight_agent_end_to_end_pipeline():
    # Full multi-stage pipeline: Ingestion -> Cleaner -> Profiler -> InsightAgent
    df, meta = load_dataset("sample_data/messy_churn.csv")
    cleaner = DataCleaner()
    cleaned_df, clean_report = cleaner.clean(df)

    profiler = MLReadinessProfiler()
    profile_report = profiler.profile(cleaned_df, target_column="churn")

    agent = InsightAgent()
    narrative_result = agent.generate_audit_narrative(clean_report, profile_report)

    assert "narrative_markdown" in narrative_result
    assert narrative_result["health_grade"] in ["A", "B", "C", "D", "F"]
    assert len(narrative_result["narrative_markdown"]) > 100
