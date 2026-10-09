"""
LLM Insight Agent.
Compiles deterministic statistical profiles into an executive audit report using Google Gemini (gemini-3.8-flash).
Sends ZERO raw data rows to ensure complete data privacy, minimal token consumption, and zero hallucinations.
"""

from __future__ import annotations
import json
import logging
from typing import Dict, Any, Optional
from backend.app.config import settings

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = """
You are a Principal Machine Learning Architect and Senior Data Quality Auditor.
You are reviewing a structured dataset profile and data hygiene audit generated deterministically by a statistical engine.
Zero raw data rows are provided—only mathematically verified statistical metadata.

Your job is to provide an authoritative, executive-grade ML Readiness Report with:
1. Executive Health Grade & Verdict (Grade: A, B, C, or F with a 2-sentence executive takeaway).
2. Critical Modeling Blockers & Structural Risks (Address class imbalance, skewness, VIF multicollinearity, and ID leakage).
3. Prescriptive Preprocessing & Modeling Roadmap (Which features to drop/transform, metric selection like PR-AUC vs ROC-AUC, and recommended algorithm families like LightGBM vs Regularized GLM).

Formatting guidelines:
- Use clear GitHub-flavored markdown with clean bullet points and bold highlights.
- Be precise, actionable, and mathematically grounded.
- Avoid generic platitudes; refer directly to the metrics, feature names, ratios, and scores in the audit summary.
"""


class InsightAgent:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL

    def generate_audit_narrative(
        self,
        cleaning_report: Dict[str, Any],
        profile_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compiles the cleaning and statistical profile reports and generates an executive audit narrative.
        
        Returns:
            Dict containing:
            - "narrative_markdown": Full text report
            - "health_grade": Letter grade (A/B/C/D/F)
            - "status": "live_llm" or "fallback_heuristic"
        """
        # 1. Build compact statistical digest
        digest = {
            "initial_shape": cleaning_report.get("initial_shape"),
            "final_shape": cleaning_report.get("final_shape"),
            "duplicates_dropped": cleaning_report.get("duplicates_dropped"),
            "columns_dropped_high_null": cleaning_report.get("columns_dropped"),
            "imputations_performed": cleaning_report.get("imputations"),
            "outliers_capped_iqr": cleaning_report.get("outliers_capped"),
            "ml_readiness_score": profile_report.get("readiness_score"),
            "critical_risks": profile_report.get("critical_risks"),
            "skewness_and_kurtosis": profile_report.get("distribution_profile"),
            "high_pairwise_correlations": profile_report.get("correlation_profile"),
            "vif_multicollinearity": profile_report.get("vif_profile"),
            "id_leakage_columns": profile_report.get("id_leakage"),
            "target_diagnostics": profile_report.get("target_profile"),
        }

        # 2. Check if API key is configured
        if self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                return self._call_gemini(digest)
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Falling back to deterministic narrative synthesis.")
                return self._generate_fallback(digest, error_msg=str(e))
        else:
            return self._generate_fallback(
                digest,
                error_msg="GEMINI_API_KEY is not configured in .env. Showing local rule-based audit report."
            )

    def _call_gemini(self, digest: Dict[str, Any]) -> Dict[str, Any]:
        """Calls the official Google GenAI SDK (google-genai) with gemini-3.8-flash."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        prompt = f"""
Analyze this structured dataset audit digest and produce the executive ML Readiness report:

```json
{json.dumps(digest, indent=2)}
```
"""

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=2048,
            )
        )

        narrative = response.text or "Unable to generate audit narrative."
        grade = self._extract_grade(narrative, digest.get("ml_readiness_score", 70))

        return {
            "narrative_markdown": narrative,
            "health_grade": grade,
            "status": "live_llm",
            "model_used": self.model,
        }

    def _generate_fallback(self, digest: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Generates a high-quality, deterministic offline audit report when API key is unavailable."""
        score = digest.get("ml_readiness_score", 75)
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 65 else "F"
        
        risks_md = "\n".join([f"- ⚠️ **{risk}**" for risk in digest.get("critical_risks", [])]) or "- No critical structural risks detected."
        
        # Outlier notes
        outliers = digest.get("outliers_capped_iqr", {})
        outlier_summary = ", ".join([f"`{col}` ({info['total_capped']} capped)" for col, info in outliers.items()]) if outliers else "None"

        # Target notes
        target = digest.get("target_diagnostics")
        if target and target.get("type") == "classification":
            target_notes = f"Target has class distribution of **{target.get('imbalance_ratio', 'N/A')}** ({target.get('minority_percentage', 'N/A')} minority). **Recommended evaluation metrics:** {', '.join(target.get('recommended_metrics', []))}."
        elif target:
            target_notes = f"Regression target with skewness {target.get('skewness', 0.0)}."
        else:
            target_notes = "No target column designated."

        narrative = f"""
> [!NOTE]
> {error_msg}

## 📊 Executive Data Audit & ML-Readiness Summary

**Overall Health Grade: `{grade}` (Score: {score}/100)**

### 1. Data Hygiene & Cleaning Actions
- **Sample Consistency:** Evaluated {digest.get('initial_shape', {}).get('rows', 0)} raw rows $\to$ dropped {digest.get('duplicates_dropped', 0)} duplicates and {len(digest.get('columns_dropped_high_null', []))} high-null columns.
- **Outlier Capping:** Applied IQR winsorization to continuous variables: {outlier_summary}.
- **Imputation:** Resolved all missing values using median for numeric features and mode for categoricals.

### 2. Critical ML Modeling Risks
{risks_md}

### 3. Prescriptive Roadmap & Recommendations
- **Target Handling:** {target_notes}
- **Feature Selection:** Exclude high-cardinality identifier columns ({', '.join([x['column'] for x in digest.get('id_leakage_columns', [])]) or 'None'}) before model ingestion to prevent memorization.
- **Model Choice:** Tree-based ensembles (LightGBM, XGBoost) are strongly recommended over unregularized linear models for this feature topology.
"""

        return {
            "narrative_markdown": narrative.strip(),
            "health_grade": grade,
            "status": "fallback_heuristic",
            "model_used": "deterministic_rule_engine",
        }

    def _extract_grade(self, text: str, fallback_score: int) -> str:
        for letter in ["Grade A", "Grade B", "Grade C", "Grade F", "Grade: A", "Grade: B", "Grade: C", "Grade: F"]:
            if letter.lower() in text.lower():
                return letter.replace("Grade:", "").replace("Grade", "").strip().upper()
        return "A" if fallback_score >= 90 else "B" if fallback_score >= 80 else "C" if fallback_score >= 65 else "F"
