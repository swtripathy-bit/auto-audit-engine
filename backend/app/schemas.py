"""
Pydantic v2 schemas for the FastAPI REST API layer.
Defines strict request/response validation contracts.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="API operational status")
    gemini_configured: bool = Field(..., description="Whether Gemini API key is present")
    model: str = Field(..., description="Designated LLM model")
    version: str = Field("1.0.0", description="API version")


class ShapeInfo(BaseModel):
    rows: int
    columns: int


class DroppedColumnInfo(BaseModel):
    column: str
    null_ratio: float
    null_percentage: str


class ImputationDetail(BaseModel):
    dtype: str
    strategy: str
    fill_value: Any
    imputed_cells: int


class OutlierCappedDetail(BaseModel):
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    below_lower_capped: int
    above_upper_capped: int
    total_capped: int


class CleaningSummary(BaseModel):
    initial_shape: ShapeInfo
    final_shape: ShapeInfo
    initial_null_cells: int
    final_null_cells: int
    duplicates_dropped: int
    columns_dropped: List[DroppedColumnInfo] = []
    type_transformations: Dict[str, Any] = {}
    imputations: Dict[str, ImputationDetail] = {}
    outliers_capped: Dict[str, OutlierCappedDetail] = {}


class DatasetMetadata(BaseModel):
    filename: str
    file_size_kb: float
    total_rows: int
    total_features: int
    numeric_features: int
    categorical_features: int
    target_column: Optional[str] = None


class AuditResponse(BaseModel):
    metadata: DatasetMetadata
    readiness_score: int = Field(..., ge=0, le=100, description="Overall ML readiness score (0-100)")
    health_grade: str = Field(..., description="Overall dataset health letter grade (A/B/C/D/F)")
    critical_risks: List[str] = Field(default_factory=list, description="Primary structural risks detected")
    cleaning_summary: CleaningSummary
    distribution_profile: Dict[str, Any] = {}
    correlation_profile: List[Dict[str, Any]] = []
    vif_profile: Dict[str, Any] = {}
    id_leakage: List[Dict[str, Any]] = []
    target_profile: Optional[Dict[str, Any]] = None
    ai_narrative_markdown: str
    narrative_status: str
    processing_time_seconds: float
