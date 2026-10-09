"""
FastAPI REST API Layer for the Autonomous Data Auditing & ML-Readiness Engine.
Exposes endpoints for dataset ingestion, deterministic hygiene, statistical profiling, and AI narrative synthesis.
"""

from __future__ import annotations
import io
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse

from backend.app.config import settings
from backend.app.core.ingestion import load_dataset, IngestionError
from backend.app.core.cleaner import DataCleaner
from backend.app.core.profiler import MLReadinessProfiler
from backend.app.core.insight_agent import InsightAgent
from backend.app.schemas import HealthCheckResponse, AuditResponse, DatasetMetadata

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Deterministic Data Hygiene, Statistical ML-Readiness Profiling, and Zero-PII LLM Insight Layer.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend clients (Streamlit, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get(
    f"{settings.API_V1_STR}/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health Check & System Configuration"
)
async def health_check():
    """Returns the operational status and Gemini API connectivity state."""
    has_gemini = bool(
        settings.GEMINI_API_KEY
        and settings.GEMINI_API_KEY.strip()
        and not settings.GEMINI_API_KEY.startswith("your_")
    )
    return HealthCheckResponse(
        status="healthy",
        gemini_configured=has_gemini,
        model=settings.GEMINI_MODEL,
        version="1.0.0",
    )


@app.post(
    f"{settings.API_V1_STR}/audit",
    response_model=AuditResponse,
    tags=["Auditing"],
    summary="Execute Multi-Stage Audit & ML-Readiness Evaluation"
)
async def audit_dataset(
    file: UploadFile = File(..., description="Raw dataset file (.csv, .xlsx, .tsv)"),
    target_column: Optional[str] = Form(None, description="Optional target variable name for ML diagnostics")
):
    """
    Executes the end-to-end autonomous audit engine:
    1. Defensive file ingestion and format parsing
    2. Deterministic data hygiene (imputation, IQR outlier capping, duplicate removal)
    3. ML-readiness statistical profiling (VIF, skewness, kurtosis, class imbalance)
    4. Zero-PII Gemini narrative generation
    """
    start_time = time.perf_counter()

    # 1. Ingestion Layer
    try:
        contents = await file.read()
        raw_df, file_meta = load_dataset(contents, filename=file.filename)
    except IngestionError as ie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File ingestion error: {str(ie)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while loading file: {str(e)}"
        )

    # Validate target column if provided
    cleaned_target = target_column.strip() if target_column else None
    if cleaned_target and cleaned_target not in raw_df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Specified target_column '{cleaned_target}' was not found in dataset columns: {list(raw_df.columns)}"
        )

    # 2. Deterministic Cleaning Engine
    cleaner = DataCleaner()
    cleaned_df, cleaning_summary = cleaner.clean(raw_df)

    # 3. Statistical & ML-Readiness Profiler
    profiler = MLReadinessProfiler()
    profile_summary = profiler.profile(cleaned_df, target_column=cleaned_target)

    # 4. LLM Insight Agent (Gemini 3.8 Flash)
    insight_agent = InsightAgent()
    narrative_result = insight_agent.generate_audit_narrative(
        cleaning_report=cleaning_summary,
        profile_report=profile_summary
    )

    elapsed_time = round(time.perf_counter() - start_time, 3)

    dataset_metadata = DatasetMetadata(
        filename=file.filename or "uploaded_dataset.csv",
        file_size_kb=file_meta["file_size_kb"],
        total_rows=profile_summary["metadata"]["total_rows"],
        total_features=profile_summary["metadata"]["total_features"],
        numeric_features=profile_summary["metadata"]["numeric_features"],
        categorical_features=profile_summary["metadata"]["categorical_features"],
        target_column=cleaned_target,
    )

    return AuditResponse(
        metadata=dataset_metadata,
        readiness_score=profile_summary["readiness_score"],
        health_grade=narrative_result["health_grade"],
        critical_risks=profile_summary["critical_risks"],
        cleaning_summary=cleaning_summary,
        distribution_profile=profile_summary.get("distribution_profile", {}),
        correlation_profile=profile_summary.get("correlation_profile", []),
        vif_profile=profile_summary.get("vif_profile", {}),
        id_leakage=profile_summary.get("id_leakage", []),
        target_profile=profile_summary.get("target_profile"),
        ai_narrative_markdown=narrative_result["narrative_markdown"],
        narrative_status=narrative_result["status"],
        processing_time_seconds=elapsed_time,
    )


@app.post(
    f"{settings.API_V1_STR}/clean-and-download",
    tags=["Auditing"],
    summary="Clean Dataset and Return Downloadable CSV"
)
async def clean_and_download(
    file: UploadFile = File(..., description="Raw dataset file (.csv, .xlsx, .tsv)")
):
    """
    Executes deterministic hygiene (deduplication, imputation, IQR outlier winsorization)
    and streams back the cleaned dataset directly as an exportable CSV.
    """
    try:
        contents = await file.read()
        raw_df, _ = load_dataset(contents, filename=file.filename)
    except IngestionError as ie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File ingestion error: {str(ie)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading file: {str(e)}"
        )

    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean(raw_df)

    # In-memory CSV streaming without writing temporary files
    csv_buffer = io.StringIO()
    cleaned_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    out_name = f"cleaned_{file.filename or 'dataset.csv'}"
    if not out_name.endswith(".csv"):
        out_name += ".csv"

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={out_name}"
        }
    )
