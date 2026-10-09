"""
Integration and API endpoint tests for the FastAPI application.
"""

import io
import pandas as pd
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemini_configured" in data
    assert "model" in data


def test_audit_endpoint_with_valid_csv():
    csv_bytes = b"age,income,churn\n25,50000,0\n30,60000,0\n45,90000,1\n35,65000,0\n50,110000,1\n"
    files = {"file": ("test_data.csv", csv_bytes, "text/csv")}
    data = {"target_column": "churn"}

    response = client.post("/api/v1/audit", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()

    assert payload["metadata"]["filename"] == "test_data.csv"
    assert payload["metadata"]["target_column"] == "churn"
    assert 0 <= payload["readiness_score"] <= 100
    assert payload["health_grade"] in ["A", "B", "C", "D", "F"]
    assert "cleaning_summary" in payload
    assert "ai_narrative_markdown" in payload
    assert payload["processing_time_seconds"] > 0


def test_audit_endpoint_invalid_target():
    csv_bytes = b"feature_a,feature_b\n1,2\n3,4\n"
    files = {"file": ("data.csv", csv_bytes, "text/csv")}
    data = {"target_column": "non_existent_column"}

    response = client.post("/api/v1/audit", files=files, data=data)
    assert response.status_code == 400
    assert "was not found" in response.json()["detail"]


def test_clean_and_download_endpoint():
    # CSV with duplicate row and null cell
    csv_bytes = b"id,value\n1,10.0\n2,\n1,10.0\n"
    files = {"file": ("raw_data.csv", csv_bytes, "text/csv")}

    response = client.post("/api/v1/clean-and-download", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "attachment; filename=cleaned_raw_data.csv" in response.headers["content-disposition"]

    # Read the returned stream into pandas
    df_clean = pd.read_csv(io.StringIO(response.text))
    # Duplicate row (1, 10.0) should be dropped -> exactly 2 rows
    assert len(df_clean) == 2
    # Null value should be imputed
    assert df_clean["value"].isna().sum() == 0


def test_audit_empty_file_returns_400():
    files = {"file": ("empty.csv", b"", "text/csv")}
    response = client.post("/api/v1/audit", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
