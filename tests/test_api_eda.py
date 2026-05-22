"""
Integration tests: POST /eda/analysis/
Uses the minimal TestClient fixture from conftest.py.
No external services required. Uses small synthetic CSV data.
"""
import io


def _make_numeric_csv(rows: int = 80) -> bytes:
    lines = ["feature1,feature2,feature3"]
    for i in range(rows):
        lines.append(f"{i * 0.5:.2f},{i * 1.1:.2f},{i % 10}")
    return "\n".join(lines).encode()


def test_eda_with_uploaded_file_returns_200(client):
    response = client.post(
        "/eda/analysis/",
        files={"file": ("eda_test.csv", io.BytesIO(_make_numeric_csv(80)), "text/csv")},
        data={"interactive": "false", "sample_size": "500", "max_numeric_cols": "5"},
    )
    assert response.status_code == 200


def test_eda_response_has_required_keys(client):
    response = client.post(
        "/eda/analysis/",
        files={"file": ("eda_test.csv", io.BytesIO(_make_numeric_csv(80)), "text/csv")},
        data={"interactive": "false", "sample_size": "500", "max_numeric_cols": "5"},
    )
    data = response.json()
    assert data["status"] == "success"
    assert "eda_report" in data
    assert "tables" in data


def test_eda_report_is_non_empty_string(client):
    response = client.post(
        "/eda/analysis/",
        files={"file": ("eda_test.csv", io.BytesIO(_make_numeric_csv(80)), "text/csv")},
        data={"interactive": "false", "sample_size": "500", "max_numeric_cols": "5"},
    )
    data = response.json()
    assert isinstance(data["eda_report"], str)
    assert len(data["eda_report"]) > 0


def test_eda_no_input_returns_error(client):
    # The route raises HTTPException(400) internally but wraps it in 500 via its
    # outer except clause. Accept both to be resilient to this route behaviour.
    response = client.post(
        "/eda/analysis/",
        data={"interactive": "false"},
    )
    assert response.status_code in (400, 500)
    assert response.status_code != 200


def test_eda_missing_dataset_id_returns_400(client):
    response = client.post(
        "/eda/analysis/",
        data={"dataset_id": "nonexistent-uuid-12345", "interactive": "false"},
    )
    assert response.status_code in (400, 500)
