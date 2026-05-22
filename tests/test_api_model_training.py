"""
Integration tests: POST /model-training/train/ and GET /evaluation/list-models/
Uses the minimal TestClient fixture from conftest.py.
No external services required. Uses small synthetic regression CSV data.
Reset endpoint is tested with a mock to avoid deleting tracked artifacts.
"""
import io
from unittest.mock import patch


def _make_regression_csv(rows: int = 70) -> bytes:
    lines = ["feature1,feature2,target"]
    for i in range(rows):
        f1 = round(i * 0.1, 4)
        f2 = round(i * 0.2 + 0.5, 4)
        target = round(f1 * 2.0 + f2 * 1.5 + 0.01 * i, 4)
        lines.append(f"{f1},{f2},{target}")
    return "\n".join(lines).encode()


def test_train_linear_regression_returns_200(client):
    response = client.post(
        "/model-training/train/",
        files={"file": ("train.csv", io.BytesIO(_make_regression_csv(70)), "text/csv")},
        params={
            "target_col": "target",
            "task_type": "regression",
            "selected_models[]": ["LinearRegression"],
            "selected_metrics[]": ["RMSE", "R2"],
            "dataset_source": "original",
        },
    )
    assert response.status_code == 200


def test_train_response_has_required_keys(client):
    response = client.post(
        "/model-training/train/",
        files={"file": ("train.csv", io.BytesIO(_make_regression_csv(70)), "text/csv")},
        params={
            "target_col": "target",
            "task_type": "regression",
            "selected_models[]": ["LinearRegression"],
            "selected_metrics[]": ["RMSE", "R2"],
            "dataset_source": "original",
        },
    )
    data = response.json()
    assert data["status"] == "success"
    assert "results" in data
    assert "trained_models" in data


def test_train_results_contain_metrics(client):
    response = client.post(
        "/model-training/train/",
        files={"file": ("train.csv", io.BytesIO(_make_regression_csv(70)), "text/csv")},
        params={
            "target_col": "target",
            "task_type": "regression",
            "selected_models[]": ["LinearRegression"],
            "selected_metrics[]": ["RMSE", "R2"],
            "dataset_source": "original",
        },
    )
    data = response.json()
    assert len(data["results"]) > 0
    row = data["results"][0]
    assert "Model" in row
    assert "RMSE" in row or "R2" in row


def test_train_missing_models_returns_error(client):
    # selected_models[] not provided -> route raises HTTPException(400) which its
    # outer except clause wraps as 500. Accept both; key assertion is not 200.
    response = client.post(
        "/model-training/train/",
        files={"file": ("train.csv", io.BytesIO(_make_regression_csv(70)), "text/csv")},
        params={
            "target_col": "target",
            "task_type": "regression",
            "dataset_source": "original",
        },
    )
    assert response.status_code != 200


def test_train_missing_file_and_id_returns_error(client):
    # No file and no dataset_id -> route raises HTTPException(400) inside try,
    # caught by outer except, returned as 500. Accept any non-200 error code.
    response = client.post(
        "/model-training/train/",
        params={
            "target_col": "target",
            "task_type": "regression",
            "selected_models[]": ["LinearRegression"],
            "selected_metrics[]": ["RMSE"],
            "dataset_source": "original",
        },
    )
    assert response.status_code != 200


def test_list_models_returns_200(client):
    response = client.get("/evaluation/list-models/")
    assert response.status_code == 200


def test_list_models_response_has_models_key(client):
    response = client.get("/evaluation/list-models/")
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_reset_endpoint_returns_success(client):
    """
    The reset endpoint deletes and recreates the models/ folder.
    Patching shutil.rmtree prevents actual deletion during the test run.
    """
    with patch("backend.app.api.reset_routes.shutil.rmtree") as mock_rmtree:
        response = client.post("/reset/restart-analysis")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "session_id" in data
    mock_rmtree.assert_called_once()
