"""
Integration tests: POST /preprocessing/preprocess/
Uses the minimal TestClient fixture from conftest.py.
No external services required. Uses small synthetic CSV data.
"""
import io


def _make_csv(rows: int = 60) -> bytes:
    lines = ["feature1,feature2,feature3,target"]
    for i in range(rows):
        lines.append(f"{i * 0.1:.2f},{i * 0.2:.2f},{i % 5},{i % 2}")
    return "\n".join(lines).encode()


def test_preprocess_valid_csv_returns_200(client):
    response = client.post(
        "/preprocessing/preprocess/",
        files={"file": ("synthetic.csv", io.BytesIO(_make_csv(60)), "text/csv")},
    )
    assert response.status_code == 200


def test_preprocess_response_has_required_keys(client):
    response = client.post(
        "/preprocessing/preprocess/",
        files={"file": ("synthetic.csv", io.BytesIO(_make_csv(60)), "text/csv")},
    )
    data = response.json()
    assert data["status"] == "success"
    assert "dataset_id" in data
    assert "raw_dataset_id" in data
    assert "shape" in data


def test_preprocess_shape_is_non_empty(client):
    response = client.post(
        "/preprocessing/preprocess/",
        files={"file": ("synthetic.csv", io.BytesIO(_make_csv(60)), "text/csv")},
    )
    data = response.json()
    rows, cols = data["shape"]
    assert rows > 0
    assert cols > 0


def test_preprocess_no_file_returns_422(client):
    response = client.post("/preprocessing/preprocess/")
    assert response.status_code == 422
