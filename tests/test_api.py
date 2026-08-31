import pytest
from api.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "city" in data


def test_missing_route_returns_safe_json(client):
    response = client.get("/missing")
    assert response.status_code == 404
    data = response.get_json()
    assert data["error"] == "resource not found"
    assert "Traceback" not in response.get_data(as_text=True)


def test_forecast_endpoint(client):
    response = client.get("/forecast")
    assert response.status_code == 200
    data = response.get_json()
    assert data["city"] == "Islamabad"
    assert "forecast" in data
    assert "1d" in data["forecast"]
    assert "2d" in data["forecast"]
    assert "3d" in data["forecast"]


def test_history_endpoint(client):
    response = client.get("/history")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
