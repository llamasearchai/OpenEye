import pytest
from fastapi.testclient import TestClient
from openvideo.main import create_app

@pytest.fixture
def test_client():
    app = create_app()
    return TestClient(app)

def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"} 