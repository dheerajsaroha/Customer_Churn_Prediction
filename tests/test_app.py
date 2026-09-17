import pytest
import os
import sys

from app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestFlaskApp:

    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Customer Churn" in response.data

    def test_predict_page(self, client):
        response = client.get("/predict")
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        import json
        data = json.loads(response.data)
        assert data["status"] == "ok"

    def test_navbar_links(self, client):
        response = client.get("/")
        assert b"Predict" in response.data
        assert b"Evaluation" in response.data
