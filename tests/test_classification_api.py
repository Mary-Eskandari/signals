import json

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from pipeline.chamber_dataset import BEATS_PATH
from pipeline.classification_models import MODEL_REGISTRY

client = TestClient(app)

pytestmark = pytest.mark.skipif(not BEATS_PATH.exists(), reason="chamber dataset not built yet")


def test_classification_status():
    response = client.get("/classification/status")
    assert response.status_code == 200
    status = response.json()
    assert status["n_beats"] > 0
    assert status["total_records"] == 83
    assert set(status["per_chamber"].keys()).issubset({"RA", "RV", "PA", "PCW"})


def test_classification_models_and_hyperparameters():
    response = client.get("/classification/models")
    assert response.status_code == 200
    body = response.json()
    assert set(body["models"].keys()) == set(MODEL_REGISTRY.keys())
    assert "cnn" in body["hyperparameters"]
    assert any(h["name"] == "epochs" for h in body["hyperparameters"]["cnn"])


def test_classification_labels():
    response = client.get("/classification/labels")
    assert response.status_code == 200
    assert response.json() == ["RA", "RV", "PA", "PCW"]


def test_classification_records_nonempty():
    response = client.get("/classification/records")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_train_classic_model_auto_split():
    response = client.post("/classification/train", json={"model": "random_forest", "split": {"test_size": 0.3}})
    assert response.status_code == 200
    result = response.json()
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n_train_beats"] > 0 and result["n_test_beats"] > 0
    assert result["hyperparameters"]["n_estimators"] == 200  # default


def test_train_with_hyperparameter_override():
    response = client.post(
        "/classification/train",
        json={"model": "random_forest", "hyperparameters": {"n_estimators": 20}},
    )
    assert response.status_code == 200
    assert response.json()["hyperparameters"]["n_estimators"] == 20


def test_train_with_cv():
    response = client.post("/classification/train", json={"model": "logistic_regression", "cv_folds": 3})
    assert response.status_code == 200
    result = response.json()
    assert len(result["cv_results"]) == 3
    assert 0.0 <= result["mean_accuracy"] <= 1.0


def test_train_manual_split_requires_both_lists():
    response = client.post(
        "/classification/train",
        json={"model": "decision_tree", "split": {"mode": "manual", "train_record_ids": ["TRM278-RHC1"]}},
    )
    assert response.status_code == 422


def test_train_rejects_unknown_model():
    response = client.post("/classification/train", json={"model": "not_a_real_model"})
    assert response.status_code == 422


def test_train_stream_yields_epoch_events_then_result():
    with client.stream(
        "POST",
        "/classification/train/stream",
        json={"model": "cnn", "hyperparameters": {"epochs": 3}},
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]

    types = [e["type"] for e in events]
    assert types.count("epoch") == 3
    assert types[-1] == "result"
    assert 0.0 <= events[-1]["data"]["accuracy"] <= 1.0
