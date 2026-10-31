import uuid

import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture
def client(temp_db):
    with TestClient(app) as c:
        yield c


def test_1_health_after_startup(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_2_valid_predict(client, valid_payload):
    r = client.post("/v1/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert {"request_id", "score", "label", "uncertainty", "meta"} <= body.keys()
    uuid.UUID(body["request_id"])
    assert 0 <= body["score"] <= 1 and body["label"] in (0, 1) and body["uncertainty"] >= 0
    assert body["meta"]["fallback_applied"] is False and body["meta"]["ood_log_density"] is not None


def test_3_protected_attribute_rejected(client, valid_payload):
    for attr, value in (("sex", "male"), ("personal_status_sex", "male: single"), ("foreign_worker", "yes")):
        r = client.post("/v1/predict", json={**valid_payload, attr: value})
        assert r.status_code == 422, attr
        assert r.json()["error_code"] == "validation_error"


def test_4_string_instead_of_number(client, valid_payload):
    r = client.post("/v1/predict", json={**valid_payload, "age": "35"})
    assert r.status_code == 422


def test_5_out_of_field_bounds(client, valid_payload):
    for field, value in (("duration", 0), ("credit_amount", 60000.0), ("age", 17), ("installment_rate", 5)):
        r = client.post("/v1/predict", json={**valid_payload, field: value})
        assert r.status_code == 422, field


def test_6_ood_fallback(client, ood_payload):
    r = client.post("/v1/predict", json=ood_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["fallback_applied"] is True and body["meta"]["segment"] is None


def test_7_batch_over_limit(client, valid_payload):
    r = client.post("/v1/predict_batch", json={"items": [valid_payload] * 257})
    assert r.status_code == 422
    r = client.post("/v1/predict_batch", json={"items": [valid_payload] * 3})
    assert r.status_code == 200 and len(r.json()["results"]) == 3


def test_8_feedback_unknown_request_id(client):
    r = client.post("/v1/feedback", json={"request_id": str(uuid.uuid4()), "y_true": 1})
    assert r.status_code == 404 and r.json()["error_code"] == "unknown_request_id"


def test_feedback_roundtrip_and_error_format(client, valid_payload):
    rid = client.post("/v1/predict", json=valid_payload).json()["request_id"]
    r = client.post("/v1/feedback", json={"request_id": rid, "y_true": 0})
    assert r.status_code == 200 and r.json()["stored"] is True
    r = client.post("/v1/feedback", json={"request_id": rid, "y_true": 2})
    assert r.status_code == 422 and set(r.json()) >= {"error_code", "message"}


def test_explain_shape(client, valid_payload):
    r = client.post("/v1/explain", json={"request": valid_payload, "lime_samples": 300})
    assert r.status_code == 200
    body = r.json()
    assert {"values", "base_value", "data", "feature_names"} <= body["shap"].keys()
    assert len(body["shap"]["values"]) == len(body["shap"]["feature_names"])
    assert body["lime"] and "top5_overlap" in body["agreement"]
    assert abs(body["shap"]["base_value"] + sum(body["shap"]["values"]) - body["prediction"]["score"]) < 1e-3


def test_health_503_with_broken_meta(tmp_path, monkeypatch, temp_db):
    import json
    import shutil
    from pathlib import Path

    src = Path("models")
    broken = tmp_path / "models"
    shutil.copytree(src, broken)
    meta = json.loads((broken / "model_meta.json").read_text(encoding="utf-8"))
    meta["features"] = meta["features"][::-1]
    (broken / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr("src.app.main.ROOT", tmp_path)
    import src.app.main as m

    monkeypatch.setattr(m, "resolve_artifacts", lambda: {"dir": broken, "onnx": broken / "pipeline.onnx", "version": "broken", "run_id": None})
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503 and "features" in " ".join(r.json()["errors"])
