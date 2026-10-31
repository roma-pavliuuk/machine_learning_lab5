import pytest
from fastapi.testclient import TestClient

from src.app.main import app


@pytest.fixture
def client(temp_db):
    with TestClient(app) as c:
        yield c


def test_same_request_same_segment(client, valid_payload):
    segs = [client.post("/v1/predict", json=valid_payload).json()["meta"]["segment"] for _ in range(5)]
    assert all(s == segs[0] for s in segs)
    assert segs[0] is not None and {"id", "name"} <= segs[0].keys() and segs[0]["name"] != f"сегмент {segs[0]['id']}"


def test_ood_request_has_no_segment(client, ood_payload):
    body = client.post("/v1/predict", json=ood_payload).json()
    assert body["meta"]["fallback_applied"] is True and body["meta"]["segment"] is None
    assert body["meta"]["ood_log_density"] is not None
