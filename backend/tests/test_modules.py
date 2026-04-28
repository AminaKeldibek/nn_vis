import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.store import store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    store.modules.clear()
    yield
    store.modules.clear()


def test_create_linear_numeric():
    r = client.post("/api/v1/modules", json={"type": "linear", "params": {"n_out": 128}})
    assert r.status_code == 201
    body = r.json()
    assert body["module"]["type"] == "linear"
    assert body["module"]["params"]["n_out"] == 128
    assert body["validation"]["ok"] is True


def test_create_linear_symbolic():
    r = client.post("/api/v1/modules", json={"type": "linear", "params": {"n_out": "d_model"}})
    assert r.status_code == 201
    assert r.json()["module"]["params"]["n_out"] == "d_model"


def test_update_module():
    create = client.post("/api/v1/modules", json={"type": "linear", "params": {"n_out": 64}})
    mid = create.json()["module"]["id"]
    r = client.patch(f"/api/v1/modules/{mid}", json={"params": {"n_out": 256}})
    assert r.status_code == 200
    assert r.json()["module"]["params"]["n_out"] == 256


def test_delete_module():
    create = client.post("/api/v1/modules", json={"type": "linear", "params": {"n_out": 64}})
    mid = create.json()["module"]["id"]
    r = client.delete(f"/api/v1/modules/{mid}")
    assert r.status_code == 204
    assert mid not in store.modules


def test_delete_missing():
    r = client.delete("/api/v1/modules/nope")
    assert r.status_code == 404
