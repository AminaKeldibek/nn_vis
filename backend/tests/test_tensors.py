import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.store import store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    store.tensors.clear()
    yield
    store.tensors.clear()


def test_create_tensor_numeric():
    r = client.post("/api/v1/tensors", json={"shape": {"dims": [4, 8]}})
    assert r.status_code == 201
    body = r.json()
    assert body["tensor"]["shape"]["dims"] == [4, 8]
    assert body["validation"]["ok"] is True


def test_create_tensor_symbolic():
    r = client.post("/api/v1/tensors", json={"shape": {"dims": ["m", "n"]}})
    assert r.status_code == 201
    assert r.json()["tensor"]["shape"]["dims"] == ["m", "n"]


def test_create_tensor_0d():
    r = client.post("/api/v1/tensors", json={"shape": {"dims": []}})
    assert r.status_code == 201
    assert r.json()["tensor"]["shape"]["dims"] == []


def test_update_tensor():
    create = client.post("/api/v1/tensors", json={"shape": {"dims": [4, 8]}})
    tid = create.json()["tensor"]["id"]
    r = client.patch(f"/api/v1/tensors/{tid}", json={"shape": {"dims": [4, 16]}})
    assert r.status_code == 200
    assert r.json()["tensor"]["shape"]["dims"] == [4, 16]


def test_delete_tensor():
    create = client.post("/api/v1/tensors", json={"shape": {"dims": [4, 8]}})
    tid = create.json()["tensor"]["id"]
    r = client.delete(f"/api/v1/tensors/{tid}")
    assert r.status_code == 204
    assert tid not in store.tensors


def test_update_missing_tensor():
    r = client.patch("/api/v1/tensors/nope", json={"shape": {"dims": [1]}})
    assert r.status_code == 404
