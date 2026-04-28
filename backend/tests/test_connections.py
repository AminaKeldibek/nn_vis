import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.store import store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    store.tensors.clear()
    store.modules.clear()
    store.connections.clear()
    yield
    store.tensors.clear()
    store.modules.clear()
    store.connections.clear()


def make_tensor(dims):
    r = client.post("/api/v1/tensors", json={"shape": {"dims": dims}})
    assert r.status_code == 201
    return r.json()["tensor"]["id"]


def make_linear(n_out):
    r = client.post("/api/v1/modules", json={"type": "linear", "params": {"n_out": n_out}})
    assert r.status_code == 201
    return r.json()["module"]["id"]


def connect(src, tgt):
    return client.post("/api/v1/connections", json={"source_id": src, "target_id": tgt})


def test_tensor_to_linear_numeric():
    t = make_tensor(["batch", 128])
    l = make_linear(64)
    r = connect(t, l)
    assert r.status_code == 201
    body = r.json()
    assert body["output_tensor"]["shape"]["dims"] == ["batch", 64]
    assert body["validation"]["ok"] is True


def test_tensor_to_linear_symbolic():
    t = make_tensor(["m", "n"])
    l = make_linear("d_out")
    r = connect(t, l)
    assert r.status_code == 201
    assert r.json()["output_tensor"]["shape"]["dims"] == ["m", "d_out"]


def test_output_tensor_stored():
    t = make_tensor([4, 8])
    l = make_linear(16)
    r = connect(t, l)
    out_id = r.json()["output_tensor"]["id"]
    assert out_id in store.tensors
    assert store.tensors[out_id].shape.dims == [4, 16]


def test_missing_source():
    l = make_linear(64)
    r = connect("nope", l)
    assert r.status_code == 404


def test_missing_target():
    t = make_tensor([4, 8])
    r = connect(t, "nope")
    assert r.status_code == 404


def test_cycle_detection():
    # T1 → L1 → T2 → L2 → T3 → L1 (cycle)
    t1 = make_tensor(["batch", 64])
    l1 = make_linear(32)
    r = connect(t1, l1)
    t2 = r.json()["output_tensor"]["id"]

    l2 = make_linear(16)
    r = connect(t2, l2)
    t3 = r.json()["output_tensor"]["id"]

    r = connect(t3, l1)
    assert r.status_code == 422
    errors = r.json()["detail"]["errors"]
    assert any(e["code"] == "CYCLE" for e in errors)


def test_delete_connection_cascades_output_tensor():
    t = make_tensor([4, 8])
    l = make_linear(16)
    r = connect(t, l)
    cid = r.json()["connection"]["id"]
    out_id = r.json()["output_tensor"]["id"]

    d = client.delete(f"/api/v1/connections/{cid}")
    assert d.status_code == 204
    assert cid not in store.connections
    assert out_id not in store.tensors


def test_chained_linear():
    # Input → L1 → T2 → L2 → T3  (2-layer MLP shape flow)
    t1 = make_tensor(["batch", 784])
    l1 = make_linear(256)
    t2 = connect(t1, l1).json()["output_tensor"]["id"]

    l2 = make_linear(10)
    r = connect(t2, l2)
    assert r.status_code == 201
    assert r.json()["output_tensor"]["shape"]["dims"] == ["batch", 10]
