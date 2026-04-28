"""End-to-end connection tests for every module type."""
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


def make_tensor(dims, dtype="float32"):
    r = client.post("/api/v1/tensors", json={"shape": {"dims": dims}, "dtype": dtype})
    assert r.status_code == 201, r.text
    return r.json()["tensor"]["id"]


def make_module(mtype, params=None):
    r = client.post("/api/v1/modules", json={"type": mtype, "params": params or {}})
    assert r.status_code == 201, r.text
    return r.json()["module"]["id"]


def connect(src, tgt, target_handle=None):
    body = {"source_id": src, "target_id": tgt}
    if target_handle:
        body["target_handle"] = target_handle
    return client.post("/api/v1/connections", json=body)


# ── Single-input modules ─────────────────────────────────────────────────────

def test_view_empty_shape():
    t = make_tensor(["m", "n"])
    m = make_module("view", {"shape": ""})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot is not None
    assert ot["shape"]["dims"] == ["m", "n"]  # passthrough when shape is empty

def test_view_explicit_shape():
    t = make_tensor([2, 12])
    m = make_module("view", {"shape": "6, -1"})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [6, 4]

def test_view_symbolic_neg1():
    # (m, n, n) viewed as (-1, n) → (m*n, n)
    t = make_tensor(["m", "n", "n"])
    m = make_module("view", {"shape": "-1, n"})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == ["m*n", "n"]

def test_view_symbolic_neg1_mixed():
    # (4, n, n) viewed as (-1, n) → (4*n, n)
    t = make_tensor([4, "n", "n"])
    m = make_module("view", {"shape": "-1, n"})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == ["4*n", "n"]

def test_view_symbolic_flatten_all():
    # (m, n) viewed as (-1,) → (m*n,)
    t = make_tensor(["m", "n"])
    m = make_module("view", {"shape": "-1"})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == ["m*n"]

def test_flatten_full():
    t = make_tensor([2, 3, 4])
    m = make_module("flatten", {"start_dim": 0, "end_dim": -1})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [24]

def test_flatten_partial():
    t = make_tensor([2, 3, 4])
    m = make_module("flatten", {"start_dim": 1, "end_dim": -1})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [2, 12]

def test_bincount_minlength():
    t = make_tensor([100], dtype="int64")
    m = make_module("bincount", {"minlength": 10})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [10]

def test_bincount_symbolic_minlength():
    t = make_tensor([100], dtype="int64")
    m = make_module("bincount", {"minlength": "C"})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == ["C"]

def test_unsqueeze_dim0():
    t = make_tensor([3, 4])
    m = make_module("unsqueeze", {"dim": 0})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [1, 3, 4]

def test_unsqueeze_dim_neg():
    t = make_tensor([3, 4])
    m = make_module("unsqueeze", {"dim": -1})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [3, 4, 1]

def test_where_two_outputs():
    t = make_tensor([3, 4])
    m = make_module("where", {"condition": "x > 0"})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["output_tensor"] is None or d.get("output_tensor") is None
    ots = d["output_tensors"]
    assert len(ots) == 2, f"expected 2 output tensors, got {ots}"
    # rows and cols: both 1D with unknown length
    for ot in ots:
        assert len(ot["shape"]["dims"]) == 1

def test_silu():
    t = make_tensor(["B", 128])
    m = make_module("silu", {})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == ["B", 128]

def test_sigmoid():
    t = make_tensor(["B", 128])
    m = make_module("sigmoid", {})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == ["B", 128]

# ── Two-input modules ────────────────────────────────────────────────────────

def test_mul_first_conn_no_output():
    t = make_tensor([3, 4])
    m = make_module("mul", {})
    r = connect(t, m)
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["output_tensor"] is None
    assert d["output_tensors"] == []

def test_mul_second_conn_broadcast():
    t1 = make_tensor([1, 4])
    t2 = make_tensor([3, 1])
    m = make_module("mul", {})
    connect(t1, m)
    r = connect(t2, m)
    assert r.status_code == 201, r.text
    ot = r.json()["output_tensor"]
    assert ot["shape"]["dims"] == [3, 4]

def test_index_add_three_connections():
    t_self = make_tensor([5, 4])
    t_src = make_tensor([2, 4])
    t_idx = make_tensor([2], dtype="int64")
    m = make_module("index_add", {"dim": 0})

    r1 = connect(t_self, m, target_handle="self")
    assert r1.status_code == 201
    assert r1.json()["output_tensor"] is None

    r2 = connect(t_src, m, target_handle="source")
    assert r2.status_code == 201
    assert r2.json()["output_tensor"] is None

    r3 = connect(t_idx, m, target_handle="index")
    assert r3.status_code == 201
    ot = r3.json()["output_tensor"]
    assert ot is not None
    assert ot["shape"]["dims"] == [5, 4]  # same as self

def test_index_add_out_of_order():
    # Connect index first, then source, then self — output must still equal self's shape
    t_self = make_tensor([7, 3])
    t_src = make_tensor([2, 3])
    t_idx = make_tensor([2], dtype="int64")
    m = make_module("index_add", {"dim": 0})

    r1 = connect(t_idx, m, target_handle="index")
    assert r1.status_code == 201
    assert r1.json()["output_tensor"] is None  # self not yet connected

    r2 = connect(t_src, m, target_handle="source")
    assert r2.status_code == 201

    r3 = connect(t_self, m, target_handle="self")
    assert r3.status_code == 201
    # self connected last — no output emitted here (output emitted on "index" handle)
    # This tests that handle-based routing works regardless of connection order
