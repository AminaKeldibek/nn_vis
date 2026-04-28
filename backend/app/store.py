import uuid
import json
from pathlib import Path
from typing import Dict, Any

STORE_FILE = Path(__file__).parent.parent / "store_data.json"


def new_id() -> str:
    return str(uuid.uuid4())[:8]


class GraphStore:
    def __init__(self):
        self.tensors: Dict[str, Any] = {}
        self.modules: Dict[str, Any] = {}
        self.connections: Dict[str, Any] = {}
        self.groups: Dict[str, Any] = {}


store = GraphStore()


def save_store() -> None:
    data = {
        "tensors": {k: v.model_dump() for k, v in store.tensors.items()},
        "modules": {k: v.model_dump() for k, v in store.modules.items()},
        "connections": {k: v.model_dump() for k, v in store.connections.items()},
        "groups": {k: v.model_dump() for k, v in store.groups.items()},
    }
    STORE_FILE.write_text(json.dumps(data))


def load_store() -> None:
    if not STORE_FILE.exists():
        return
    try:
        from app.models.tensor import TensorData
        from app.models.module import ModuleData
        from app.models.connection import ConnectionData
        from app.models.group import GroupData
        raw = json.loads(STORE_FILE.read_text())
        store.tensors = {k: TensorData(**v) for k, v in raw.get("tensors", {}).items()}
        store.modules = {k: ModuleData(**v) for k, v in raw.get("modules", {}).items()}
        store.connections = {k: ConnectionData(**v) for k, v in raw.get("connections", {}).items()}
        store.groups = {k: GroupData(**v) for k, v in raw.get("groups", {}).items()}
    except Exception:
        import traceback
        traceback.print_exc()


load_store()
