from fastapi import APIRouter, HTTPException
from app.models.module import ModuleCreate, ModuleUpdate, ModuleData, ModuleResponse
from app.models.common import ValidationResult
from app.store import store, new_id, save_store
from app.recalc import recalculate_downstream

router = APIRouter()


def _ok() -> ValidationResult:
    return ValidationResult(ok=True)


@router.post("/modules", response_model=ModuleResponse, status_code=201)
def create_module(body: ModuleCreate):
    mid = new_id()
    data = ModuleData(id=mid, type=body.type, params=body.params, input_ids=body.input_ids)
    store.modules[mid] = data
    save_store()
    return ModuleResponse(module=data, validation=_ok())


@router.patch("/modules/{mid}", response_model=ModuleResponse)
def update_module(mid: str, body: ModuleUpdate):
    if mid not in store.modules:
        raise HTTPException(404, "Module not found")
    data: ModuleData = store.modules[mid]
    data = data.model_copy(update={"params": body.params})
    store.modules[mid] = data
    updated = recalculate_downstream(store.tensors, store.modules, store.connections, mid)
    save_store()
    return ModuleResponse(module=data, updated_tensors=updated, validation=_ok())


@router.delete("/modules/{mid}", status_code=204)
def delete_module(mid: str):
    if mid not in store.modules:
        raise HTTPException(404, "Module not found")
    del store.modules[mid]
    save_store()
