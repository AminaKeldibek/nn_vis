from fastapi import APIRouter, HTTPException
from app.models.tensor import TensorCreate, TensorUpdate, TensorData, TensorResponse
from app.models.common import ValidationResult
from app.store import store, new_id, save_store
from app.recalc import recalculate_downstream

router = APIRouter()


def _ok() -> ValidationResult:
    return ValidationResult(ok=True)


@router.post("/tensors", response_model=TensorResponse, status_code=201)
def create_tensor(body: TensorCreate):
    tid = new_id()
    data = TensorData(id=tid, shape=body.shape, dtype=body.dtype)
    store.tensors[tid] = data
    save_store()
    return TensorResponse(tensor=data, validation=_ok())


@router.patch("/tensors/{tid}", response_model=TensorResponse)
def update_tensor(tid: str, body: TensorUpdate):
    if tid not in store.tensors:
        raise HTTPException(404, "Tensor not found")
    data: TensorData = store.tensors[tid]
    if body.shape is not None:
        data = data.model_copy(update={"shape": body.shape})
    if body.dtype is not None:
        data = data.model_copy(update={"dtype": body.dtype})
    store.tensors[tid] = data
    updated = recalculate_downstream(store.tensors, store.modules, store.connections, tid)
    save_store()
    return TensorResponse(tensor=data, updated_tensors=updated, validation=_ok())


@router.delete("/tensors/{tid}", status_code=204)
def delete_tensor(tid: str):
    if tid not in store.tensors:
        raise HTTPException(404, "Tensor not found")
    del store.tensors[tid]
    save_store()
