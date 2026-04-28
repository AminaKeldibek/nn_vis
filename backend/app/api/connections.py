from fastapi import APIRouter, HTTPException
from app.models.connection import ConnectionCreate, ConnectionData, ConnectionResponse
from app.models.module import ModuleType
from app.models.tensor import TensorData
from app.models.common import ValidationResult, TensorShape
from app.graph import compute_module_output, compute_broadcast_shape, compute_indexing_output, validate_graph
from app.store import store, new_id, save_store

router = APIRouter()


@router.post("/connections", response_model=ConnectionResponse, status_code=201)
def create_connection(body: ConnectionCreate):
    src, tgt = body.source_id, body.target_id

    src_is_tensor = src in store.tensors
    src_is_module = src in store.modules
    tgt_is_module = tgt in store.modules
    tgt_is_tensor = tgt in store.tensors

    if not src_is_tensor and not src_is_module:
        raise HTTPException(404, f"Source {src} not found")
    if not tgt_is_module and not tgt_is_tensor:
        raise HTTPException(404, f"Target {tgt} not found")

    validation = validate_graph(store.connections, proposed=(src, tgt))
    if not validation.ok:
        raise HTTPException(
            422,
            detail={"errors": [e.model_dump() for e in validation.errors]},
        )

    output_tensor = None
    output_tensor_id = None
    output_tensors: list[TensorData] = []
    output_tensor_ids: list[str] = []

    if src_is_tensor and tgt_is_module:
        source_tensor = store.tensors[src]
        target_module = store.modules[tgt]

        if target_module.type == ModuleType.topk:
            try:
                output_shape = compute_module_output(
                    target_module.type, target_module.params, source_tensor.shape
                )
            except ValueError as e:
                raise HTTPException(422, detail={"errors": [{"code": "SHAPE_ERROR", "message": str(e)}]})
            vals_id, idx_id = new_id(), new_id()
            values_tensor = TensorData(id=vals_id, shape=output_shape, dtype=source_tensor.dtype)
            indices_tensor = TensorData(id=idx_id, shape=output_shape, dtype="int64")
            store.tensors[vals_id] = values_tensor
            store.tensors[idx_id] = indices_tensor
            output_tensors = [values_tensor, indices_tensor]
            output_tensor_ids = [vals_id, idx_id]

        elif target_module.type == ModuleType.where:
            # torch.where(condition) → two 1D index tensors (rows, cols), length unknown
            rows_id, cols_id = new_id(), new_id()
            k_shape = TensorShape(dims=["K"])
            rows_t = TensorData(id=rows_id, shape=k_shape, dtype="int64")
            cols_t = TensorData(id=cols_id, shape=k_shape, dtype="int64")
            store.tensors[rows_id] = rows_t
            store.tensors[cols_id] = cols_t
            output_tensors = [rows_t, cols_t]
            output_tensor_ids = [rows_id, cols_id]

        elif target_module.type in (ModuleType.mul, ModuleType.sum):
            # First connection: no output; second connection: broadcast both inputs
            existing = [c for c in store.connections.values()
                        if c.target_id == tgt and c.source_id in store.tensors]
            if existing:
                first_tensor = store.tensors[existing[0].source_id]
                try:
                    out_shape = compute_broadcast_shape(first_tensor.shape, source_tensor.shape)
                except ValueError as e:
                    raise HTTPException(422, detail={"errors": [{"code": "SHAPE_ERROR", "message": str(e)}]})
                tid = new_id()
                output_tensor = TensorData(id=tid, shape=out_shape, dtype=first_tensor.dtype)
                store.tensors[tid] = output_tensor
                output_tensor_id = tid

        elif target_module.type == ModuleType.index_add:
            # Emit output only when "index" handle is connected; output shape = self's shape
            if body.target_handle == "index":
                self_conn = next(
                    (c for c in store.connections.values()
                     if c.target_id == tgt and c.target_handle == "self"),
                    None,
                )
                if self_conn and self_conn.source_id in store.tensors:
                    self_tensor = store.tensors[self_conn.source_id]
                    tid = new_id()
                    output_tensor = TensorData(id=tid, shape=self_tensor.shape, dtype=self_tensor.dtype)
                    store.tensors[tid] = output_tensor
                    output_tensor_id = tid

        elif target_module.type == ModuleType.indexing:
            # Emit output immediately when "tensor" handle is connected.
            # dim_0_idxs / dim_1_idxs are optional; recalc updates the output when they change.
            if body.target_handle == "tensor":
                def _get_shape(handle):
                    c = next((x for x in store.connections.values()
                               if x.target_id == tgt and x.target_handle == handle), None)
                    return store.tensors[c.source_id].shape if (c and c.source_id in store.tensors) else None
                output_shape = compute_indexing_output(
                    source_tensor.shape,
                    _get_shape("dim_0_idxs"),
                    _get_shape("dim_1_idxs"),
                )
                tid = new_id()
                output_tensor = TensorData(id=tid, shape=output_shape, dtype=source_tensor.dtype)
                store.tensors[tid] = output_tensor
                output_tensor_id = tid

        else:
            try:
                output_shape = compute_module_output(
                    target_module.type, target_module.params, source_tensor.shape
                )
            except ValueError as e:
                raise HTTPException(422, detail={"errors": [{"code": "SHAPE_ERROR", "message": str(e)}]})
            tid = new_id()
            output_tensor = TensorData(id=tid, shape=output_shape, dtype=source_tensor.dtype)
            store.tensors[tid] = output_tensor
            output_tensor_id = tid

    cid = new_id()
    connection = ConnectionData(
        id=cid,
        source_id=src,
        target_id=tgt,
        target_handle=body.target_handle,
        output_tensor_id=output_tensor_id,
        output_tensor_ids=output_tensor_ids,
    )
    store.connections[cid] = connection
    save_store()

    return ConnectionResponse(
        connection=connection,
        output_tensor=output_tensor,
        output_tensors=output_tensors,
        validation=ValidationResult(ok=True),
    )


@router.delete("/connections/{cid}", status_code=204)
def delete_connection(cid: str):
    if cid not in store.connections:
        raise HTTPException(404, "Connection not found")
    conn = store.connections[cid]
    if conn.output_tensor_id and conn.output_tensor_id in store.tensors:
        del store.tensors[conn.output_tensor_id]
    for tid in conn.output_tensor_ids:
        if tid in store.tensors:
            del store.tensors[tid]
    del store.connections[cid]
    save_store()
