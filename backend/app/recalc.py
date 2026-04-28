from typing import List
from app.models.tensor import TensorData
from app.models.module import ModuleType
from app.models.common import TensorShape
from app.graph import compute_module_output, compute_broadcast_shape, compute_indexing_output

# Modules whose output_tensor_id lives on a later connection (not the first)
DEFERRED_OUTPUT_MODULES = {ModuleType.mul, ModuleType.sum, ModuleType.index_add, ModuleType.indexing}


def recalculate_downstream(
    tensors: dict,
    modules: dict,
    connections: dict,
    changed_id: str,
) -> List[TensorData]:
    updated: List[TensorData] = []
    queue = [changed_id]
    seen: set = set()

    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)

        for conn in list(connections.values()):
            recompute = False
            src_id = conn.source_id
            tgt_id = conn.target_id

            if conn.source_id == nid and (conn.output_tensor_id or conn.output_tensor_ids):
                if nid in tensors and tgt_id in modules:
                    recompute = True

            elif conn.target_id == nid and (conn.output_tensor_id or conn.output_tensor_ids):
                if src_id in tensors and nid in modules:
                    src_id = conn.source_id
                    recompute = True

            # Input to a deferred-output module changed (no output on this conn)
            elif (conn.source_id == nid and nid in tensors
                  and tgt_id in modules
                  and modules[tgt_id].type in DEFERRED_OUTPUT_MODULES
                  and not conn.output_tensor_id and not conn.output_tensor_ids):
                for out_conn in connections.values():
                    if out_conn.target_id == tgt_id and out_conn.output_tensor_id:
                        if out_conn.source_id not in tensors:
                            continue
                        try:
                            mod_type = modules[tgt_id].type
                            if mod_type in (ModuleType.mul, ModuleType.sum):
                                new_shape = compute_broadcast_shape(
                                    tensors[nid].shape, tensors[out_conn.source_id].shape
                                )
                            elif mod_type == ModuleType.index_add:
                                self_conn = next(
                                    (c for c in connections.values()
                                     if c.target_id == tgt_id and c.target_handle == "self"),
                                    None,
                                )
                                if not self_conn or self_conn.source_id not in tensors:
                                    continue
                                new_shape = tensors[self_conn.source_id].shape
                            elif mod_type == ModuleType.indexing:
                                # out_conn is the "tensor" connection (carries output_tensor_id)
                                if out_conn.target_handle != "tensor" or out_conn.source_id not in tensors:
                                    continue
                                def _gs(h):
                                    c = next((x for x in connections.values() if x.target_id == tgt_id and x.target_handle == h), None)
                                    return tensors[c.source_id].shape if (c and c.source_id in tensors) else None
                                new_shape = compute_indexing_output(tensors[out_conn.source_id].shape, _gs("dim_0_idxs"), _gs("dim_1_idxs"))
                            else:
                                continue
                        except ValueError:
                            continue
                        if out_conn.output_tensor_id in tensors:
                            old = tensors[out_conn.output_tensor_id]
                            new_t = TensorData(id=old.id, shape=new_shape, dtype=old.dtype)
                            tensors[out_conn.output_tensor_id] = new_t
                            updated.append(new_t)
                            if out_conn.output_tensor_id not in seen:
                                queue.append(out_conn.output_tensor_id)
                continue

            if recompute:
                try:
                    mod = modules[tgt_id]
                    if mod.type in (ModuleType.mul, ModuleType.sum):
                        inputs = [c for c in connections.values()
                                  if c.target_id == tgt_id and c.source_id in tensors]
                        if len(inputs) < 2:
                            continue
                        new_shape = compute_broadcast_shape(
                            tensors[inputs[0].source_id].shape,
                            tensors[inputs[1].source_id].shape,
                        )
                    elif mod.type == ModuleType.index_add:
                        self_conn = next(
                            (c for c in connections.values()
                             if c.target_id == tgt_id and c.target_handle == "self"),
                            None,
                        )
                        if not self_conn or self_conn.source_id not in tensors:
                            continue
                        new_shape = tensors[self_conn.source_id].shape
                    elif mod.type == ModuleType.indexing:
                        tensor_conn = next((c for c in connections.values() if c.target_id == tgt_id and c.target_handle == "tensor"), None)
                        if not tensor_conn or tensor_conn.source_id not in tensors:
                            continue
                        def _gs2(h):
                            c = next((x for x in connections.values() if x.target_id == tgt_id and x.target_handle == h), None)
                            return tensors[c.source_id].shape if (c and c.source_id in tensors) else None
                        new_shape = compute_indexing_output(tensors[tensor_conn.source_id].shape, _gs2("dim_0_idxs"), _gs2("dim_1_idxs"))
                    elif mod.type == ModuleType.where:
                        # Output indices are always 1D of unknown length
                        new_shape = TensorShape(dims=["K"])
                    else:
                        new_shape = compute_module_output(mod.type, mod.params, tensors[src_id].shape)
                except (ValueError, KeyError):
                    continue

                if conn.output_tensor_id and conn.output_tensor_id in tensors:
                    old = tensors[conn.output_tensor_id]
                    new_tensor = TensorData(id=old.id, shape=new_shape, dtype=old.dtype)
                    tensors[conn.output_tensor_id] = new_tensor
                    updated.append(new_tensor)
                    queue.append(conn.output_tensor_id)

                for tid in conn.output_tensor_ids:
                    if tid in tensors:
                        old = tensors[tid]
                        new_tensor = TensorData(id=old.id, shape=new_shape, dtype=old.dtype)
                        tensors[tid] = new_tensor
                        updated.append(new_tensor)
                        queue.append(tid)

    return updated
