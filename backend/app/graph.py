from typing import Dict, List, Set, Optional, Tuple
from app.models.common import TensorShape, ValidationResult, ValidationError
from app.models.module import ModuleType


def compute_broadcast_shape(shape_a: TensorShape, shape_b: TensorShape) -> TensorShape:
    dims_a = list(shape_a.dims)
    dims_b = list(shape_b.dims)
    while len(dims_a) < len(dims_b):
        dims_a.insert(0, 1)
    while len(dims_b) < len(dims_a):
        dims_b.insert(0, 1)
    result = []
    for a, b in zip(dims_a, dims_b):
        if isinstance(a, int) and isinstance(b, int):
            if a != b and a != 1 and b != 1:
                raise ValueError(f"Shapes not broadcastable: {shape_a.dims} vs {shape_b.dims}")
            result.append(max(a, b))
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            result.append(a)
    return TensorShape(dims=result)


def _factorize(dims) -> Optional[Tuple[int, Dict[str, int]]]:
    """Return (int_factor, {symbol: exponent}) for a list of dims, or None on failure."""
    int_factor = 1
    sym_counts: Dict[str, int] = {}
    for d in dims:
        if isinstance(d, int):
            int_factor *= d
        elif isinstance(d, str):
            sym_counts[d] = sym_counts.get(d, 0) + 1
        else:
            return None
    return int_factor, sym_counts


def _infer_neg_dim(input_dims, output_dims, neg_idx) -> Optional[object]:
    """Infer the -1 dim in view with symbolic support.

    Factorises input and non-(-1) output dims into int * symbols^exp, then
    cancels to produce the inferred dimension (int or symbolic string).
    """
    inp = _factorize(input_dims)
    if inp is None:
        return None
    inp_int, inp_syms = inp

    other_dims = [d for i, d in enumerate(output_dims) if i != neg_idx]
    other = _factorize(other_dims)
    if other is None:
        return None
    other_int, other_syms = other

    if other_int == 0:
        return None
    if inp_int % other_int != 0:
        return None

    result_int = inp_int // other_int
    result_syms = dict(inp_syms)
    for sym, count in other_syms.items():
        new_exp = result_syms.get(sym, 0) - count
        if new_exp < 0:
            return None  # symbol appears more in denominator — can't simplify
        if new_exp == 0:
            result_syms.pop(sym, None)
        else:
            result_syms[sym] = new_exp

    if not result_syms:
        return result_int

    parts: list = []
    if result_int != 1:
        parts.append(str(result_int))
    for sym in sorted(result_syms):
        parts.extend([sym] * result_syms[sym])
    return "*".join(parts)


def compute_view_output(params: dict, input_shape: TensorShape) -> TensorShape:
    shape_str = str(params.get("shape", "")).strip()
    if not shape_str:
        return TensorShape(dims=list(input_shape.dims))
    parts = [s.strip() for s in shape_str.split(",") if s.strip()]
    if not parts:
        return TensorShape(dims=list(input_shape.dims))
    new_dims: list = []
    neg_idx = None
    for i, p in enumerate(parts):
        try:
            v = int(p)
            if v == -1:
                neg_idx = i
            new_dims.append(v)
        except ValueError:
            new_dims.append(p)

    if neg_idx is not None:
        inferred = _infer_neg_dim(input_shape.dims, new_dims, neg_idx)
        if inferred is not None:
            new_dims[neg_idx] = inferred

    return TensorShape(dims=new_dims)


def compute_flatten_output(params: dict, input_shape: TensorShape) -> TensorShape:
    dims = list(input_shape.dims)
    ndim = len(dims)
    if ndim == 0:
        return TensorShape(dims=[1])

    start_dim = params.get("start_dim", 0)
    end_dim = params.get("end_dim", -1)
    try:
        sd = int(start_dim)
        ed = int(end_dim)
    except (ValueError, TypeError):
        return TensorShape(dims=["N"])

    if sd < 0:
        sd = ndim + sd
    if ed < 0:
        ed = ndim + ed
    sd = max(0, min(sd, ndim - 1))
    ed = max(0, min(ed, ndim - 1))
    if sd > ed:
        return TensorShape(dims=dims)

    product = 1
    for d in dims[sd:ed + 1]:
        if isinstance(d, int):
            product *= d
        else:
            return TensorShape(dims=dims[:sd] + ["N"] + dims[ed + 1:])

    return TensorShape(dims=dims[:sd] + [product] + dims[ed + 1:])


def compute_bincount_output(params: dict) -> TensorShape:
    minlength = params.get("minlength", 0)
    if isinstance(minlength, str) and not minlength.lstrip("-").isdigit():
        # Symbolic value like "C" — pass through as dimension name
        return TensorShape(dims=[minlength if minlength else "N"])
    try:
        ml = int(minlength)
    except (ValueError, TypeError):
        ml = 0
    return TensorShape(dims=[ml if ml > 0 else "N"])


def compute_unsqueeze_output(params: dict, input_shape: TensorShape) -> TensorShape:
    dims = list(input_shape.dims)
    ndim = len(dims)
    dim = params.get("dim", 0)
    try:
        d = int(dim)
    except (ValueError, TypeError):
        return TensorShape(dims=["?"] * (ndim + 1))
    # PyTorch valid range: [-ndim-1, ndim]
    actual = d if d >= 0 else ndim + d + 1
    actual = max(0, min(actual, ndim))
    return TensorShape(dims=dims[:actual] + [1] + dims[actual:])


def compute_indexing_output(
    data_shape: TensorShape,
    d0_shape: Optional[TensorShape] = None,
    d1_shape: Optional[TensorShape] = None,
) -> TensorShape:
    data = list(data_shape.dims)
    d0 = list(d0_shape.dims) if d0_shape and d0_shape.dims else None
    d1 = list(d1_shape.dims) if d1_shape and d1_shape.dims else None
    if d0 and d1:
        return TensorShape(dims=d0[:1] + data[2:])
    if d0:
        return TensorShape(dims=d0[:1] + data[1:])
    return TensorShape(dims=data)


def compute_module_output(module_type: str, params: dict, input_shape: TensorShape) -> TensorShape:
    if module_type == ModuleType.linear:
        if not input_shape.dims:
            raise ValueError("Linear requires at least 1-D input")
        n_out = params.get("n_out", "n_out")
        return TensorShape(dims=list(input_shape.dims[:-1]) + [n_out])
    if module_type == ModuleType.topk:
        k = params.get("k", "k")
        dim = params.get("dim", -1)
        dims = list(input_shape.dims)
        ndim = len(dims)
        if isinstance(dim, int) and ndim > 0:
            actual = dim if dim >= 0 else ndim + dim
            actual = max(0, min(actual, ndim - 1))
            return TensorShape(dims=dims[:actual] + [k] + dims[actual + 1:])
        return TensorShape(dims=dims[:-1] + [k])
    if module_type == ModuleType.view:
        return compute_view_output(params, input_shape)
    if module_type == ModuleType.flatten:
        return compute_flatten_output(params, input_shape)
    if module_type == ModuleType.bincount:
        return compute_bincount_output(params)
    if module_type == ModuleType.unsqueeze:
        return compute_unsqueeze_output(params, input_shape)
    # silu, sigmoid, softmax, mul, where, index_add, identity: shape-preserving fallback
    return TensorShape(dims=list(input_shape.dims))


def _build_adj(connections: dict, proposed: Optional[Tuple[str, str]] = None) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for conn in connections.values():
        adj.setdefault(conn.source_id, []).append(conn.target_id)
        if conn.output_tensor_id:
            adj.setdefault(conn.target_id, []).append(conn.output_tensor_id)
    if proposed:
        src, tgt = proposed
        adj.setdefault(src, []).append(tgt)
    return adj


def _has_cycle(adj: Dict[str, List[str]]) -> bool:
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs(v: str) -> bool:
        visited.add(v)
        rec_stack.add(v)
        for w in adj.get(v, []):
            if w not in visited:
                if dfs(w):
                    return True
            elif w in rec_stack:
                return True
        rec_stack.discard(v)
        return False

    for node in list(adj):
        if node not in visited:
            if dfs(node):
                return True
    return False


def validate_graph(connections: dict, proposed: Optional[Tuple[str, str]] = None) -> ValidationResult:
    adj = _build_adj(connections, proposed)
    if _has_cycle(adj):
        return ValidationResult(
            ok=False,
            errors=[ValidationError(code="CYCLE", message="Cycle detected in graph")]
        )
    return ValidationResult(ok=True)
