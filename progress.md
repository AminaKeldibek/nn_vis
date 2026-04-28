# nn_vis — Progress & Plan

> **Scope discipline:** Only implement what is listed here. When new requirements arrive, update this file first. No scope creep.

## Decisions locked in

- **"Data flow" = shape propagation only**, not values.
- **No tensor auto-sizing.** Each tensor rank (0–3D) has a distinct visual; all tensors smaller than modules by default.
- **PyTorch code export + custom user blocks: parked.** Not on roadmap until core is stable.
- **`torch` dep dropped** — shape math is hardcoded; no runtime PyTorch needed.
- **`Linear.n_in` inferred** from input tensor; only `n_out` is user input.
- **Symbolic shapes: atomic symbols only** (`m`, `n`, `batch`); expressions like `2*m` only appear as view inference output, not user input.
- **Transport: HTTP/REST.** WebSocket deferred.
- **Validation: async, on every change.** Every mutating endpoint returns `{result, validation}`; frontend renders warnings inline without blocking edits.
- **Group registry persisted to `localStorage`** — survives page refresh and backend restart; templates store module types/params, not backend IDs.

---

## Backend API

Base: `/api/v1`. Frontend owns x/y/zoom; backend owns graph + shapes + persistence.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| POST | `/tensors` | `{shape, dtype?}` → `{tensor, validation}` |
| PATCH | `/tensors/{id}` | update shape/dtype, recomputes downstream |
| DELETE | `/tensors/{id}` | |
| POST | `/modules` | `{type, params}` → `{module, validation}` |
| PATCH | `/modules/{id}` | update params, recomputes outputs |
| DELETE | `/modules/{id}` | |
| POST | `/connections` | `{source_id, target_id, target_handle?}` → `{connection, output_tensor?, output_tensors?, validation}` |
| DELETE | `/connections/{id}` | removes connection + its output tensor(s) |
| POST | `/groups` | `{member_ids, name}` → `{group, validation}` |
| PATCH | `/groups/{id}` | rename / update member list |
| DELETE | `/groups/{id}` | |

`target_handle` on connections is used by `index_add` to identify which semantic input (`self`, `source`, `index`) is being connected, regardless of connection order.

---

## Module types

| Type | Inputs | Outputs | Notes |
|---|---|---|---|
| `linear` | 1 | 1 | n_out param; n_in inferred |
| `softmax` | 1 | 1 | dim param |
| `silu` | 1 | 1 | SiLU(x) = x·σ(x) |
| `sigmoid` | 1 | 1 | |
| `topk` | 1 | 2 (values, indices) | k, dim params |
| `mul` | 2 | 1 | broadcast output on 2nd connection |
| `view` | 1 | 1 | shape string; -1 infers from total elements including symbolic (e.g. `(m,n,n)` + `(-1,n)` → `(m*n, n)`) |
| `flatten` | 1 | 1 | start_dim, end_dim params |
| `bincount` | 1 | 1 | minlength param; symbolic minlength passed through to output dim |
| `unsqueeze` | 1 | 1 | dim param (supports negative) |
| `where` | 1 | 2 (rows, cols) | condition string; outputs are 1D index tensors of shape `[K]` |
| `index_add` | 3 (self, source, index) | 1 | dim param; output shape = self shape; identified via `target_handle` |
| `indexing` | 2 (data, idx) | 1 | expr param (cosmetic); output shape = `[idx_dims[0], *data_dims[1:]]`; pill/oval shape; golden color; in Data section |

---

## Frontend node palette

**Data (no grouping header)**
- `TensorNode` — 0–3D visual with rank selector and dim inputs; golden color scheme
- `VarNode` — annotation variable
- `TextNode` — free text label

**MODULES section (teal)**
- `LinearNode`

**ACTIVATIONS section (teal)**
- `SoftmaxNode`, `SiluNode`, `SigmoidNode`

**OTHER section (rose)**
- `TopkNode`, `MulNode`, `ViewNode`, `FlattenNode`, `BincountNode`, `UnsqueezeNode`, `WhereNode`, `IndexAddNode`

**CUSTOM section**
- User-created group templates (saved to `localStorage`)

Sidebar has a search bar; filtered view shows flat list across all sections.

---

## Groups

- Marquee-select nodes → "Create group" → collapsible `GroupNode`
- `moduleChain` (sequential single-in/single-out modules only) drives auto-wiring when a tensor is connected to the group
- `ALL_MODULE_TYPES` (all module types including multi-input/output) are snapshotted and cloned when the group is dragged from sidebar
- Group templates persist in `localStorage` across restarts
- Rename inline; ungroup restores original nodes

---

## Output tensor placement

- Single output: placed centered below module using `measured.width`; a `pendingCenter` ref corrects position to pixel-perfect after React Flow measures the new node
- Two outputs (topk, where): symmetric layout — `nodeA.x = centerX − TENSOR_EST_W/2 − GAP/2`, `nodeB.x = centerX + TENSOR_EST_W/2 + GAP/2` — guarantees 24px gap between them regardless of actual measured width; also post-corrected via `pendingCenter`

---

## Test suite

`backend/tests/test_all_modules.py` — 18 integration tests covering all module connection scenarios including symbolic shape propagation, multi-output modules, out-of-order connections, and symbolic view inference.
