# Iterations

## Iteration 1 — Core canvas + MLP flow ✅

**Exit criteria:** user can build a 2-layer MLP (Tensor → Linear → Softmax → Linear → Softmax); shapes propagate; a subset can be grouped, collapsed, renamed, reused.

### Completed
- FastAPI + uv + pytest scaffold; `/health` endpoint
- Core types: `TensorShape`, `TensorData`, `ModuleData`, `ConnectionData`, `GroupData`
- Modules: `linear`, `softmax`
- CRUD endpoints: tensors, modules, connections, groups
- Shape propagation on connection create; downstream recalculation on tensor/param change
- Cycle detection + validation returned inline on every mutation
- React + React Flow canvas: dark theme, zoom/pan, keyboard Delete
- Sidebar with draggable blocks
- Custom nodes: `TensorNode` (0–3D visuals), `LinearNode`, `SoftmaxNode`, `TextNode`
- On connect: POST /connections → render output tensor → draw edge
- Group system: marquee-select → collapsible `GroupNode`; rename, collapse/expand, reuse from sidebar
- In-memory store (no persistence yet)

---

## Iteration 2 — Activation + operator blocks ✅

**Exit criteria:** full set of common PyTorch ops available on canvas; all nodes have correct color theming; sidebar organised.

### Completed
- Added modules: `silu`, `sigmoid`, `topk`, `mul`, `view`, `flatten`, `bincount`, `unsqueeze`, `where`, `index_add`
- SiLU replaces SELU; accurate SiLU(x) = x·σ(x) curve in node visual
- `mul`: circular node, broadcasts on 2nd connection
- `view`: shape string with `-1` inference; handles symbolic dims (e.g. `(m,n,n)` + `(-1,n)` → `(m*n,n)`)
- `flatten`: start_dim / end_dim with PyTorch semantics
- `bincount`: minlength accepts numeric or symbolic; symbolic passed through to output dim
- `unsqueeze`: inserts dim of size 1 at given index
- `where`: single tensor input, two 1D index outputs (rows, cols)
- `index_add`: 3 handles (self, source, index); output = self shape; `target_handle` sent to backend to identify "self" regardless of connection order
- Tensor node golden color scheme (`#F5C518`); subtle golden background tint
- Color theming: MODULES/ACTIVATIONS = teal; OTHER = rose; matches sidebar and canvas
- Sidebar redesign: search bar + MODULES / ACTIVATIONS / OTHER / CUSTOM sections
- Backend test suite: 18 integration tests, all passing
- Group snapshotting extended to all module types (`ALL_MODULE_TYPES`)
- Group registry persisted to `localStorage`
- Output tensor placement: `pendingCenter` post-measure correction; two-tensor symmetric spacing
- Group topology fix: `EdgeSnapshot` + `internalEdges` capture parallel/fan-out topologies; cloned groups replay the correct graph (not sequential chain); legacy groups fall back gracefully
- `IndexingNode`: pill/oval shape, golden color, Data section; inputs `data` + `idx`; expr `[rows, ...]`; output `[K, *data_dims[1:]]`

---

## Iteration 3 — Persistence & export (planned)

**Exit criteria:** user can save and reload a named graph; optionally export to PyTorch sketch.

### Planned
- Backend persistence: SQLite via SQLModel; `projects` table (id, name, graph_json, timestamps)
- `POST /projects`, `GET /projects`, `GET /projects/{id}`, `PUT /projects/{id}`, `DELETE /projects/{id}`
- Frontend: Save / Load menu; named projects list
- On load: restore nodes, edges, groups, and group registry from saved JSON
- (stretch) PyTorch code export: walk DAG, emit `nn.Module` skeleton

---

## Iteration 4 — Polish & animation (planned)

**Exit criteria:** smooth step-through animation of a forward pass; validation errors shown inline.

### Planned
- Forward-pass animation: step through nodes/edges with highlight; prev/next/end controls in sidebar
- Validation display: offending nodes highlighted; non-blocking error panel
- Edge routing improvements
- Mobile / touch support (stretch)
