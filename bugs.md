# Known Bugs

## 1. Parallel-entry group: only one entry receives a visible wire when attaching a tensor (`screenshots/bug_grouping_2.png`)

**Symptom**: Group has parallel entry modules (e.g. `EXPERT` containing two `Linear` blocks fed in parallel + a `SiLU` downstream of the left Linear). Connecting a tensor to the (expanded) group only shows one visible edge from the source tensor, even though two backend connections are made.

**What we know**:
- `onConnect` BFS in `App.tsx` (`onConnect`, ~line 959) queues all `entryModuleIds`, calls `api.createConnection` for each, and pushes one canvas edge per iteration with a unique `cr.connection.id`. The arithmetic produces two distinct edge objects.
- In the expanded case the code reaches the `else` branch (`allNewEdges.push({ id: cr.connection.id, source, target: item.moduleId, ... })`) for both iterations.
- Backend connections both get created (visible in the recalculated tensor shapes).

**Suspected root cause**: visual-only — likely a React Flow rendering quirk when one edge connects an outside-group source to an inside-group target (`parentId` set, `extent: 'parent'`). The two parallel edges may overlap or the second is clipped by the parent group's bounds.

**Next step to try**: add unique `sourceHandle` per edge or render edges *outside* the parent group with `zIndex` overrides; reproduce in a minimal React Flow sandbox.

---

## 3. Group's expanded position drifts left after running an animation (`screenshots/bug_expand_mode.png`)

**Symptom**: Collapse a group, run an animation, end the animation, then click expand. The group's expanded box appears shifted to the left of where it was previously.

**What we know**:
- `onGroupToggle` reads `gd.expandedX` (saved on collapse) to restore the position. `applyAnimStep` only edits `style` (opacity / pointerEvents / transition) — it spreads `{ ...n }` so `data.expandedX` should survive.
- `endAnimation` restores `style` from `animNodeStyleBackup` (empty `{}` for a collapsed group).
- The bug only reproduces *after* an animation cycle; raw collapse → expand without animation works.

**Suspected root cause**: unclear. Possibly:
- React Flow's measurement (`measured.width`) of the collapsed pill is recomputed differently after the opacity transitions, so the `centeredX → exactX` correction in the collapse branch lands at a slightly different pixel and the next expand reads a stale `expandedX`.
- `pendingCenter` ref interaction with the animation (the post-measure correction effect runs while opacity is animating).

**Next step to try**: log `gd.expandedX`, `group.measured.width`, and `group.position.x` before/after animation to see which value diverges.

---

## 5a. Cloning a group that contains another group flattens the nested structure (`screenshots/group_bug_3.png`)

**Symptom**: Create a parent group whose selection includes a sidebar-cloned group; save the parent to the sidebar; drop the parent on a new canvas. Result: all modules from the inner group appear at the parent group's top level (no inner group), and connections are wired wrong.

**Root cause (intentional simplification — needs revisit)**: `onGroupCreate` flattens nested groups' `moduleChain` and `internalEdges` into the parent's flat topology when storing the registry snapshot. Visual nesting is lost.

**To fix properly**: the registry needs to preserve a recursive structure (`ModuleSnapshot | GroupSnapshot`), and the `onDrop` cloning logic must recursively recreate inner groups (their own `moduleChain` / `internalEdges` / collapsed state). This is a substantial refactor.

---

## 5b. Expanding an inner group inside a parent group causes block overlaps (`screenshots/group_bug_3.png`)

**Symptom**: Open a parent group, then expand one of its child groups. The child group's expanded view exceeds the parent's bounds and visually overlaps the parent's other members.

**Root cause**: `onGroupToggle` (expand branch) only resizes the group being toggled; it doesn't propagate a size change to ancestor groups, and there is no auto-relayout of the parent's other members.

**To fix**: when expanding a nested group, walk up `parentId`, push siblings down/right by the size delta, and grow the parent group's `expandedHeight` / `expandedWidth` to fit. Bounds-based collision resolution is non-trivial and should account for `extent: 'parent'`.
