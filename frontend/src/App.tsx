import { useCallback, useRef, useState, useMemo, useEffect } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  Panel,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
  type Connection,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { Sidebar } from './components/Sidebar'
import { TextNode, type TextNodeData } from './nodes/TextNode'
import { TensorNode, type TensorNodeData } from './nodes/TensorNode'
import { LinearNode, type LinearNodeData } from './nodes/LinearNode'
import { SoftmaxNode, type SoftmaxNodeData } from './nodes/SoftmaxNode'
import { GroupNode, type GroupNodeData, type EdgeSnapshot, type SavedEdge, type ReflowSnapshot } from './nodes/GroupNode'
import { SiluNode } from './nodes/SiluNode'
import { SigmoidNode } from './nodes/SigmoidNode'
import { TopkNode, type TopkNodeData } from './nodes/TopkNode'
import { VarNode, type VarNodeData } from './nodes/VarNode'
import { MulNode } from './nodes/MulNode'
import { SumNode } from './nodes/SumNode'
import { ViewNode, type ViewNodeData } from './nodes/ViewNode'
import { FlattenNode, type FlattenNodeData } from './nodes/FlattenNode'
import { BincountNode, type BincountNodeData } from './nodes/BincountNode'
import { UnsqueezeNode, type UnsqueezeNodeData } from './nodes/UnsqueezeNode'
import { WhereNode, type WhereNodeData } from './nodes/WhereNode'
import { IndexAddNode, type IndexAddNodeData } from './nodes/IndexAddNode'
import { IndexingNode, type IndexingNodeData } from './nodes/IndexingNode'
import { colors } from './theme/colors'
import { api, type Dim } from './api/client'
import { AppContext } from './AppContext'
import { type SidebarAnimState } from './components/Sidebar'

// 'blockGroup' avoids React Flow's reserved 'group' type which adds its own background
const nodeTypes = {
  text: TextNode,
  tensor: TensorNode,
  var: VarNode,
  linear: LinearNode,
  softmax: SoftmaxNode,
  silu: SiluNode,
  sigmoid: SigmoidNode,
  topk: TopkNode,
  mul: MulNode,
  sum: SumNode,
  view: ViewNode,
  flatten: FlattenNode,
  bincount: BincountNode,
  unsqueeze: UnsqueezeNode,
  where: WhereNode,
  index_add: IndexAddNode,
  indexing: IndexingNode,
  blockGroup: GroupNode,
}

// Sequential single-in/single-out modules — used for auto-wiring when tensor connects to a group
const MODULE_TYPES = new Set(['linear', 'softmax', 'silu', 'sigmoid', 'topk', 'view', 'flatten', 'bincount', 'unsqueeze'])
// All module types that can be snapshotted and cloned as part of a group template
const ALL_MODULE_TYPES = new Set([...MODULE_TYPES, 'mul', 'sum', 'index_add', 'where', 'indexing'])

// Animation builder — topological sort (Kahn's algorithm), supports DAG with multiple roots
type AnimStep = { nodeIds: string[]; edgeIds: string[] }

function buildAnimTopoSteps(selectedNodes: Node[], selectedEdges: Edge[]): AnimStep[] | string {
  const selIds = new Set(selectedNodes.map(n => n.id))
  const inDegree = new Map<string, number>()
  const adj = new Map<string, { targetId: string; edgeId: string }[]>()
  const incoming = new Map<string, { sourceId: string; edgeId: string }[]>()

  for (const n of selectedNodes) {
    inDegree.set(n.id, 0)
    adj.set(n.id, [])
    incoming.set(n.id, [])
  }
  for (const e of selectedEdges) {
    if (!selIds.has(e.source) || !selIds.has(e.target)) continue
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
    adj.get(e.source)!.push({ targetId: e.target, edgeId: e.id })
    incoming.get(e.target)!.push({ sourceId: e.source, edgeId: e.id })
  }

  const queue: string[] = selectedNodes
    .filter(n => (inDegree.get(n.id) ?? 0) === 0)
    .sort((a, b) => (a.position?.x ?? 0) - (b.position?.x ?? 0))
    .map(n => n.id)

  if (queue.length === 0) return 'Cycle detected in selection'

  const processed = new Set<string>()
  const steps: AnimStep[] = []

  while (queue.length > 0) {
    const nodeId = queue.shift()!
    if (processed.has(nodeId)) continue
    processed.add(nodeId)

    // Edges whose source was already processed (revealed in earlier steps)
    const stepEdgeIds = (incoming.get(nodeId) ?? [])
      .filter(e => processed.has(e.sourceId))
      .map(e => e.edgeId)

    steps.push({ nodeIds: [nodeId], edgeIds: stepEdgeIds })

    for (const { targetId } of (adj.get(nodeId) ?? [])) {
      if (!selIds.has(targetId)) continue
      const nd = (inDegree.get(targetId) ?? 1) - 1
      inDegree.set(targetId, nd)
      if (nd === 0) queue.push(targetId)
    }
  }

  if (processed.size !== selectedNodes.length) return 'Cycle or disconnected nodes in selection'
  return steps
}

// Collect all transitive member IDs from a set of group IDs (including nested group members)
function collectTransitiveMembers(ids: Iterable<string>, allNodes: Node[]): Set<string> {
  const result = new Set<string>()
  const queue = [...ids]
  while (queue.length > 0) {
    const id = queue.shift()!
    if (result.has(id)) continue
    result.add(id)
    const node = allNodes.find(n => n.id === id)
    if (node?.type === 'blockGroup') {
      for (const mid of (node.data as GroupNodeData).memberIds) {
        if (!result.has(mid)) queue.push(mid)
      }
    }
  }
  return result
}

// nodeId → should be visible, when expanding a group (respects child group collapsed states)
function computeExpandVisibility(groupId: string, allNodes: Node[]): Map<string, boolean> {
  const result = new Map<string, boolean>()
  const node = allNodes.find(n => n.id === groupId)
  if (!node || node.type !== 'blockGroup') return result
  const gd = node.data as GroupNodeData
  for (const memberId of gd.memberIds) {
    result.set(memberId, true)
    const memberNode = allNodes.find(n => n.id === memberId)
    if (memberNode?.type === 'blockGroup') {
      if (!(memberNode.data as GroupNodeData).collapsed) {
        const subMap = computeExpandVisibility(memberId, allNodes)
        for (const [k, v] of subMap) if (!result.has(k)) result.set(k, v)
      }
    }
  }
  return result
}

// Absolute canvas position for a node (traverses parentId chain)
function getAbsolutePosition(nodeId: string, allNodes: Node[]): { x: number; y: number } {
  const node = allNodes.find(n => n.id === nodeId)
  if (!node) return { x: 0, y: 0 }
  if (!node.parentId) return { x: node.position.x, y: node.position.y }
  const pp = getAbsolutePosition(node.parentId, allNodes)
  return { x: node.position.x + pp.x, y: node.position.y + pp.y }
}

// Exit modules of a group: modules whose output flows out (no downstream in internalEdges)
function getGroupExitModules(gd: GroupNodeData): string[] {
  const ie = gd.internalEdges
  if (!ie || ie.length === 0) return (gd.moduleChain ?? []).slice(-1)
  const fromSet = new Set(ie.map(e => e.from))
  const allIds = new Set([
    ...(gd.moduleChain ?? []),
    ...ie.map(e => e.from),
    ...ie.map(e => e.to),
  ])
  return [...allIds].filter(id => !fromSet.has(id))
}

// Recursively flatten a group's moduleChain to actual backend module IDs (ENTRY modules only).
// Used for resolving edge endpoints when a connection lands on a group node.
function getFlatModuleIds(moduleChain: string[], allNodes: Node[]): string[] {
  const result: string[] = []
  for (const id of moduleChain) {
    const node = allNodes.find(n => n.id === id)
    if (!node) continue
    if (node.type === 'blockGroup') {
      result.push(...getFlatModuleIds((node.data as GroupNodeData).moduleChain ?? [], allNodes))
    } else if (ALL_MODULE_TYPES.has(node.type ?? '')) {
      result.push(id)
    }
  }
  return result
}

// Visual footprint (width/height) of a node in its parent's coordinate space.
// Collapsed group → pill; expanded group → its style w/h; module → measured.
function getNodeFootprint(n: Node): { w: number; h: number } {
  if (n.type === 'blockGroup') {
    const gd = n.data as GroupNodeData
    if (gd.collapsed) {
      return { w: (n.measured as any)?.width ?? COLLAPSED_NODE_W, h: (n.measured as any)?.height ?? COLLAPSED_HEIGHT }
    }
    return {
      w: (n.style?.width as number) ?? gd.expandedWidth ?? 300,
      h: (n.style?.height as number) ?? gd.expandedHeight ?? 200,
    }
  }
  return {
    w: (n.measured as any)?.width ?? 180,
    h: (n.measured as any)?.height ?? 120,
  }
}

// After a group's footprint changes (expand/collapse), reflow:
//   1. Push later same-level siblings down by Δh; right by Δw if vertically overlapping.
//   2. Grow the immediate parent group's expanded size to fit (collapsed parent untouched).
//   3. Recurse on the parent (its own footprint just changed).
// Only down/right shifts — never reflows above the changed node.
// Returns both the updated nodes and a SNAPSHOT of every change made (sibling shifts + ancestor resizes),
// so the caller (expand) can store it on the toggled group and the collapse can perfectly undo.
function reflowAfterFootprintChange(
  nodes: Node[],
  changedId: string,
  prevFootprint: { w: number; h: number },
  newFootprint: { w: number; h: number },
  acc?: ReflowSnapshot,
): { nodes: Node[]; snapshot: ReflowSnapshot } {
  const snapshot: ReflowSnapshot = acc ?? { shifts: [], resizes: [] }
  const dw = newFootprint.w - prevFootprint.w
  const dh = newFootprint.h - prevFootprint.h
  if (dw === 0 && dh === 0) return { nodes, snapshot }

  const changedNode = nodes.find(n => n.id === changedId)
  if (!changedNode) return { nodes, snapshot }

  const parentId = changedNode.parentId
  // No parent → don't touch canvas-root siblings (independent standalone nodes shouldn't shift
  // when a top-level group toggles).
  if (!parentId) return { nodes, snapshot }

  const changedX = changedNode.position.x
  const changedY = changedNode.position.y
  const PADDING = 24

  const updated = nodes.map(n => {
    if (n.id === changedId) return n
    if (n.parentId !== parentId) return n
    const nFoot = getNodeFootprint(n)
    let nx = n.position.x
    let ny = n.position.y

    if (dh > 0 && ny >= changedY) {
      ny += dh
    } else if (dh < 0 && ny + dh >= changedY) {
      ny += dh
    }
    const sibTop = n.position.y
    const sibBot = sibTop + nFoot.h
    const verticallyOverlaps = (chgTop: number, chgBot: number) => sibBot > chgTop && sibTop < chgBot
    if (dw > 0) {
      if (verticallyOverlaps(changedY, changedY + Math.max(prevFootprint.h, newFootprint.h)) && nx >= changedX + prevFootprint.w) {
        nx += dw
      }
    } else if (dw < 0) {
      if (verticallyOverlaps(changedY, changedY + prevFootprint.h) && nx + dw >= changedX + newFootprint.w) {
        nx += dw
      }
    }

    if (nx === n.position.x && ny === n.position.y) return n
    snapshot.shifts.push({ id: n.id, dx: nx - n.position.x, dy: ny - n.position.y })
    return { ...n, position: { x: nx, y: ny } }
  })

  const parentNode = updated.find(n => n.id === parentId)
  if (!parentNode || parentNode.type !== 'blockGroup') return { nodes: updated, snapshot }
  const parentGd = parentNode.data as GroupNodeData
  if (parentGd.collapsed) return { nodes: updated, snapshot }

  // Recompute parent bounds from all visible direct children
  let maxRight = 0, maxBottom = 0
  for (const n of updated) {
    if (n.parentId !== parentId) continue
    if (n.hidden) continue
    const f = getNodeFootprint(n)
    maxRight = Math.max(maxRight, n.position.x + f.w)
    maxBottom = Math.max(maxBottom, n.position.y + f.h)
  }
  const requiredW = maxRight + PADDING
  const requiredH = maxBottom + PADDING

  const parentPrevFoot = getNodeFootprint(parentNode)
  const newW = Math.max(parentPrevFoot.w, requiredW)
  const newH = Math.max(parentPrevFoot.h, requiredH)

  if (newW === parentPrevFoot.w && newH === parentPrevFoot.h) return { nodes: updated, snapshot }

  snapshot.resizes.push({ id: parentId, prevW: parentPrevFoot.w, prevH: parentPrevFoot.h })

  // Update only style — keep data.expandedWidth/Height as the immutable "natural" intent
  let withParentResized = updated.map(n => {
    if (n.id !== parentId) return n
    return {
      ...n,
      style: { ...(n.style ?? {}), width: newW, height: newH },
    }
  })

  return reflowAfterFootprintChange(
    withParentResized,
    parentId,
    parentPrevFoot,
    { w: newW, h: newH },
    snapshot,
  )
}

// All module ids reachable from a node by walking memberIds recursively (every module at every depth).
// Used for collecting full module set when snapshotting a parent group containing nested groups.
function getAllModuleIdsRecursive(rootId: string, allNodes: Node[]): string[] {
  const result: string[] = []
  const seen = new Set<string>()
  const visit = (id: string) => {
    if (seen.has(id)) return
    seen.add(id)
    const node = allNodes.find(n => n.id === id)
    if (!node) return
    if (ALL_MODULE_TYPES.has(node.type ?? '')) {
      result.push(id)
    } else if (node.type === 'blockGroup') {
      for (const mid of ((node.data as GroupNodeData).memberIds ?? [])) visit(mid)
    }
    // tensors and others ignored
  }
  visit(rootId)
  return result
}

// Parsed once when module loads — avoids repeated JSON.parse on re-renders
const _savedCanvas = (() => {
  try {
    const s = localStorage.getItem('nn_vis_canvas')
    return s ? JSON.parse(s) as { nodes: Node[]; edges: Edge[] } : null
  } catch { return null }
})()
const _savedConnMap = (() => {
  try {
    const s = localStorage.getItem('nn_vis_conn_map')
    return s ? JSON.parse(s) as [string, string][] : null
  } catch { return null }
})()

const EDGE_STYLE = { stroke: colors.teal, strokeWidth: 2 }
const EDGE_MARKER = { type: MarkerType.ArrowClosed, color: colors.teal }
const COLLAPSED_HEIGHT = 44
const COLLAPSED_NODE_W = 220  // estimated rendered width of the compact group block
const TENSOR_EST_W = 140      // initial placement estimate; corrected once React Flow measures
const TENSOR_TWO_GAP = 24     // gap between two side-by-side output tensors (topk, where)

function rankFromDims(dims: Dim[]): 0 | 1 | 2 | 3 {
  return Math.min(dims.length, 3) as 0 | 1 | 2 | 3
}

function getDownstreamNodeIds(startIds: string[], allEdges: Edge[], excludeIds: Set<string>): Set<string> {
  const result = new Set<string>()
  const queue = [...startIds]
  while (queue.length) {
    const id = queue.shift()!
    if (result.has(id) || excludeIds.has(id)) continue
    result.add(id)
    for (const ed of allEdges) {
      if (ed.source === id && !result.has(ed.target) && !excludeIds.has(ed.target)) {
        queue.push(ed.target)
      }
    }
  }
  return result
}

// Template stores only modules (not tensors) — ordered by chain position
type ModuleSnapshot = {
  tmpId: string
  type: string
  data: Record<string, unknown>
  relativePosition: { x: number; y: number }
}

// Visual-hierarchy-only snapshot (used IN ADDITION to the flat moduleChain/internalEdges).
// When present, the drop path does a re-parenting pass to recreate the nested visual structure.
// Flat moduleChain/internalEdges (used for wiring) are unchanged in nested mode.
type NestedMember =
  | {
      kind: 'module'
      tmpId: string
      relativePosition: { x: number; y: number }   // relative to immediate parent group
    }
  | {
      kind: 'group'
      tmpId: string                                  // not referenced by moduleChain — purely a hierarchy id
      label: string
      relativePosition: { x: number; y: number }   // relative to immediate parent group
      expandedWidth: number
      expandedHeight: number
      members: NestedMember[]
    }

type NestedGroupSnapshot = {
  members: NestedMember[]
  expandedWidth: number
  expandedHeight: number
}

type GroupRegistryEntry = {
  id: string
  label: string
  moduleChain: ModuleSnapshot[]
  internalEdges?: EdgeSnapshot[]
  nestedSnapshot?: NestedGroupSnapshot   // present iff the saved group contained nested blockGroups
}

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(_savedCanvas?.nodes ?? [])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(_savedCanvas?.edges ?? [])
  const [error, setError] = useState<string | null>(null)
  const [selectedCount, setSelectedCount] = useState(0)
  const [groupRegistry, setGroupRegistry] = useState<GroupRegistryEntry[]>(() => {
    try {
      const saved = localStorage.getItem('nn_vis_group_registry')
      return saved ? (JSON.parse(saved) as GroupRegistryEntry[]) : []
    } catch {
      return []
    }
  })
  useEffect(() => {
    localStorage.setItem('nn_vis_group_registry', JSON.stringify(groupRegistry))
  }, [groupRegistry])

  // Persist canvas and connection map whenever the graph changes
  useEffect(() => {
    const nodesToSave = nodes.map(({ measured: _, ...n }: any) => n)
    localStorage.setItem('nn_vis_canvas', JSON.stringify({ nodes: nodesToSave, edges }))
    localStorage.setItem('nn_vis_conn_map', JSON.stringify(Array.from(connectionOutputMap.current.entries())))
  }, [nodes, edges])

  // Maps output tensor node ID → target center X; resolved once React Flow measures the node
  const pendingCenter = useRef<Map<string, number>>(new Map())
  useEffect(() => {
    if (pendingCenter.current.size === 0) return
    const corrections = new Map<string, number>()
    for (const n of nodes) {
      if (!pendingCenter.current.has(n.id)) continue
      const w = (n.measured as any)?.width
      if (!w) continue
      corrections.set(n.id, pendingCenter.current.get(n.id)! - w / 2)
      pendingCenter.current.delete(n.id)
    }
    if (corrections.size === 0) return
    setNodes(ns => ns.map(n => corrections.has(n.id) ? { ...n, position: { ...n.position, x: corrections.get(n.id)! } } : n))
  }, [nodes, setNodes])

  const [animState, setAnimState] = useState<{ steps: AnimStep[]; currentStep: number } | null>(null)
  const animNodeStyleBackup = useRef<Map<string, React.CSSProperties>>(new Map())
  const animEdgeStyleBackup = useRef<Map<string, React.CSSProperties>>(new Map())
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const rfRef = useRef<any>(null)
  const connectionOutputMap = useRef<Map<string, string>>(
    _savedConnMap ? new Map(_savedConnMap) : new Map()
  )
  const historyRef = useRef<{ nodes: Node[]; edges: Edge[] }[]>([])

  const onInit = useCallback((instance: any) => { rfRef.current = instance }, [])

  const pushHistory = useCallback(() => {
    const ns = rfRef.current?.getNodes() ?? []
    const es = rfRef.current?.getEdges() ?? []
    historyRef.current = [...historyRef.current.slice(-19), { nodes: ns, edges: es }]
  }, [])

  const applyAnimStep = useCallback((steps: AnimStep[], step: number) => {
    const visibleNodeIds = new Set(steps.slice(0, step + 1).flatMap(s => s.nodeIds))
    const visibleEdgeIds = new Set(steps.slice(0, step + 1).flatMap(s => s.edgeIds))
    const allNodeIds = new Set(steps.flatMap(s => s.nodeIds))
    const allEdgeIds = new Set(steps.flatMap(s => s.edgeIds))
    setNodes(ns => ns.map(n => {
      if (!allNodeIds.has(n.id)) return n
      const vis = visibleNodeIds.has(n.id)
      return { ...n, style: { ...animNodeStyleBackup.current.get(n.id), opacity: vis ? 1 : 0, pointerEvents: vis ? undefined : 'none' as const, transition: 'opacity 0.35s ease' } }
    }))
    setEdges(es => es.map(e => {
      if (!allEdgeIds.has(e.id)) return e
      const vis = visibleEdgeIds.has(e.id)
      return { ...e, style: { ...animEdgeStyleBackup.current.get(e.id), opacity: vis ? 1 : 0, transition: 'opacity 0.35s ease' } }
    }))
  }, [setNodes, setEdges])

  const startAnimation = useCallback(() => {
    const allNodes: Node[] = rfRef.current?.getNodes() ?? []
    const allEdges: Edge[] = rfRef.current?.getEdges() ?? []
    // Include blockGroup so it participates in topo order; exclude hidden nodes
    // (internal members of collapsed groups would otherwise stall the animation)
    const selectedNodes = allNodes.filter(n =>
      n.selected && !n.hidden && n.type !== 'text' && n.type !== 'var'
    )
    const selectedNodeIds = new Set(selectedNodes.map(n => n.id))
    const relevantEdges = allEdges.filter(e =>
      selectedNodeIds.has(e.source) && selectedNodeIds.has(e.target) && !e.hidden
    )

    const result = buildAnimTopoSteps(selectedNodes, relevantEdges)
    if (typeof result === 'string') { showError(result); return }
    if (result.length === 0) { showError('No nodes to animate'); return }

    const allAnimNodeIds = new Set(result.flatMap(s => s.nodeIds))
    const allAnimEdgeIds = new Set(result.flatMap(s => s.edgeIds))
    animNodeStyleBackup.current.clear()
    animEdgeStyleBackup.current.clear()
    for (const n of allNodes) {
      if (allAnimNodeIds.has(n.id)) animNodeStyleBackup.current.set(n.id, n.style ?? {})
    }
    for (const e of allEdges) {
      if (allAnimEdgeIds.has(e.id)) animEdgeStyleBackup.current.set(e.id, e.style ?? {})
    }

    setAnimState({ steps: result, currentStep: 0 })
    applyAnimStep(result, 0)
  }, [applyAnimStep])

  const nextAnimStep = useCallback(() => {
    if (!animState || animState.currentStep >= animState.steps.length - 1) return
    const next = animState.currentStep + 1
    applyAnimStep(animState.steps, next)
    setAnimState(prev => prev ? { ...prev, currentStep: next } : null)
  }, [animState, applyAnimStep])

  const prevAnimStep = useCallback(() => {
    if (!animState || animState.currentStep <= 0) return
    const prev = animState.currentStep - 1
    applyAnimStep(animState.steps, prev)
    setAnimState(p => p ? { ...p, currentStep: prev } : null)
  }, [animState, applyAnimStep])

  const endAnimation = useCallback(() => {
    if (!animState) return
    const allNodeIds = new Set(animState.steps.flatMap(s => s.nodeIds))
    const allEdgeIds = new Set(animState.steps.flatMap(s => s.edgeIds))
    setNodes(ns => ns.map(n => {
      if (!allNodeIds.has(n.id)) return n
      return { ...n, style: animNodeStyleBackup.current.get(n.id) ?? {} }
    }))
    setEdges(es => es.map(e => {
      if (!allEdgeIds.has(e.id)) return e
      return { ...e, style: animEdgeStyleBackup.current.get(e.id) ?? {} }
    }))
    setAnimState(null)
  }, [animState, setNodes, setEdges])

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  function showError(msg: string) {
    setError(msg)
    setTimeout(() => setError(null), 4000)
  }

  const applyUpdatedTensors = useCallback((updated: { id: string; shape: { dims: Dim[] }; dtype: string }[]) => {
    if (!updated.length) return
    const map = new Map(updated.map(t => [t.id, t]))
    setNodes(ns => ns.map(n => {
      const ut = map.get(n.id)
      if (ut && n.type === 'tensor') {
        const dims = ut.shape.dims
        return { ...n, data: { ...n.data as TensorNodeData, dims, rank: rankFromDims(dims) } }
      }
      return n
    }))
  }, [setNodes])

  const onTensorDimsChange = useCallback(async (nodeId: string, dims: Dim[]) => {
    try {
      const res = await api.patchTensor(nodeId, { dims })
      applyUpdatedTensors([res.tensor, ...res.updated_tensors])
    } catch (err) {
      showError((err as Error).message)
    }
  }, [applyUpdatedTensors])

  const onModuleParamChange = useCallback(async (nodeId: string, params: Record<string, unknown>) => {
    try {
      const res = await api.patchModule(nodeId, params)
      applyUpdatedTensors(res.updated_tensors)
      setNodes(ns => ns.map(n => n.id === nodeId ? { ...n, data: { ...n.data, ...params } } : n))
    } catch (err) {
      showError((err as Error).message)
    }
  }, [applyUpdatedTensors, setNodes])

  const onGroupToggle = useCallback((groupId: string) => {
    pushHistory()
    const allNodes = rfRef.current?.getNodes() ?? []
    const allEdges: Edge[] = rfRef.current?.getEdges() ?? []
    const group = allNodes.find((n: Node) => n.id === groupId)
    if (!group) return

    const gd = group.data as GroupNodeData
    const isCollapsed = gd.collapsed
    const memberSet = new Set(gd.memberIds)

    if (isCollapsed) {
      // ── Expand ────────────────────────────────────────────────────────────
      const saved = gd.savedBoundaryEdges ?? []
      const savedMap = new Map(saved.map(e => [e.id, e]))
      const delta = gd.positionDelta ?? 0

      // Downstream nodes to shift back down
      const outgoingOrigTargets = saved
        .filter(e => memberSet.has(e.source) && !memberSet.has(e.target))
        .map(e => e.target)
      const downstreamIds = getDownstreamNodeIds(outgoingOrigTargets, allEdges, memberSet)

      const expandedH = gd.expandedHeight ?? 200
      const expandedBottom = group.position.y + expandedH
      const visMap = computeExpandVisibility(groupId, allNodes)

      const prevFootprint = getNodeFootprint(group)
      const newFootprint = { w: gd.expandedWidth, h: gd.expandedHeight }

      setNodes(ns => {
        const stepped = ns.map((n: Node) => {
          if (n.id === groupId) {
            const collapsedW = (group.measured as any)?.width ?? 180
            const restoredX = gd.expandedX !== undefined
              ? gd.expandedX
              : n.position.x + collapsedW / 2 - gd.expandedWidth / 2
            return {
              ...n,
              position: { x: restoredX, y: n.position.y },
              style: { width: gd.expandedWidth, height: gd.expandedHeight },
              data: { ...n.data, collapsed: false, savedBoundaryEdges: undefined, positionDelta: 0, expandedX: undefined },
            }
          }
          if (visMap.has(n.id)) return { ...n, hidden: false }
          if (downstreamIds.has(n.id)) {
            if (delta) {
              return { ...n, position: { ...n.position, y: n.position.y + delta } }
            } else if (n.position.y < expandedBottom) {
              return { ...n, position: { ...n.position, y: n.position.y + expandedH } }
            }
          }
          return n
        })
        const { nodes: reflowed, snapshot } = reflowAfterFootprintChange(stepped, groupId, prevFootprint, newFootprint)
        // Stash the snapshot of changes on the toggled group so collapse can perfectly undo
        return reflowed.map(n =>
          n.id === groupId ? { ...n, data: { ...(n.data as GroupNodeData), reflowSnapshot: snapshot } } : n
        )
      })
      setEdges(es => es.map((ed: Edge) => {
        const orig = savedMap.get(ed.id)
        if (orig) {
          return {
            ...ed,
            source: orig.source,
            sourceHandle: orig.sourceHandle ?? undefined,
            target: orig.target,
            targetHandle: orig.targetHandle ?? undefined,
            hidden: false,
          }
        }
        if (memberSet.has(ed.source) && memberSet.has(ed.target)) {
          return { ...ed, hidden: false }
        }
        return ed
      }))
    } else {
      // ── Collapse ──────────────────────────────────────────────────────────
      const incoming = allEdges.filter(ed =>
        !memberSet.has(ed.source) && memberSet.has(ed.target) && !ed.hidden
      )
      const outgoing = allEdges.filter(ed =>
        memberSet.has(ed.source) && !memberSet.has(ed.target) && !ed.hidden
      )
      const incomingIds = new Set(incoming.map(ed => ed.id))
      const outgoingIds = new Set(outgoing.map(ed => ed.id))

      const savedBoundaryEdges = [...incoming, ...outgoing].map(ed => ({
        id: ed.id,
        source: ed.source,
        sourceHandle: ed.sourceHandle,
        target: ed.target,
        targetHandle: ed.targetHandle,
      }))

      const w = (group.style?.width as number) ?? gd.expandedWidth
      const h = (group.style?.height as number) ?? gd.expandedHeight
      const delta = Math.max(0, h - COLLAPSED_HEIGHT)

      // BFS downstream from outgoing targets to shift up
      const outgoingTargets = outgoing.map(ed => ed.target)
      const downstreamIds = getDownstreamNodeIds(outgoingTargets, allEdges, memberSet)

      // Center collapsed block: start with estimate, then correct once React Flow measures it
      const centeredX = group.position.x + w / 2 - COLLAPSED_NODE_W / 2
      const savedExpandedX = group.position.x

      // Hide all transitive members (including nested group members)
      const allGroupMembers = collectTransitiveMembers(memberSet, allNodes)

      // If there's a snapshot from this group's last expand, apply the inverse to perfectly
      // restore siblings + ancestor sizes to the pre-expand state.
      const reflowSnapshot = gd.reflowSnapshot

      setNodes(ns => {
        return ns.map((n: Node) => {
          if (n.id === groupId) {
            return {
              ...n,
              position: { x: centeredX, y: n.position.y },
              style: {},
              data: {
                ...n.data,
                collapsed: true,
                expandedWidth: w,
                expandedHeight: h,
                expandedX: savedExpandedX,
                savedBoundaryEdges,
                positionDelta: delta,
                reflowSnapshot: undefined,  // consumed
              },
            }
          }
          if (allGroupMembers.has(n.id)) return { ...n, hidden: true }
          if (delta && downstreamIds.has(n.id)) {
            return { ...n, position: { ...n.position, y: n.position.y - delta } }
          }
          // Inverse-apply the snapshot: pull back any sibling that was shifted, restore any ancestor's size
          if (reflowSnapshot) {
            const shift = reflowSnapshot.shifts.find(s => s.id === n.id)
            const resize = reflowSnapshot.resizes.find(r => r.id === n.id)
            if (shift || resize) {
              let result: Node = n
              if (shift) {
                result = { ...result, position: { x: result.position.x - shift.dx, y: result.position.y - shift.dy } }
              }
              if (resize) {
                result = { ...result, style: { ...(result.style ?? {}), width: resize.prevW, height: resize.prevH } }
              }
              return result
            }
          }
          return n
        })
      })
      setEdges(es => es.map((ed: Edge) => {
        if (incomingIds.has(ed.id)) {
          return { ...ed, target: groupId, targetHandle: 'input', hidden: false }
        }
        if (outgoingIds.has(ed.id)) {
          return { ...ed, source: groupId, sourceHandle: 'output', hidden: false }
        }
        if (memberSet.has(ed.source) && memberSet.has(ed.target)) {
          return { ...ed, hidden: true }
        }
        return ed
      }))

      // After React Flow measures the collapsed node, apply exact centering
      setTimeout(() => {
        const collapsedNode = rfRef.current?.getNode(groupId)
        if (!collapsedNode || !(collapsedNode.data as GroupNodeData).collapsed) return
        const measuredW = (collapsedNode.measured as any)?.width ?? COLLAPSED_NODE_W
        const exactX = savedExpandedX + w / 2 - measuredW / 2
        setNodes(ns => ns.map((n: Node) =>
          n.id === groupId ? { ...n, position: { x: exactX, y: n.position.y } } : n
        ))
      }, 50)
    }
  }, [pushHistory, setNodes, setEdges])

  const onDeleteGroupFromSidebar = useCallback((id: string) => {
    setGroupRegistry(prev => prev.filter(g => g.id !== id))
  }, [])

  const onGroupRename = useCallback((groupId: string, label: string) => {
    setNodes(ns => ns.map((n: Node) =>
      n.id === groupId ? { ...n, data: { ...n.data, label } } : n
    ))
    setGroupRegistry(prev => prev.map(g => g.id === groupId ? { ...g, label } : g))
    api.patchGroup(groupId, { name: label }).catch(console.error)
  }, [setNodes])

  const onGroupUngroup = useCallback((groupId: string) => {
    pushHistory()
    const allNodes: Node[] = rfRef.current?.getNodes() ?? []
    const allEdges: Edge[] = rfRef.current?.getEdges() ?? []
    const group = allNodes.find((n: Node) => n.id === groupId)
    if (!group) return

    const gd = group.data as GroupNodeData
    const memberSet = new Set(gd.memberIds)
    const gx = group.position.x
    const gy = group.position.y

    // If collapsed, restore boundary edges before dissolving
    let updatedEdges = allEdges
    if (gd.collapsed) {
      const saved = gd.savedBoundaryEdges ?? []
      const savedMap = new Map(saved.map(e => [e.id, e]))
      updatedEdges = allEdges.map(ed => {
        const orig = savedMap.get(ed.id)
        if (orig) {
          return {
            ...ed,
            source: orig.source,
            sourceHandle: orig.sourceHandle ?? undefined,
            target: orig.target,
            targetHandle: orig.targetHandle ?? undefined,
            hidden: false,
          }
        }
        if (memberSet.has(ed.source) && memberSet.has(ed.target)) {
          return { ...ed, hidden: false }
        }
        return ed
      })
    }

    setNodes(ns =>
      ns
        .filter((n: Node) => n.id !== groupId)
        .map((n: Node) => {
          if (!memberSet.has(n.id)) return n
          return {
            ...n,
            hidden: false,
            position: { x: n.position.x + gx, y: n.position.y + gy },
            parentId: undefined,
            extent: undefined,
          }
        })
    )
    setEdges(updatedEdges.filter(ed => !ed.hidden))
    api.deleteGroup(groupId).catch(console.error)
    // groupRegistry intentionally not modified — template stays in sidebar
  }, [pushHistory, setNodes, setEdges])

  const appCallbacks = useMemo(() => ({
    onTensorDimsChange,
    onModuleParamChange,
    onGroupToggle,
    onGroupRename,
    onGroupUngroup,
  }), [onTensorDimsChange, onModuleParamChange, onGroupToggle, onGroupRename, onGroupUngroup])

  const onDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    const type = e.dataTransfer.getData('application/nnvis-node')
    if (!type || !rfRef.current) return

    const rect = reactFlowWrapper.current!.getBoundingClientRect()
    const position = rfRef.current.screenToFlowPosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    })

    pushHistory()

    try {
      if (type === 'tensor') {
        const res = await api.createTensor(['m', 'n'])
        const data: TensorNodeData = { name: '', rank: 2, dims: res.tensor.shape.dims, dtype: res.tensor.dtype }
        setNodes(ns => [...ns, { id: res.tensor.id, type, position, data }])

      } else if (type === 'linear') {
        const res = await api.createModule('linear', { n_out: 64 })
        const data: LinearNodeData = { n_out: (res.module.params.n_out as number | string) ?? 64 }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'softmax') {
        const res = await api.createModule('softmax', { dim: -1 })
        const data: SoftmaxNodeData = { dim: (res.module.params.dim as number) ?? -1 }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'var') {
        const data: VarNodeData = { value: 'n' }
        setNodes(ns => [...ns, { id: `var-${Date.now()}`, type, position, data }])

      } else if (type === 'silu') {
        const res = await api.createModule('silu', {})
        setNodes(ns => [...ns, { id: res.module.id, type, position, data: {} }])

      } else if (type === 'mul') {
        const res = await api.createModule('mul', {})
        setNodes(ns => [...ns, { id: res.module.id, type, position, data: {} }])

      } else if (type === 'sum') {
        const res = await api.createModule('sum', {})
        setNodes(ns => [...ns, { id: res.module.id, type, position, data: {} }])

      } else if (type === 'view') {
        const res = await api.createModule('view', { shape: '' })
        const data: ViewNodeData = { shape: '' }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'flatten') {
        const res = await api.createModule('flatten', { start_dim: 0, end_dim: -1 })
        const data: FlattenNodeData = { start_dim: 0, end_dim: -1 }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'bincount') {
        const res = await api.createModule('bincount', { minlength: 0 })
        const data: BincountNodeData = { minlength: 0 }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'unsqueeze') {
        const res = await api.createModule('unsqueeze', { dim: 0 })
        const data: UnsqueezeNodeData = { dim: 0 }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'where') {
        const res = await api.createModule('where', { condition: '' })
        const data: WhereNodeData = { condition: '' }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'index_add') {
        const res = await api.createModule('index_add', { dim: 0 })
        const data: IndexAddNodeData = { dim: 0 }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'indexing') {
        const res = await api.createModule('indexing', { expr: 'rows, ...' })
        const data: IndexingNodeData = { expr: 'rows, ...' }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'sigmoid') {
        const res = await api.createModule('sigmoid', {})
        setNodes(ns => [...ns, { id: res.module.id, type, position, data: {} }])

      } else if (type === 'topk') {
        const res = await api.createModule('topk', { k: 1, dim: -1 })
        const data: TopkNodeData = {
          k: (res.module.params.k as number) ?? 1,
          dim: (res.module.params.dim as number) ?? -1,
        }
        setNodes(ns => [...ns, { id: res.module.id, type, position, data }])

      } else if (type === 'blockGroup') {
        const label = e.dataTransfer.getData('application/nnvis-group-label') || 'Group'
        const sourceGroupId = e.dataTransfer.getData('application/nnvis-group-id')
        const template = sourceGroupId ? groupRegistry.find(g => g.id === sourceGroupId) : undefined

        if (template && template.moduleChain.length > 0) {
          // Clone: create only modules at their relative positions, start collapsed
          // Tensors are NOT pre-created — they are generated when the user connects a tensor to the group
          const res = await api.createGroup(label, [])
          const gid = res.group.id

          const PADDING = 24
          const PADDING_TOP = 44
          let maxRelX = 0, maxRelY = 0
          for (const m of template.moduleChain) {
            maxRelX = Math.max(maxRelX, m.relativePosition.x + 200)
            maxRelY = Math.max(maxRelY, m.relativePosition.y + 120)
          }
          // For nested templates, the flat moduleChain contains inner modules whose root-relative
          // positions extend deep below the user's grouped layout — using them yields a too-tall
          // bounding box. Prefer the nestedSnapshot's saved bounds (the user-visible group rect).
          const gw = template.nestedSnapshot
            ? template.nestedSnapshot.expandedWidth
            : maxRelX + PADDING
          const gh = template.nestedSnapshot
            ? template.nestedSnapshot.expandedHeight
            : maxRelY + PADDING_TOP + PADDING

          const newMemberNodes: Node[] = []
          const newMemberIds: string[] = []
          const tmpIdToNewId = new Map<string, string>()

          for (const m of template.moduleChain) {
            try {
              let params: Record<string, unknown> = {}
              if (m.type === 'linear') {
                params = { n_out: (m.data as LinearNodeData).n_out ?? 64 }
              } else if (m.type === 'softmax') {
                params = { dim: (m.data as SoftmaxNodeData).dim ?? -1 }
              } else if (m.type === 'topk') {
                params = { k: (m.data as TopkNodeData).k ?? 1, dim: (m.data as TopkNodeData).dim ?? -1 }
              } else if (m.type === 'view') {
                params = { shape: (m.data as ViewNodeData).shape ?? '' }
              } else if (m.type === 'flatten') {
                params = { start_dim: (m.data as FlattenNodeData).start_dim ?? 0, end_dim: (m.data as FlattenNodeData).end_dim ?? -1 }
              } else if (m.type === 'bincount') {
                params = { minlength: (m.data as BincountNodeData).minlength ?? 0 }
              } else if (m.type === 'unsqueeze') {
                params = { dim: (m.data as UnsqueezeNodeData).dim ?? 0 }
              } else if (m.type === 'where') {
                params = { condition: (m.data as WhereNodeData).condition ?? '' }
              } else if (m.type === 'index_add') {
                params = { dim: (m.data as IndexAddNodeData).dim ?? 0 }
              } else if (m.type === 'indexing') {
                params = { expr: (m.data as IndexingNodeData).expr ?? 'rows, ...' }
              }
              // mul, silu, sigmoid: no params needed
              const mr = await api.createModule(m.type, params)
              tmpIdToNewId.set(m.tmpId, mr.module.id)
              newMemberIds.push(mr.module.id)
              // Rebuild node data from response params
              let nodeData: Record<string, unknown> = {}
              if (m.type === 'linear') nodeData = { n_out: mr.module.params.n_out ?? 64 }
              else if (m.type === 'softmax') nodeData = { dim: mr.module.params.dim ?? -1 }
              else if (m.type === 'topk') nodeData = { k: mr.module.params.k ?? 1, dim: mr.module.params.dim ?? -1 }
              else if (m.type === 'view') nodeData = { shape: mr.module.params.shape ?? '' }
              else if (m.type === 'flatten') nodeData = { start_dim: mr.module.params.start_dim ?? 0, end_dim: mr.module.params.end_dim ?? -1 }
              else if (m.type === 'bincount') nodeData = { minlength: mr.module.params.minlength ?? 0 }
              else if (m.type === 'unsqueeze') nodeData = { dim: mr.module.params.dim ?? 0 }
              else if (m.type === 'where') nodeData = { condition: mr.module.params.condition ?? '' }
              else if (m.type === 'index_add') nodeData = { dim: mr.module.params.dim ?? 0 }
              else if (m.type === 'indexing') nodeData = { expr: mr.module.params.expr ?? 'rows, ...' }
              newMemberNodes.push({
                id: mr.module.id,
                type: m.type,
                position: m.relativePosition,
                data: nodeData,
                parentId: gid,
                extent: 'parent' as const,
                hidden: true,
              })
            } catch { /* skip failed members */ }
          }

          if (newMemberIds.length > 0) {
            await api.patchGroup(gid, { member_ids: newMemberIds })
          }

          // Build topology for new group instance
          let groupModuleChain: string[]
          let groupInternalEdges: EdgeSnapshot[] | undefined

          if (template.internalEdges !== undefined) {
            const newInternalEdges: EdgeSnapshot[] = template.internalEdges.map(e => ({
              from: tmpIdToNewId.get(e.from) ?? e.from,
              fromHandle: e.fromHandle,
              to: tmpIdToNewId.get(e.to) ?? e.to,
              toHandle: e.toHandle,
            }))
            const targetedInNew = new Set(newInternalEdges.map(e => e.to))
            groupModuleChain = newMemberIds.filter(id => !targetedInNew.has(id))
            groupInternalEdges = newInternalEdges
          } else {
            // Legacy sequential: moduleChain = sequential MODULE_TYPES only, in template order
            groupModuleChain = template.moduleChain
              .filter(m => MODULE_TYPES.has(m.type) && tmpIdToNewId.has(m.tmpId))
              .map(m => tmpIdToNewId.get(m.tmpId)!)
            groupInternalEdges = undefined
          }

          // ── Nested re-parenting pass ─────────────────────────────────────
          // If the template captured a nested visual hierarchy, recreate it now.
          // Strategy: the flat clone above already created every backend module and parented it to `gid`.
          // We walk the nested tree, create one backend group per nested visual group, and re-parent
          // modules into their immediate parent group. Wiring (moduleChain/internalEdges) is unchanged.
          const nestedGroupNodes: Node[] = []
          let rootDirectMemberIds: string[] | null = null
          if (template.nestedSnapshot) {
            const cloneNestedLevel = async (members: NestedMember[], parentGid: string): Promise<string[]> => {
              const ids: string[] = []
              for (const m of members) {
                if (m.kind === 'module') {
                  const newId = tmpIdToNewId.get(m.tmpId)
                  if (!newId) continue
                  const idx = newMemberNodes.findIndex(n => n.id === newId)
                  if (idx >= 0) {
                    newMemberNodes[idx] = {
                      ...newMemberNodes[idx],
                      parentId: parentGid,
                      position: m.relativePosition,
                    }
                  }
                  ids.push(newId)
                } else {
                  // Create backend group for this nested visual group
                  let subGid: string
                  try {
                    const subRes = await api.createGroup(m.label, [])
                    subGid = subRes.group.id
                  } catch {
                    continue
                  }
                  const subMemberIds = await cloneNestedLevel(m.members, subGid)
                  nestedGroupNodes.push({
                    id: subGid,
                    type: 'blockGroup',
                    position: m.relativePosition,
                    style: {},
                    data: {
                      label: m.label,
                      collapsed: true,
                      memberIds: subMemberIds,
                      expandedWidth: m.expandedWidth,
                      expandedHeight: m.expandedHeight,
                    } as GroupNodeData,
                    parentId: parentGid,
                    extent: 'parent' as const,
                    hidden: true,  // root starts collapsed → all nested levels hidden initially
                  })
                  await api.patchGroup(subGid, { member_ids: subMemberIds }).catch(console.error)
                  ids.push(subGid)
                }
              }
              return ids
            }
            rootDirectMemberIds = await cloneNestedLevel(template.nestedSnapshot.members, gid)
            // Re-patch the root with its DIRECT memberIds (top-level visual children)
            await api.patchGroup(gid, { member_ids: rootDirectMemberIds }).catch(console.error)
          }

          const groupNode: Node = {
            id: gid,
            type: 'blockGroup',
            position,
            style: {},
            data: {
              label,
              collapsed: true,
              memberIds: rootDirectMemberIds ?? newMemberIds,
              expandedWidth: gw,
              expandedHeight: gh,
              moduleChain: groupModuleChain,
              ...(groupInternalEdges !== undefined ? { internalEdges: groupInternalEdges } : {}),
            } as GroupNodeData,
          }

          // Order matters: parent group nodes must appear BEFORE child nodes in React Flow's array
          // for parentId resolution. Root first, then nested groups (each before its members), then modules.
          setNodes(ns => [groupNode, ...nestedGroupNodes, ...ns, ...newMemberNodes])
        } else {
          // Empty group (no template or empty template) — starts collapsed, shows compact block with handles
          const res = await api.createGroup(label, [])
          const data: GroupNodeData = {
            label,
            collapsed: true,
            memberIds: [],
            expandedWidth: 300,
            expandedHeight: 200,
          }
          setNodes(ns => [...ns, { id: res.group.id, type: 'blockGroup', position, data }])
        }

      } else if (type === 'text') {
        const data: TextNodeData = { text: '', fontSize: 14, color: colors.offwhite }
        setNodes(ns => [...ns, {
          id: `text-${Date.now()}`,
          type,
          position,
          data,
          style: { minWidth: 120, minHeight: 40 },
        }])
      }
    } catch (err) {
      showError((err as Error).message)
    }
  }, [setNodes, pushHistory, groupRegistry])

  const onConnect = useCallback(async (connection: Connection) => {
    const { source, target, sourceHandle, targetHandle } = connection
    if (!source || !target) return

    // ── Var source: visual-only annotation edge, no backend ───────────────────
    const sourceNode = rfRef.current?.getNode(source)
    if (sourceNode?.type === 'var') {
      pushHistory()
      setEdges(es => [...es, {
        id: `var-${source}-${target}-${Date.now()}`,
        source, target, sourceHandle: sourceHandle ?? undefined, targetHandle: targetHandle ?? undefined,
        style: { stroke: colors.violet, strokeWidth: 1, strokeDasharray: '5 4', opacity: 0.7 },
        markerEnd: { type: MarkerType.ArrowClosed, color: colors.violet },
      }])
      return
    }

    // ── Group target: chain source tensor through all modules ──────────────────
    // Also intercept direct connections to the first module of a group chain
    const rawTargetNode = rfRef.current?.getNode(target)
    let groupTarget = target
    if (rawTargetNode?.type !== 'blockGroup' && rawTargetNode?.parentId) {
      const parentNode = rfRef.current?.getNode(rawTargetNode.parentId)
      if (parentNode?.type === 'blockGroup') {
        const parentChain = (parentNode.data as GroupNodeData).moduleChain ?? []
        if (parentChain[0] === target) groupTarget = rawTargetNode.parentId
      }
    }
    const targetNode = rfRef.current?.getNode(groupTarget)

    if (targetNode?.type === 'blockGroup') {
      const gd = targetNode.data as GroupNodeData
      const entryModuleIds = gd.moduleChain ?? []
      if (entryModuleIds.length === 0) { showError('Group has no modules'); return }

      pushHistory()
      const isCollapsed = gd.collapsed
      const allNewNodes: Node[] = []
      const allNewEdges: Edge[] = []
      const newMemberIds: string[] = []
      const newBoundaryEdges: SavedEdge[] = [...(gd.savedBoundaryEdges ?? [])]
      // For modules nested inside child groups, intermediate auto-tensors must be parented
      // to the IMMEDIATE parent (the child group), not the outer groupTarget — otherwise
      // module-relative position math lands the tensor in the wrong coordinate space.
      // Track per-parent member additions so each nested group's memberIds & backend record stay correct.
      const newMembersByParent = new Map<string, string[]>()
      const recordMember = (parentId: string, mid: string) => {
        if (parentId === groupTarget) {
          newMemberIds.push(mid)
        } else {
          let arr = newMembersByParent.get(parentId)
          if (!arr) { arr = []; newMembersByParent.set(parentId, arr) }
          arr.push(mid)
        }
      }
      // A node is hidden if ANY ancestor group (incl. itself if it's a group) is collapsed.
      const isAnyAncestorCollapsed = (nodeId: string | undefined): boolean => {
        let cur: string | undefined = nodeId
        while (cur) {
          const n = rfRef.current?.getNode(cur)
          if (!n) return false
          if (n.type === 'blockGroup' && (n.data as GroupNodeData).collapsed) return true
          cur = n.parentId
        }
        return false
      }
      // First visible ancestor walking up the parentId chain (or self if visible).
      const visibleAncestor = (nodeId: string): string => {
        let cur: string | undefined = nodeId
        while (cur) {
          const n: Node | undefined = rfRef.current?.getNode(cur)
          if (!n) return nodeId
          if (!n.hidden) return cur
          cur = n.parentId
        }
        return nodeId
      }
      // Per-collapsed-sub-group savedBoundaryEdges so expanding a sub-group later restores edges.
      const newBoundaryByGroup = new Map<string, SavedEdge[]>()
      const addBoundaryForGroup = (groupId: string, se: SavedEdge) => {
        let arr = newBoundaryByGroup.get(groupId)
        if (!arr) { arr = []; newBoundaryByGroup.set(groupId, arr) }
        arr.push(se)
      }

      // Position below the source module's nearest visible ancestor (so each module's final
      // outputs land below ITS visible representation — e.g. EXPERT outputs below EXPERT pill,
      // GATE outputs below GATE pill — instead of all stacking at root group's bottom).
      const getFinalOutputBase = (moduleId?: string) => {
        // No moduleId or outer group is collapsed → fall back to root group placement (legacy)
        if (!moduleId || isCollapsed) {
          const grpNode = rfRef.current?.getNode(groupTarget)
          const expandedH = isCollapsed ? 0 : ((grpNode?.style?.height as number) ?? gd.expandedHeight ?? 200)
          const expandedW = isCollapsed
            ? ((grpNode?.measured as any)?.width ?? COLLAPSED_NODE_W)
            : ((grpNode?.style?.width as number) ?? gd.expandedWidth ?? 300)
          return {
            centerX: (grpNode?.position.x ?? 400) + expandedW / 2,
            y: (grpNode?.position.y ?? 0) + expandedH + 60,
          }
        }
        const refId = visibleAncestor(moduleId)
        const refNode = rfRef.current?.getNode(refId)
        if (!refNode) return { centerX: 400, y: 300 }
        let w = 0, h = 0
        if (refNode.type === 'blockGroup') {
          const rgd = refNode.data as GroupNodeData
          if (rgd.collapsed) {
            w = (refNode.measured as any)?.width ?? COLLAPSED_NODE_W
            h = (refNode.measured as any)?.height ?? COLLAPSED_HEIGHT
          } else {
            w = (refNode.style?.width as number) ?? rgd.expandedWidth ?? 300
            h = (refNode.style?.height as number) ?? rgd.expandedHeight ?? 200
          }
        } else {
          w = (refNode.measured as any)?.width ?? 180
          h = (refNode.measured as any)?.height ?? 120
        }
        const allCurrentNodes = rfRef.current?.getNodes() ?? []
        const abs = getAbsolutePosition(refId, allCurrentNodes)
        return { centerX: abs.x + w / 2, y: abs.y + h + 60 }
      }

      try {
        if (gd.internalEdges !== undefined) {
          // ── Graph-based auto-wiring (topology-aware) ──────────────────────────
          // Build downstream adjacency from internalEdges
          const downstream = new Map<string, Array<{to: string; fromHandle: string; toHandle: string}>>()
          for (const e of gd.internalEdges) {
            if (!downstream.has(e.from)) downstream.set(e.from, [])
            downstream.get(e.from)!.push({ to: e.to, fromHandle: e.fromHandle, toHandle: e.toHandle })
          }

          type QItem = { tensorId: string; moduleId: string; targetHandle: string | null; externalIdx: number }
          const queue: QItem[] = entryModuleIds.map((modId, idx) => ({
            tensorId: source, moduleId: modId, targetHandle: null, externalIdx: idx,
          }))

          while (queue.length > 0) {
            const item = queue.shift()!
            const apiHandle = (item.targetHandle && item.targetHandle !== 'input') ? item.targetHandle : null
            const cr = await api.createConnection(item.tensorId, item.moduleId, apiHandle)

            // Canvas edge for this connection
            if (item.externalIdx >= 0) {
              if (isCollapsed) {
                if (item.externalIdx === 0) {
                  allNewEdges.push({ id: cr.connection.id, source, target: groupTarget, sourceHandle: sourceHandle ?? undefined, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                }
                newBoundaryEdges.push({ id: cr.connection.id, source, sourceHandle: sourceHandle ?? null, target: item.moduleId, targetHandle: 'input' })
              } else {
                // Outer expanded — but item.moduleId might be hidden inside a collapsed nested group.
                // Redirect target to the visible pill and save a boundary for the sub-group.
                const visTarget = visibleAncestor(item.moduleId)
                allNewEdges.push({
                  id: cr.connection.id, source, target: visTarget,
                  sourceHandle: sourceHandle ?? undefined, targetHandle: 'input',
                  style: EDGE_STYLE, markerEnd: EDGE_MARKER,
                })
                if (visTarget !== item.moduleId) {
                  addBoundaryForGroup(visTarget, {
                    id: cr.connection.id, source, sourceHandle: sourceHandle ?? null,
                    target: item.moduleId, targetHandle: 'input',
                  })
                }
              }
            } else {
              // Internal tensor → module edge.
              // Redirect to a collapsed INNER sub-group's pill if target is hidden there. Don't
              // redirect to groupTarget itself — the existing collapsed-root flow (root's
              // memberSet unhides direct-member edges on expand) handles that case correctly.
              const visTarget = visibleAncestor(item.moduleId)
              const targetRedirected = visTarget !== item.moduleId && visTarget !== groupTarget
              allNewEdges.push({
                id: cr.connection.id,
                source: item.tensorId,
                target: targetRedirected ? visTarget : item.moduleId,
                targetHandle: targetRedirected ? 'input' : (item.targetHandle ?? undefined),
                style: EDGE_STYLE, markerEnd: EDGE_MARKER,
                hidden: isCollapsed,
              })
              if (targetRedirected) {
                addBoundaryForGroup(visTarget, {
                  id: cr.connection.id,
                  source: item.tensorId, sourceHandle: null,
                  target: item.moduleId, targetHandle: item.targetHandle ?? null,
                })
              }
            }

            const isLast = !downstream.has(item.moduleId)

            if (cr.output_tensor) {
              const ot = cr.output_tensor
              connectionOutputMap.current.set(cr.connection.id, ot.id)
              if (!isLast) {
                const modNode = rfRef.current?.getNode(item.moduleId)
                const modW = (modNode?.measured as any)?.width ?? 180
                const cX = modNode ? modNode.position.x + modW / 2 : 30
                const modParentId = modNode?.parentId ?? groupTarget
                const tensorHidden = isAnyAncestorCollapsed(modParentId)
                recordMember(modParentId, ot.id)
                allNewNodes.push({
                  id: ot.id, type: 'tensor',
                  position: modNode ? { x: cX - TENSOR_EST_W / 2, y: modNode.position.y + 160 } : { x: 30, y: 120 },
                  data: { name: '', rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData,
                  parentId: modParentId, hidden: tensorHidden,
                })
                pendingCenter.current.set(ot.id, cX)
                allNewEdges.push({ id: `auto-${cr.connection.id}`, source: item.moduleId, target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: tensorHidden })
                for (const d of (downstream.get(item.moduleId) ?? [])) {
                  const th = d.toHandle === 'input' ? null : d.toHandle
                  queue.push({ tensorId: ot.id, moduleId: d.to, targetHandle: th, externalIdx: -1 })
                }
              } else {
                const { centerX, y: finalY } = getFinalOutputBase(item.moduleId)
                pendingCenter.current.set(ot.id, centerX)
                allNewNodes.push({ id: ot.id, type: 'tensor', position: { x: centerX - TENSOR_EST_W / 2, y: finalY }, data: { name: '', rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData })
                if (isCollapsed) {
                  newBoundaryEdges.push({ id: `auto-${cr.connection.id}`, source: item.moduleId, sourceHandle: 'output', target: ot.id, targetHandle: 'input' })
                  allNewEdges.push({ id: `auto-${cr.connection.id}`, source: groupTarget, sourceHandle: 'output', target: ot.id, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                } else {
                  // Source module might be hidden inside a collapsed sub-group → redirect.
                  const visSource = visibleAncestor(item.moduleId)
                  allNewEdges.push({ id: `auto-${cr.connection.id}`, source: visSource, sourceHandle: 'output', target: ot.id, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                  if (visSource !== item.moduleId) {
                    addBoundaryForGroup(visSource, {
                      id: `auto-${cr.connection.id}`,
                      source: item.moduleId, sourceHandle: 'output',
                      target: ot.id, targetHandle: 'input',
                    })
                  }
                }
              }
            } else if (cr.output_tensors && cr.output_tensors.length >= 2) {
              const moduleNodeType = rfRef.current?.getNode(item.moduleId)?.type
              const isWhere = moduleNodeType === 'where'
              const srcHandles = isWhere ? ['rows', 'cols'] : ['values', 'indices']
              const suffixes = isWhere ? ['r', 'c'] : ['v', 'i']
              const downForMod = downstream.get(item.moduleId) ?? []

              for (const [i, ot] of cr.output_tensors.entries()) {
                const srcHandle = srcHandles[i]
                const sfx = suffixes[i]
                const pendingForHandle = downForMod.filter(d => d.fromHandle === srcHandle)
                const hasDownstream = pendingForHandle.length > 0
                const autoEdgeId = `auto-${cr.connection.id}-${sfx}`

                if (!isLast || hasDownstream) {
                  const modNode = rfRef.current?.getNode(item.moduleId)
                  const modW = (modNode?.measured as any)?.width ?? 200
                  const modX = modNode?.position.x ?? 50
                  const modY = modNode?.position.y ?? 50
                  const offsetX = i === 0 ? -TENSOR_EST_W / 2 - TENSOR_TWO_GAP / 2 : TENSOR_EST_W / 2 + TENSOR_TWO_GAP / 2
                  const cX = modX + modW / 2 + offsetX
                  const modParentId = modNode?.parentId ?? groupTarget
                  const tensorHidden = isAnyAncestorCollapsed(modParentId)
                  recordMember(modParentId, ot.id)
                  allNewNodes.push({
                    id: ot.id, type: 'tensor',
                    position: { x: cX - TENSOR_EST_W / 2, y: modY + 160 },
                    data: { name: srcHandle, rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData,
                    parentId: modParentId, hidden: tensorHidden,
                  })
                  pendingCenter.current.set(ot.id, cX)
                  allNewEdges.push({ id: autoEdgeId, source: item.moduleId, sourceHandle: srcHandle, target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: tensorHidden })
                  for (const d of pendingForHandle) {
                    queue.push({ tensorId: ot.id, moduleId: d.to, targetHandle: d.toHandle === 'input' ? null : d.toHandle, externalIdx: -1 })
                  }
                } else {
                  const { centerX: fCX, y: finalY } = getFinalOutputBase(item.moduleId)
                  const offsetX = i === 0 ? -TENSOR_EST_W / 2 - TENSOR_TWO_GAP / 2 : TENSOR_EST_W / 2 + TENSOR_TWO_GAP / 2
                  const otCX = fCX + offsetX
                  allNewNodes.push({
                    id: ot.id, type: 'tensor',
                    position: { x: otCX - TENSOR_EST_W / 2, y: finalY },
                    data: { name: srcHandle, rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData,
                  })
                  pendingCenter.current.set(ot.id, otCX)
                  if (isCollapsed) {
                    newBoundaryEdges.push({ id: autoEdgeId, source: item.moduleId, sourceHandle: srcHandle, target: ot.id })
                    allNewEdges.push({ id: autoEdgeId, source: groupTarget, sourceHandle: 'output', target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                  } else {
                    // Source module may be hidden inside a collapsed sub-group → redirect.
                    const visSource = visibleAncestor(item.moduleId)
                    const sourceRedirected = visSource !== item.moduleId
                    allNewEdges.push({
                      id: autoEdgeId,
                      source: visSource,
                      sourceHandle: sourceRedirected ? 'output' : srcHandle,
                      target: ot.id,
                      style: EDGE_STYLE, markerEnd: EDGE_MARKER,
                    })
                    if (sourceRedirected) {
                      addBoundaryForGroup(visSource, {
                        id: autoEdgeId,
                        source: item.moduleId, sourceHandle: srcHandle,
                        target: ot.id, targetHandle: null,
                      })
                    }
                  }
                }
              }
            }
            // No output: multi-input module waiting for more inputs — queue continues
          }
        } else {
          // ── Legacy sequential auto-wiring ─────────────────────────────────────
          const chain = entryModuleIds
          let currentSrc = source

          for (let i = 0; i < chain.length; i++) {
            const moduleId = chain[i]
            const isFirst = i === 0
            const isLast = i === chain.length - 1
            const cr = await api.createConnection(currentSrc, moduleId)

            if (isFirst) {
              if (isCollapsed) {
                allNewEdges.push({ id: cr.connection.id, source, target: groupTarget, sourceHandle: sourceHandle ?? undefined, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                newBoundaryEdges.push({ id: cr.connection.id, source, sourceHandle: sourceHandle ?? null, target: moduleId, targetHandle: 'input' })
              } else {
                allNewEdges.push({ id: cr.connection.id, source, target: moduleId, sourceHandle: sourceHandle ?? undefined, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
              }
            } else {
              allNewEdges.push({ id: cr.connection.id, source: currentSrc, target: moduleId, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
            }

            if (cr.output_tensor) {
              const ot = cr.output_tensor
              connectionOutputMap.current.set(cr.connection.id, ot.id)
              if (!isLast) {
                const modNode = rfRef.current?.getNode(moduleId)
                const modW = (modNode?.measured as any)?.width ?? 180
                const cX = modNode ? modNode.position.x + modW / 2 : 30
                newMemberIds.push(ot.id)
                allNewNodes.push({
                  id: ot.id, type: 'tensor',
                  position: modNode ? { x: cX - TENSOR_EST_W / 2, y: modNode.position.y + 160 } : { x: 30, y: 120 },
                  data: { name: '', rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData,
                  parentId: groupTarget, hidden: isCollapsed,
                })
                pendingCenter.current.set(ot.id, cX)
                allNewEdges.push({ id: `auto-${cr.connection.id}`, source: moduleId, target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
                currentSrc = ot.id
              } else {
                const { centerX, y: finalY } = getFinalOutputBase()
                pendingCenter.current.set(ot.id, centerX)
                allNewNodes.push({ id: ot.id, type: 'tensor', position: { x: centerX - TENSOR_EST_W / 2, y: finalY }, data: { name: '', rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData })
                if (isCollapsed) {
                  newBoundaryEdges.push({ id: `auto-${cr.connection.id}`, source: moduleId, sourceHandle: 'output', target: ot.id, targetHandle: 'input' })
                  allNewEdges.push({ id: `auto-${cr.connection.id}`, source: groupTarget, sourceHandle: 'output', target: ot.id, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                } else {
                  allNewEdges.push({ id: `auto-${cr.connection.id}`, source: moduleId, sourceHandle: 'output', target: ot.id, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                }
              }
            } else if (cr.output_tensors && cr.output_tensors.length >= 2) {
              const [valsTensor, idxTensor] = cr.output_tensors
              if (!isLast) {
                const modNode = rfRef.current?.getNode(moduleId)
                const modW = (modNode?.measured as any)?.width ?? 200
                const modX = modNode?.position.x ?? 50
                const modY = modNode?.position.y ?? 50
                const grpCX = modX + modW / 2
                const vCX = grpCX - TENSOR_EST_W / 2 - TENSOR_TWO_GAP / 2
                const iCX = grpCX + TENSOR_EST_W / 2 + TENSOR_TWO_GAP / 2
                newMemberIds.push(valsTensor.id, idxTensor.id)
                allNewNodes.push({ id: valsTensor.id, type: 'tensor', position: { x: vCX - TENSOR_EST_W / 2, y: modY + 160 }, data: { name: 'values', rank: rankFromDims(valsTensor.shape.dims), dims: valsTensor.shape.dims, dtype: valsTensor.dtype } as TensorNodeData, parentId: groupTarget, hidden: isCollapsed })
                allNewNodes.push({ id: idxTensor.id, type: 'tensor', position: { x: iCX - TENSOR_EST_W / 2, y: modY + 160 }, data: { name: 'indices', rank: rankFromDims(idxTensor.shape.dims), dims: idxTensor.shape.dims, dtype: idxTensor.dtype } as TensorNodeData, parentId: groupTarget, hidden: isCollapsed })
                pendingCenter.current.set(valsTensor.id, vCX)
                pendingCenter.current.set(idxTensor.id, iCX)
                allNewEdges.push({ id: `auto-${cr.connection.id}-v`, source: moduleId, sourceHandle: 'values', target: valsTensor.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
                allNewEdges.push({ id: `auto-${cr.connection.id}-i`, source: moduleId, sourceHandle: 'indices', target: idxTensor.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
                currentSrc = valsTensor.id
              } else {
                const { centerX, y: finalY } = getFinalOutputBase()
                const lvCX = centerX - TENSOR_EST_W / 2 - TENSOR_TWO_GAP / 2
                const liCX = centerX + TENSOR_EST_W / 2 + TENSOR_TWO_GAP / 2
                allNewNodes.push({ id: valsTensor.id, type: 'tensor', position: { x: lvCX - TENSOR_EST_W / 2, y: finalY }, data: { name: 'values', rank: rankFromDims(valsTensor.shape.dims), dims: valsTensor.shape.dims, dtype: valsTensor.dtype } as TensorNodeData })
                allNewNodes.push({ id: idxTensor.id, type: 'tensor', position: { x: liCX - TENSOR_EST_W / 2, y: finalY }, data: { name: 'indices', rank: rankFromDims(idxTensor.shape.dims), dims: idxTensor.shape.dims, dtype: idxTensor.dtype } as TensorNodeData })
                pendingCenter.current.set(valsTensor.id, lvCX)
                pendingCenter.current.set(idxTensor.id, liCX)
                if (isCollapsed) {
                  newBoundaryEdges.push({ id: `auto-${cr.connection.id}-v`, source: moduleId, sourceHandle: 'values', target: valsTensor.id })
                  newBoundaryEdges.push({ id: `auto-${cr.connection.id}-i`, source: moduleId, sourceHandle: 'indices', target: idxTensor.id })
                  allNewEdges.push({ id: `auto-${cr.connection.id}-v`, source: groupTarget, sourceHandle: 'output', target: valsTensor.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                  allNewEdges.push({ id: `auto-${cr.connection.id}-i`, source: groupTarget, sourceHandle: 'output', target: idxTensor.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                } else {
                  allNewEdges.push({ id: `auto-${cr.connection.id}-v`, source: moduleId, sourceHandle: 'values', target: valsTensor.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                  allNewEdges.push({ id: `auto-${cr.connection.id}-i`, source: moduleId, sourceHandle: 'indices', target: idxTensor.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                }
              }
            }
          }
        }

        const updatedMemberIds = [...gd.memberIds, ...newMemberIds]
        if (newMemberIds.length > 0) api.patchGroup(groupTarget, { member_ids: updatedMemberIds }).catch(console.error)
        // Patch each nested parent group (whose modules produced intermediate tensors during this connect)
        for (const [parentId, additions] of newMembersByParent) {
          const parentNode = rfRef.current?.getNode(parentId)
          const existing = ((parentNode?.data as GroupNodeData)?.memberIds ?? [])
          const merged = [...existing, ...additions]
          api.patchGroup(parentId, { member_ids: merged }).catch(console.error)
        }
        setNodes(ns => [
          ...ns.map(n => {
            if (n.id === groupTarget) {
              return { ...n, data: { ...n.data, memberIds: updatedMemberIds, savedBoundaryEdges: newBoundaryEdges } }
            }
            const memberAdds = newMembersByParent.get(n.id)
            const boundaryAdds = newBoundaryByGroup.get(n.id)
            if (memberAdds || boundaryAdds) {
              const ngd = n.data as GroupNodeData
              return {
                ...n,
                data: {
                  ...ngd,
                  ...(memberAdds ? { memberIds: [...(ngd.memberIds ?? []), ...memberAdds] } : {}),
                  ...(boundaryAdds ? { savedBoundaryEdges: [...(ngd.savedBoundaryEdges ?? []), ...boundaryAdds] } : {}),
                },
              }
            }
            return n
          }),
          ...allNewNodes,
        ])
        setEdges(es => [...es, ...allNewEdges])
      } catch (err) { showError((err as Error).message) }
      return
    }

    pushHistory()

    try {
      const res = await api.createConnection(source, target, targetHandle)

      const userEdge: Edge = {
        id: res.connection.id,
        source,
        target,
        sourceHandle,
        targetHandle,
        style: EDGE_STYLE,
        markerEnd: EDGE_MARKER,
      }

      if (res.output_tensors && res.output_tensors.length >= 2) {
        // Multi-output module (topk, where): two tensors centered under module
        const [outA, outB] = res.output_tensors
        const targetType = rfRef.current?.getNode(target)?.type
        const isWhere = targetType === 'where'
        const [nameA, nameB] = isWhere ? ['rows', 'cols'] : ['values', 'indices']
        const [handleA, handleB] = isWhere ? ['rows', 'cols'] : ['values', 'indices']

        const moduleNode = rfRef.current?.getNode(target)
        const modW = (moduleNode?.measured as any)?.width ?? 200
        const centerX = (moduleNode?.position.x ?? 400) + modW / 2
        const baseY = (moduleNode?.position.y ?? 240) + 160
        const aCenterX = centerX - TENSOR_EST_W / 2 - TENSOR_TWO_GAP / 2
        const bCenterX = centerX + TENSOR_EST_W / 2 + TENSOR_TWO_GAP / 2

        const nodeA: Node = {
          id: outA.id, type: 'tensor',
          position: { x: aCenterX - TENSOR_EST_W / 2, y: baseY },
          data: { name: nameA, rank: rankFromDims(outA.shape.dims), dims: outA.shape.dims, dtype: outA.dtype } as TensorNodeData,
        }
        const nodeB: Node = {
          id: outB.id, type: 'tensor',
          position: { x: bCenterX - TENSOR_EST_W / 2, y: baseY },
          data: { name: nameB, rank: rankFromDims(outB.shape.dims), dims: outB.shape.dims, dtype: outB.dtype } as TensorNodeData,
        }
        pendingCenter.current.set(outA.id, aCenterX)
        pendingCenter.current.set(outB.id, bCenterX)
        const edgeA: Edge = { id: `auto-${res.connection.id}-v`, source: target, target: outA.id, sourceHandle: handleA, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER }
        const edgeB: Edge = { id: `auto-${res.connection.id}-i`, source: target, target: outB.id, sourceHandle: handleB, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER }

        setNodes(ns => [...ns, nodeA, nodeB])
        setEdges(es => [...es, userEdge, edgeA, edgeB])
      } else if (res.output_tensor) {
        const ot = res.output_tensor
        const moduleNode = rfRef.current?.getNode(target)
        const modW = (moduleNode?.measured as any)?.width ?? 180
        const moduleCenterX = moduleNode ? moduleNode.position.x + modW / 2 : 400
        const pos = moduleNode
          ? { x: moduleCenterX - TENSOR_EST_W / 2, y: moduleNode.position.y + 160 }
          : { x: 400, y: 300 }
        pendingCenter.current.set(ot.id, moduleCenterX)

        const outputTensorData: TensorNodeData = {
          name: '',
          rank: rankFromDims(ot.shape.dims),
          dims: ot.shape.dims,
          dtype: ot.dtype,
        }

        const outputNode: Node = { id: ot.id, type: 'tensor', position: pos, data: outputTensorData }

        const autoEdge: Edge = {
          id: `auto-${res.connection.id}`,
          source: target,
          target: ot.id,
          sourceHandle: 'output',
          targetHandle: 'input',
          style: EDGE_STYLE,
          markerEnd: EDGE_MARKER,
        }

        connectionOutputMap.current.set(res.connection.id, ot.id)
        setNodes(ns => [...ns, outputNode])
        setEdges(es => [...es, userEdge, autoEdge])
      } else {
        setEdges(es => [...es, userEdge])
      }
    } catch (err) {
      showError((err as Error).message)
    }
  }, [pushHistory, setNodes, setEdges])

  const onGroupCreate = useCallback(async () => {
    const allNodes = rfRef.current?.getNodes() ?? []
    // Allow top-level groups (no parentId) in selection alongside modules/tensors
    const selected = allNodes.filter((n: Node) => n.selected && (n.type !== 'blockGroup' || !n.parentId))
    if (selected.length < 2) return

    pushHistory()

    const PADDING = 24
    const PADDING_TOP = 44

    // Bounding box: use style dimensions for expanded groups, measured for collapsed
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const n of selected) {
      const isGrp = n.type === 'blockGroup'
      const gd = isGrp ? n.data as GroupNodeData : null
      const w = isGrp
        ? (gd?.collapsed ? ((n.measured as any)?.width ?? COLLAPSED_NODE_W) : (n.style?.width as number ?? 200))
        : ((n.measured as any)?.width ?? 180)
      const h = isGrp
        ? (gd?.collapsed ? ((n.measured as any)?.height ?? COLLAPSED_HEIGHT) : (n.style?.height as number ?? 120))
        : ((n.measured as any)?.height ?? 120)
      minX = Math.min(minX, n.position.x)
      minY = Math.min(minY, n.position.y)
      maxX = Math.max(maxX, n.position.x + w)
      maxY = Math.max(maxY, n.position.y + h)
    }

    const gx = minX - PADDING
    const gy = minY - PADDING_TOP
    const gw = maxX - minX + PADDING * 2
    const gh = maxY - minY + PADDING_TOP + PADDING

    try {
      const selectedIds = selected.map((n: Node) => n.id)
      const res = await api.createGroup('Group', selectedIds)
      const gid = res.group.id
      const selectedSet = new Set(selectedIds)

      // Direct modules (not inside a selected group)
      const directModuleNodes = selected
        .filter((n: Node) => ALL_MODULE_TYPES.has(n.type!))
      // Groups in selection
      const selectedGroupNodes = selected.filter((n: Node) => n.type === 'blockGroup')
      const selectedGroupIdSet = new Set(selectedGroupNodes.map((n: Node) => n.id))

      // Collect ALL modules from selected groups (every member at every depth — not just entries).
      // Without this, non-entry inner modules (e.g., the SiLU after the first Linear in EXPERT) are
      // missing from the snapshot, so on clone the BFS can't reach them and outputs never get wired.
      const groupFlatModuleIds = selectedGroupNodes.flatMap((g: Node) =>
        getAllModuleIdsRecursive(g.id, allNodes)
      )
      const groupFlatModules = groupFlatModuleIds
        .map((id: string) => allNodes.find((n: Node) => n.id === id))
        .filter(Boolean) as Node[]

      // All modules (direct + flattened), sorted by absolute Y position
      const allModuleNodes = [...directModuleNodes, ...groupFlatModules]
        .filter((n, i, arr) => arr.findIndex(m => m.id === n.id) === i) // deduplicate
        .sort((a: Node, b: Node) => {
          const aAbs = getAbsolutePosition(a.id, allNodes)
          const bAbs = getAbsolutePosition(b.id, allNodes)
          return aAbs.y - bAbs.y
        })

      const moduleChainSnapshot: ModuleSnapshot[] = allModuleNodes.map((n: Node) => {
        const absPos = getAbsolutePosition(n.id, allNodes)
        return {
          tmpId: n.id,
          type: n.type!,
          data: { ...n.data } as Record<string, unknown>,
          relativePosition: { x: absPos.x - gx, y: absPos.y - gy },
        }
      })

      const allCanvasEdges = rfRef.current?.getEdges() ?? []
      const selectedModuleIdSet = new Set(directModuleNodes.map((n: Node) => n.id))
      const selectedTensorIdSet = new Set(
        selected.filter((n: Node) => n.type === 'tensor').map((n: Node) => n.id)
      )
      const targetNodeSet = new Set([...selectedModuleIdSet, ...selectedGroupIdSet])

      const internalEdges: EdgeSnapshot[] = []

      // 1. Direct module → module/group edges
      for (const modA of directModuleNodes) {
        const autoEdgesOut = allCanvasEdges.filter((e: Edge) =>
          e.source === modA.id && e.id.startsWith('auto-') && selectedTensorIdSet.has(e.target)
        )
        for (const autoEdge of autoEdgesOut) {
          const edgesFromTensor = allCanvasEdges.filter((e: Edge) =>
            e.source === autoEdge.target && !e.id.startsWith('auto-') && targetNodeSet.has(e.target)
          )
          for (const outEdge of edgesFromTensor) {
            const toNode = allNodes.find((n: Node) => n.id === outEdge.target)
            const toIds = toNode?.type === 'blockGroup'
              ? getFlatModuleIds((toNode.data as GroupNodeData).moduleChain ?? [], allNodes)
              : [outEdge.target]
            for (const toId of toIds) {
              internalEdges.push({ from: modA.id, fromHandle: autoEdge.sourceHandle ?? 'output', to: toId, toHandle: outEdge.targetHandle ?? 'input' })
            }
          }
        }
      }

      // 2. Each selected group's own internal edges (already flat module IDs)
      for (const g of selectedGroupNodes) {
        const gd = g.data as GroupNodeData
        if (gd.internalEdges) internalEdges.push(...gd.internalEdges)
      }

      // 3. Cross-group edges: selected group → (auto) → selected tensor → module/group
      for (const g of selectedGroupNodes) {
        const gd = g.data as GroupNodeData
        const exitMods = getGroupExitModules(gd)
        const autoEdgesFromGroup = allCanvasEdges.filter((e: Edge) =>
          e.source === g.id && e.id.startsWith('auto-') && selectedTensorIdSet.has(e.target)
        )
        for (const autoEdge of autoEdgesFromGroup) {
          const edgesFromTensor = allCanvasEdges.filter((e: Edge) =>
            e.source === autoEdge.target && !e.id.startsWith('auto-') && targetNodeSet.has(e.target)
          )
          for (const outEdge of edgesFromTensor) {
            const toNode = allNodes.find((n: Node) => n.id === outEdge.target)
            const toIds = toNode?.type === 'blockGroup'
              ? getFlatModuleIds((toNode.data as GroupNodeData).moduleChain ?? [], allNodes)
              : [outEdge.target]
            for (const exitMod of exitMods) {
              for (const toId of toIds) {
                internalEdges.push({ from: exitMod, fromHandle: 'output', to: toId, toHandle: outEdge.targetHandle ?? 'input' })
              }
            }
          }
        }
      }

      // Entry modules = not targeted by any internal edge
      const targetedByInternal = new Set(internalEdges.map(e => e.to))
      const instanceModuleChain = allModuleNodes
        .filter((n: Node) => !targetedByInternal.has(n.id))
        .map((n: Node) => n.id)

      // Build a visual-hierarchy snapshot ONLY when at least one nested group was selected.
      // This lives alongside the flat moduleChain/internalEdges (which still drive wiring).
      const buildNestedMembers = (parent: Node | null, parentX: number, parentY: number): NestedMember[] => {
        const childNodes: Node[] = parent === null
          ? selected.filter((n: Node) => ALL_MODULE_TYPES.has(n.type!) || n.type === 'blockGroup')
          : ((parent.data as GroupNodeData).memberIds ?? [])
              .map((mid: string) => allNodes.find((x: Node) => x.id === mid))
              .filter((x): x is Node => Boolean(x))
              .filter((x: Node) => ALL_MODULE_TYPES.has(x.type!) || x.type === 'blockGroup')

        return childNodes.map((cn: Node): NestedMember => {
          const abs = getAbsolutePosition(cn.id, allNodes)
          const relPos = { x: abs.x - parentX, y: abs.y - parentY }
          if (cn.type === 'blockGroup') {
            const cgd = cn.data as GroupNodeData
            return {
              kind: 'group',
              tmpId: cn.id,
              label: cgd.label ?? 'Group',
              relativePosition: relPos,
              expandedWidth: cgd.expandedWidth ?? (cn.style?.width as number) ?? 300,
              expandedHeight: cgd.expandedHeight ?? (cn.style?.height as number) ?? 200,
              members: buildNestedMembers(cn, abs.x, abs.y),
            }
          }
          return { kind: 'module', tmpId: cn.id, relativePosition: relPos }
        })
      }
      const nestedSnapshot: NestedGroupSnapshot | undefined = selectedGroupNodes.length > 0
        ? {
            members: buildNestedMembers(null, gx, gy),
            expandedWidth: gw,
            expandedHeight: gh,
          }
        : undefined

      const groupNode: Node = {
        id: gid,
        type: 'blockGroup',
        position: { x: gx, y: gy },
        style: { width: gw, height: gh },
        data: {
          label: 'Group',
          collapsed: false,
          memberIds: selectedIds,
          expandedWidth: gw,
          expandedHeight: gh,
          moduleChain: instanceModuleChain,
          internalEdges,
        } as GroupNodeData,
        selected: false,
        zIndex: -1,
      }

      setNodes(ns => [
        groupNode,
        ...ns.map((n: Node) => {
          if (!selectedSet.has(n.id)) return n
          return {
            ...n,
            position: { x: n.position.x - gx, y: n.position.y - gy },
            parentId: gid,
            extent: 'parent' as const,
            selected: false,
          }
        }),
      ])

      setGroupRegistry(prev =>
        prev.some(g => g.id === gid)
          ? prev
          : [...prev, {
              id: gid,
              label: 'Group',
              moduleChain: moduleChainSnapshot,
              internalEdges,
              ...(nestedSnapshot ? { nestedSnapshot } : {}),
            }]
      )
    } catch (err) {
      showError((err as Error).message)
    }
  }, [pushHistory, setNodes])

  const onKeyDown = useCallback(async (e: React.KeyboardEvent) => {
    // Ctrl+Z / Cmd+Z undo (canvas only — backend not reversed)
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
      e.preventDefault()
      const snapshot = historyRef.current.pop()
      if (snapshot) {
        setNodes(snapshot.nodes)
        setEdges(snapshot.edges)
      }
      return
    }

    if (e.key !== 'Delete' && e.key !== 'Backspace') return

    const allNodes: Node[] = rfRef.current?.getNodes() ?? []
    const selectedNodes = allNodes.filter((n: Node) => n.selected)
    const selectedEdges = rfRef.current?.getEdges().filter((ed: Edge) => ed.selected) ?? []

    if (selectedNodes.length === 0 && selectedEdges.length === 0) return
    pushHistory()

    const backendEdges = selectedEdges.filter((ed: Edge) => !ed.id.startsWith('auto-'))

    const extraNodeIds = new Set<string>()
    const extraEdgeIds = new Set<string>()
    for (const ed of backendEdges) {
      const outId = connectionOutputMap.current.get(ed.id)
      if (outId) {
        extraNodeIds.add(outId)
        extraEdgeIds.add(`auto-${ed.id}`)
        connectionOutputMap.current.delete(ed.id)
      }
      // topk multi-output auto-edges
      extraEdgeIds.add(`auto-${ed.id}-v`)
      extraEdgeIds.add(`auto-${ed.id}-i`)
    }

    const selectedNodeIds = new Set(selectedNodes.map((n: Node) => n.id))
    const selectedEdgeIds = new Set(selectedEdges.map((ed: Edge) => ed.id))

    // Collect ALL transitive member IDs of deleted groups
    const deletedGroupIds = selectedNodes.filter(n => n.type === 'blockGroup').map(n => n.id)
    const deletedMemberIds = collectTransitiveMembers(deletedGroupIds, allNodes)

    // Backend deletes
    for (const n of selectedNodes) {
      if (n.type === 'tensor') api.deleteTensor(n.id).catch(console.error)
      else if (ALL_MODULE_TYPES.has(n.type!)) api.deleteModule(n.id).catch(console.error)
      else if (n.type === 'blockGroup') {
        api.deleteGroup(n.id).catch(console.error)
        for (const mid of deletedMemberIds) {
          const mn = allNodes.find((m: Node) => m.id === mid)
          if (mn?.type === 'tensor') api.deleteTensor(mid).catch(console.error)
          else if (mn?.type === 'blockGroup') api.deleteGroup(mid).catch(console.error)
          else if (mn?.type && ALL_MODULE_TYPES.has(mn.type)) api.deleteModule(mid).catch(console.error)
        }
      }
    }
    for (const ed of backendEdges) {
      api.deleteConnection(ed.id).catch(console.error)
    }
    for (const [connId, outId] of connectionOutputMap.current.entries()) {
      if (deletedMemberIds.has(outId)) connectionOutputMap.current.delete(connId)
    }

    const allDeletedIds = new Set([...selectedNodeIds, ...extraNodeIds, ...deletedMemberIds])

    setNodes(ns => ns.filter(n => !allDeletedIds.has(n.id)))
    setEdges(es => es.filter(ed =>
      !selectedEdgeIds.has(ed.id) &&
      !extraEdgeIds.has(ed.id) &&
      !allDeletedIds.has(ed.source) &&
      !allDeletedIds.has(ed.target)
    ))
    // groupRegistry not touched — sidebar entries survive canvas deletion
  }, [pushHistory, setNodes, setEdges])

  return (
    <AppContext.Provider value={appCallbacks}>
    <div style={{ display: 'flex', width: '100%', height: '100%' }}>
      <Sidebar
        customGroups={groupRegistry}
        onDeleteGroup={onDeleteGroupFromSidebar}
        animState={animState ? ({ currentStep: animState.currentStep, totalSteps: animState.steps.length } satisfies SidebarAnimState) : null}
        onAnimPrev={prevAnimStep}
        onAnimNext={nextAnimStep}
        onAnimEnd={endAnimation}
      />
      <div
        ref={reactFlowWrapper}
        style={{ flex: 1, position: 'relative' }}
        onKeyDown={onKeyDown}
        tabIndex={0}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onInit={onInit}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onSelectionChange={({ nodes: sel }) =>
            setSelectedCount(sel.filter(n => n.type !== 'blockGroup').length)
          }
          nodeTypes={nodeTypes}
          deleteKeyCode={null}
          fitView={false}
          selectionOnDrag={true}
          panOnDrag={[2]}
          panOnScroll={true}
          style={{ background: colors.bg }}
        >
          <Background variant={BackgroundVariant.Dots} gap={24} size={1} color={colors.border} />
          <Controls />

          {selectedCount >= 2 && !animState && (
            <Panel position="top-center">
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={onGroupCreate}
                  style={{
                    background: colors.bgPanel,
                    border: `1px solid ${colors.orange}99`,
                    borderRadius: 6,
                    color: colors.orange,
                    fontSize: 12,
                    padding: '5px 14px',
                    cursor: 'pointer',
                    letterSpacing: '0.04em',
                  }}
                >
                  Group selected
                </button>
                <button
                  onClick={startAnimation}
                  style={{
                    background: colors.bgPanel,
                    border: `1px solid ${colors.teal}99`,
                    borderRadius: 6,
                    color: colors.teal,
                    fontSize: 12,
                    padding: '5px 14px',
                    cursor: 'pointer',
                    letterSpacing: '0.04em',
                  }}
                >
                  Animate
                </button>
              </div>
            </Panel>
          )}
        </ReactFlow>

        {error && (
          <div style={{
            position: 'absolute',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            background: colors.bgPanel,
            border: `1px solid ${colors.error}`,
            borderRadius: 8,
            padding: '8px 16px',
            color: colors.error,
            fontSize: 13,
            pointerEvents: 'none',
            zIndex: 100,
          }}>
            {error}
          </div>
        )}
      </div>
    </div>
    </AppContext.Provider>
  )
}
