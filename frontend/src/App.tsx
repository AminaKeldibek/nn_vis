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
import { GroupNode, type GroupNodeData, type EdgeSnapshot, type SavedEdge } from './nodes/GroupNode'
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

// Recursively flatten a group's moduleChain to actual backend module IDs
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

type GroupRegistryEntry = {
  id: string
  label: string
  moduleChain: ModuleSnapshot[]
  internalEdges?: EdgeSnapshot[]  // optional for backward compat with old localStorage entries
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
      // Visibility map: which transitive members should be visible (respects child group states)
      const visMap = computeExpandVisibility(groupId, allNodes)

      // Footprints before/after for reflow
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
        // Propagate the footprint change up the parent chain (only meaningful when this group has a parent)
        return reflowAfterFootprintChange(stepped, groupId, prevFootprint, newFootprint)
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

      const prevFootprintCol = getNodeFootprint(group)
      // Approximated collapsed footprint (will be measured precisely later, but this is enough for reflow)
      const newFootprintCol = { w: COLLAPSED_NODE_W, h: COLLAPSED_HEIGHT }

      setNodes(ns => {
        const stepped = ns.map((n: Node) => {
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
              },
            }
          }
          if (allGroupMembers.has(n.id)) return { ...n, hidden: true }
          if (delta && downstreamIds.has(n.id)) {
            return { ...n, position: { ...n.position, y: n.position.y - delta } }
          }
          return n
        })
        return reflowAfterFootprintChange(stepped, groupId, prevFootprintCol, newFootprintCol)
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

        const hasV2 = !!(template?.rootMembers && template.rootMembers.length > 0)
        const hasV1 = !!(template?.moduleChain && template.moduleChain.length > 0)

        if (template && (hasV2 || hasV1)) {
          // Adapt v1 (legacy flat) into v2 (recursive) on the fly so we have one code path.
          const v2Root: { members: MemberSnapshot[]; moduleChain: string[]; internalEdges: EdgeSnapshot[] } =
            hasV2 ? {
              members: template.rootMembers!,
              moduleChain: template.rootModuleChain ?? [],
              internalEdges: template.rootInternalEdges ?? [],
            } : {
              members: (template.moduleChain ?? []).map(m => ({ kind: 'module' as const, ...m })),
              moduleChain: template.internalEdges !== undefined
                ? (() => {
                    const targets = new Set((template.internalEdges ?? []).map(e => e.to))
                    return (template.moduleChain ?? []).map(m => m.tmpId).filter(id => !targets.has(id))
                  })()
                : (template.moduleChain ?? []).filter(m => MODULE_TYPES.has(m.type)).map(m => m.tmpId),
              internalEdges: template.internalEdges ?? [],
            }

          // Module backend factory — params and node data builder per type
          const buildModuleParams = (m: ModuleSnapshot) => {
            switch (m.type) {
              case 'linear': return { n_out: (m.data as LinearNodeData).n_out ?? 64 }
              case 'softmax': return { dim: (m.data as SoftmaxNodeData).dim ?? -1 }
              case 'topk': return { k: (m.data as TopkNodeData).k ?? 1, dim: (m.data as TopkNodeData).dim ?? -1 }
              case 'view': return { shape: (m.data as ViewNodeData).shape ?? '' }
              case 'flatten': return { start_dim: (m.data as FlattenNodeData).start_dim ?? 0, end_dim: (m.data as FlattenNodeData).end_dim ?? -1 }
              case 'bincount': return { minlength: (m.data as BincountNodeData).minlength ?? 0 }
              case 'unsqueeze': return { dim: (m.data as UnsqueezeNodeData).dim ?? 0 }
              case 'where': return { condition: (m.data as WhereNodeData).condition ?? '' }
              case 'index_add': return { dim: (m.data as IndexAddNodeData).dim ?? 0 }
              case 'indexing': return { expr: (m.data as IndexingNodeData).expr ?? 'rows, ...' }
              default: return {}
            }
          }
          const buildNodeData = (m: ModuleSnapshot, returned: any) => {
            switch (m.type) {
              case 'linear': return { n_out: returned.n_out ?? 64 }
              case 'softmax': return { dim: returned.dim ?? -1 }
              case 'topk': return { k: returned.k ?? 1, dim: returned.dim ?? -1 }
              case 'view': return { shape: returned.shape ?? '' }
              case 'flatten': return { start_dim: returned.start_dim ?? 0, end_dim: returned.end_dim ?? -1 }
              case 'bincount': return { minlength: returned.minlength ?? 0 }
              case 'unsqueeze': return { dim: returned.dim ?? 0 }
              case 'where': return { condition: returned.condition ?? '' }
              case 'index_add': return { dim: returned.dim ?? 0 }
              case 'indexing': return { expr: returned.expr ?? 'rows, ...' }
              default: return {}
            }
          }

          // Recursive clone — creates one backend group per visual group at every depth.
          // All groups start collapsed; all non-top-level nodes start hidden.
          const allCreatedNodes: Node[] = []
          const cloneLevel = async (
            members: MemberSnapshot[],
            levelModuleChain: string[],
            levelInternalEdges: EdgeSnapshot[],
            parentGid: string,
            depth: number,
          ): Promise<{ memberIds: string[]; bounds: { w: number; h: number } }> => {
            const tmpToNew = new Map<string, string>()
            const memberIds: string[] = []
            let maxRelX = 0, maxRelY = 0

            for (const member of members) {
              try {
                if (member.kind === 'module') {
                  const mr = await api.createModule(member.type, buildModuleParams(member))
                  tmpToNew.set(member.tmpId, mr.module.id)
                  memberIds.push(mr.module.id)
                  allCreatedNodes.push({
                    id: mr.module.id,
                    type: member.type,
                    position: member.relativePosition,
                    data: buildNodeData(member, mr.module.params),
                    parentId: parentGid,
                    extent: 'parent' as const,
                    hidden: depth >= 1,  // hidden if not at top level (parent starts collapsed)
                  })
                  maxRelX = Math.max(maxRelX, member.relativePosition.x + 200)
                  maxRelY = Math.max(maxRelY, member.relativePosition.y + 120)
                } else {
                  // Nested group — recurse
                  const subRes = await api.createGroup(member.label, [])
                  const subGid = subRes.group.id
                  tmpToNew.set(member.tmpId, subGid)
                  memberIds.push(subGid)
                  const sub = await cloneLevel(
                    member.members,
                    member.moduleChain,
                    member.internalEdges,
                    subGid,
                    depth + 1,
                  )
                  allCreatedNodes.push({
                    id: subGid,
                    type: 'blockGroup',
                    position: member.relativePosition,
                    style: {},
                    data: {
                      label: member.label,
                      collapsed: true,
                      memberIds: sub.memberIds,
                      expandedWidth: member.expandedWidth,
                      expandedHeight: member.expandedHeight,
                      moduleChain: member.moduleChain.map(id => tmpToNew.get(id) ?? id),
                      internalEdges: member.internalEdges.map(e => ({
                        from: tmpToNew.get(e.from) ?? e.from,
                        fromHandle: e.fromHandle,
                        to: tmpToNew.get(e.to) ?? e.to,
                        toHandle: e.toHandle,
                      })),
                    } as GroupNodeData,
                    parentId: parentGid,
                    extent: 'parent' as const,
                    hidden: depth >= 1,
                  })
                  // For nested groups, use their COLLAPSED footprint for the bounding box at the parent level
                  // (parent starts collapsed → all children hidden, so this only matters when expanded later)
                  maxRelX = Math.max(maxRelX, member.relativePosition.x + COLLAPSED_NODE_W)
                  maxRelY = Math.max(maxRelY, member.relativePosition.y + COLLAPSED_HEIGHT)
                }
              } catch (err) {
                console.error('clone level failure', err)
              }
            }

            if (memberIds.length > 0) {
              await api.patchGroup(parentGid, { member_ids: memberIds }).catch(console.error)
            }

            // Remap chain + edges using THIS level's tmpId map
            const remappedChain = levelModuleChain.map(id => tmpToNew.get(id) ?? id)
            const remappedEdges = levelInternalEdges.map(e => ({
              from: tmpToNew.get(e.from) ?? e.from,
              fromHandle: e.fromHandle,
              to: tmpToNew.get(e.to) ?? e.to,
              toHandle: e.toHandle,
            }))
            // Patch the parent group node we already pushed (last group with id=parentGid),
            // since at the time of push we used the un-remapped chain/edges.
            const idx = allCreatedNodes.findIndex(n => n.id === parentGid)
            if (idx >= 0) {
              const gd = allCreatedNodes[idx].data as GroupNodeData
              allCreatedNodes[idx] = {
                ...allCreatedNodes[idx],
                data: { ...gd, moduleChain: remappedChain, internalEdges: remappedEdges, memberIds },
              }
            }
            return { memberIds, bounds: { w: maxRelX, h: maxRelY } }
          }

          // Create the root group
          const res = await api.createGroup(label, [])
          const gid = res.group.id
          const PADDING = 24
          const PADDING_TOP = 44

          // Push the root group up front so cloneLevel can patch it after remapping
          allCreatedNodes.push({
            id: gid,
            type: 'blockGroup',
            position,
            style: {},
            data: {
              label,
              collapsed: true,
              memberIds: [],
              expandedWidth: 300,
              expandedHeight: 200,
              moduleChain: [],
              internalEdges: [],
            } as GroupNodeData,
          })

          const rootResult = await cloneLevel(v2Root.members, v2Root.moduleChain, v2Root.internalEdges, gid, 0)

          // Final size adjustment for the root from observed bounds
          const rootIdx = allCreatedNodes.findIndex(n => n.id === gid)
          if (rootIdx >= 0) {
            const gd = allCreatedNodes[rootIdx].data as GroupNodeData
            allCreatedNodes[rootIdx] = {
              ...allCreatedNodes[rootIdx],
              data: {
                ...gd,
                expandedWidth: rootResult.bounds.w + PADDING,
                expandedHeight: rootResult.bounds.h + PADDING_TOP + PADDING,
              },
            }
          }

          // Order: parents must appear before children for React Flow's parentId resolution
          // We pushed parents before children naturally (root first, then descendants in BFS-ish order),
          // so just append the whole batch.
          setNodes(ns => [...ns, ...allCreatedNodes])
        } else if (template) {
          // Template exists but is empty — fall through to empty group
          const res = await api.createGroup(label, [])
          const data: GroupNodeData = {
            label,
            collapsed: true,
            memberIds: [],
            expandedWidth: 300,
            expandedHeight: 200,
          }
          setNodes(ns => [...ns, { id: res.group.id, type: 'blockGroup', position, data }])
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

      // Helper: get position below group for final outputs
      const getFinalOutputBase = () => {
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

            // ── If this item targets a sub-group, expand it inline ─────────────
            // The sub-group's own moduleChain entries are queued with the same input tensor;
            // its internalEdges merge into downstream; its exit modules carry the parent's downstream forward.
            const itemNode = rfRef.current?.getNode(item.moduleId)
            if (itemNode?.type === 'blockGroup') {
              const subGd = itemNode.data as GroupNodeData
              const subChain = subGd.moduleChain ?? []
              const subEdges = subGd.internalEdges ?? []
              // Merge sub-group's edges into downstream (same map; sub-group ids are namespace-distinct)
              for (const e of subEdges) {
                if (!downstream.has(e.from)) downstream.set(e.from, [])
                downstream.get(e.from)!.push({ to: e.to, fromHandle: e.fromHandle, toHandle: e.toHandle })
              }
              // Sub-group exit nodes inherit the PARENT's downstream that was attached to the sub-group id
              const parentDown = downstream.get(item.moduleId) ?? []
              if (parentDown.length > 0) {
                const subFromSet = new Set(subEdges.map(e => e.from))
                const subAllIds = new Set([...subChain, ...subEdges.map(e => e.from), ...subEdges.map(e => e.to)])
                const exitIds = subEdges.length === 0 ? subChain : [...subAllIds].filter(id => !subFromSet.has(id))
                for (const exitId of exitIds) {
                  if (!downstream.has(exitId)) downstream.set(exitId, [])
                  downstream.get(exitId)!.push(...parentDown)
                }
              }
              // Queue each entry of the sub-group with the same input tensor.
              // For sub-group entries we are still on the EXTERNAL boundary if this item came from outside,
              // so propagate the externalIdx; otherwise it's an internal feed (-1).
              for (let i = 0; i < subChain.length; i++) {
                const subEntry = subChain[i]
                queue.push({
                  tensorId: item.tensorId,
                  moduleId: subEntry,
                  targetHandle: i === 0 ? item.targetHandle : null,
                  externalIdx: item.externalIdx >= 0 && i === 0 ? item.externalIdx : -1,
                })
              }
              continue
            }

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
                allNewEdges.push({ id: cr.connection.id, source, target: item.moduleId, sourceHandle: sourceHandle ?? undefined, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
              }
            } else {
              // Internal tensor → module edge
              allNewEdges.push({ id: cr.connection.id, source: item.tensorId, target: item.moduleId, targetHandle: item.targetHandle ?? undefined, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
            }

            const isLast = !downstream.has(item.moduleId)

            if (cr.output_tensor) {
              const ot = cr.output_tensor
              connectionOutputMap.current.set(cr.connection.id, ot.id)
              if (!isLast) {
                const modNode = rfRef.current?.getNode(item.moduleId)
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
                allNewEdges.push({ id: `auto-${cr.connection.id}`, source: item.moduleId, target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
                for (const d of (downstream.get(item.moduleId) ?? [])) {
                  const th = d.toHandle === 'input' ? null : d.toHandle
                  queue.push({ tensorId: ot.id, moduleId: d.to, targetHandle: th, externalIdx: -1 })
                }
              } else {
                const { centerX, y: finalY } = getFinalOutputBase()
                pendingCenter.current.set(ot.id, centerX)
                allNewNodes.push({ id: ot.id, type: 'tensor', position: { x: centerX - TENSOR_EST_W / 2, y: finalY }, data: { name: '', rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData })
                if (isCollapsed) {
                  newBoundaryEdges.push({ id: `auto-${cr.connection.id}`, source: item.moduleId, sourceHandle: 'output', target: ot.id, targetHandle: 'input' })
                  allNewEdges.push({ id: `auto-${cr.connection.id}`, source: groupTarget, sourceHandle: 'output', target: ot.id, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
                } else {
                  allNewEdges.push({ id: `auto-${cr.connection.id}`, source: item.moduleId, sourceHandle: 'output', target: ot.id, targetHandle: 'input', style: EDGE_STYLE, markerEnd: EDGE_MARKER })
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
                  newMemberIds.push(ot.id)
                  allNewNodes.push({
                    id: ot.id, type: 'tensor',
                    position: { x: cX - TENSOR_EST_W / 2, y: modY + 160 },
                    data: { name: srcHandle, rank: rankFromDims(ot.shape.dims), dims: ot.shape.dims, dtype: ot.dtype } as TensorNodeData,
                    parentId: groupTarget, hidden: isCollapsed,
                  })
                  pendingCenter.current.set(ot.id, cX)
                  allNewEdges.push({ id: autoEdgeId, source: item.moduleId, sourceHandle: srcHandle, target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER, hidden: isCollapsed })
                  for (const d of pendingForHandle) {
                    queue.push({ tensorId: ot.id, moduleId: d.to, targetHandle: d.toHandle === 'input' ? null : d.toHandle, externalIdx: -1 })
                  }
                } else {
                  const { centerX: fCX, y: finalY } = getFinalOutputBase()
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
                    allNewEdges.push({ id: autoEdgeId, source: item.moduleId, sourceHandle: srcHandle, target: ot.id, style: EDGE_STYLE, markerEnd: EDGE_MARKER })
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
        setNodes(ns => [
          ...ns.map(n => n.id === groupTarget ? { ...n, data: { ...n.data, memberIds: updatedMemberIds, savedBoundaryEdges: newBoundaryEdges } } : n),
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

      const directModuleNodes = selected.filter((n: Node) => ALL_MODULE_TYPES.has(n.type!))
      const selectedGroupNodes = selected.filter((n: Node) => n.type === 'blockGroup')
      const directMemberNodes = [...directModuleNodes, ...selectedGroupNodes]
      const directMemberIdSet = new Set(directMemberNodes.map((n: Node) => n.id))
      const selectedTensorIdSet = new Set(
        selected.filter((n: Node) => n.type === 'tensor').map((n: Node) => n.id)
      )

      // Build internalEdges at THIS LEVEL only — references direct member ids (modules or sub-groups).
      const allCanvasEdges = rfRef.current?.getEdges() ?? []
      const internalEdges: EdgeSnapshot[] = []
      for (const fromNode of directMemberNodes) {
        const autoEdgesOut = allCanvasEdges.filter((e: Edge) =>
          e.source === fromNode.id && e.id.startsWith('auto-') && selectedTensorIdSet.has(e.target)
        )
        for (const autoEdge of autoEdgesOut) {
          const edgesFromTensor = allCanvasEdges.filter((e: Edge) =>
            e.source === autoEdge.target && !e.id.startsWith('auto-') && directMemberIdSet.has(e.target)
          )
          for (const outEdge of edgesFromTensor) {
            internalEdges.push({
              from: fromNode.id,
              fromHandle: autoEdge.sourceHandle ?? 'output',
              to: outEdge.target,
              toHandle: outEdge.targetHandle ?? 'input',
            })
          }
        }
      }

      // Entry members = direct members not targeted by any direct internal edge
      const targetedByInternal = new Set(internalEdges.map(e => e.to))
      const instanceModuleChain = directMemberNodes
        .slice()
        .sort((a: Node, b: Node) => {
          const aAbs = getAbsolutePosition(a.id, allNodes)
          const bAbs = getAbsolutePosition(b.id, allNodes)
          return aAbs.y - bAbs.y
        })
        .filter((n: Node) => !targetedByInternal.has(n.id))
        .map((n: Node) => n.id)

      // Recursive snapshot for the registry — captures structure for arbitrary nesting depth.
      const buildMemberSnapshot = (n: Node, parentX: number, parentY: number): MemberSnapshot => {
        const absPos = getAbsolutePosition(n.id, allNodes)
        const relPos = { x: absPos.x - parentX, y: absPos.y - parentY }
        if (n.type === 'blockGroup') {
          const childGd = n.data as GroupNodeData
          const childMembers: MemberSnapshot[] = (childGd.memberIds ?? [])
            .map((mid: string) => allNodes.find((x: Node) => x.id === mid))
            .filter((x): x is Node => Boolean(x))
            .filter((x: Node) => x.type === 'blockGroup' || ALL_MODULE_TYPES.has(x.type!))
            .map((x: Node) => buildMemberSnapshot(x, absPos.x, absPos.y))
          return {
            kind: 'group',
            tmpId: n.id,
            label: childGd.label ?? 'Group',
            relativePosition: relPos,
            expandedWidth: childGd.expandedWidth ?? (n.style?.width as number) ?? 300,
            expandedHeight: childGd.expandedHeight ?? (n.style?.height as number) ?? 200,
            members: childMembers,
            moduleChain: childGd.moduleChain ?? [],
            internalEdges: childGd.internalEdges ?? [],
          }
        }
        return {
          kind: 'module',
          tmpId: n.id,
          type: n.type!,
          data: { ...n.data } as Record<string, unknown>,
          relativePosition: relPos,
        }
      }

      const rootMembers: MemberSnapshot[] = directMemberNodes.map(n => buildMemberSnapshot(n, gx, gy))

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
              rootMembers,
              rootModuleChain: instanceModuleChain,
              rootInternalEdges: internalEdges,
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
