import { useState, useEffect, useRef } from 'react'
import { type NodeProps, Handle, Position, NodeResizer } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type SavedEdge = {
  id: string
  source: string
  sourceHandle?: string | null
  target: string
  targetHandle?: string | null
}

export type EdgeSnapshot = {
  from: string        // module ID (tmpId in registry; actual backend ID in GroupNodeData)
  fromHandle: string  // 'output', 'values', 'indices', 'rows', 'cols'
  to: string          // module ID
  toHandle: string    // 'input', 'self', 'source', 'index'
}

export type GroupNodeData = {
  label: string
  collapsed: boolean
  memberIds: string[]
  expandedWidth: number
  expandedHeight: number
  savedBoundaryEdges?: SavedEdge[]
  positionDelta?: number
  moduleChain?: string[]    // entry module IDs — used when connecting tensors to the group
  internalEdges?: EdgeSnapshot[]  // module→module topology; undefined = legacy sequential
  expandedX?: number        // saved left-edge x when collapsing, restored on expand
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.orange,
  border: `2px solid ${colors.bgNode}`,
}

export function GroupNode({ data, id, selected }: NodeProps) {
  const d = data as GroupNodeData
  const [name, setName] = useState(d.label ?? 'Group')
  const [editing, setEditing] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { onGroupToggle, onGroupRename, onGroupUngroup } = useAppCallbacks()

  useEffect(() => { setName(d.label) }, [d.label])

  function startEdit(e: React.MouseEvent) {
    e.stopPropagation()
    setEditing(true)
    setTimeout(() => inputRef.current?.select(), 0)
  }

  function commitRename() {
    setEditing(false)
    const trimmed = name.trim() || 'Group'
    setName(trimmed)
    onGroupRename(id, trimmed)
  }

  const nameEl = editing ? (
    <input
      ref={inputRef}
      value={name}
      onChange={e => setName(e.target.value)}
      onBlur={commitRename}
      onKeyDown={e => { if (e.key === 'Enter') commitRename(); e.stopPropagation() }}
      onClick={e => e.stopPropagation()}
      onMouseDown={e => e.stopPropagation()}
      style={{
        background: 'transparent',
        border: 'none',
        borderBottom: `1px solid ${colors.orange}99`,
        color: colors.orange,
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: '0.06em',
        outline: 'none',
        textTransform: 'uppercase',
        minWidth: 0,
        width: 90,
      }}
    />
  ) : (
    <span
      onDoubleClick={startEdit}
      style={{
        color: colors.orange,
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: '0.06em',
        cursor: 'text',
        textTransform: 'uppercase',
        userSelect: 'none',
      }}
    >
      {name}
    </span>
  )

  const toggleBtn = (
    <button
      onClick={e => { e.stopPropagation(); onGroupToggle(id) }}
      onMouseDown={e => e.stopPropagation()}
      style={{
        background: `${colors.orange}22`,
        border: `1px solid ${colors.orange}99`,
        borderRadius: 3,
        color: colors.orange,
        fontSize: 15,
        fontWeight: 700,
        width: 20,
        height: 20,
        flexShrink: 0,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        lineHeight: 1,
        padding: 0,
      }}
    >
      {d.collapsed ? '+' : '−'}
    </button>
  )

  const ungroupBtn = d.memberIds.length > 0 ? (
    <button
      onClick={e => { e.stopPropagation(); onGroupUngroup(id) }}
      onMouseDown={e => e.stopPropagation()}
      style={{
        background: 'transparent',
        border: 'none',
        color: `${colors.orange}99`,
        fontSize: 10,
        cursor: 'pointer',
        padding: 0,
        letterSpacing: '0.04em',
        textDecoration: 'underline',
        flexShrink: 0,
      }}
    >
      ungroup
    </button>
  ) : null

  // ── Expanded: bounding-box container ──────────────────────────────────────
  if (!d.collapsed) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          borderRadius: 10,
          border: `1.5px dashed ${colors.orange}88`,
          background: `${colors.orange}08`,
          boxSizing: 'border-box',
          position: 'relative',
          overflow: 'visible',
        }}
      >
        <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
        <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />
        <NodeResizer
          isVisible={!!selected}
          minWidth={160}
          minHeight={80}
          color={colors.orange}
        />

        {/* Label bar in top-left */}
        <div
          style={{
            position: 'absolute',
            top: 8,
            left: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            pointerEvents: 'all',
          }}
        >
          {nameEl}
          {toggleBtn}
          {ungroupBtn}
        </div>
      </div>
    )
  }

  // ── Collapsed: compact orange module block ─────────────────────────────────
  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1px solid ${colors.orange}99`,
        borderRadius: 8,
        padding: '10px 16px',
        minWidth: 180,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 10,
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />
      {nameEl}
      {toggleBtn}
      {ungroupBtn}
    </div>
  )
}
