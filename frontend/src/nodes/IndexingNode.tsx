import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type IndexingNodeData = {
  expr: string  // content inside brackets, e.g. "rows, ..."
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.tensorDark,
  border: `2px solid ${colors.bgNode}`,
}

const LABEL_STYLE: React.CSSProperties = {
  color: `${colors.tensor}99`,
  fontSize: 9,
  letterSpacing: '0.04em',
  textAlign: 'center' as const,
  flex: 1,
}

export function IndexingNode({ data, id }: NodeProps) {
  const d = data as IndexingNodeData
  const [expr, setExpr] = useState(d.expr ?? 'rows, ...')
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(val: string) {
    setExpr(val)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onModuleParamChange(id, { expr: val }), 500)
  }

  return (
    <div
      style={{
        background: colors.bgNode,
        border: `2px solid ${colors.tensor}`,
        borderRadius: 50,
        padding: '10px 20px',
        minWidth: 230,
        boxShadow: `0 3px 0 ${colors.tensorDark}, 0 5px 18px rgba(0,0,0,0.45)`,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        alignItems: 'center',
      }}
    >
      {/* Handle labels — evenly spaced across width */}
      <div style={{ display: 'flex', width: '100%' }}>
        <span style={LABEL_STYLE}>tensor</span>
        <span style={LABEL_STYLE}>dim_0_idxs</span>
        <span style={LABEL_STYLE}>dim_1_idxs</span>
      </div>

      <Handle type="target" position={Position.Top} id="tensor"     style={{ ...HANDLE_STYLE, left: '17%' }} />
      <Handle type="target" position={Position.Top} id="dim_0_idxs" style={{ ...HANDLE_STYLE, left: '50%' }} />
      <Handle type="target" position={Position.Top} id="dim_1_idxs" style={{ ...HANDLE_STYLE, left: '83%' }} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />

      {/* Bracket expression row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <span style={{ color: colors.tensor, fontSize: 17, fontWeight: 700, lineHeight: 1, userSelect: 'none' }}>
          [
        </span>
        <input
          value={expr}
          onChange={e => handleChange(e.target.value)}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onKeyDown={e => e.stopPropagation()}
          style={{
            width: 100,
            padding: '3px 6px',
            fontSize: 11,
            fontFamily: 'monospace',
            background: `${colors.tensor}15`,
            border: `1px solid ${colors.tensorDark}`,
            borderRadius: 4,
            color: colors.offwhite,
            textAlign: 'center',
            outline: 'none',
          }}
        />
        <span style={{ color: colors.tensor, fontSize: 17, fontWeight: 700, lineHeight: 1, userSelect: 'none' }}>
          ]
        </span>
      </div>
    </div>
  )
}
