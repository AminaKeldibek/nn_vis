import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type SoftmaxNodeData = {
  dim: number | string
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.teal,
  border: `2px solid ${colors.bgNode}`,
}

const HIST_HEIGHTS = [0.25, 0.45, 1.0, 0.6, 0.2]

export function SoftmaxNode({ data, id }: NodeProps) {
  const d = data as SoftmaxNodeData
  const [dim, setDim] = useState<number | string>(d.dim ?? -1)
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(val: string) {
    const parsed = /^-?\d+$/.test(val) && val !== '' ? parseInt(val, 10) : val
    setDim(parsed)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onModuleParamChange(id, { dim: parsed }), 400)
  }

  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1.5px solid ${colors.tealDark}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 180,
        boxShadow: `0 3px 0 ${colors.tealDark}, 0 5px 18px rgba(0,0,0,0.45)`,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        alignItems: 'center',
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />

      <span style={{ color: colors.teal, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
        SOFTMAX
      </span>

      {/* Histogram glyph */}
      <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', height: 36 }}>
        {HIST_HEIGHTS.map((h, i) => (
          <div
            key={i}
            style={{
              width: 12,
              height: Math.round(h * 32),
              background: i === 2 ? colors.teal : `${colors.teal}55`,
              border: `1px solid ${colors.tealDark}`,
              borderRadius: '2px 2px 0 0',
            }}
          />
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ color: colors.textMuted, fontSize: 11 }}>dim</span>
        <input
          value={String(dim)}
          maxLength={4}
          onChange={e => handleChange(e.target.value)}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onKeyDown={e => e.stopPropagation()}
          style={{
            width: 36,
            padding: '2px 4px',
            fontSize: 11,
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            color: colors.offwhite,
            textAlign: 'center',
            outline: 'none',
          }}
        />
      </div>
    </div>
  )
}
