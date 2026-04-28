import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type ViewNodeData = {
  shape: string
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.rose,
  border: `2px solid ${colors.bgNode}`,
}

export function ViewNode({ data, id }: NodeProps) {
  const d = data as ViewNodeData
  const [shape, setShape] = useState(d.shape ?? '')
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(val: string) {
    setShape(val)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onModuleParamChange(id, { shape: val }), 500)
  }

  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1.5px solid ${colors.roseDark}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 160,
        boxShadow: `0 3px 0 ${colors.roseDark}, 0 5px 18px rgba(0,0,0,0.45)`,
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        alignItems: 'center',
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />

      <span style={{ color: colors.rose, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
        VIEW
      </span>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, width: '100%' }}>
        <span style={{ color: colors.textMuted, fontSize: 10, letterSpacing: '0.04em' }}>new shape</span>
        <input
          value={shape}
          placeholder="3, -1"
          onChange={e => handleChange(e.target.value)}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onKeyDown={e => e.stopPropagation()}
          style={{
            width: '100%',
            padding: '3px 6px',
            fontSize: 12,
            fontFamily: 'monospace',
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 3,
            color: colors.offwhite,
            textAlign: 'center',
            outline: 'none',
            boxSizing: 'border-box' as const,
          }}
        />
      </div>
    </div>
  )
}
