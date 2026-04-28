import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type LinearNodeData = {
  n_out: number | string
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.teal,
  border: `2px solid ${colors.bgNode}`,
}

export function LinearNode({ data, id }: NodeProps) {
  const d = data as LinearNodeData
  const [nOut, setNOut] = useState<number | string>(d.n_out ?? 64)
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(val: string) {
    const parsed = /^\d+$/.test(val) && val !== '' ? parseInt(val, 10) : val
    setNOut(parsed)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onModuleParamChange(id, { n_out: parsed }), 400)
  }

  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1.5px solid ${colors.tealDark}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 220,
        boxShadow: `0 3px 0 ${colors.tealDark}, 0 5px 18px rgba(0,0,0,0.45)`,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />

      <span style={{ color: colors.teal, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textAlign: 'center' }}>
        LINEAR
      </span>

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', justifyContent: 'center' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <div style={{
            width: 42, height: 34,
            background: `${colors.teal}22`,
            border: `1px solid ${colors.tealDark}`,
            borderRadius: 4,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ color: colors.teal, fontSize: 13, fontStyle: 'italic', fontWeight: 600 }}>W</span>
          </div>
          <span style={{ color: colors.textMuted, fontSize: 9 }}>n_in × n_out</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
          <div style={{
            width: 42, height: 14,
            background: `${colors.teal}22`,
            border: `1px solid ${colors.tealDark}`,
            borderRadius: 4,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ color: colors.teal, fontSize: 13, fontStyle: 'italic', fontWeight: 600 }}>b</span>
          </div>
          <span style={{ color: colors.textMuted, fontSize: 9 }}>n_out</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
        <span style={{ color: colors.textMuted, fontSize: 11 }}>n_out</span>
        <input
          value={String(nOut)}
          maxLength={12}
          onChange={e => handleChange(e.target.value)}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onKeyDown={e => e.stopPropagation()}
          style={{
            width: Math.max(36, Math.min(7, String(nOut).length) * 8 + 12),
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
