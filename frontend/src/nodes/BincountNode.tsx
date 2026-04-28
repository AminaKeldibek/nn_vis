import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type BincountNodeData = {
  minlength: number | string
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.rose,
  border: `2px solid ${colors.bgNode}`,
}

export function BincountNode({ data, id }: NodeProps) {
  const d = data as BincountNodeData
  const [minlength, setMinlength] = useState<number | string>(d.minlength ?? 0)
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(val: string) {
    const parsed = /^\d+$/.test(val) && val !== '' ? parseInt(val, 10) : val
    setMinlength(parsed)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onModuleParamChange(id, { minlength: parsed }), 400)
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
        BINCOUNT
      </span>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ color: colors.textMuted, fontSize: 10 }}>minlength</span>
        <input
          value={String(minlength)}
          maxLength={6}
          onChange={e => handleChange(e.target.value)}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onKeyDown={e => e.stopPropagation()}
          style={{
            width: 44,
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

      <span style={{ color: `${colors.rose}66`, fontSize: 9, letterSpacing: '0.04em' }}>
        → 1D count tensor
      </span>
    </div>
  )
}
