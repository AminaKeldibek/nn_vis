import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type FlattenNodeData = {
  start_dim: number | string
  end_dim: number | string
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.rose,
  border: `2px solid ${colors.bgNode}`,
}

export function FlattenNode({ data, id }: NodeProps) {
  const d = data as FlattenNodeData
  const [startDim, setStartDim] = useState<number | string>(d.start_dim ?? 0)
  const [endDim, setEndDim] = useState<number | string>(d.end_dim ?? -1)
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(field: 'start_dim' | 'end_dim', val: string) {
    const parsed = /^-?\d+$/.test(val) && val !== '' ? parseInt(val, 10) : val
    if (field === 'start_dim') setStartDim(parsed)
    else setEndDim(parsed)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      const params = {
        start_dim: field === 'start_dim' ? parsed : startDim,
        end_dim: field === 'end_dim' ? parsed : endDim,
      }
      onModuleParamChange(id, params)
    }, 400)
  }

  const inputStyle = {
    width: 36,
    padding: '2px 4px',
    fontSize: 11,
    background: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: 3,
    color: colors.offwhite,
    textAlign: 'center' as const,
    outline: 'none',
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
        FLATTEN
      </span>

      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: colors.textMuted, fontSize: 10 }}>start</span>
          <input
            value={String(startDim)}
            maxLength={4}
            onChange={e => handleChange('start_dim', e.target.value)}
            onClick={e => e.stopPropagation()}
            onMouseDown={e => e.stopPropagation()}
            onKeyDown={e => e.stopPropagation()}
            style={inputStyle}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: colors.textMuted, fontSize: 10 }}>end</span>
          <input
            value={String(endDim)}
            maxLength={4}
            onChange={e => handleChange('end_dim', e.target.value)}
            onClick={e => e.stopPropagation()}
            onMouseDown={e => e.stopPropagation()}
            onKeyDown={e => e.stopPropagation()}
            style={inputStyle}
          />
        </div>
      </div>
    </div>
  )
}
