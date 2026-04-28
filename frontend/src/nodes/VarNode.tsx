import { useState } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'

export type VarNodeData = {
  value: string
}

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.violet,
  border: `2px solid ${colors.bgNode}`,
}

export function VarNode({ data }: NodeProps) {
  const d = data as VarNodeData
  const [value, setValue] = useState(d.value ?? 'n')

  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1px solid ${colors.violetDark}`,
        borderRadius: 8,
        padding: '8px 14px',
        minWidth: 100,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 6,
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />

      <span style={{ color: colors.violet, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
        VAR
      </span>

      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <span style={{ color: `${colors.violet}99`, fontSize: 12, fontFamily: 'monospace' }}>=</span>
        <input
          value={value}
          onChange={e => setValue(e.target.value)}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onKeyDown={e => e.stopPropagation()}
          placeholder="n"
          style={{
            width: 64,
            padding: '3px 6px',
            fontSize: 13,
            fontFamily: 'monospace',
            fontWeight: 600,
            background: colors.bg,
            border: `1px solid ${colors.violetDark}`,
            borderRadius: 4,
            color: colors.violet,
            textAlign: 'center',
            outline: 'none',
          }}
        />
      </div>
    </div>
  )
}
