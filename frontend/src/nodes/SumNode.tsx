import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'

export type SumNodeData = Record<string, never>

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.rose,
  border: `2px solid ${colors.bgNode}`,
}

export function SumNode(_: NodeProps) {
  return (
    <div
      style={{
        width: 68,
        height: 68,
        borderRadius: '50%',
        background: colors.bgNode,
        border: `1.5px solid ${colors.roseDark}`,
        boxShadow: `0 3px 0 ${colors.roseDark}, 0 5px 18px rgba(0,0,0,0.45)`,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        position: 'relative',
      }}
    >
      <Handle type="target" position={Position.Top}    id="input_a" style={{ ...HANDLE_STYLE, left: '28%' }} />
      <Handle type="target" position={Position.Top}    id="input_b" style={{ ...HANDLE_STYLE, left: '72%' }} />
      <Handle type="source" position={Position.Bottom} id="output"  style={HANDLE_STYLE} />

      <span style={{ color: colors.rose, fontSize: 22, fontWeight: 700, lineHeight: 1, userSelect: 'none' }}>
        +
      </span>
      <span style={{ color: `${colors.rose}88`, fontSize: 9, fontWeight: 600, letterSpacing: '0.06em', lineHeight: 1 }}>
        SUM
      </span>
    </div>
  )
}
