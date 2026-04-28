import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'

export type SigmoidNodeData = Record<string, never>

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.teal,
  border: `2px solid ${colors.bgNode}`,
}

// σ(x) = 1/(1+e^(-x)) sampled at x = -3..3
const SIG_PTS = '3,30.6 10.3,28.4 17.6,23.9 24.9,17 32.2,10.1 39.5,5.6 46.8,3.4'

export function SigmoidNode(_: NodeProps) {
  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1.5px solid ${colors.tealDark}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 160,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        alignItems: 'center',
        boxShadow: `0 3px 0 ${colors.tealDark}, 0 5px 18px rgba(0,0,0,0.45)`,
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />

      <span style={{ color: colors.teal, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
        SIGMOID
      </span>

      <svg width="52" height="36" viewBox="0 0 52 36">
        {/* axes */}
        <line x1="1" y1="17" x2="51" y2="17" stroke={`${colors.teal}33`} strokeWidth="1" />
        <line x1="25" y1="1" x2="25" y2="35" stroke={`${colors.teal}33`} strokeWidth="1" />
        {/* asymptote hints */}
        <line x1="1" y1="3" x2="51" y2="3" stroke={`${colors.teal}22`} strokeWidth="1" strokeDasharray="3 3" />
        <line x1="1" y1="31" x2="51" y2="31" stroke={`${colors.teal}22`} strokeWidth="1" strokeDasharray="3 3" />
        {/* sigmoid curve */}
        <polyline
          points={SIG_PTS}
          fill="none"
          stroke={colors.teal}
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* midpoint dot */}
        <circle cx="24.9" cy="17" r="2" fill={colors.teal} />
      </svg>
    </div>
  )
}
