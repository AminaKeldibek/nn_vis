import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'

export type SiluNodeData = Record<string, never>

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.teal,
  border: `2px solid ${colors.bgNode}`,
}

// SiLU(x) = x * σ(x) — slight negative dip near x≈-1.3, supralinear for x>0
// x∈[-3,3] → canvas x∈[2,54], y=0 at canvas y=19, scale 4.5px per unit
const NEG_PTS = '2,19.6 10.7,20.1 15,20.2 17.3,20.3 19.3,20.2 23.7,19.7 28,19'
const POS_PTS = '28,19 32.3,17.6 36.7,15.7 41,14.1 45.3,11.1 49.7,8.8 54,6.1'

export function SiluNode(_: NodeProps) {
  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1.5px solid ${colors.tealDark}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 160,
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
        SiLU
      </span>

      <svg width="56" height="38" viewBox="0 0 56 38">
        <line x1="2" y1="19" x2="54" y2="19" stroke={`${colors.teal}33`} strokeWidth="1" />
        <line x1="28" y1="2" x2="28" y2="36" stroke={`${colors.teal}33`} strokeWidth="1" />
        {/* negative-x: slight dip below zero */}
        <polyline points={NEG_PTS} fill="none" stroke={`${colors.teal}77`} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
        {/* positive-x: supralinear growth */}
        <polyline points={POS_PTS} fill="none" stroke={colors.teal} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx="28" cy="19" r="2" fill={colors.teal} />
      </svg>
    </div>
  )
}
