import { useState, useRef } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

export type TopkNodeData = {
  k: number | string
  dim: number | string
}

const INPUT_HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.rose,
  border: `2px solid ${colors.bgNode}`,
}

const OUTPUT_HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.rose,
  border: `2px solid ${colors.bgNode}`,
}

// Sorted descending bars — top-k highlighted
const BAR_HEIGHTS = [1.0, 0.82, 0.65, 0.42, 0.25, 0.12]
const DEFAULT_K = 3

export function TopkNode({ data, id }: NodeProps) {
  const d = data as TopkNodeData
  const [k, setK] = useState<number | string>(d.k ?? 1)
  const [dim, setDim] = useState<number | string>(d.dim ?? -1)
  const { onModuleParamChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  function handleChange(field: 'k' | 'dim', val: string) {
    const parsed = /^-?\d+$/.test(val) && val !== '' ? parseInt(val, 10) : val
    if (field === 'k') setK(parsed)
    else setDim(parsed)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      const params = { k: field === 'k' ? parsed : k, dim: field === 'dim' ? parsed : dim }
      onModuleParamChange(id, params)
    }, 400)
  }

  const kNum = typeof k === 'number' ? k : DEFAULT_K

  return (
    <div
      style={{
        background: colors.bgNode,
        border: `1.5px solid ${colors.roseDark}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 200,
        boxShadow: `0 3px 0 ${colors.roseDark}, 0 5px 18px rgba(0,0,0,0.45)`,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        alignItems: 'center',
      }}
    >
      <Handle type="target" position={Position.Top} id="input" style={INPUT_HANDLE_STYLE} />

      <span style={{ color: colors.rose, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
        TOP-K
      </span>

      {/* Bar chart glyph — sorted descending, top-k highlighted */}
      <div style={{ display: 'flex', gap: 3, alignItems: 'flex-end', height: 34, position: 'relative' }}>
        {BAR_HEIGHTS.map((h, i) => (
          <div
            key={i}
            style={{
              width: 11,
              height: Math.round(h * 30),
              background: i < kNum ? colors.rose : `${colors.rose}28`,
              border: `1px solid ${i < kNum ? colors.roseDark : `${colors.roseDark}55`}`,
              borderRadius: '2px 2px 0 0',
            }}
          />
        ))}
        {/* cutoff line after k-th bar */}
        {kNum >= 1 && kNum < BAR_HEIGHTS.length && (
          <div style={{
            position: 'absolute',
            left: kNum * 14 - 1,
            top: 0,
            bottom: 0,
            borderLeft: `1.5px dashed ${colors.rose}88`,
          }} />
        )}
      </div>

      {/* Params */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ color: colors.textMuted, fontSize: 11 }}>k</span>
          <input
            value={String(k)}
            maxLength={6}
            onChange={e => handleChange('k', e.target.value)}
            onClick={e => e.stopPropagation()}
            onMouseDown={e => e.stopPropagation()}
            onKeyDown={e => e.stopPropagation()}
            style={{
              width: 40,
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
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ color: colors.textMuted, fontSize: 11 }}>dim</span>
          <input
            value={String(dim)}
            maxLength={4}
            onChange={e => handleChange('dim', e.target.value)}
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

      {/* Output labels */}
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', paddingTop: 2 }}>
        <span style={{ color: `${colors.rose}cc`, fontSize: 10, letterSpacing: '0.04em' }}>values</span>
        <span style={{ color: `${colors.rose}cc`, fontSize: 10, letterSpacing: '0.04em' }}>indices</span>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        id="values"
        style={{ ...OUTPUT_HANDLE_STYLE, left: '30%' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="indices"
        style={{ ...OUTPUT_HANDLE_STYLE, left: '70%' }}
      />
    </div>
  )
}
