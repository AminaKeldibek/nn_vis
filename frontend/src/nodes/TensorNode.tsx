import { useState, useRef, useEffect } from 'react'
import { type NodeProps, Handle, Position } from '@xyflow/react'
import { colors } from '../theme/colors'
import { useAppCallbacks } from '../AppContext'

const HANDLE_STYLE = {
  width: 10,
  height: 10,
  background: colors.tensorDark,
  border: `2px solid ${colors.bgNode}`,
}

type Dim = number | string
type Rank = 0 | 1 | 2 | 3

export type TensorNodeData = {
  name: string
  rank: Rank
  dims: Dim[]
  dtype: string
}

const CELL = 8
const GAP = 2
const VEC_CAP = 8
const ROW_CAP = 4
const COL_CAP = 6
const STACK_CAP = 5
const LAYER_OFFSET_X = 10
const LAYER_OFFSET_Y = 7

function cellCount(dim: Dim, cap: number): { n: number; truncated: boolean } {
  if (typeof dim === 'string') return { n: 3, truncated: true }
  return { n: Math.min(dim, cap), truncated: dim > cap }
}

function Grid2D({ rows, cols, rowTrunc, colTrunc }: {
  rows: number; cols: number; rowTrunc: boolean; colTrunc: boolean
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: GAP }}>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} style={{ display: 'flex', gap: GAP, alignItems: 'center' }}>
          {Array.from({ length: cols }).map((_, c) => (
            <div key={c} style={{ width: CELL, height: CELL, background: colors.tensor, borderRadius: 2 }} />
          ))}
          {colTrunc && <span style={{ color: colors.tensorDark, fontSize: 9 }}>…</span>}
        </div>
      ))}
      {rowTrunc && <div style={{ color: colors.tensorDark, fontSize: 9, lineHeight: 1 }}>…</div>}
    </div>
  )
}

function Stack3D({ dims }: { dims: Dim[] }) {
  const { n: depth, truncated: depthTrunc } = cellCount(dims[0] ?? 1, STACK_CAP)
  const { n: rows } = cellCount(dims[1] ?? 1, ROW_CAP)
  const { n: cols } = cellCount(dims[2] ?? 1, COL_CAP)

  const layerW = cols * CELL + Math.max(0, cols - 1) * GAP
  const layerH = rows * CELL + Math.max(0, rows - 1) * GAP
  const totalW = layerW + (depth - 1) * LAYER_OFFSET_X + (depthTrunc ? 10 : 0)
  const totalH = layerH + (depth - 1) * LAYER_OFFSET_Y

  return (
    <div style={{ position: 'relative', width: totalW, height: totalH }}>
      {/* Render back→front so front layer paints last and sits on top */}
      {Array.from({ length: depth }).map((_, i) => {
        const left = (depth - 1 - i) * LAYER_OFFSET_X
        const top = i * LAYER_OFFSET_Y
        const opacity = depth > 1 ? 0.35 + (i / (depth - 1)) * 0.65 : 1
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left,
              top,
              width: layerW,
              height: layerH,
              background: colors.tensor,
              opacity,
              borderRadius: 3,
              border: `1px solid ${colors.tensorDark}`,
            }}
          />
        )
      })}
      {depthTrunc && (
        <span style={{
          position: 'absolute',
          left: totalW - 8,
          top: 0,
          color: colors.tensorDark,
          fontSize: 9,
          lineHeight: 1,
        }}>…</span>
      )}
    </div>
  )
}

function TensorCells({ rank, dims }: { rank: Rank; dims: Dim[] }) {
  if (rank === 0) {
    return <div style={{ width: CELL, height: CELL, background: colors.tensor, borderRadius: 2 }} />
  }
  if (rank === 1) {
    const { n, truncated } = cellCount(dims[0] ?? 1, VEC_CAP)
    return (
      <div style={{ display: 'flex', gap: GAP, alignItems: 'center' }}>
        {Array.from({ length: n }).map((_, i) => (
          <div key={i} style={{ width: CELL, height: CELL, background: colors.tensor, borderRadius: 2 }} />
        ))}
        {truncated && <span style={{ color: colors.tensorDark, fontSize: 9 }}>…</span>}
      </div>
    )
  }
  if (rank === 2) {
    const { n: rows, truncated: rowTrunc } = cellCount(dims[0] ?? 1, ROW_CAP)
    const { n: cols, truncated: colTrunc } = cellCount(dims[1] ?? 1, COL_CAP)
    return <Grid2D rows={rows} cols={cols} rowTrunc={rowTrunc} colTrunc={colTrunc} />
  }
  return <Stack3D dims={dims} />
}

function DimInput({ value, onChange }: { value: Dim; onChange: (v: string) => void }) {
  const len = Math.max(2, Math.min(7, String(value).length))
  const width = len * 8 + 12
  return (
    <input
      value={String(value)}
      maxLength={12}
      onChange={e => onChange(e.target.value)}
      onClick={e => e.stopPropagation()}
      onMouseDown={e => e.stopPropagation()}
      onKeyDown={e => e.stopPropagation()}
      style={{
        width,
        padding: '1px 3px',
        fontSize: 11,
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: 3,
        color: colors.offwhite,
        textAlign: 'center',
        outline: 'none',
        transition: 'width 0.1s',
      }}
    />
  )
}

export function TensorNode({ data, id }: NodeProps) {
  const d = data as TensorNodeData
  const [name, setName] = useState(d.name ?? '')
  const [rank, setRank] = useState<Rank>(d.rank ?? 2)
  const [dims, setDims] = useState<Dim[]>(d.dims ?? ['m', 'n'])
  const { onTensorDimsChange } = useAppCallbacks()
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  useEffect(() => {
    setDims(d.dims)
    setRank(d.rank)
  }, [d.dims, d.rank])

  function fireDimsChange(newDims: Dim[]) {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => onTensorDimsChange(id, newDims), 400)
  }

  function handleRankChange(r: Rank) {
    setRank(r)
    let newDims: Dim[] = []
    if (r === 0) newDims = []
    else if (r === 1) newDims = [dims[0] ?? 'm']
    else if (r === 2) newDims = [dims[0] ?? 'm', dims[1] ?? 'n']
    else newDims = [dims[0] ?? 'k', dims[1] ?? 'm', dims[2] ?? 'n']
    setDims(newDims)
    fireDimsChange(newDims)
  }

  function handleDimChange(idx: number, val: string) {
    const parsed: Dim = /^\d+$/.test(val) && val !== '' ? parseInt(val, 10) : val
    const newDims = dims.map((d, i) => (i === idx ? parsed : d))
    setDims(newDims)
    fireDimsChange(newDims)
  }

  return (
    <div
      style={{
        background: `${colors.tensor}18`,
        border: `1px solid ${colors.tensorDark}`,
        borderRadius: 8,
        padding: '8px 10px',
        minWidth: 80,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
      }}
    >
      <Handle type="target" position={Position.Top}    id="input"  style={HANDLE_STYLE} />
      <Handle type="source" position={Position.Bottom} id="output" style={HANDLE_STYLE} />
      {/* Name */}
      <input
        value={name}
        placeholder="name"
        onChange={e => setName(e.target.value)}
        onClick={e => e.stopPropagation()}
        onMouseDown={e => e.stopPropagation()}
        onKeyDown={e => e.stopPropagation()}
        style={{
          background: 'transparent',
          border: 'none',
          borderBottom: `1px solid ${colors.border}`,
          color: colors.offwhite,
          fontSize: 12,
          fontWeight: 600,
          outline: 'none',
          width: '100%',
          paddingBottom: 2,
        }}
      />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
        <span style={{ color: colors.tensor, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
          TENSOR
        </span>
        <div style={{ display: 'flex', gap: 2 }}>
          {([0, 1, 2, 3] as Rank[]).map(r => (
            <button
              key={r}
              onMouseDown={e => e.stopPropagation()}
              onClick={e => { e.stopPropagation(); handleRankChange(r) }}
              style={{
                padding: '1px 4px',
                fontSize: 9,
                borderRadius: 3,
                background: rank === r ? colors.tensor : 'transparent',
                color: rank === r ? colors.bg : colors.textMuted,
                border: `1px solid ${rank === r ? colors.tensor : colors.border}`,
                cursor: 'pointer',
                lineHeight: '14px',
              }}
            >
              {r}D
            </button>
          ))}
        </div>
      </div>

      {/* Visual */}
      <div style={{ display: 'flex', justifyContent: 'center', padding: '4px 0', minHeight: 14 }}>
        <TensorCells rank={rank} dims={dims} />
      </div>

      {/* Dim inputs */}
      {rank === 0 && (
        <div style={{ color: colors.tensorDark, fontSize: 10, textAlign: 'center' }}>scalar</div>
      )}
      {rank === 1 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <span style={{ color: colors.textMuted, fontSize: 11 }}>[</span>
          <DimInput value={dims[0] ?? 'm'} onChange={v => handleDimChange(0, v)} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>]</span>
        </div>
      )}
      {rank === 2 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <span style={{ color: colors.textMuted, fontSize: 11 }}>[</span>
          <DimInput value={dims[0] ?? 'm'} onChange={v => handleDimChange(0, v)} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>×</span>
          <DimInput value={dims[1] ?? 'n'} onChange={v => handleDimChange(1, v)} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>]</span>
        </div>
      )}
      {rank === 3 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexWrap: 'wrap' }}>
          <span style={{ color: colors.textMuted, fontSize: 11 }}>[</span>
          <DimInput value={dims[0] ?? 'k'} onChange={v => handleDimChange(0, v)} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>×</span>
          <DimInput value={dims[1] ?? 'm'} onChange={v => handleDimChange(1, v)} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>×</span>
          <DimInput value={dims[2] ?? 'n'} onChange={v => handleDimChange(2, v)} />
          <span style={{ color: colors.textMuted, fontSize: 11 }}>]</span>
        </div>
      )}
    </div>
  )
}
