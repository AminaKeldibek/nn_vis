import { useState } from 'react'
import { colors } from '../theme/colors'

type BlockDef = { type: string; label: string; color: string }

const DATA_BLOCKS: BlockDef[] = [
  { type: 'tensor',   label: 'Tensor',     color: colors.tensor },
  { type: 'indexing', label: 'Indexing [ ]', color: colors.tensor },
  { type: 'var',      label: 'Var',        color: colors.violet },
  { type: 'text',     label: 'Text',       color: colors.textMuted },
]

const BLOCK_GROUPS: { label: string; blocks: BlockDef[] }[] = [
  {
    label: 'MODULES',
    blocks: [
      { type: 'linear',  label: 'Linear',  color: colors.teal },
    ],
  },
  {
    label: 'ACTIVATIONS',
    blocks: [
      { type: 'softmax', label: 'Softmax', color: colors.teal },
      { type: 'silu',    label: 'SiLU',    color: colors.teal },
      { type: 'sigmoid', label: 'Sigmoid', color: colors.teal },
    ],
  },
  {
    label: 'OTHER',
    blocks: [
      { type: 'topk',      label: 'Top-K',     color: colors.rose },
      { type: 'mul',       label: 'Mul ×',     color: colors.rose },
      { type: 'sum',       label: 'Sum +',     color: colors.rose },
      { type: 'view',      label: 'View',      color: colors.rose },
      { type: 'flatten',   label: 'Flatten',   color: colors.rose },
      { type: 'bincount',  label: 'Bincount',  color: colors.rose },
      { type: 'unsqueeze', label: 'Unsqueeze', color: colors.rose },
      { type: 'where',     label: 'Where',     color: colors.rose },
      { type: 'index_add', label: 'Index Add_',color: colors.rose },
    ],
  },
]

const ALL_BLOCKS: BlockDef[] = [...DATA_BLOCKS, ...BLOCK_GROUPS.flatMap(g => g.blocks)]

const SECTION_LABEL: React.CSSProperties = {
  color: colors.textMuted,
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: '0.10em',
  paddingLeft: 2,
  marginTop: 4,
  marginBottom: 3,
}

type CustomGroup = { id: string; label: string }

export type SidebarAnimState = {
  currentStep: number
  totalSteps: number
}

type SidebarProps = {
  customGroups?: CustomGroup[]
  onDeleteGroup?: (id: string) => void
  animState?: SidebarAnimState | null
  onAnimPrev?: () => void
  onAnimNext?: () => void
  onAnimEnd?: () => void
}

export function Sidebar({
  customGroups = [],
  onDeleteGroup,
  animState,
  onAnimPrev,
  onAnimNext,
  onAnimEnd,
}: SidebarProps) {
  const [query, setQuery] = useState('')

  function onDragStart(e: React.DragEvent, type: string, label?: string, groupId?: string) {
    e.dataTransfer.setData('application/nnvis-node', type)
    if (label) e.dataTransfer.setData('application/nnvis-group-label', label)
    if (groupId) e.dataTransfer.setData('application/nnvis-group-id', groupId)
    e.dataTransfer.effectAllowed = 'move'
  }

  function blockItem(b: BlockDef, key: string) {
    return (
      <div
        key={key}
        draggable
        onDragStart={e => onDragStart(e, b.type)}
        style={{
          padding: '7px 10px',
          background: colors.bgNode,
          border: `1px solid ${colors.border}`,
          borderRadius: 6,
          cursor: 'grab',
          color: b.color,
          fontSize: 12,
          transition: 'border-color 0.15s',
        }}
        onMouseEnter={e => (e.currentTarget.style.borderColor = b.color)}
        onMouseLeave={e => (e.currentTarget.style.borderColor = colors.border)}
      >
        {b.label}
      </div>
    )
  }

  function customGroupItem(g: CustomGroup) {
    return (
      <div key={g.id} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <div
          draggable
          onDragStart={e => onDragStart(e, 'blockGroup', g.label, g.id)}
          style={{
            flex: 1,
            padding: '7px 10px',
            background: colors.bgNode,
            border: `1px solid ${colors.border}`,
            borderRadius: 6,
            cursor: 'grab',
            color: colors.orange,
            fontSize: 12,
            transition: 'border-color 0.15s',
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          onMouseEnter={e => (e.currentTarget.style.borderColor = colors.orange)}
          onMouseLeave={e => (e.currentTarget.style.borderColor = colors.border)}
        >
          {g.label}
        </div>
        {onDeleteGroup && (
          <button
            onClick={() => onDeleteGroup(g.id)}
            title="Remove from sidebar"
            style={{
              background: 'transparent',
              border: 'none',
              color: colors.textMuted,
              fontSize: 14,
              cursor: 'pointer',
              padding: '2px 4px',
              borderRadius: 4,
              flexShrink: 0,
              lineHeight: 1,
            }}
            onMouseEnter={e => (e.currentTarget.style.color = colors.error ?? '#e05')}
            onMouseLeave={e => (e.currentTarget.style.color = colors.textMuted)}
          >
            ×
          </button>
        )}
      </div>
    )
  }

  const q = query.trim().toLowerCase()

  const blocksContent = q ? (
    // ── Flat filtered list ──────────────────────────────────────────────────
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      {ALL_BLOCKS
        .filter(b => b.label.toLowerCase().includes(q) || b.type.toLowerCase().includes(q))
        .map(b => blockItem(b, b.type))}
      {customGroups
        .filter(g => g.label.toLowerCase().includes(q))
        .map(g => customGroupItem(g))}
    </div>
  ) : (
    // ── Organized sections ──────────────────────────────────────────────────
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      {DATA_BLOCKS.map(b => blockItem(b, b.type))}

      {BLOCK_GROUPS.map(group => (
        <div key={group.label}>
          <div style={SECTION_LABEL}>{group.label}</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {group.blocks.map(b => blockItem(b, b.type))}
          </div>
        </div>
      ))}

      {customGroups.length > 0 && (
        <div>
          <div style={SECTION_LABEL}>CUSTOM</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {customGroups.map(g => customGroupItem(g))}
          </div>
        </div>
      )}
    </div>
  )

  const ANIM_BTN: React.CSSProperties = {
    flex: 1,
    padding: '7px 0',
    background: `${colors.teal}18`,
    border: `1px solid ${colors.teal}55`,
    borderRadius: 5,
    color: colors.teal,
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    letterSpacing: '0.04em',
    transition: 'border-color 0.15s, background 0.15s',
  }

  const ANIM_BTN_DISABLED: React.CSSProperties = {
    ...ANIM_BTN,
    opacity: 0.35,
    cursor: 'default',
  }

  return (
    <aside
      style={{
        width: 180,
        background: colors.bgPanel,
        borderRight: `1px solid ${colors.border}`,
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        userSelect: 'none',
      }}
    >
      {/* Search */}
      <div style={{ padding: '10px 8px 6px', borderBottom: `1px solid ${colors.border}33` }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search blocks…"
          style={{
            width: '100%',
            padding: '5px 8px',
            fontSize: 11,
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 5,
            color: colors.offwhite,
            outline: 'none',
            boxSizing: 'border-box',
          }}
          onFocus={e => (e.currentTarget.style.borderColor = `${colors.teal}88`)}
          onBlur={e => (e.currentTarget.style.borderColor = colors.border)}
        />
      </div>

      {/* Scrollable blocks area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '8px 8px',
          display: 'flex',
          flexDirection: 'column',
          gap: 0,
        }}
      >
        {blocksContent}
      </div>

      {/* Animation controls — sticky at bottom */}
      {animState && (
        <div
          style={{
            borderTop: `1px solid ${colors.border}`,
            padding: '10px 8px 12px',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ color: colors.teal, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em' }}>
              ANIMATE
            </span>
            <span style={{ color: colors.textMuted, fontSize: 11 }}>
              {animState.currentStep + 1} / {animState.totalSteps}
            </span>
          </div>

          <div style={{ display: 'flex', gap: 4, justifyContent: 'center', flexWrap: 'wrap' }}>
            {Array.from({ length: animState.totalSteps }).map((_, i) => (
              <div
                key={i}
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  background: i <= animState.currentStep ? colors.teal : `${colors.teal}33`,
                  border: `1px solid ${i <= animState.currentStep ? colors.teal : `${colors.teal}44`}`,
                  transition: 'background 0.3s, border-color 0.3s',
                }}
              />
            ))}
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={onAnimPrev}
              disabled={animState.currentStep === 0}
              style={animState.currentStep === 0 ? ANIM_BTN_DISABLED : ANIM_BTN}
            >
              ← Prev
            </button>
            <button
              onClick={onAnimNext}
              disabled={animState.currentStep === animState.totalSteps - 1}
              style={animState.currentStep === animState.totalSteps - 1 ? ANIM_BTN_DISABLED : ANIM_BTN}
            >
              Next →
            </button>
          </div>

          <button
            onClick={onAnimEnd}
            style={{
              padding: '5px 0',
              background: 'transparent',
              border: `1px solid ${colors.border}`,
              borderRadius: 5,
              color: colors.textMuted,
              fontSize: 11,
              cursor: 'pointer',
              letterSpacing: '0.04em',
            }}
          >
            ✕ End animation
          </button>
        </div>
      )}
    </aside>
  )
}
