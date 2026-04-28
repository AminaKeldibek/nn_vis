import { useState, useRef, useEffect } from 'react'
import { type NodeProps, NodeResizer } from '@xyflow/react'
import { colors } from '../theme/colors'

export type TextNodeData = {
  text: string
  fontSize: number
  color: string
}

export function TextNode({ data, selected }: NodeProps) {
  const d = data as TextNodeData
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState(d.text ?? 'Text')
  const [fontSize, setFontSize] = useState(d.fontSize ?? 14)
  const [color, setColor] = useState(d.color ?? colors.offwhite)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (editing) textareaRef.current?.focus()
  }, [editing])

  return (
    <>
      <NodeResizer
        isVisible={selected as boolean}
        minWidth={80}
        minHeight={32}
        lineStyle={{ borderColor: colors.teal }}
        handleStyle={{ background: colors.teal, border: 'none', width: 8, height: 8 }}
      />
      <div
        style={{
          width: '100%',
          height: '100%',
          padding: '6px 8px',
          position: 'relative',
        }}
      >
        {selected && (
          <div
            style={{
              position: 'absolute',
              top: -34,
              left: 0,
              display: 'flex',
              gap: 6,
              background: colors.bgPanel,
              border: `1px solid ${colors.border}`,
              borderRadius: 6,
              padding: '4px 8px',
              alignItems: 'center',
              zIndex: 10,
              whiteSpace: 'nowrap',
            }}
          >
            <label style={{ color: colors.textMuted, fontSize: 11 }}>size</label>
            <input
              type="number"
              value={fontSize}
              min={8}
              max={72}
              onChange={e => setFontSize(Number(e.target.value))}
              onKeyDown={e => e.stopPropagation()}
              style={{
                width: 44,
                background: colors.bgNode,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                color: colors.text,
                fontSize: 11,
                padding: '1px 4px',
              }}
            />
            <label style={{ color: colors.textMuted, fontSize: 11 }}>color</label>
            <input
              type="color"
              value={color}
              onChange={e => setColor(e.target.value)}
              style={{ width: 24, height: 20, border: 'none', background: 'none', cursor: 'pointer', padding: 0 }}
            />
          </div>
        )}

        {editing ? (
          <textarea
            ref={textareaRef}
            value={text}
            onChange={e => setText(e.target.value)}
            onBlur={() => setEditing(false)}
            onKeyDown={e => { e.stopPropagation(); if (e.key === 'Escape') setEditing(false) }}
            style={{
              width: '100%',
              height: '100%',
              background: 'transparent',
              border: 'none',
              outline: 'none',
              resize: 'none',
              fontSize,
              color,
              fontFamily: 'inherit',
            }}
          />
        ) : (
          <div
            onDoubleClick={() => setEditing(true)}
            style={{
              fontSize,
              color,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              cursor: 'text',
              minHeight: 20,
            }}
          >
            {text || <span style={{ color: colors.textMuted, fontStyle: 'italic' }}>double-click to edit</span>}
          </div>
        )}
      </div>
    </>
  )
}
