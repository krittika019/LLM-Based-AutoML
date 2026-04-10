import React, { useMemo } from 'react'
import { marked } from 'marked'
import ChartPanel from './ChartPanel.jsx'
import ModelTable from './ModelTable.jsx'

// Configure marked: GitHub-flavored markdown, line breaks preserved
marked.use({ gfm: true, breaks: true })

const STATUS_ICON = {
  running: <span className="spin-ring" aria-hidden />,
  done:    <span aria-label="done">✅</span>,
  failed:  <span aria-label="failed">❌</span>,
  waiting: <span aria-label="waiting">⏸</span>,
  pending: <span aria-label="pending" style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>○</span>,
}

// Map phase status to CSS class for left-border colour + pulse animation
const STATUS_CLASS = {
  running: 'status-running',
  done:    'status-done',
  failed:  'status-failed',
  waiting: 'status-waiting',
  pending: 'status-pending',
}

/** Simple markdown → HTML conversion for display inside phase cards. */
function renderMarkdown(text) {
  // The content comes from our own server, so dangerouslySetInnerHTML is acceptable.
  try {
    return marked.parse(String(text ?? ''))
  } catch {
    return String(text ?? '')
  }
}

export default function PhaseCard({ phase, runId, onToggle }) {
  const { num, name, status, messages, charts, modelResults, expanded } = phase

  const statusCls = STATUS_CLASS[status] ?? 'status-waiting'
  const isPending = status === 'pending'

  // Deduplicate consecutive identical messages (node labels can repeat on retry)
  const dedupedMessages = useMemo(() => {
    const out = []
    for (const m of messages) {
      if (out.length === 0 || out[out.length - 1] !== m) out.push(m)
    }
    return out
  }, [messages])

  return (
    <div className={`phase-card ${statusCls}`}>

      {/* ── Header ── */}
      <div
        className="phase-card-header"
        onClick={isPending ? undefined : onToggle}
        role={isPending ? undefined : 'button'}
        tabIndex={isPending ? undefined : 0}
        onKeyDown={isPending ? undefined : e => e.key === 'Enter' && onToggle()}
      >
        <div className="phase-header-left">
          <span
            className="phase-num-badge"
          >
            Phase {num}
          </span>
          <span className="phase-name">{name}</span>
        </div>

        <div className="phase-header-right">
          {STATUS_ICON[status] ?? null}
          <button
            className="phase-toggle-btn"
            onClick={e => { e.stopPropagation(); onToggle() }}
            aria-label={expanded ? 'Collapse' : 'Expand'}
          >
            {expanded ? '⌄ Collapse' : '› Expand'}
          </button>
        </div>
      </div>

      {/* ── Body ── */}
      {expanded && !isPending && (
        <div className="phase-card-body">

          {dedupedMessages.length === 0 && status === 'running' && (
            <p className="phase-processing">Processing…</p>
          )}

          {dedupedMessages.map((msg, i) => (
            <div key={i} className="agent-msg-row">
              <div className="agent-avatar-dot" title="Agent">✦</div>
              <div
                className="phase-message"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(msg) }}
              />
            </div>
          ))}

          {charts && <ChartPanel data={charts} />}

          {modelResults && <ModelTable data={modelResults} />}

        </div>
      )}

    </div>
  )
}
