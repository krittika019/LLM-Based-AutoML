import React from 'react'

export const PHASES = [
  { num: 1, short: 'Understand', title: 'Understand the data' },
  { num: 2, short: 'Plan', title: 'Plan preprocessing' },
  { num: 3, short: 'Prepare', title: 'Prepare the data' },
  { num: 4, short: 'Model plan', title: 'Plan the models' },
  { num: 5, short: 'Train', title: 'Train & evaluate' },
  { num: 6, short: 'Check', title: 'Check & revise' },
  { num: 7, short: 'Deliver', title: 'Deliver' },
]

// Canvas coordinates (viewBox 920 × 380). Snake layout: 1-4 on top, 5-7 back along the bottom.
const W = 920, H = 380, NW = 184, NH = 96
const POS = { 1: [32, 56], 2: [256, 56], 3: [480, 56], 4: [704, 56], 5: [704, 236], 6: [480, 236], 7: [256, 236] }
const EDGES = [
  [1, 2, 'h'], [2, 3, 'h'], [3, 4, 'h'], [4, 5, 'v'], [5, 6, 'hr'], [6, 7, 'hr'],
]

function edgePath(a, b, kind) {
  const [ax, ay] = POS[a], [bx, by] = POS[b]
  if (kind === 'h') return `M${ax + NW} ${ay + NH / 2} L${bx} ${by + NH / 2}`
  if (kind === 'hr') return `M${ax} ${ay + NH / 2} L${bx + NW} ${by + NH / 2}`
  return `M${ax + NW / 2} ${ay + NH} L${bx + NW / 2} ${by}`
}

export function fmtSecs(s) {
  if (s == null || Number.isNaN(s)) return ''
  s = Math.round(s)
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`
}

function nodeSubtitle(num, st, details, now) {
  if (st.status === 'running') {
    const secs = st.startedAt ? (now - st.startedAt) / 1000 : null
    return `Running${secs != null ? ` · ${fmtSecs(secs)}` : ''}`
  }
  if (st.status === 'pending') return 'Waiting'
  if (st.status === 'failed') return 'Failed'
  const stats = details?.stats || []
  if (!details?.available || !stats.length) return 'Done'
  if (num === 5 && stats[1]) return `${stats[0].value} · ${stats[1].value}`
  if (num === 1) return `${stats[0].value} rows · ${stats[1]?.value ?? '?'} cols`
  return `${stats[0].label}: ${stats[0].value}`
}

export default function PipelineCanvas({ status, details, selected, onSelect, minimized, onToggle, rounds, now, running }) {
  const doneCount = PHASES.filter(p => status[p.num]?.status === 'done').length
  const head = (
    <div className="rw-canvas-head">
      <span className="title">Pipeline</span>
      <span className="rw-hint mono">{doneCount}/7</span>
      <div className="rw-progress" aria-hidden="true"><div style={{ width: `${(doneCount / 7) * 100}%` }} /></div>
      {rounds > 1 && <span className="rw-chip">{rounds} training rounds</span>}
      <div className="rw-grow" />
      <button className="rw-btn sm" onClick={onToggle} aria-expanded={!minimized}
        title={minimized ? 'Show the full pipeline graph' : 'Collapse the pipeline to a strip'}>
        {minimized ? (
          <><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M4 6l4 4 4-4" /></svg>Expand</>
        ) : (
          <><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"><path d="M4 10l4-4 4 4" /></svg>Minimize</>
        )}
      </button>
    </div>
  )

  if (minimized) {
    return (
      <div className="rw-card">
        {head}
        <div className="rw-strip" role="tablist" aria-label="Pipeline phases">
          {PHASES.map((p, i) => {
            const st = status[p.num] || {}
            return (
              <React.Fragment key={p.num}>
                {i > 0 && <span className={`rw-step-line${status[PHASES[i - 1].num]?.status === 'done' ? ' done' : ''}`} />}
                <button role="tab" aria-selected={selected === p.num}
                  className={`rw-step${selected === p.num ? ' sel' : ''}`} onClick={() => onSelect(p.num)}>
                  <span className={`rw-dot ${st.status || 'pending'}`} />
                  <span className="n">{p.num}</span>{p.short}
                </button>
              </React.Fragment>
            )
          })}
        </div>
      </div>
    )
  }

  const p6 = POS[6], p3 = POS[3]
  return (
    <div className="rw-card">
      {head}
      <div className="rw-canvas">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <marker id="rw-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0L10 5L0 10z" fill="var(--border-strong)" />
            </marker>
          </defs>
          {EDGES.map(([a, b, k]) => {
            const done = status[a]?.status === 'done' && status[b]?.status && status[b].status !== 'pending'
            return <path key={`${a}-${b}`} d={edgePath(a, b, k)} fill="none" strokeWidth="1.6"
              stroke={done ? 'var(--ok)' : 'var(--border-strong)'} markerEnd="url(#rw-arrow)" vectorEffect="non-scaling-stroke" />
          })}
          <path d={`M${p6[0] + NW / 2 + 40} ${p6[1]} L${p3[0] + NW / 2 + 40} ${p3[1] + NH}`} fill="none"
            stroke={rounds > 1 ? 'var(--accent)' : 'var(--border-strong)'} strokeWidth="1.4" strokeDasharray="5 5"
            markerEnd="url(#rw-arrow)" vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="rw-loop-tag" style={{ left: `${((p6[0] + NW / 2 + 40) / W) * 100}%`, top: `${((p3[1] + NH + p6[1]) / 2 / H) * 100}%` }}>
          {rounds > 1 ? `revised ×${rounds - 1}` : 'revise if below target'}
        </span>
        {PHASES.map(p => {
          const st = status[p.num] || { status: 'pending' }
          const [x, y] = POS[p.num]
          return (
            <button key={p.num} className={`rw-node ${st.status}${selected === p.num ? ' sel' : ''}`}
              style={{ left: `${(x / W) * 100}%`, top: `${(y / H) * 100}%` }}
              onClick={() => onSelect(p.num)} aria-pressed={selected === p.num}>
              <span className="rw-node-top"><span className={`rw-dot ${st.status}`} />Phase {p.num}</span>
              <span className="rw-node-name">{p.title}</span>
              <span className="rw-node-sub">{nodeSubtitle(p.num, st, details?.[String(p.num)], now)}</span>
            </button>
          )
        })}
        {!running && doneCount === 0 && (
          <span className="rw-hint" style={{ position: 'absolute', left: 16, bottom: 12 }}>Upload a dataset and describe your goal to start.</span>
        )}
      </div>
    </div>
  )
}
