import React, { useEffect, useState } from 'react'
import PhaseBody from './PhaseBody.jsx'
import { PHASE_COLORS } from '../constants.js'

const STATUS_ICON = {
  running: <span className="spin-ring" aria-hidden />,
  done:    <span aria-label="done">✅</span>,
  failed:  <span aria-label="failed">❌</span>,
  waiting: <span aria-label="waiting">⏸</span>,
  pending: <span className="tab-pending-dot" aria-label="pending">○</span>,
}

function formatElapsed(ms) {
  if (ms == null || ms < 0) return null
  const totalSec = Math.max(0, Math.round(ms / 1000))
  if (totalSec < 60) return `${totalSec}s`
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  return `${min}m ${sec}s`
}

/** Horizontal tab bar over the pipeline phases — one tab per phase, active tab's
 * content shown below. Auto-follows the running phase until the user clicks a tab
 * themselves, at which point it stays put so they can read at their own pace. */
export default function PhaseTabs({ phases }) {
  const [activeIdx, setActiveIdx] = useState(0)
  const [pinned, setPinned] = useState(false)
  const [, setTick] = useState(0)

  // Live-tick the elapsed time on the currently running phase's tab.
  useEffect(() => {
    if (!phases.some(p => p.status === 'running')) return
    const id = setInterval(() => setTick(t => t + 1), 1000)
    return () => clearInterval(id)
  }, [phases])

  useEffect(() => {
    if (pinned) return
    const runningIdx = phases.findIndex(p => p.status === 'running')
    if (runningIdx !== -1) {
      setActiveIdx(runningIdx)
      return
    }
    // Nothing running — settle on the furthest phase that has actually finished.
    const lastDoneIdx = phases.reduce(
      (acc, p, i) => (p.status === 'done' || p.status === 'failed' ? i : acc), -1
    )
    if (lastDoneIdx !== -1) setActiveIdx(lastDoneIdx)
  }, [phases, pinned])

  if (phases.length === 0) return null
  const active = phases[Math.min(activeIdx, phases.length - 1)]

  return (
    <div className="phase-tabs-wrap">
      <div className="phase-tab-bar" role="tablist">
        {phases.map((p, i) => {
          const isActive = i === activeIdx
          const color = PHASE_COLORS[p.num] ?? '#6366f1'
          const elapsedMs = p.status === 'running' && p.startedAt
            ? Date.now() - p.startedAt
            : (p.finishedAt && p.startedAt ? p.finishedAt - p.startedAt : null)
          const elapsed = formatElapsed(elapsedMs)
          return (
            <button
              key={`${p.num}-${i}`}
              role="tab"
              aria-selected={isActive}
              className={`phase-tab-item status-${p.status}${isActive ? ' active' : ''}`}
              style={isActive ? { '--tab-color': color } : undefined}
              onClick={() => { setActiveIdx(i); setPinned(true) }}
            >
              <span className="phase-tab-icon">{STATUS_ICON[p.status] ?? null}</span>
              <span className="phase-tab-label">
                <span className="phase-tab-num">
                  Phase {p.num}{elapsed && <span className="phase-tab-time"> · {elapsed}</span>}
                </span>
                <span className="phase-tab-name">{p.name}</span>
              </span>
            </button>
          )
        })}
      </div>

      <div className="phase-tab-panel">
        {active.status === 'failed' && (
          <div className="phase-tab-failed-banner">⚠️ This phase failed.</div>
        )}
        <PhaseBody phase={active} />
      </div>
    </div>
  )
}
