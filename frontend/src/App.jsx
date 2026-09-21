import React, { useState, useRef, useCallback, useEffect } from 'react'
import Sidebar from './components/Sidebar.jsx'
import PhaseTabs from './components/PhaseTabs.jsx'
import InlineScript from './components/InlineScript.jsx'

// ── Helpers ───────────────────────────────────────────────────────────────────

function findLastIdx(arr, pred) {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return i
  }
  return -1
}

// ── Token Cost Table ──────────────────────────────────────────────────────────
const PHASE_LABELS = {
  planning: 'Phase 2 — Planning',
  eda: 'Phase 3 — EDA',
  feature_selection: 'Phase 3 — Feature Selection',
  deep_cleaning: 'Phase 3 — Deep Cleaning',
  feature_engineering: 'Phase 3 — Feature Engineering',
  execution_planning: 'Phase 4 — Execution Planning',
  model_training: 'Phase 5 — Model Training',
  report_generation: 'Phase 7 — Report',
}

function TokenCostTable({ data }) {
  if (!data) return null
  const total = data._total ?? 0
  const cacheHits = data._cache_hits ?? 0
  const rows = Object.entries(data)
    .filter(([k]) => !k.startsWith('_'))
    .sort((a, b) => b[1] - a[1])

  return (
    <div className="token-cost-wrap">
      <div className="token-cost-title">LLM Call Budget</div>
      <div className="token-cost-summary">
        <span><strong>{total}</strong> total calls</span>
        <span className="token-cost-sep">·</span>
        <span><strong>{cacheHits}</strong> cache hits saved</span>
      </div>
      <table className="token-cost-table">
        <thead>
          <tr><th>Phase</th><th>LLM Calls</th><th>Share</th></tr>
        </thead>
        <tbody>
          {rows.map(([phase, count]) => {
            const pct = total > 0 ? Math.round(count / total * 100) : 0
            return (
              <tr key={phase}>
                <td>{PHASE_LABELS[phase] ?? phase}</td>
                <td style={{ textAlign: 'right', fontWeight: 600 }}>{count}</td>
                <td>
                  <div className="token-bar-wrap">
                    <div className="token-bar" style={{ width: `${pct}%` }} />
                    <span className="token-bar-pct">{pct}%</span>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

const TOTAL_PHASES = 7
const PHASE_STUB_NAMES = [
  'Task & Data Understanding',
  'Planning',
  'Preprocessing',
  'Feature Engineering',
  'Modeling',
  'Evaluation',
  'Report Generation',
]

// Feature cards shown on the landing hero
const FEATURES = [
  {
    icon: '🔍',
    title: 'Intelligent EDA',
    desc: 'Automatically discovers distributions, correlations and data quality issues.',
  },
  {
    icon: '🧹',
    title: 'Smart Cleaning',
    desc: 'Imputation, outlier removal and encoding handled end-to-end.',
  },
  {
    icon: '⚙️',
    title: 'Feature Engineering',
    desc: 'Creates and selects the best features guided by your prediction goal.',
  },
  {
    icon: '🏆',
    title: 'Model Selection',
    desc: 'Trains and ranks multiple models, choosing the best for your task.',
  },
]

// ── Run summary (copy-to-clipboard) ────────────────────────────────────────────

function buildRunSummary({ instruction, fileInfo, phases, bestModel }) {
  const lines = ['AutoML Assistant — Run Summary']
  if (fileInfo) {
    lines.push(`Dataset: ${fileInfo.filename} (${fileInfo.rows.toLocaleString()} rows × ${fileInfo.cols} columns)`)
  }
  if (instruction) lines.push(`Goal: ${instruction}`)

  const modelPhase = [...phases].reverse().find(p => p.modelResults?.comparison_table?.length)
  if (modelPhase) {
    const mr = modelPhase.modelResults
    const best = mr.comparison_table.find(r => r.is_best) ?? mr.comparison_table[0]
    if (best) {
      const rawScore = best.test_display_score ?? best.test_score
      const score = rawScore != null && !isNaN(parseFloat(rawScore)) ? parseFloat(rawScore).toFixed(4) : 'N/A'
      lines.push(`Best model: ${(best.model_name ?? '').replace(/^\*BEST\*\s*/, '')}`)
      lines.push(`${mr.metric_display || mr.metric || 'Score'}: ${score}`)
    }
    if (mr.task_type) lines.push(`Task type: ${mr.task_type}`)
  } else if (bestModel) {
    lines.push(`Best model: ${bestModel}`)
  }

  return lines.join('\n')
}

function CopySummaryButton({ instruction, fileInfo, phases, bestModel }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    const text = buildRunSummary({ instruction, fileInfo, phases, bestModel })
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      /* clipboard permission denied — silently ignore */
    }
  }

  return (
    <button
      className={`dl-btn dl-btn-outline${copied ? ' copied' : ''}`}
      onClick={handleCopy}
      title="Copy a plain-text run summary to clipboard"
    >
      {copied ? 'Copied' : '📋 Copy Summary'}
    </button>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Dataset
  const [fileInfo, setFileInfo] = useState(null)

  // Input
  const [instructions, setInstructions] = useState('')

  // Pipeline state
  const [phases, setPhases] = useState([])
  const [isRunning, setIsRunning] = useState(false)
  const [runId, setRunId] = useState(null)
  const [sessionDict, setSessionDict] = useState(null)
  const [scriptAvailable, setScriptAvail] = useState(false)
  const [reportAvailable, setReportAvail] = useState(false)
  const [runError, setRunError] = useState('')
  const [pipelineDone, setPipelineDone] = useState(false)
  const [tokenCost, setTokenCost] = useState(null)
  const [pipelineStarted, setPipelineStarted] = useState(false)
  const [submittedInstruction, setSubmittedInstruction] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const [runSeq, setRunSeq] = useState(0)
  const [bestModel, setBestModel] = useState('')

  const eventSourceRef = useRef(null)
  const bottomRef = useRef(null)

  // Close SSE on unmount
  useEffect(() => () => eventSourceRef.current?.close(), [])

  // Scroll to bottom when new phase appears
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [phases.length])

  // Derived: show landing vs pipeline
  const showLanding = !pipelineStarted

  // ── SSE event dispatcher ────────────────────────────────────────────────────

  const handleEvent = useCallback((event) => {
    if (!event?.type) return

    switch (event.type) {
      case 'phase_start':
        setPhases(prev => {
          // Normal path: find a non-finished stub for this phase and activate it
          const pendingIdx = prev.findIndex(
            p => p.num === event.phase && p.status !== 'done' && p.status !== 'failed'
          )
          if (pendingIdx !== -1) {
            const updated = [...prev]
            updated[pendingIdx] = {
              ...updated[pendingIdx],
              name: event.phase_name,
              status: 'running',
              startedAt: updated[pendingIdx].startedAt ?? Date.now(),
            }
            return updated
          }
          // Loop re-entry: phase already ran and is 'done' — insert a fresh card after it
          const newEntry = {
            num: event.phase, name: event.phase_name, status: 'running',
            messages: [], charts: null, modelResults: null, startedAt: Date.now(),
          }
          const lastIdx = findLastIdx(prev, p => p.num === event.phase)
          if (lastIdx !== -1) {
            const updated = [...prev]
            updated.splice(lastIdx + 1, 0, newEntry)
            return updated
          }
          // Fallback: append at end
          return [...prev, newEntry]
        })
        break

      case 'node_start':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase && p.status === 'running')
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = {
            ...updated[idx],
            messages: [...updated[idx].messages, `_${event.node}_`],
          }
          return updated
        })
        break

      case 'message':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase && p.status === 'running')
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = {
            ...updated[idx],
            messages: [...updated[idx].messages, event.text],
          }
          return updated
        })
        break

      case 'charts':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase)
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], charts: event.data }
          return updated
        })
        break

      case 'phase1_summary':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase)
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], phase1Summary: event.data }
          return updated
        })
        break

      case 'model_results':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase)
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], modelResults: event.data }
          return updated
        })
        break

      case 'phase_done':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase && p.status === 'running')
          if (idx === -1) return prev
          const updated = [...prev]
          const statusMap = { done: 'done', failed: 'failed', waiting: 'waiting' }
          updated[idx] = {
            ...updated[idx],
            status: statusMap[event.status] ?? 'done',
            finishedAt: Date.now(),
          }
          return updated
        })
        break

      case 'token_cost':
        setTokenCost(event.data)
        break

      case 'final':
        setPipelineDone(true)
        setScriptAvail(event.script_available ?? false)
        setReportAvail(event.report_available ?? false)
        setBestModel(event.best_model ?? '')
        if (event.session) setSessionDict(event.session)
        setIsRunning(false)
        eventSourceRef.current?.close()
        eventSourceRef.current = null
        break

      case 'error':
        setRunError(event.message ?? 'Pipeline error')
        setIsRunning(false)
        eventSourceRef.current?.close()
        eventSourceRef.current = null
        break

      case 'heartbeat':
        break

      default:
        break
    }
  }, [])

  // ── Run pipeline ────────────────────────────────────────────────────────────

  const handleRun = useCallback(async () => {
    if (!instructions.trim() || !fileInfo || isRunning) return

    const trimmedInstr = instructions.trim()

    // Archive the current run into history before resetting
    if (pipelineStarted) {
      setChatHistory(prev => [...prev, {
        instruction: submittedInstruction,
        phases,
        runId,
        scriptAvailable,
        reportAvailable,
        runError,
        pipelineDone,
        bestModel,
        fileInfo,
      }])
    }

    // ── Instant visual transition — switch to pipeline view BEFORE any API call ──
    setIsRunning(true)
    setPipelineDone(false)
    setScriptAvail(false)
    setReportAvail(false)
    setRunError('')
    setBestModel('')
    setSubmittedInstruction(trimmedInstr)
    setInstructions('')
    setPipelineStarted(true)
    setRunSeq(s => s + 1)
    setPhases(PHASE_STUB_NAMES.map((name, i) => ({
      num: i + 1,
      name,
      status: i === 0 ? 'running' : 'pending',
      messages: [],
      charts: null,
      modelResults: null,
      startedAt: i === 0 ? Date.now() : null,
    })))

    try {
      const res = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: fileInfo.file_path,
          instructions: trimmedInstr,
          session: sessionDict ?? null,
          filename: fileInfo.filename ?? null,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail ?? `Server error ${res.status}`)
      }

      const { run_id } = await res.json()
      setRunId(run_id)

      eventSourceRef.current?.close()
      const es = new EventSource(`/stream/${run_id}`)
      eventSourceRef.current = es

      es.onmessage = (e) => {
        if (!e.data || e.data.trim() === '' || e.data.trim() === '{}') return
        try {
          handleEvent(JSON.parse(e.data))
        } catch { /* malformed event — ignore */ }
      }

      es.onerror = () => {
        setRunError('Connection to server lost. Is the FastAPI server running?')
        setIsRunning(false)
        es.close()
      }
    } catch (err) {
      setRunError(err.message ?? 'Failed to start pipeline')
      setIsRunning(false)
    }
  }, [instructions, fileInfo, isRunning, sessionDict, handleEvent])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleRun()
    }
  }

  const handleNewFile = (info) => {
    setFileInfo(info)
    setSessionDict(null)
    setPhases([])
    setPipelineDone(false)
    setScriptAvail(false)
    setReportAvail(false)
    setRunError('')
    setPipelineStarted(false)
    setSubmittedInstruction('')
    setChatHistory([])
    setBestModel('')
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="app-root">

      {/* Sidebar backdrop */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        fileInfo={fileInfo}
        onFileInfo={handleNewFile}
      />

      {/* ── Navbar ── */}
      <nav className="navbar">
        <div className="navbar-left">
          <button
            className="hamburger-btn"
            onClick={() => setSidebarOpen(o => !o)}
            aria-label="Toggle data panel"
          >
            ☰
          </button>
          <div className="navbar-brand">
            <div className="navbar-brand-icon">✦</div>
            AutoML Assistant
          </div>
        </div>

        <div className="navbar-right">
          {/* File indicator — visible when a file is loaded */}
          {fileInfo && (
            <div className="file-indicator" title={fileInfo.filename}>
              <span className="file-indicator-dot" />
              <span className="file-indicator-name">{fileInfo.filename}</span>
            </div>
          )}
        </div>
      </nav>

      {/* ── Main content ── */}
      <main className="main-content">

        {/* Landing hero — shown when no pipeline has been started */}
        {showLanding && (
          <div className="landing">
            <div className="hero">
              <div className="hero-icon-wrap">✦</div>
              <h1 className="hero-title">AutoML Assistant</h1>
              <p className="hero-subtitle">
                Upload a CSV dataset, describe your prediction goal, and watch a full
                machine learning pipeline reason and execute end-to-end.
              </p>
            </div>

            <div className="feature-grid">
              {FEATURES.map(f => (
                <div className="feature-card" key={f.title}>
                  <div className="feature-icon">{f.icon}</div>
                  <div>
                    <div className="feature-title">{f.title}</div>
                    <div className="feature-desc">{f.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pipeline view — slides in when running begins */}
        {!showLanding && (
          <div className="pipeline-view">

            {/* ── Historical runs ── */}
            {chatHistory.map((run, ri) => (
              <div key={ri} className="historical-run">
                {run.instruction && (
                  <div className="instruction-pill-wrap">
                    <div className="instruction-pill">{run.instruction}</div>
                    <div className="user-avatar-dot" title="You">👤</div>
                  </div>
                )}
                <section className="phases-section">
                  <PhaseTabs key={`hist-${ri}`} phases={run.phases} />
                  {run.runError && (
                    <div className="error-banner" role="alert">⚠️ {run.runError}</div>
                  )}
                  {run.pipelineDone && (
                    <div className="download-bar">
                      <span className="download-label">🎉 Pipeline complete!</span>
                      <div className="download-buttons">
                        {run.scriptAvailable && (
                          <a href={`/download/script/${run.runId}`} className="dl-btn" download="automl_pipeline.py">
                            ⬇ pipeline.py
                          </a>
                        )}
                        {run.reportAvailable && (
                          <a href={`/download/report/${run.runId}`} className="dl-btn dl-btn-outline" download="automl_stakeholder_report.pdf">
                            📋 Stakeholder PDF
                          </a>
                        )}
                        <CopySummaryButton
                          instruction={run.instruction}
                          fileInfo={run.fileInfo}
                          phases={run.phases}
                          bestModel={run.bestModel}
                        />
                      </div>
                    </div>
                  )}
                  <InlineScript runId={run.runId} visible={run.pipelineDone && run.scriptAvailable} />
                </section>
                <hr className="run-divider" />
              </div>
            ))}

            {/* ── Current run ── */}

            {/* Instruction pill — current run */}
            {submittedInstruction && (
              <div className="instruction-pill-wrap">
                <div className="instruction-pill">{submittedInstruction}</div>
                <div className="user-avatar-dot" title="You">👤</div>
              </div>
            )}

            {/* Progress bar */}
            {(() => {
              const doneCount = phases.filter(p => p.status === 'done').length
              return (
                <div className="progress-bar-wrap">
                  <div className="progress-bar-header">
                    <span>Pipeline Progress</span>
                    <span className={`progress-counter${pipelineDone ? ' complete' : ''}`}>
                      {doneCount}/{TOTAL_PHASES} phases{pipelineDone ? ' ✓' : ''}
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{ width: `${(doneCount / TOTAL_PHASES) * 100}%` }}
                    />
                  </div>
                </div>
              )
            })()}

            <section className="phases-section">

              <PhaseTabs key={runSeq} phases={phases} />

              {runError && (
                <div className="error-banner" role="alert">
                  ⚠️ {runError}
                </div>
              )}

              {pipelineDone && (
                <div className="download-bar">
                  <span className="download-label">🎉 Pipeline complete!</span>
                  <div className="download-buttons">
                    {scriptAvailable && (
                      <a
                        href={`/download/script/${runId}`}
                        className="dl-btn"
                        download="automl_pipeline.py"
                      >
                        ⬇ pipeline.py
                      </a>
                    )}
                    {reportAvailable && (
                      <a
                        href={`/download/report/${runId}`}
                        className="dl-btn dl-btn-outline"
                        download="automl_stakeholder_report.pdf"
                      >
                        📋 Stakeholder PDF
                      </a>
                    )}
                    <CopySummaryButton
                      instruction={submittedInstruction}
                      fileInfo={fileInfo}
                      phases={phases}
                      bestModel={bestModel}
                    />
                  </div>
                </div>
              )}

              {/* Token cost table — shown after pipeline completes */}
              {pipelineDone && tokenCost && <TokenCostTable data={tokenCost} />}

              {/* Inline pipeline script — always visible after completion */}
              <InlineScript runId={runId} visible={pipelineDone && scriptAvailable} />

              <div ref={bottomRef} style={{ height: 1 }} />
            </section>
          </div>
        )}
      </main>

      {/* ── Fixed input bar ── */}
      <div className="input-bar">
        <textarea
          className="instruction-input"
          placeholder={
            fileInfo
              ? 'Describe your prediction goal — e.g. "Predict Survived"'
              : 'Upload a dataset first (☰), then describe your goal…'
          }
          value={instructions}
          onChange={e => setInstructions(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          disabled={isRunning}
        />
        <button
          className={`send-btn${isRunning ? ' running' : ''}`}
          onClick={handleRun}
          disabled={isRunning || !fileInfo || !instructions.trim()}
          title="Run pipeline (Enter)"
          aria-label="Run pipeline"
        >
          {isRunning ? <span className="send-spinner" /> : '➤'}
        </button>
      </div>

    </div>
  )
}
