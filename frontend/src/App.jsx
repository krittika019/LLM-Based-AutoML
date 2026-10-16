import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import Dashboard from './run/Dashboard.jsx'
import Predict from './run/Predict.jsx'
import ConfigRail from './run/ConfigRail.jsx'
import PipelineCanvas, { PHASES, fmtSecs } from './run/PipelineCanvas.jsx'
import PhaseInspector from './run/PhaseInspector.jsx'
import './run/run.css'

// ── Helpers ───────────────────────────────────────────────────────────────────

function findLastIdx(arr, pred) {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (pred(arr[i])) return i
  }
  return -1
}

const load = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v) } catch { return d } }
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* storage unavailable */ } }

const stubPhases = () => PHASES.map((p, i) => ({
  num: p.num, name: p.title, status: i === 0 ? 'running' : 'pending',
  messages: [], charts: null, modelResults: null, startedAt: i === 0 ? Date.now() : null,
}))

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  // Dataset + input
  const [fileInfo, setFileInfo] = useState(null)
  const [instructions, setInstructions] = useState('')
  const [mode, setMode] = useState('predict') // 'predict' | 'patterns'

  // Pipeline state
  const [phases, setPhases] = useState([])
  const [phaseDetails, setPhaseDetails] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [runId, setRunId] = useState(null)
  const [sessionDict, setSessionDict] = useState(null)
  const [scriptAvailable, setScriptAvail] = useState(false)
  const [scriptValidation, setScriptValidation] = useState(null)
  const [reportAvailable, setReportAvail] = useState(false)
  const [runError, setRunError] = useState('')
  const [pipelineDone, setPipelineDone] = useState(false)
  const [pipelineStarted, setPipelineStarted] = useState(false)
  const [submittedInstruction, setSubmittedInstruction] = useState('')
  const [runMode, setRunMode] = useState('predict')
  const [chatHistory, setChatHistory] = useState([])
  const [bestModel, setBestModel] = useState('')
  const [predict, setPredict] = useState({ available: false, note: '' })
  const [runStartedAt, setRunStartedAt] = useState(null)
  const [runFinishedAt, setRunFinishedAt] = useState(null)

  // View state
  const [mainView, setMainView] = useState('pipeline') // 'pipeline' | 'analytics'
  const [viewing, setViewing] = useState(null) // index into chatHistory, null = current run
  const [selected, setSelected] = useState(1)
  const [dark, setDark] = useState(() => load('rw-theme', null) ?? !!window.matchMedia?.('(prefers-color-scheme: dark)').matches)
  const [railCollapsed, setRailCollapsed] = useState(() => load('rw-rail-collapsed', false))
  const [canvasMin, setCanvasMin] = useState(() => load('rw-canvas-min', false))
  const [now, setNow] = useState(Date.now())

  const eventSourceRef = useRef(null)
  const followRef = useRef(true) // inspector follows the running phase until the user picks one

  useEffect(() => () => eventSourceRef.current?.close(), [])
  useEffect(() => save('rw-theme', dark), [dark])
  useEffect(() => save('rw-rail-collapsed', railCollapsed), [railCollapsed])
  useEffect(() => save('rw-canvas-min', canvasMin), [canvasMin])
  useEffect(() => {
    if (!isRunning) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [isRunning])
  useEffect(() => {
    document.body.style.background = dark ? '#0c0c0e' : '#f6f6f4'
    return () => { document.body.style.background = '' }
  }, [dark])

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
              status: 'running',
              startedAt: updated[pendingIdx].startedAt ?? Date.now(),
            }
            return updated
          }
          // Loop re-entry: phase already ran and is 'done' — insert a fresh entry after it
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
          return [...prev, newEntry]
        })
        if (followRef.current) setSelected(event.phase)
        break

      case 'node_start':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase && p.status === 'running')
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], messages: [...updated[idx].messages, `_${event.node}_`] }
          return updated
        })
        break

      case 'message':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase && p.status === 'running')
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], messages: [...updated[idx].messages, event.text] }
          return updated
        })
        break

      case 'charts':
      case 'phase1_summary':
      case 'model_results': {
        const key = { charts: 'charts', phase1_summary: 'phase1Summary', model_results: 'modelResults' }[event.type]
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase)
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], [key]: event.data }
          return updated
        })
        break
      }

      case 'phase_details':
        if (event.data) setPhaseDetails(event.data)
        break

      case 'phase_done':
        setPhases(prev => {
          const idx = findLastIdx(prev, p => p.num === event.phase && p.status === 'running')
          if (idx === -1) return prev
          const updated = [...prev]
          const statusMap = { done: 'done', failed: 'failed', waiting: 'waiting' }
          updated[idx] = { ...updated[idx], status: statusMap[event.status] ?? 'done', finishedAt: Date.now() }
          return updated
        })
        break

      case 'final':
        setPipelineDone(true)
        setScriptAvail(event.script_available ?? false)
        setScriptValidation(event.script_validation ?? null)
        setReportAvail(event.report_available ?? false)
        setBestModel(event.best_model ?? '')
        setPredict({ available: !!event.predict_available, note: event.predict_note || '' })
        if (event.phase_details && Object.keys(event.phase_details).length) setPhaseDetails(event.phase_details)
        if (event.session) setSessionDict(event.session)
        setIsRunning(false)
        setRunFinishedAt(Date.now())
        // Phases that never started (e.g. skipped in pattern mode) should not look like they are still waiting.
        setPhases(prev => prev.map(p => (p.status === 'pending' ? { ...p, status: 'done', skipped: true } : p)))
        eventSourceRef.current?.close()
        eventSourceRef.current = null
        break

      case 'error':
        setRunError(event.message ?? 'Pipeline error')
        setIsRunning(false)
        setRunFinishedAt(Date.now())
        setPhases(prev => prev.map(p => (p.status === 'running' ? { ...p, status: 'failed' } : p)))
        eventSourceRef.current?.close()
        eventSourceRef.current = null
        break

      default:
        break
    }
  }, [])

  // ── Run pipeline ────────────────────────────────────────────────────────────

  const handleRun = useCallback(async () => {
    if (!instructions.trim() || !fileInfo || isRunning) return
    const trimmedInstr = instructions.trim()

    // Archive the current run before resetting
    if (pipelineStarted) {
      setChatHistory(prev => [...prev, {
        instruction: submittedInstruction, phases, phaseDetails, runId, scriptAvailable, scriptValidation,
        reportAvailable, runError, pipelineDone, bestModel, fileInfo, sessionDict, mode: runMode,
        startedAt: runStartedAt, finishedAt: runFinishedAt, predict,
      }])
    }

    setIsRunning(true)
    setPipelineDone(false)
    setScriptAvail(false)
    setScriptValidation(null)
    setReportAvail(false)
    setRunError('')
    setBestModel('')
    setPredict({ available: false, note: '' })
    setPhaseDetails(null)
    setSubmittedInstruction(trimmedInstr)
    setRunMode(mode)
    setInstructions('')
    setPipelineStarted(true)
    setMainView('pipeline')
    setViewing(null)
    setSelected(1)
    followRef.current = true
    setRunStartedAt(Date.now())
    setRunFinishedAt(null)
    setPhases(stubPhases())

    try {
      const res = await fetch('/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: fileInfo.file_path,
          instructions: trimmedInstr,
          session: sessionDict ?? null,
          filename: fileInfo.filename ?? null,
          // Sent only in "Find patterns" mode — Phase 1 then skips target resolution.
          task_type: mode === 'patterns' ? 'clustering' : null,
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
        try { handleEvent(JSON.parse(e.data)) } catch { /* malformed event — ignore */ }
      }
      es.onerror = () => {
        setRunError('Connection to the server was lost. Is the backend running?')
        setIsRunning(false)
        es.close()
      }
    } catch (err) {
      setRunError(err.message ?? 'Failed to start pipeline')
      setIsRunning(false)
      setPhases(prev => prev.map(p => (p.status === 'running' ? { ...p, status: 'failed' } : p)))
    }
  }, [instructions, fileInfo, isRunning, sessionDict, handleEvent, mode, pipelineStarted, submittedInstruction,
    phases, phaseDetails, runId, scriptAvailable, scriptValidation, reportAvailable, runError, pipelineDone,
    bestModel, runMode, runStartedAt, runFinishedAt])

  const handleNewFile = (info) => {
    setFileInfo(info)
    setSessionDict(null)
    setPhases([])
    setPhaseDetails(null)
    setPipelineDone(false)
    setScriptAvail(false)
    setScriptValidation(null)
    setReportAvail(false)
    setRunError('')
    setPipelineStarted(false)
    setSubmittedInstruction('')
    setChatHistory([])
    setBestModel('')
    setMainView('pipeline')
    setViewing(null)
    setSelected(1)
  }

  // ── What is on screen: the current run or an archived one ──────────────────

  const current = {
    instruction: submittedInstruction, phases, phaseDetails, runId, scriptAvailable, scriptValidation,
    reportAvailable, runError, pipelineDone, bestModel, sessionDict, mode: runMode, isRunning,
    startedAt: runStartedAt, finishedAt: runFinishedAt, predict,
  }
  const allRuns = pipelineStarted ? [...chatHistory, current] : []
  const shown = viewing != null && chatHistory[viewing] ? chatHistory[viewing] : current
  const shownLive = shown === current

  const status = useMemo(() => {
    const out = {}
    for (const p of shown.phases || []) out[p.num] = { status: p.skipped ? 'done' : p.status, startedAt: p.startedAt, finishedAt: p.finishedAt }
    return out
  }, [shown.phases])
  const rounds = (shown.phases || []).filter(p => p.num === 5 && p.status !== 'pending').length

  const selectPhase = (n) => { setSelected(n); if (shownLive) followRef.current = false }
  const nav = (d) => selectPhase(Math.min(7, Math.max(1, selected + d)))

  // header status pill
  const elapsed = shown.startedAt ? ((shown.finishedAt || now) - shown.startedAt) / 1000 : null
  let pill = <span className="rw-pill idle">Not started</span>
  if (shown.runError) pill = <span className="rw-pill err">Error</span>
  else if (shown.pipelineDone) pill = <span className="rw-pill ok">✓ Completed{elapsed ? ` · ${fmtSecs(elapsed)}` : ''}</span>
  else if (shownLive && isRunning) {
    const running = PHASES.find(p => status[p.num]?.status === 'running')
    pill = <span className="rw-pill run">Phase {running?.num ?? '…'} · {fmtSecs(elapsed)}</span>
  }

  const runCtx = {
    runId: shown.runId, scriptAvailable: shown.scriptAvailable, scriptValidation: shown.scriptValidation,
    reportAvailable: shown.reportAvailable, fallbackCode: shown.sessionDict?.final_pipeline_script, mode: shown.mode,
  }

  return (
    <div className={`rw${dark ? ' rw-dark' : ''}`}>
      <header className="rw-header">
        <a className="rw-brand" href="#/">
          <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"><rect width="28" height="28" rx="8" fill="var(--primary)" /><circle cx="9" cy="9" r="2.6" fill="var(--on-primary)" /><circle cx="19" cy="14" r="2.6" fill="var(--accent)" /><circle cx="9" cy="19" r="2.6" fill="var(--on-primary)" /><path d="M11.4 10.2L16.6 12.8M11.4 17.8L16.6 15.2" stroke="var(--on-primary)" strokeWidth="1.6" strokeLinecap="round" /></svg>
          AutoML Assistant
        </a>
        <nav className="rw-crumb" aria-label="Breadcrumb">
          <span className="sep">/</span><span>Runs</span>
          {fileInfo && <><span className="sep">/</span><b title={fileInfo.filename}>{fileInfo.filename}</b></>}
          {allRuns.length > 1 && <><span className="sep">/</span><span>Run {(viewing ?? allRuns.length - 1) + 1}</span></>}
        </nav>
        {pill}
        <div className="rw-grow" />
        {shown.pipelineDone && shown.sessionDict && (
          <div className="rw-seg" role="tablist">
            <button role="tab" aria-selected={mainView === 'pipeline'} onClick={() => setMainView('pipeline')}>Pipeline</button>
            <button role="tab" aria-selected={mainView === 'analytics'} onClick={() => setMainView('analytics')}>Analytics</button>
            {shown.predict?.available && (
              <button role="tab" aria-selected={mainView === 'predict'} onClick={() => setMainView('predict')}>Try the model</button>
            )}
          </div>
        )}
        <button className="rw-icon-btn" onClick={() => setDark(d => !d)} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'} title="Toggle theme">
          {dark
            ? <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><circle cx="8" cy="8" r="3" /><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" /></svg>
            : <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M13.5 9.5A5.5 5.5 0 016.5 2.5a5.5 5.5 0 107 7z" /></svg>}
        </button>
      </header>

      <div className="rw-body">
        <ConfigRail
          collapsed={railCollapsed} onToggle={() => setRailCollapsed(c => !c)}
          fileInfo={fileInfo} onFileInfo={handleNewFile} mode={mode} setMode={setMode}
          instructions={instructions} setInstructions={setInstructions}
          onRun={handleRun} isRunning={isRunning}
          history={allRuns} viewing={viewing ?? allRuns.length - 1}
          onView={i => { setViewing(i === allRuns.length - 1 ? null : i); setSelected(7); setMainView('pipeline') }}
        />

        <main className="rw-main">
          {mainView === 'predict' && shown.pipelineDone ? (
            <Predict runId={shown.runId} available={!!shown.predict?.available} note={shown.predict?.note} />
          ) : mainView === 'analytics' && shown.sessionDict ? (
            <Dashboard session={shown.sessionDict} fileName={fileInfo?.filename} runId={shown.runId}
              reportAvailable={shown.reportAvailable} />
          ) : !pipelineStarted ? (
            <>
              <div className="rw-card rw-welcome">
                <span className="h3">New run</span>
                <h1>Upload a dataset and describe your goal</h1>
                <p>The pipeline runs seven phases — from understanding the data to a verified, standalone script.
                  Click any phase to see what it found, which decisions it made and who made them: a fixed rule, a statistical test or the LLM.</p>
              </div>
              <PipelineCanvas status={{}} details={null} selected={selected} onSelect={setSelected}
                minimized={canvasMin} onToggle={() => setCanvasMin(m => !m)} rounds={0} now={now} running={false} />
            </>
          ) : (
            <>
              {shown.instruction && (
                <div className="rw-hint" style={{ fontSize: 13 }}>
                  <span className="h3" style={{ marginRight: 8 }}>Goal</span>{shown.instruction}
                </div>
              )}
              {shown.runError && <div className="rw-error" role="alert">{shown.runError}</div>}
              <PipelineCanvas status={status} details={shown.phaseDetails} selected={selected} onSelect={selectPhase}
                minimized={canvasMin} onToggle={() => setCanvasMin(m => !m)} rounds={rounds} now={now}
                running={shownLive && isRunning} />
              <PhaseInspector
                num={selected}
                entries={(shown.phases || []).filter(p => p.num === selected)}
                details={shown.phaseDetails?.[String(selected)]}
                onNav={nav} run={runCtx} now={now}
              />
            </>
          )}
        </main>
      </div>
    </div>
  )
}
