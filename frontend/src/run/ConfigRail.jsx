import React, { useRef, useState } from 'react'

const ALLOWED = ['.csv', '.tsv', '.parquet', '.xlsx', '.xls']

const Icon = ({ d, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
)
const I_DATA = 'M4 5c0-1.1 2.7-2 6-2s6 .9 6 2-2.7 2-6 2-6-.9-6-2zm0 0v10c0 1.1 2.7 2 6 2s6-.9 6-2V5M4 10c0 1.1 2.7 2 6 2s6-.9 6-2'
const I_GOAL = 'M10 17a7 7 0 100-14 7 7 0 000 14zm0-4a3 3 0 100-6 3 3 0 000 6z'
const I_PLAY = 'M6 4l10 6-10 6z'
const I_SIDE = 'M3 4h14v12H3zM8 4v12'

export default function ConfigRail({
  collapsed, onToggle, fileInfo, onFileInfo, mode, setMode, instructions, setInstructions,
  onRun, isRunning, history, viewing, onView,
}) {
  const inputRef = useRef(null)
  const [over, setOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  async function upload(file) {
    if (!file) return
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED.includes(ext)) { setError(`Unsupported format "${ext}". Use ${ALLOWED.join(', ')}.`); return }
    setUploading(true); setError('')
    const form = new FormData(); form.append('file', file)
    try {
      const res = await fetch('/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Upload failed (${res.status})`)
      }
      onFileInfo(await res.json())
    } catch (err) {
      setError(err.message ?? 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const canRun = !!fileInfo && !!instructions.trim() && !isRunning
  const onKey = e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && canRun) { e.preventDefault(); onRun() } }

  return (
    <aside className={`rw-rail${collapsed ? ' collapsed' : ''}`} aria-label="Run configuration">
      <div className="rw-rail-head">
        <button className="rw-icon-btn" onClick={onToggle} aria-label={collapsed ? 'Expand configuration' : 'Collapse configuration'}
          title={collapsed ? 'Expand' : 'Collapse'}><Icon d={I_SIDE} /></button>
        <span className="rw-rail-title" style={{ fontWeight: 600 }}>Configure run</span>
      </div>

      <div className="rw-rail-mini">
        <button className="rw-icon-btn" onClick={onToggle} title={fileInfo?.filename || 'Dataset'}><Icon d={I_DATA} /></button>
        <button className="rw-icon-btn" onClick={onToggle} title="Goal"><Icon d={I_GOAL} /></button>
        <button className="rw-icon-btn" onClick={onRun} disabled={!canRun} title="Run (Ctrl+Enter)" style={canRun ? { color: 'var(--accent)' } : undefined}><Icon d={I_PLAY} /></button>
      </div>

      <div className="rw-rail-body">
        <div className="rw-field">
          <span className="h3">Dataset</span>
          <input ref={inputRef} type="file" accept={ALLOWED.join(',')} style={{ display: 'none' }}
            onChange={e => { upload(e.target.files?.[0]); e.target.value = '' }} />
          {fileInfo ? (
            <>
              <div className="rw-file">
                <Icon d={I_DATA} />
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div className="rw-file-name" title={fileInfo.filename}>{fileInfo.filename}</div>
                  <div className="rw-file-meta">{Number(fileInfo.rows).toLocaleString()} rows · {fileInfo.cols} columns{fileInfo.size_kb != null ? ` · ${fileInfo.size_kb} KB` : ''}</div>
                </div>
                <button className="rw-btn sm" onClick={() => inputRef.current?.click()} disabled={isRunning || uploading}>Replace</button>
              </div>
              {fileInfo.columns?.length > 0 && (
                <div className="rw-cols">
                  {fileInfo.columns.map(c => <span key={c} className="rw-chip mono">{c}</span>)}
                  {fileInfo.extra_cols > 0 && <span className="rw-chip">+{fileInfo.extra_cols}</span>}
                </div>
              )}
            </>
          ) : (
            <div className={`rw-drop${over ? ' over' : ''}`} role="button" tabIndex={0}
              onClick={() => inputRef.current?.click()} onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
              onDrop={e => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files?.[0]) }}>
              {uploading ? 'Uploading…' : <>Drop a file here or <b style={{ color: 'var(--accent-text)' }}>browse</b><br /><span className="rw-hint">CSV, TSV, Parquet or Excel · up to 200 MB</span></>}
            </div>
          )}
          {error && <div className="rw-error">{error}</div>}
        </div>

        <div className="rw-field">
          <span className="h3">What do you want?</span>
          <div className="rw-seg" role="tablist">
            <button role="tab" aria-selected={mode === 'predict'} onClick={() => setMode('predict')} disabled={isRunning}>Predict a target</button>
            <button role="tab" aria-selected={mode === 'patterns'} onClick={() => setMode('patterns')} disabled={isRunning}>Find patterns</button>
          </div>
          <span className="rw-hint">{mode === 'predict' ? 'Supervised: classification or regression on a column you name.' : 'Unsupervised: clusters and structure — no target column needed.'}</span>
        </div>

        <div className="rw-field">
          <span className="h3">Goal</span>
          <textarea className="rw-textarea" value={instructions} onChange={e => setInstructions(e.target.value)} onKeyDown={onKey}
            disabled={isRunning}
            placeholder={mode === 'predict'
              ? 'e.g. Predict which patients have heart disease. Missing a real case is worse than a false alarm.'
              : 'e.g. Segment customers by spending behaviour.'} />
          <span className="rw-hint">Name the column to predict and what matters most. Ctrl+Enter runs.</span>
        </div>

        <button className="rw-btn accent" style={{ justifyContent: 'center', height: 40 }} onClick={onRun} disabled={!canRun}>
          {isRunning ? 'Running…' : 'Run pipeline'}
        </button>

        {history.length > 1 && (
          <div className="rw-field">
            <span className="h3">Runs in this session</span>
            <div className="rw-history">
              {history.map((h, i) => (
                <button key={i} aria-current={viewing === i} onClick={() => onView(i)}>
                  <b>Run {i + 1}</b> · {h.bestModel || (h.isRunning ? 'running' : h.runError ? 'error' : '—')}<br />
                  <span style={{ color: 'var(--faint)' }}>{(h.instruction || '').slice(0, 70)}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
