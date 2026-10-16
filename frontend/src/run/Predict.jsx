import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Bars } from './Sections.jsx'

const MAX_ROWS = 500

async function api(url, opts) {
  const res = await fetch(url, opts)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`)
  return body
}

/** Small RFC-4180 CSV parser (quoted fields, escaped quotes, CRLF). */
function parseCsv(text) {
  const rows = []
  let row = [], field = '', q = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (ch === '"') q = false
      else field += ch
    } else if (ch === '"') q = true
    else if (ch === ',') { row.push(field); field = '' }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++
      row.push(field); field = ''
      if (row.some(v => v !== '')) rows.push(row)
      row = []
    } else field += ch
  }
  row.push(field)
  if (row.some(v => v !== '')) rows.push(row)
  if (!rows.length) return { header: [], records: [] }
  const header = rows[0].map(h => h.trim())
  return { header, records: rows.slice(1).map(r => Object.fromEntries(header.map((h, i) => [h, r[i] ?? '']))) }
}

const toCsv = (header, rows) => [header, ...rows].map(r => r.map(v => {
  const s = v == null ? '' : String(v)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}).join(',')).join('\n')

const pct = p => `${Math.round(p * 100)}%`

// ── one row ─────────────────────────────────────────────────────────────────

function Field({ col, value, onChange }) {
  const id = `pf-${col.name}`
  const opts = col.options || []
  let input
  if (col.type === 'category' && opts.length <= 4 && !col.more) {
    input = (
      <div className="rw-seg pr-seg" role="group" aria-label={col.name}>
        {opts.map(o => (
          <button key={String(o)} type="button" aria-selected={String(value) === String(o)} onClick={() => onChange(o)}>{String(o)}</button>
        ))}
      </div>
    )
  } else if (col.type === 'category') {
    input = (
      <select id={id} className="pr-input" value={value ?? ''} onChange={e => onChange(e.target.value)}>
        <option value="">(blank)</option>
        {opts.map(o => <option key={String(o)} value={o}>{String(o)}</option>)}
      </select>
    )
  } else {
    input = (
      <input id={id} className="pr-input mono" type="number" value={value ?? ''} step={col.integer ? 1 : 'any'}
        onChange={e => onChange(e.target.value)} placeholder={col.default != null ? String(col.default) : ''} />
    )
  }
  return (
    <div className="pr-field">
      <label htmlFor={id} className="pr-label">
        <span className="mono" title={col.name}>{col.name}</span>
        {col.type === 'number' && col.min != null && <span className="rw-hint">{col.min} – {col.max}</span>}
      </label>
      {input}
    </div>
  )
}

function Result({ result, schema, busy, actual }) {
  if (!result) return <div className="rw-card pr-result rw-empty">{busy ? 'Predicting…' : 'Fill in the details to get a prediction.'}</div>
  const probs = result.probabilities
  const classes = schema.classes || []
  const binary = classes.length === 2 && probs
  const pos = binary ? classes[1] : null
  const pPos = binary ? probs[pos] : null
  const thr = schema.decision_threshold ?? 0.5
  const contrib = result.contributions
  const actualMatch = actual != null && String(actual) === String(result.prediction)
  return (
    <div className="pr-col">
      <section className="rw-card pr-result" aria-live="polite" style={{ opacity: busy ? 0.6 : 1 }}>
        <span className="h3">Prediction · {schema.target}</span>
        <div className="pr-big">
          <span className="pr-label-big">{String(result.prediction)}</span>
          {binary && <span className="pr-prob"><b className="mono">{pct(pPos)}</b> chance of {pos}</span>}
          {!probs && typeof result.prediction === 'number' && <span className="rw-hint">predicted value</span>}
        </div>
        {binary && (
          <div className="pr-gauge" title={`Probability of ${pos}: ${pPos.toFixed(3)} · threshold ${thr}`}>
            <div className="pr-gauge-fill" style={{ width: pct(pPos) }} />
            <div className="pr-gauge-thr" style={{ left: pct(thr) }}><span className="mono">threshold {thr}</span></div>
          </div>
        )}
        {probs && !binary && (
          <Bars s={{ items: Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([label, value]) => ({ label, value })) }} />
        )}
        {actual != null && (
          <span className={`rw-pill ${actualMatch ? 'ok' : 'err'}`} style={{ alignSelf: 'flex-start' }}>
            Actual value in the test set: {String(actual)}{actualMatch ? ' — correct' : ''}
          </span>
        )}
      </section>
      <section className="rw-card pr-result">
        <div>
          <h2 className="pr-h2">Why this prediction</h2>
          <p className="rw-hint" style={{ margin: '2px 0 0', fontSize: 13 }}>
            {contrib ? `How each model input pushes this prediction ${binary ? `towards or away from ${pos}` : 'up or down'} (${contrib.method}).`
              : `A per-row explanation is only computed for single tree-based and linear models, and ${schema.best_model} is neither.`}
          </p>
        </div>
        {contrib && contrib.items.length > 0 && (
          <Bars s={{ signed: true, digits: 3, items: contrib.items.map(it => ({ label: `${it.feature} = ${it.value ?? '(blank)'}`, value: it.effect })) }} />
        )}
      </section>
    </div>
  )
}

function OneRow({ runId, schema }) {
  const defaults = useMemo(() => Object.fromEntries(schema.columns.map(c => [c.name, c.default ?? ''])), [schema])
  const [values, setValues] = useState(defaults)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [example, setExample] = useState(-1)
  const seq = useRef(0)

  useEffect(() => {
    const my = ++seq.current
    const t = setTimeout(async () => {
      setBusy(true)
      try {
        const body = await api(`/predict/${runId}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rows: [values] }),
        })
        if (my === seq.current) { setResult(body.predictions[0]); setError('') }
      } catch (e) {
        if (my === seq.current) setError(e.message)
      } finally {
        if (my === seq.current) setBusy(false)
      }
    }, 350)
    return () => clearTimeout(t)
  }, [values, runId])

  const set = (k, v) => { setValues(prev => ({ ...prev, [k]: v })); setExample(-1) }
  const loadExample = () => {
    const ex = schema.examples || []
    if (!ex.length) return
    const i = (example + 1) % ex.length
    setValues({ ...defaults, ...ex[i].values })
    setExample(i)
  }
  const actual = example >= 0 ? schema.examples[example]?.actual : null

  return (
    <div className="pr-grid">
      <section className="rw-card pr-form" aria-label="Input row">
        <div className="pr-form-head">
          <h2 className="pr-h2">Input row</h2>
          <div className="rw-grow" />
          {schema.examples?.length > 0 && (
            <button className="rw-btn sm" onClick={loadExample}>{example >= 0 ? `Next test row (${example + 1}/${schema.examples.length})` : 'Load a real test row'}</button>
          )}
          <button className="rw-btn sm" onClick={() => { setValues(defaults); setExample(-1) }}>Reset</button>
        </div>
        <span className="rw-hint">One field per column of the training data. Defaults are typical training values; leave a field blank to let the pipeline fill it.</span>
        <div className="pr-fields">
          {schema.columns.map(c => <Field key={c.name} col={c} value={values[c.name]} onChange={v => set(c.name, v)} />)}
        </div>
        {error && <div className="rw-error">{error}</div>}
      </section>
      <Result result={result} schema={schema} busy={busy} actual={actual} />
    </div>
  )
}

// ── batch ───────────────────────────────────────────────────────────────────

function Batch({ runId, schema }) {
  const [file, setFile] = useState(null)
  const [rows, setRows] = useState(null)
  const [out, setOut] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [over, setOver] = useState(false)
  const inputRef = useRef(null)
  const cols = schema.columns.map(c => c.name)

  const run = async (f) => {
    setError(''); setOut(null); setFile(f)
    try {
      const { header, records } = parseCsv(await f.text())
      const missing = cols.filter(c => !header.includes(c))
      if (!records.length) throw new Error('The file has no data rows.')
      if (missing.length === cols.length) throw new Error(`None of the model's columns were found. Expected: ${cols.slice(0, 6).join(', ')}…`)
      if (records.length > MAX_ROWS) throw new Error(`At most ${MAX_ROWS} rows at a time (this file has ${records.length}).`)
      setRows({ header, records, missing })
      setBusy(true)
      const body = await api(`/predict/${runId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows: records.map(r => Object.fromEntries(cols.map(c => [c, r[c] ?? '']))), explain: false }),
      })
      setOut(body.predictions)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const classes = schema.classes || []
  const download = () => {
    const extra = classes.length ? classes.map(c => `proba_${c}`) : []
    const header = [...rows.header, 'prediction', ...extra]
    const body = rows.records.map((r, i) => [...rows.header.map(h => r[h]), out[i].prediction,
      ...classes.map(c => out[i].probabilities?.[c] ?? '')])
    const url = URL.createObjectURL(new Blob([toCsv(header, body)], { type: 'text/csv' }))
    const a = Object.assign(document.createElement('a'), { href: url, download: 'predictions.csv' })
    a.click(); URL.revokeObjectURL(url)
  }
  const counts = useMemo(() => {
    if (!out || !classes.length) return null
    const c = {}
    out.forEach(o => { c[o.prediction] = (c[o.prediction] || 0) + 1 })
    return c
  }, [out, classes.length])
  const shownCols = cols.slice(0, 5)

  return (
    <div className="pr-col">
      <section className="rw-card pr-form">
        <h2 className="pr-h2">Predict for many rows</h2>
        <div className={`rw-drop${over ? ' over' : ''}`} role="button" tabIndex={0}
          onClick={() => inputRef.current?.click()} onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) run(f) }}>
          {busy ? 'Predicting…' : <>Drop a CSV here or <b style={{ color: 'var(--accent-text)' }}>browse</b><br />
            <span className="rw-hint">Same columns as the training data ({cols.length}) · the {schema.target} column is not needed · up to {MAX_ROWS} rows</span></>}
        </div>
        <input ref={inputRef} type="file" accept=".csv,text/csv" style={{ display: 'none' }}
          onChange={e => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = '' }} />
        {rows?.missing?.length > 0 && (
          <div className="rw-hint">Missing columns are filled by the pipeline: {rows.missing.slice(0, 8).join(', ')}{rows.missing.length > 8 ? '…' : ''}</div>
        )}
        {error && <div className="rw-error">{error}</div>}
      </section>
      {out && (
        <section className="rw-card pr-form">
          <div className="pr-form-head">
            <h2 className="pr-h2">{file?.name}</h2>
            <span className="rw-hint">{out.length} rows{counts ? ' · ' + Object.entries(counts).map(([k, v]) => `${v} × ${k}`).join(' · ') : ''}</span>
            <div className="rw-grow" />
            <button className="rw-btn primary sm" onClick={download}>Download predictions.csv</button>
          </div>
          <div className="rw-table-wrap">
            <table className="rw-table">
              <thead><tr><th>#</th>{shownCols.map(c => <th key={c} className="mono">{c}</th>)}
                {classes.length === 2 && <th>P({classes[1]})</th>}<th>Prediction</th></tr></thead>
              <tbody>
                {out.slice(0, 50).map((o, i) => (
                  <tr key={i}>
                    <td className="mono">{i + 1}</td>
                    {shownCols.map(c => <td key={c}>{rows.records[i][c]}</td>)}
                    {classes.length === 2 && <td className="mono">{o.probabilities ? o.probabilities[classes[1]].toFixed(2) : '—'}</td>}
                    <td style={{ fontWeight: 500 }}>{String(o.prediction)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {out.length > 50 && <div className="rw-hint" style={{ marginTop: 6 }}>Showing 50 of {out.length} — the download has all rows.</div>}
          </div>
        </section>
      )}
    </div>
  )
}

// ── code ────────────────────────────────────────────────────────────────────

function FromCode({ runId, schema }) {
  const origin = window.location.origin
  const sample = Object.fromEntries(schema.columns.slice(0, 8).map(c => [c.name, c.default ?? null]))
  const body = JSON.stringify({ rows: [sample] }, null, 2)
  const curl = `curl -X POST ${origin}/predict/${runId} \\\n  -H "Content-Type: application/json" \\\n  -d '${body.replace(/'/g, "'\\''")}'`
  const py = `import requests\n\nrows = [${JSON.stringify(sample)}]\nr = requests.post("${origin}/predict/${runId}", json={"rows": rows})\nfor p in r.json()["predictions"]:\n    print(p["prediction"], p.get("probabilities"))`
  const [copied, setCopied] = useState('')
  const copy = async (k, text) => { try { await navigator.clipboard.writeText(text); setCopied(k); setTimeout(() => setCopied(''), 1500) } catch { /* no clipboard */ } }
  return (
    <section className="rw-card pr-form">
      <h2 className="pr-h2">Call this model from another app</h2>
      <p className="rw-hint" style={{ margin: 0, fontSize: 13 }}>
        POST up to {MAX_ROWS} rows as JSON. Each result has the prediction{schema.classes ? ', class probabilities' : ''} and the
        top contributing inputs. Columns you leave out are filled by the pipeline, exactly as in training. The model stays available
        while this server keeps the run's files.
      </p>
      {[['curl', curl], ['Python', py]].map(([k, text]) => (
        <div key={k} className="pr-code">
          <div className="pr-code-head"><span className="h3">{k}</span><div className="rw-grow" />
            <button className="rw-btn sm" onClick={() => copy(k, text)}>{copied === k ? 'Copied' : 'Copy'}</button></div>
          <pre className="mono">{text}</pre>
        </div>
      ))}
    </section>
  )
}

// ── page ────────────────────────────────────────────────────────────────────

export default function Predict({ runId, available, note }) {
  const [schema, setSchema] = useState(null)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('one')
  const load = useCallback(async () => {
    setError(''); setSchema(null)
    try { setSchema(await api(`/predict/${runId}/schema`)) } catch (e) { setError(e.message) }
  }, [runId])
  useEffect(() => { if (available && runId) load() }, [available, runId, load])

  if (!available) {
    return (
      <div className="db"><div className="rw-card rw-empty">
        The model from this run can't be used for predictions{note ? `: ${note}` : '.'}
      </div></div>
    )
  }
  return (
    <div className="db">
      <div className="db-head">
        <div>
          <h1>Try the model</h1>
          <span className="rw-hint" style={{ fontSize: 13 }}>
            {schema ? <>{schema.best_model} from this run · predicts <b>{schema.target}</b> · same preprocessing as the verified pipeline script</> : 'Loading the trained model…'}
          </span>
        </div>
        <div className="rw-grow" />
        <div className="rw-seg" role="tablist" aria-label="Prediction mode">
          <button role="tab" aria-selected={tab === 'one'} onClick={() => setTab('one')}>One row</button>
          <button role="tab" aria-selected={tab === 'batch'} onClick={() => setTab('batch')}>Upload a file</button>
          <button role="tab" aria-selected={tab === 'code'} onClick={() => setTab('code')}>Use from code</button>
        </div>
      </div>
      {error && <div className="rw-error">{error} <button className="rw-btn sm" onClick={load} style={{ marginLeft: 8 }}>Retry</button></div>}
      {!schema && !error && <div className="rw-skel" style={{ height: 320 }} />}
      {schema && tab === 'one' && <OneRow runId={runId} schema={schema} />}
      {schema && tab === 'batch' && <Batch runId={runId} schema={schema} />}
      {schema && tab === 'code' && <FromCode runId={runId} schema={schema} />}
    </div>
  )
}
