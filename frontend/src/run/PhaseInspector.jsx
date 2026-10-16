import React, { useEffect, useMemo, useState } from 'react'
import { marked } from 'marked'
import ChartPanel from '../components/ChartPanel.jsx'
import ModelTable from '../components/ModelTable.jsx'
import Phase1Summary from '../components/Phase1Summary.jsx'
import InlineScript from '../components/InlineScript.jsx'
import { Section, SourceBadge } from './Sections.jsx'
import { PHASES, fmtSecs } from './PipelineCanvas.jsx'

marked.use({ gfm: true, breaks: true })
const md = t => { try { return marked.parse(String(t ?? '')) } catch { return String(t ?? '') } }

// What each phase does — shown before the phase has produced anything.
const PLAN = {
  1: ['Profile every column: type, missing values, distinct values', 'Find the target column from your goal', 'Decide the task type and the metric to optimise'],
  2: ['Spot columns that need encoding, scaling or a transform', 'Flag possible leakage and identifiers', 'Decide whether engineered features are worth trying'],
  3: ['Remove duplicates and fix types', 'Split off a test set before learning anything', 'Explore, drop useless columns, fill gaps, encode and scale', 'Propose new features and keep only the ones that pass cross-validation', 'Verify the result: no nulls, no leakage, consistent shapes'],
  4: ['Pick model families suited to the data', 'Choose the validation scheme and search strategy', 'Set the score the best model has to reach'],
  5: ['Tune and cross-validate every candidate', 'Try ensembles and keep them only if they win', 'Score the winner on the untouched test set and explain it'],
  6: ['Compare the best score with the threshold', 'If it falls short, run probe experiments and diagnose why', 'Send the run back to the responsible phase, or stop'],
  7: ['Assemble a standalone pipeline script from the run', 'Re-run it in an isolated sandbox and compare scores', 'Write the report'],
}

function Stats({ stats }) {
  if (!stats?.length) return null
  return (
    <div className="rw-stats">
      {stats.map((s, i) => (
        <div className="rw-tile" key={i}>
          <span className="label">{s.label}</span>
          <span className="value" title={s.value}>{s.value}</span>
          {s.note && <span className="note">{s.note}</span>}
        </div>
      ))}
    </div>
  )
}

function Decisions({ items, onEntry }) {
  if (!items?.length) return null
  return (
    <section className="rw-sec wide">
      <h3 className="h3">Decisions made</h3>
      <div className="rw-decs">
        {items.map((d, i) => (
          <div className="rw-dec" key={i}>
            <SourceBadge by={d.by} />
            <div><div className="what">{d.what}</div>{d.why && <div className="why">{d.why}</div>}</div>
            {d.entry != null
              ? <button className="rw-entry" onClick={() => onEntry(d.entry)} title="Show this entry in the decision log">#{d.entry}</button>
              : <span />}
          </div>
        ))}
      </div>
    </section>
  )
}

function LiveLog({ entries }) {
  const blocks = []
  entries.forEach((e, ri) => {
    if (entries.length > 1) blocks.push({ kind: 'round', text: `Round ${ri + 1}` })
    let prev = null
    for (const m of e.messages || []) {
      if (m === prev) continue
      prev = m
      const node = /^_(.+)_$/.exec(m)
      blocks.push(node ? { kind: 'node', text: node[1] } : { kind: 'msg', text: m })
    }
  })
  if (!blocks.length) return <div className="rw-empty">No messages yet.</div>
  return (
    <div className="rw-log">
      {blocks.map((b, i) => b.kind === 'msg'
        ? <div key={i} className="msg" dangerouslySetInnerHTML={{ __html: md(b.text) }} />
        : <div key={i} className={b.kind}>{b.text}</div>)}
    </div>
  )
}

function DecisionLog({ log, highlight }) {
  useEffect(() => {
    if (highlight != null) document.getElementById(`rw-log-${highlight}`)?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [highlight])
  if (!log?.length) return <div className="rw-empty">This phase wrote no entries to the decision log.</div>
  return (
    <div className="rw-declog">
      {log.map(e => (
        <div key={e.n} id={`rw-log-${e.n}`} className={`row${e.n === highlight ? ' hl' : ''}`}>
          <span className="n">#{e.n}</span><span className="a">{e.action}</span><span className="t">{e.note || '—'}</span>
        </div>
      ))}
    </div>
  )
}

function Downloads({ runId, scriptAvailable, reportAvailable }) {
  if (!runId || !scriptAvailable) return null
  return (
    <div className="rw-downloads">
      <a className="rw-btn primary" href={`/download/script/${runId}`} download="automl_pipeline.py">Download pipeline.py</a>
      <a className="rw-btn" href={`/download/split/${runId}/train`} download="train.csv">train.csv</a>
      <a className="rw-btn" href={`/download/split/${runId}/test`} download="test.csv">test.csv</a>
      {reportAvailable && <a className="rw-btn" href={`/download/report/${runId}`} download="automl_report.pdf">Report PDF</a>}
    </div>
  )
}

export default function PhaseInspector({ num, entries, details, onNav, run, now }) {
  const [tab, setTab] = useState('overview')
  const [hl, setHl] = useState(null)
  useEffect(() => { setTab('overview'); setHl(null) }, [num])

  const meta = PHASES.find(p => p.num === num)
  const last = entries[entries.length - 1] || { status: 'pending', messages: [] }
  const status = last.status || 'pending'
  const charts = [...entries].reverse().find(e => e.charts)?.charts
  const modelResults = [...entries].reverse().find(e => e.modelResults)?.modelResults
  const p1 = entries.find(e => e.phase1Summary)?.phase1Summary
  const msgCount = entries.reduce((n, e) => n + (e.messages || []).filter(m => !/^_.+_$/.test(m)).length, 0)
  const hasDetails = !!(charts || modelResults || p1 || (num === 7 && run.scriptAvailable))
  const avail = details?.available
  const live = status === 'running' && last.startedAt ? (now - last.startedAt) / 1000 : null
  const secs = details?.seconds ?? live ?? (last.finishedAt && last.startedAt ? (last.finishedAt - last.startedAt) / 1000 : null)
  const sections = details?.sections || []

  const statusPill = {
    done: <span className="rw-pill ok">Done</span>,
    running: <span className="rw-pill run">Running</span>,
    failed: <span className="rw-pill err">Failed</span>,
  }[status] || <span className="rw-pill idle">Waiting</span>

  const goEntry = n => { setHl(n); setTab('log2') }

  return (
    <div className="rw-card rw-insp">
      <div className="rw-insp-head">
        <div style={{ minWidth: 0 }}>
          <div className="h3">Phase {num} of 7</div>
          <h2 className="rw-insp-title">{meta?.title}</h2>
          <div className="rw-insp-meta">
            {statusPill}
            {secs != null && secs > 0 && <span className="rw-pill idle">{fmtSecs(secs)}</span>}
            {details?.llm_calls > 0 && <span className="rw-pill idle">{details.llm_calls} LLM call{details.llm_calls > 1 ? 's' : ''}</span>}
            {entries.length > 1 && <span className="rw-pill idle">{entries.length} rounds</span>}
          </div>
        </div>
        <div className="rw-nav">
          <button className="rw-icon-btn" onClick={() => onNav(-1)} disabled={num === 1} aria-label="Previous phase">‹</button>
          <button className="rw-icon-btn" onClick={() => onNav(1)} disabled={num === 7} aria-label="Next phase">›</button>
        </div>
      </div>

      <div className="rw-tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'overview'} onClick={() => setTab('overview')}>Overview</button>
        {hasDetails && <button role="tab" aria-selected={tab === 'details'} onClick={() => setTab('details')}>
          {num === 7 ? 'Script' : 'Charts & tables'}</button>}
        <button role="tab" aria-selected={tab === 'log'} onClick={() => setTab('log')}>Live log<span className="count">{msgCount || ''}</span></button>
        <button role="tab" aria-selected={tab === 'log2'} onClick={() => setTab('log2')}>Decision log<span className="count">{details?.log?.length || ''}</span></button>
      </div>

      {tab === 'overview' && (
        <>
          {avail ? (
            <>
              <Stats stats={details.stats} />
              {details.takeaway && <div className="rw-takeaway">{details.takeaway}</div>}
              {num === 7 && <Downloads {...run} />}
              <div className="rw-grid2">
                {sections.map((s, i) => <Section key={i} s={s} />)}
                <Decisions items={details.decisions} onEntry={goEntry} />
              </div>
            </>
          ) : status === 'running' ? (
            <>
              <div className="rw-stats">{[0, 1, 2, 3].map(i => <div key={i} className="rw-skel" />)}</div>
              <section className="rw-sec">
                <h3 className="h3">In progress</h3>
                <ul className="rw-list">{PLAN[num].map(t => <li key={t}>{t}</li>)}</ul>
                <span className="rw-hint">Results appear here when the phase finishes. Follow along in the live log.</span>
              </section>
            </>
          ) : status === 'done' || status === 'failed' ? (
            <div className="rw-empty">{run.mode === 'patterns' && [2, 4, 6].includes(num)
              ? 'This phase is skipped when finding patterns.'
              : 'This phase did not record any results for this run. The live log has what it printed.'}</div>
          ) : (
            <section className="rw-sec">
              <h3 className="h3">What this phase will do</h3>
              <ul className="rw-list">{PLAN[num].map(t => <li key={t}>{t}</li>)}</ul>
            </section>
          )}
        </>
      )}

      {tab === 'details' && (
        <div className="rw-legacy">
          {p1 && <Phase1Summary data={p1} />}
          {charts && <ChartPanel data={charts} />}
          {modelResults && <ModelTable data={modelResults} />}
          {num === 7 && <InlineScript runId={run.runId} visible={run.scriptAvailable}
            validation={run.scriptValidation} fallbackCode={run.fallbackCode} />}
        </div>
      )}
      {tab === 'log' && <LiveLog entries={entries} />}
      {tab === 'log2' && <DecisionLog log={details?.log} highlight={hl} />}
    </div>
  )
}
