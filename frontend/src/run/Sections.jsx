import React from 'react'

const BY_LABEL = { rule: 'Rule', stat: 'Statistic', llm: 'LLM' }

export function SourceBadge({ by }) {
  const k = BY_LABEL[by] ? by : 'rule'
  return <span className={`rw-badge ${k}`}>{BY_LABEL[k]}</span>
}

const fmt = (v, d = 3) => {
  if (v == null || Number.isNaN(Number(v))) return '—'
  const n = Number(v)
  if (Number.isInteger(n) && Math.abs(n) >= 1) return n.toLocaleString()
  return n.toFixed(d)
}

export function Bars({ s }) {
  const items = s.items || []
  const max = Math.max(...items.map(i => Math.abs(Number(i.value) || 0)), 1e-9)
  return (
    <div className="rw-bars">
      {items.map((it, i) => {
        const v = Number(it.value) || 0
        const w = (Math.abs(v) / max) * (s.signed ? 50 : 100)
        const style = s.signed
          ? (v >= 0 ? { left: '50%', width: `${w}%` } : { left: `${50 - w}%`, width: `${w}%` })
          : { left: 0, width: `${w}%` }
        return (
          <div className="rw-bar-row" key={i}>
            <span className="lab" title={it.label}>{it.label}</span>
            <div className="rw-bar-track">
              {s.signed && <div className="rw-bar-mid" />}
              <div className={`rw-bar-fill${s.signed && v < 0 ? ' neg' : ''}`} style={style} />
            </div>
            <span className="val">{s.signed && v > 0 ? '+' : ''}{fmt(v, s.digits ?? 3)}{s.unit || ''}</span>
          </div>
        )
      })}
      {s.signed && (
        <div className="rw-legend"><span><i style={{ background: 'var(--series-1)' }} />moves the target up</span>
          <span><i style={{ background: 'var(--series-2)' }} />moves it down</span></div>
      )}
    </div>
  )
}

function Table({ s }) {
  return (
    <div className="rw-table-wrap">
      <table className="rw-table">
        <thead><tr>{(s.columns || []).map(c => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {(s.rows || []).map((r, i) => (
            <tr key={i}>{r.map((c, j) => (
              <td key={j}>{j === s.source_col ? <SourceBadge by={c} /> : (c || '—')}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
      {s.more > 0 && <div className="rw-hint" style={{ marginTop: 6 }}>+ {s.more} more</div>}
    </div>
  )
}

function Steps({ s }) {
  return (
    <ol className="rw-steps">
      {(s.items || []).map((it, i) => (
        <li key={i}>
          <span className={`num${it.ok === false ? ' bad' : ''}`}>{i + 1}</span>
          <div><div className="name">{it.name}</div>{it.detail && <div className="detail">{it.detail}</div>}</div>
        </li>
      ))}
    </ol>
  )
}

function KV({ s }) {
  return (
    <div className="rw-kv">
      {(s.items || []).map((it, i) => (
        <div key={i}><span className="k">{it.label}</span><span className="v">{it.value}</span></div>
      ))}
    </div>
  )
}

function Diff({ s }) {
  const col = (title, cells, n) => (
    <div className="rw-diff-col">
      <div className="hd">{title} · {n} columns</div>
      {(cells || []).map((c, i) => <div className="row" key={i}><span>{c.column}</span><span>{c.value}</span></div>)}
    </div>
  )
  return <div className="rw-diff">{col('Raw row', s.before, s.n_before)}{col('After preprocessing', s.after, s.n_after)}</div>
}

export function Features({ s }) {
  const items = s.items || []
  const max = Math.max(...items.map(i => Math.abs(Number(i.value) || 0)), 1e-9)
  return (
    <div className="rw-bars">
      {items.map((it, i) => {
        const v = Number(it.value)
        const has = it.value != null && !Number.isNaN(v)
        return (
          <div className="rw-bar-row" key={i} style={{ gridTemplateColumns: 'minmax(90px, 30%) 1fr auto' }}>
            <span className="lab mono" title={it.label}>{it.label}</span>
            <div className="rw-bar-track">
              {has && <div className={`rw-bar-fill${v < 0 ? ' neg' : ''}`}
                style={{ left: 0, width: `${(Math.abs(v) / max) * 100}%`, background: it.accepted ? 'var(--ok)' : undefined }} />}
            </div>
            <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <span className="mono" style={{ fontSize: 12 }}>{has ? `${v >= 0 ? '+' : ''}${v.toFixed(4)}` : ''}</span>
              <span className={`rw-status${it.accepted ? ' ok' : ''}`}>{it.accepted ? 'added' : it.status}</span>
            </span>
          </div>
        )
      })}
      <div className="rw-hint">Bar = cross-validated gain over the current feature set. Only features with a reliable gain are added.</div>
    </div>
  )
}

/** Dot plot — scores sit close together, so a truncated axis with dots (not bars) is honest. */
export function DotPlot({ items, threshold, lowerBetter, showCv }) {
  const vals = items.flatMap(i => [i.value, showCv ? i.cv : null]).concat(threshold != null ? [threshold] : [])
    .filter(v => v != null && !Number.isNaN(Number(v))).map(Number)
  if (!vals.length) return null
  let lo = Math.min(...vals), hi = Math.max(...vals)
  const pad = Math.max((hi - lo) * 0.15, Math.abs(hi) * 0.005, 0.002)
  lo -= pad; hi += pad
  const pos = v => `${((Number(v) - lo) / (hi - lo)) * 100}%`
  return (
    <div className="rw-dotplot">
      {items.map((it, i) => (
        <div className={`rw-dp-row${it.best ? ' best' : ''}`} key={i}>
          <span className="lab" title={it.label}>{it.label}{it.model ? ` · ${it.model}` : ''}</span>
          <div className="rw-dp-track">
            {threshold != null && <div className="rw-dp-thr" style={{ left: pos(threshold) }} />}
            {showCv && it.cv != null && <div className="rw-dp-dot cv" style={{ left: pos(it.cv) }} title={`CV ${fmt(it.cv, 4)}`} />}
            {it.value != null && <div className={`rw-dp-dot${it.best ? ' best' : ''}`} style={{ left: pos(it.value) }} />}
          </div>
          <span className="mono" style={{ textAlign: 'right', fontSize: 12 }}>{fmt(it.value, 4)}</span>
        </div>
      ))}
      <div className="rw-dp-axis"><div /><div><span>{fmt(lo, 3)}</span><span>{fmt(hi, 3)}</span></div><div /></div>
      <div className="rw-legend" style={{ marginTop: 6 }}>
        <span><i style={{ background: 'var(--accent)' }} />best</span>
        <span><i style={{ background: 'var(--series-1)' }} />test score</span>
        {showCv && <span><i style={{ border: '1.5px solid var(--faint)' }} />cross-validation</span>}
        {threshold != null && <span><i style={{ borderRadius: 0, width: 0, height: 10, borderLeft: '1.5px dashed var(--accent)' }} />threshold</span>}
        {lowerBetter && <span>lower is better</span>}
      </div>
    </div>
  )
}

function Split({ s }) {
  const tr = Number(s.train) || 0, te = Number(s.test) || 0
  const tot = tr + te || 1
  return (
    <div>
      <div className="rw-split">
        <div style={{ width: `${(tr / tot) * 100}%`, background: 'var(--series-1)', color: '#fff' }}>Train {fmt(tr)}</div>
        <div style={{ width: `${(te / tot) * 100}%`, background: 'var(--series-2)', color: '#fff' }}>Test {fmt(te)}</div>
      </div>
      <div className="rw-hint" style={{ marginTop: 6 }}>{s.kind ? `${s.kind} split — ` : ''}the test rows were kept aside and never used for training or tuning.</div>
    </div>
  )
}

function Probes({ s }) {
  const items = s.items || []
  const max = Math.max(...items.map(i => Math.abs(Number(i.value) || 0) + (Number(i.se) || 0)), 1e-9)
  return (
    <div className="rw-bars">
      {items.map((it, i) => {
        const v = Number(it.value) || 0
        const w = (Math.abs(v) / max) * 50
        return (
          <div className="rw-bar-row" key={i}>
            <span className="lab" title={it.label}>{it.label}</span>
            <div className="rw-bar-track">
              <div className="rw-bar-mid" />
              <div className={`rw-bar-fill${v < 0 ? ' neg' : ''}`}
                style={{ left: v >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`, opacity: it.significant ? 1 : 0.45 }} />
            </div>
            <span className="val">{v >= 0 ? '+' : ''}{v.toFixed(4)}{it.significant ? ' ✓' : ''}</span>
          </div>
        )
      })}
      <div className="rw-hint">Change in cross-validated score for each experiment{s.baseline != null ? ` (baseline ${fmt(s.baseline, 4)})` : ''}. ✓ = larger than its noise.</div>
    </div>
  )
}

export function Section({ s }) {
  let body = null
  switch (s.type) {
    case 'bars': body = <Bars s={s} />; break
    case 'table': body = <Table s={s} />; break
    case 'steps': body = <Steps s={s} />; break
    case 'list': body = <ul className="rw-list">{(s.items || []).map((t, i) => <li key={i}>{t}</li>)}</ul>; break
    case 'chips': body = <div className="rw-cols">{(s.items || []).map((t, i) => <span className="rw-chip mono" key={i}>{t}</span>)}</div>; break
    case 'kv': body = <KV s={s} />; break
    case 'diff': body = <Diff s={s} />; break
    case 'features': body = <Features s={s} />; break
    case 'leaderboard': body = <DotPlot items={s.items || []} lowerBetter={s.lower_better} showCv />; break
    case 'rounds': body = <DotPlot items={s.items || []} threshold={s.threshold} />; break
    case 'split': body = <Split s={s} />; break
    case 'probes': body = <Probes s={s} />; break
    default: return null
  }
  const wide = ['table', 'diff', 'leaderboard'].includes(s.type)
  return (
    <section className={`rw-sec${wide ? ' wide' : ''}`}>
      <h3 className="h3">{s.title}</h3>
      {body}
    </section>
  )
}
