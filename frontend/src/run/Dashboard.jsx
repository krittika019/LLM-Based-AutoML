import React, { useMemo, useState } from 'react'
import UnsupervisedResults from '../components/UnsupervisedResults.jsx'
import { Bars, DotPlot, Features } from './Sections.jsx'
import { fmtSecs } from './PipelineCanvas.jsx'

const f4 = v => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(4))
const pct = v => (v == null || !Number.isFinite(v) ? '—' : `${Math.round(v * 100)}%`)
const human = s => String(s ?? '').replace(/_/g, ' ')

// graph node -> label for "Where the time went"
const NODE_GROUP = {
  planning: 'Planning', basic_cleaning: 'Cleaning & EDA', split_data: 'Cleaning & EDA', eda: 'Cleaning & EDA',
  feature_selection: 'Feature selection', deep_cleaning: 'Deep cleaning', feature_engineering: 'Feature engineering',
  verification: 'Verification', phase4_planning: 'Model plan', train_evaluate: 'Model training',
  unsupervised_train: 'Model training', interpret_results: 'Model training', threshold_check: 'Check & revise',
  diagnose_failure: 'Check & revise', targeted_revision: 'Check & revise', phase7_final_output: 'Script & report',
}

function Card({ title, sub, aside, children, grow = 1, basis = 420 }) {
  return (
    <section className="rw-card db-card" style={{ flex: `${grow} 1 ${basis}px` }}>
      <div className="db-card-head">
        <div style={{ flex: 1, minWidth: 'min(240px, 100%)' }}><h2>{title}</h2>{sub && <p>{sub}</p>}</div>
        {aside}
      </div>
      {children}
    </section>
  )
}

function Kpi({ label, value, note, tone }) {
  return (
    <div className="rw-card db-kpi">
      <span className="label">{label}</span>
      <span className={`value${tone ? ` ${tone}` : ''}`} title={String(value)}>{value}</span>
      {note && <span className="note">{note}</span>}
    </div>
  )
}

// ── classification: threshold explorer, ROC, confusion matrix ────────────────

function countsAt(c) {
  const { tp, fp, fn, tn } = c
  return {
    ...c,
    recall: tp + fn ? tp / (tp + fn) : null, precision: tp + fp ? tp / (tp + fp) : null,
    fpr: fp + tn ? fp / (fp + tn) : 0, tpr: tp + fn ? tp / (tp + fn) : 0,
  }
}

function ThresholdExplorer({ curves, idx, setIdx, chosen, goalRecall, target }) {
  const sweep = curves.sweep
  const c = countsAt(sweep[idx])
  const pos = `${target ? `${target} = ` : ''}${curves.positive_label}`
  const nPos = c.tp + c.fn, nNeg = c.fp + c.tn
  return (
    <Card grow={3} basis={560} title="Decision threshold explorer"
      sub={`A row is predicted as ${pos} when the model's probability is at or above the threshold. Move it to trade missed cases against false alarms.`}
      aside={goalRecall && <span className="rw-pill run">Your goal: catch every real case</span>}>
      <div className="db-thr">
        <label htmlFor="db-thr" className="db-thr-label">
          Threshold <span className="mono db-chip">{c.t.toFixed(2)}</span>
          {Math.abs(c.t - chosen) < 0.011 && <span className="rw-hint">— the model uses this one</span>}
        </label>
        <input id="db-thr" type="range" min={0} max={sweep.length - 1} step={1} value={idx}
          onChange={e => setIdx(Number(e.target.value))} aria-valuetext={`threshold ${c.t.toFixed(2)}`} />
        <div className="mono db-thr-scale"><span>{sweep[0].t.toFixed(2)} · flag more</span><span>{sweep[sweep.length - 1].t.toFixed(2)} · flag fewer</span></div>
      </div>
      <div className="db-thr-tiles">
        <div><span className="label">Missed cases</span><span className="big mono">{c.fn}</span><span className="note">of {nPos} real {pos}</span></div>
        <div><span className="label">False alarms</span><span className="big mono">{c.fp}</span><span className="note">of {nNeg} others</span></div>
        <div><span className="label">Recall (cases caught)</span><span className="big mono">{pct(c.recall)}</span><span className="note">precision {pct(c.precision)}</span></div>
      </div>
      <div>
        <div className="db-stack" title={`Caught ${c.tp}, missed ${c.fn} of ${nPos}`}>
          <span style={{ flex: `${c.tp} 1 0`, background: 'var(--series-1)' }} />
          {c.fn > 0 && <span style={{ flex: `${c.fn} 1 0`, background: 'var(--series-2)' }} />}
        </div>
        <div className="rw-legend" style={{ marginTop: 6 }}>
          <span><i style={{ background: 'var(--series-1)', borderRadius: 3 }} />Caught {c.tp}</span>
          <span><i style={{ background: 'var(--series-2)', borderRadius: 3 }} />Missed {c.fn}</span>
        </div>
      </div>
    </Card>
  )
}

function RocCurve({ curves, current, auc }) {
  const S = 240, P = 28
  const x = v => P + v * S, y = v => P + (1 - v) * S
  const path = curves.roc.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)} ${y(b).toFixed(1)}`).join('')
  const cur = countsAt(current)
  return (
    <Card grow={2} basis={320} title="ROC curve" sub="True-positive rate against false-positive rate. The orange dot is the threshold on the left.">
      <svg viewBox={`0 0 ${S + P * 2} ${S + P * 2}`} role="img" className="db-roc"
        aria-label={`ROC curve, area under curve ${f4(auc)}`}>
        {[0.25, 0.5, 0.75].map(g => <path key={g} d={`M${x(g)} ${y(0)}V${y(1)}M${x(0)} ${y(g)}H${x(1)}`} stroke="var(--grid)" />)}
        <path d={`M${x(0)} ${y(1)}V${y(0)}H${x(1)}`} fill="none" stroke="var(--border-strong)" />
        <path d={`M${x(0)} ${y(0)}L${x(1)} ${y(1)}`} stroke="var(--neutral-bar)" strokeWidth="1.5" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" />
        {curves.roc.map(([a, b], i) => (
          <circle key={i} cx={x(a)} cy={y(b)} r="6" fill="transparent">
            <title>{`false-positive rate ${a.toFixed(2)} · true-positive rate ${b.toFixed(2)}`}</title>
          </circle>
        ))}
        <circle cx={x(cur.fpr)} cy={y(cur.tpr)} r="6" fill="var(--accent)" stroke="var(--surface)" strokeWidth="2">
          <title>{`threshold ${cur.t.toFixed(2)}: TPR ${cur.tpr.toFixed(2)}, FPR ${cur.fpr.toFixed(2)}`}</title>
        </circle>
        <text x={x(0.5)} y={S + P * 2 - 6} fontSize="10" fill="var(--muted)" textAnchor="middle">false-positive rate</text>
        <text x={10} y={y(0.5)} fontSize="10" fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 10 ${y(0.5)})`}>true-positive rate</text>
        {auc != null && <text x={x(0.52)} y={y(0.22)} fontSize="13" fill="var(--text)" fontWeight="600">AUC {f4(auc)}</text>}
        <text x={x(0.52)} y={y(0.22) + 16} fontSize="10" fill="var(--muted)">diagonal = random guess</text>
      </svg>
    </Card>
  )
}

function Matrix({ grid, rowLabels, colLabels, names, sub }) {
  const max = Math.max(...grid.flat(), 1)
  return (
    <Card grow={1} basis={320} title="Confusion matrix" sub={sub}>
      <div className="db-cm" style={{ gridTemplateColumns: `minmax(60px, auto) repeat(${grid.length}, minmax(0, 1fr))` }}>
        <span />
        {colLabels.map(l => <span key={l} className="db-cm-ax">Predicted {l}</span>)}
        {grid.map((row, i) => (
          <React.Fragment key={i}>
            <span className="db-cm-ax" style={{ textAlign: 'left' }}>Actual {rowLabels[i]}</span>
            {row.map((v, j) => {
              const share = Math.round(18 + (v / max) * 72)
              return (
                <div key={j} className="db-cm-cell" title={`Actual ${rowLabels[i]}, predicted ${colLabels[j]}: ${v}`}
                  style={{ background: `color-mix(in srgb, var(--series-1) ${share}%, var(--surface))`, color: share > 55 ? '#fff' : 'var(--text)' }}>
                  <span className="mono">{v}</span>
                  {names && <span className="lab">{names[i][j]}</span>}
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>
    </Card>
  )
}

// ── regression ───────────────────────────────────────────────────────────────

function PredVsActual({ sample, metrics }) {
  const pts = (sample || []).filter(p => Number.isFinite(p.actual) && Number.isFinite(p.predicted))
  if (!pts.length) return null
  const vals = pts.flatMap(p => [p.actual, p.predicted])
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1
  const S = 260, P = 30
  const sx = v => P + ((v - lo) / span) * S, sy = v => P + (1 - (v - lo) / span) * S
  return (
    <Card grow={3} basis={520} title="Predicted vs actual"
      sub={`${pts.length} test rows. Points on the dashed line are perfect predictions; distance from it is the error.`}>
      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        <svg viewBox={`0 0 ${S + P * 2} ${S + P * 2}`} className="db-roc" role="img" aria-label="Scatter of predicted against actual values">
          <path d={`M${sx(lo)} ${sy(hi)}V${sy(lo)}H${sx(hi)}`} fill="none" stroke="var(--border-strong)" />
          <path d={`M${sx(lo)} ${sy(lo)}L${sx(hi)} ${sy(hi)}`} stroke="var(--neutral-bar)" strokeDasharray="4 4" strokeWidth="1.5" />
          {pts.map((p, i) => (
            <circle key={i} cx={sx(p.actual)} cy={sy(p.predicted)} r="4" fill="var(--series-1)" fillOpacity="0.55" stroke="var(--surface)" strokeWidth="1">
              <title>{`actual ${p.actual} · predicted ${p.predicted}`}</title>
            </circle>
          ))}
          <text x={sx(lo + span / 2)} y={S + P * 2 - 6} fontSize="10" fill="var(--muted)" textAnchor="middle">actual</text>
          <text x={10} y={sy(lo + span / 2)} fontSize="10" fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 10 ${sy(lo + span / 2)})`}>predicted</text>
        </svg>
        <div className="rw-kv" style={{ flex: 1, minWidth: 180 }}>
          {[['R²', metrics.r2], ['MAE', metrics.mae], ['RMSE', metrics.rmse], ['Median error', metrics.median_ae], ['MAPE', metrics.mape_pct != null ? `${Number(metrics.mape_pct).toFixed(1)}%` : null]]
            .filter(([, v]) => v != null)
            .map(([k, v]) => <div key={k}><span className="k">{k}</span><span className="v mono">{typeof v === 'string' ? v : f4(v)}</span></div>)}
        </div>
      </div>
    </Card>
  )
}

// ── shared panels ─────────────────────────────────────────────────────────────

function Stability({ rows, metric }) {
  const items = rows.filter(r => r.cv != null && r.sd != null)
  if (!items.length) return null
  const lo0 = Math.min(...items.map(r => r.cv - r.sd)), hi0 = Math.max(...items.map(r => r.cv + r.sd))
  const pad = Math.max((hi0 - lo0) * 0.1, 0.002)
  const lo = lo0 - pad, hi = hi0 + pad
  const pos = v => `${((v - lo) / (hi - lo)) * 100}%`
  return (
    <Card title="How stable is each model?" sub={`Cross-validation ${metric}: dot = mean over folds, line = ± one standard deviation. Short lines mean the score holds up on different slices of the data.`}>
      <div className="rw-dotplot">
        {items.map(r => (
          <div className={`rw-dp-row${r.best ? ' best' : ''}`} key={r.label}
            title={`${r.label}: CV ${f4(r.cv)} ± ${f4(r.sd)}`}>
            <span className="lab">{r.label}</span>
            <div className="rw-dp-track">
              <div className="db-whisker" style={{ left: pos(r.cv - r.sd), width: `calc(${pos(r.cv + r.sd)} - ${pos(r.cv - r.sd)})` }} />
              <div className={`rw-dp-dot${r.best ? ' best' : ''}`} style={{ left: pos(r.cv) }} />
            </div>
            <span className="mono" style={{ textAlign: 'right', fontSize: 12, color: 'var(--muted)' }}>± {r.sd.toFixed(3)}</span>
          </div>
        ))}
        <div className="rw-dp-axis"><div /><div><span>{lo.toFixed(3)}</span><span>{hi.toFixed(3)}</span></div><div /></div>
      </div>
    </Card>
  )
}

function Router({ stats }) {
  const parts = [
    { k: 'rule', label: 'Rule', v: stats.rule_decisions ?? 0 },
    { k: 'stat', label: 'Statistical test', v: stats.stat_decisions ?? 0 },
    { k: 'llm', label: 'LLM', v: stats.llm_decisions ?? 0 },
  ]
  const total = parts.reduce((a, p) => a + p.v, 0)
  if (!total) return null
  return (
    <Card basis={340} title="Decision router" sub={`How the ${total} preprocessing decisions were made${stats.avoided_pct != null ? ` — ${Math.round(stats.avoided_pct)}% without the LLM` : ''}.`}>
      <div className="db-stack" style={{ height: 26 }}>
        {parts.filter(p => p.v).map(p => <span key={p.k} title={`${p.label}: ${p.v}`} style={{ flex: `${p.v} 1 0`, background: `var(--${p.k}-fill)` }} />)}
      </div>
      <div className="rw-legend">
        {parts.map(p => <span key={p.k}><i style={{ background: `var(--${p.k}-fill)`, borderRadius: 3 }} />{p.label} <b className="mono">{p.v}</b></span>)}
      </div>
    </Card>
  )
}

// ── page ──────────────────────────────────────────────────────────────────────

export default function Dashboard({ session, fileName, runId, reportAvailable }) {
  const res = session.execution_results || {}
  const unsup = session.unsupervised_results || {}
  const table = (res.comparison_table || []).filter(r => r && r.status === 'ok' && Number.isFinite(Number(r.test_score)))
  const lower = !!res.metric_is_lower_better
  const disp = v => (v == null ? null : lower && v < 0 ? -v : Number(v))
  const metric = res.metric_display || res.metric || 'score'
  const best = table.find(r => r.is_best) || table.find(r => r.model_name === res.best_model_name) || null
  const bm = res.best_model_metrics || best?.detailed_metrics || {}
  const curves = bm.curves || best?.detailed_metrics?.curves || null
  const chosen = Number(best?.best_params?.decision_threshold ?? 0.5)
  const task = res.task_type || session.problem_summary?.task_type || session.task_config?.task_type || ''
  const target = session.task_config?.target_column || session.problem_summary?.target
  const goalRecall = !!session.execution_plan?.constraint_signals?.prioritize_recall

  const defaultIdx = curves ? curves.sweep.reduce((bi, s, i, a) => (Math.abs(s.t - chosen) < Math.abs(a[bi].t - chosen) ? i : bi), 0) : 0
  const [thrIdx, setThrIdx] = useState(defaultIdx)

  const timings = session.phase_timings || {}
  const total = Object.values(timings).reduce((a, v) => a + (Number(v) || 0), 0)
  const timeGroups = useMemo(() => {
    const g = {}
    for (const [k, v] of Object.entries(timings)) { const n = NODE_GROUP[k] || human(k); g[n] = (g[n] || 0) + (Number(v) || 0) }
    return Object.entries(g).sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value: Math.round(value * 10) / 10 }))
  }, [timings])
  const tc = session.token_cost_table || {}
  const llmCalls = tc._total ?? Object.entries(tc).filter(([k]) => !k.startsWith('_')).reduce((a, [, v]) => a + (Number(v) || 0), 0)
  const router = session.preprocessing_meta?.router_stats
  const sv = session.script_validation || {}
  const nTest = res.split_meta?.n_test

  const assignments = Object.values(session.deep_cleaning_assignments || {}).filter(a => a && typeof a === 'object')
  const quality = [
    ['Missing cells filled', session.problem_summary?.missing_values],
    ['Duplicate rows removed', session.cleaning_report?.duplicate_rows_removed],
    ['Type fixes', (session.cleaning_report?.coercion_log || []).length],
    ['Columns dropped', (session.feature_selection_log || []).length],
    ['Columns encoded', assignments.filter(a => a.encoding && a.encoding !== 'none').length],
    ['Outliers handled', assignments.filter(a => a.outlier && a.outlier !== 'leave').length],
  ].filter(([, v]) => Number(v) > 0).map(([label, value]) => ({ label, value: Number(value) }))

  const corr = Object.entries(session.eda_results?.target_correlations || {})
    .filter(([, v]) => Number.isFinite(Number(v))).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10)
    .map(([label, v]) => ({ label, value: Math.round(Number(v) * 1000) / 1000 }))
  const shapFeats = res.shap_summary?.available ? (res.shap_summary.features || []) : []
  const drivers = shapFeats.length
    ? shapFeats.slice(0, 10).map(f => ({ label: f.name, value: Math.round(Number(f.mean_abs_shap) * 10000) / 10000 }))
    : (res.feature_importances?.top_features || []).slice(0, 10).map(([n, v]) => ({ label: n, value: Math.round(Number(v) * 10) / 10 }))
  const fe = (session.feature_engineering_log || []).filter(e => e && e.feature)
    .map(e => ({ label: e.feature, value: e.cv_gain ?? null, accepted: e.status === 'added', status: human(e.status) }))

  const header = (
    <div className="db-head">
      <div>
        <h1>Analytics</h1>
        <span className="rw-hint" style={{ fontSize: 13 }}>
          {[fileName, human(task), res.metric_display && `metric ${metric}`, nTest && `test set ${Number(nTest).toLocaleString()} rows`].filter(Boolean).join(' · ')}
        </span>
      </div>
      <div className="rw-grow" />
      {reportAvailable && runId && <a className="rw-btn" href={`/download/report/${runId}`} download="automl_report.pdf">Export report</a>}
    </div>
  )

  if (!table.length && Object.keys(unsup).length) {
    return (
      <div className="db">
        {header}
        <section className="rw-card db-card"><div className="rw-legacy"><UnsupervisedResults results={unsup} session={session} /></div></section>
        {router && <div className="db-row"><Router stats={router} /></div>}
      </div>
    )
  }
  if (!table.length) return <div className="db">{header}<div className="rw-card rw-empty">This run has no trained models to analyse.</div></div>

  const boardItems = [...table].sort((a, b) => (lower ? -1 : 1) * (Number(b.test_score) - Number(a.test_score)))
    .map(r => ({ label: r.model_name, value: disp(Number(r.test_score)), cv: r.cv_score != null ? disp(Number(r.cv_score)) : null, best: r === best }))
  const stability = table.map(r => ({
    label: r.model_name, best: r === best,
    cv: r.cv_score != null && Number.isFinite(Number(r.cv_score)) ? disp(Number(r.cv_score)) : null,
    sd: r.fold_variance != null && Number(r.fold_variance) >= 0 ? Math.sqrt(Number(r.fold_variance)) : null,
  }))

  let matrix = null
  if (curves) {
    const c = curves.sweep[thrIdx]
    const n = curves.negative_label, p = curves.positive_label
    matrix = <Matrix grid={[[c.tn, c.fp], [c.fn, c.tp]]} rowLabels={[n, p]} colLabels={[n, p]}
      names={[['Correct', 'False alarm'], ['Missed', 'Caught']]}
      sub={`${best?.model_name} on ${(c.tn + c.fp + c.fn + c.tp).toLocaleString()} test rows at threshold ${c.t.toFixed(2)}.`} />
  } else if (Array.isArray(bm.confusion_matrix) && bm.confusion_matrix.length > 1) {
    const labels = bm.confusion_matrix.map((_, i) => `class ${i}`)
    matrix = <Matrix grid={bm.confusion_matrix} rowLabels={labels} colLabels={labels} sub={`${best?.model_name} on the test set (classes in sorted order).`} />
  }

  return (
    <div className="db">
      {header}
      <div className="db-kpis">
        <Kpi label="Best model" value={best?.model_name || res.best_model_name || '—'} note={`Best of ${table.length} candidates`} />
        <Kpi label={`Test ${metric}`} value={f4(disp(Number(best?.test_score ?? res.best_test_score)))} note="On rows never seen in training" />
        <Kpi label="Pipeline script" value={sv.status ? human(sv.status) : '—'} tone={sv.status === 'verified' ? 'ok' : undefined}
          note={sv.gap != null ? `Reproduced score · gap ${f4(sv.gap)}` : sv.message ? String(sv.message).slice(0, 60) : ''} />
        <Kpi label="Total run time" value={total ? fmtSecs(total) : '—'}
          note={timeGroups[0] && total ? `${Math.round((timeGroups[0].value / total) * 100)}% in ${timeGroups[0].label.toLowerCase()}` : ''} />
        <Kpi label="LLM calls" value={llmCalls || 0}
          note={router?.avoided_pct != null ? `${Math.round(router.avoided_pct)}% of cleaning decisions made without it` : ''} />
      </div>

      {task === 'classification' && curves && (
        <div className="db-row">
          <ThresholdExplorer curves={curves} idx={thrIdx} setIdx={setThrIdx} chosen={chosen} goalRecall={goalRecall} target={target} />
          <RocCurve curves={curves} current={curves.sweep[thrIdx]} auc={bm.roc_auc} />
        </div>
      )}
      {task === 'regression' && bm.scatter_sample && (
        <div className="db-row"><PredVsActual sample={bm.scatter_sample} metrics={bm} /></div>
      )}

      <div className="db-row">
        <Card grow={2} basis={560} title="Model comparison"
          sub={`Test ${metric} for every candidate, with its cross-validation score. The axis is zoomed in — dots, not bars, so small gaps are not exaggerated.`}>
          <DotPlot items={boardItems} lowerBetter={lower} showCv />
        </Card>
        {matrix}
      </div>

      <div className="db-row">
        {drivers.length > 0 && (
          <Card title="What drives predictions"
            sub={shapFeats.length ? `Mean |SHAP| — how much each feature moves ${best?.model_name}'s predictions on average.` : 'Model feature importance (% of total).'}>
            <Bars s={{ items: drivers, unit: shapFeats.length ? '' : '%' }} />
          </Card>
        )}
        {corr.length > 0 && (
          <Card title={task === 'regression' ? 'What raises or lowers the target' : 'What raises or lowers the outcome'}
            sub={`Correlation of each (cleaned) feature with ${target || 'the target'}. Association, not cause.`}>
            <Bars s={{ items: corr, signed: true }} />
          </Card>
        )}
      </div>

      <div className="db-row">
        <Stability rows={stability} metric={metric} />
        {fe.length > 0 && (
          <Card title="Engineered features" sub="Each proposed feature is kept only if it improves the cross-validated score by more than its noise.">
            <Features s={{ items: fe }} />
          </Card>
        )}
      </div>

      <div className="db-row">
        {quality.length > 0 && (
          <Card basis={340} title="Data quality fixed" sub="Problems found and handled before training.">
            <Bars s={{ items: quality }} />
          </Card>
        )}
        {router && <Router stats={router} />}
        {timeGroups.length > 0 && (
          <Card basis={340} title="Where the time went" sub={`Seconds per stage · ${fmtSecs(total)} total.`}>
            <Bars s={{ items: timeGroups, unit: ' s', digits: 1 }} />
          </Card>
        )}
      </div>
    </div>
  )
}
