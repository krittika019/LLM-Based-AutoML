import React, { useState } from 'react'

function scoreCell(v, lowerIsBetter = false, allScores = null) {
  if (v == null || v === '' || v === 'N/A') return <span className="score-na">—</span>
  const num = parseFloat(v)
  if (isNaN(num)) return <span>{v}</span>
  let color
  // For raw regression metrics (RMSE/MAE in data units), fixed 0-1 thresholds don't apply.
  // Use relative rank among all non-null scores when allScores is provided and range > 2.
  if (lowerIsBetter && allScores && allScores.length > 1) {
    const valid = allScores.filter(s => s != null && !isNaN(s))
    const range = Math.max(...valid) - Math.min(...valid)
    if (range > 2) {
      // Relative: best (min) = green, worst (max) = red
      const mn = Math.min(...valid), mx = Math.max(...valid)
      const rel = range > 0 ? (num - mn) / range : 0
      color = rel <= 0.15 ? '#10b981' : rel <= 0.4 ? '#3b82f6' : rel <= 0.7 ? '#f59e0b' : '#ef4444'
    } else {
      color = num <= 0.1 ? '#10b981' : num <= 0.5 ? '#3b82f6' : num <= 1.5 ? '#f59e0b' : '#ef4444'
    }
  } else if (lowerIsBetter) {
    color = num <= 0.1 ? '#10b981' : num <= 0.5 ? '#3b82f6' : num <= 1.5 ? '#f59e0b' : '#ef4444'
  } else {
    color = num >= 0.9 ? '#10b981' : num >= 0.75 ? '#3b82f6' : num >= 0.6 ? '#f59e0b' : '#ef4444'
  }
  // Format: use exponential notation for very large numbers
  const display = Math.abs(num) >= 1e6 ? num.toExponential(3) : num.toFixed(4)
  return <span style={{ color, fontWeight: 600 }}>{display}</span>
}

function StatusBadge({ status }) {
  if (!status) return <span className="score-na">—</span>
  const s = String(status).toLowerCase()
  let cls = 'model-status-badge'
  let label = status
  if (s === 'ok' || s === 'done') { cls += ' badge-ok'; label = 'ok' }
  else if (s.startsWith('skip')) { cls += ' badge-skip'; label = 'skipped' }
  else if (s.startsWith('fail')) { cls += ' badge-fail'; label = 'failed' }
  else if (s.startsWith('elim')) { cls += ' badge-skip'; label = 'eliminated' }
  else { cls += ' badge-skip'; label = s }
  return <span className={cls}>{label}</span>
}

function FrameworkBadge({ source }) {
  if (source !== 'automl_framework') return null
  return <span className="framework-badge" title="External AutoML framework benchmark">AutoML</span>
}

// ── Metric label maps ─────────────────────────────────────────────────────────
const CLS_LABELS = {
  accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1: 'F1',
  roc_auc: 'ROC-AUC', pr_auc: 'PR-AUC', log_loss: 'Log Loss', brier_score: 'Brier Score',
}
const REG_LABELS = {
  mae: 'MAE', mse: 'MSE', rmse: 'RMSE', r2: 'R²',
  median_ae: 'Median AE', mape_pct: 'MAPE (%)',
}
// Lower-is-better keys (render with inverted color)
const LOWER_BETTER = new Set(['log_loss', 'brier_score', 'mae', 'mse', 'rmse', 'median_ae', 'mape_pct'])

// Key metrics to always show (secondary importance — primary is always included)
const CLS_IMPORTANT = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
const REG_IMPORTANT = ['r2', 'rmse', 'mae']

function metricColor(key, val) {
  const n = parseFloat(val)
  if (isNaN(n)) return '#94a3b8'
  if (LOWER_BETTER.has(key)) {
    return n <= 0.1 ? '#10b981' : n <= 0.3 ? '#3b82f6' : n <= 0.6 ? '#f59e0b' : '#ef4444'
  }
  return n >= 0.9 ? '#10b981' : n >= 0.75 ? '#3b82f6' : n >= 0.6 ? '#f59e0b' : '#ef4444'
}

function MetricChip({ label, value, isPrimary }) {
  const n = parseFloat(value)
  const display = isNaN(n) ? '—' : n.toFixed(4)
  return (
    <div className={`metric-chip${isPrimary ? ' metric-chip-primary' : ''}`}>
      <span className="metric-chip-label">{label}{isPrimary && <span className="metric-chip-star"> ★</span>}</span>
      <span className="metric-chip-value" style={{ color: metricColor(label.toLowerCase().replace(/[^a-z0-9]/g, '_'), n) }}>
        {display}
      </span>
    </div>
  )
}

function ConfusionMatrix({ cm }) {
  if (!cm || !cm.length) return null
  const n = cm.length
  return (
    <div className="confusion-wrap">
      <div className="confusion-title">Confusion Matrix</div>
      <div className="confusion-grid" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
        {cm.map((row, ri) =>
          row.map((cell, ci) => {
            const isDiag = ri === ci
            return (
              <div
                key={`${ri}-${ci}`}
                className={`confusion-cell ${isDiag ? 'confusion-cell-diag' : ''}`}
              >
                {cell}
              </div>
            )
          })
        )}
      </div>
      <div className="confusion-legend">Rows = Actual · Cols = Predicted</div>
    </div>
  )
}

function DetailedMetrics({ metrics, taskType, primaryMetric }) {
  if (!metrics || !Object.keys(metrics).length) return null
  const labelMap = taskType === 'regression' ? REG_LABELS : CLS_LABELS
  const important = taskType === 'regression' ? REG_IMPORTANT : CLS_IMPORTANT
  const cm = metrics.confusion_matrix

  // Normalise primary metric key (e.g. "roc-auc" → "roc_auc")
  const normPrimary = (primaryMetric || '').toLowerCase().replace(/[^a-z0-9]/g, '_')

  // Build ordered list: primary first, then important set, dedup
  const orderedKeys = [
    normPrimary,
    ...important,
  ].filter((k, idx, arr) => arr.indexOf(k) === idx && labelMap[k] && metrics[k] != null)

  const chips = orderedKeys.map(k => (
    <MetricChip
      key={k}
      label={labelMap[k]}
      value={metrics[k]}
      isPrimary={k === normPrimary}
    />
  ))

  return (
    <div className="metrics-detail">
      <div className="metrics-chip-grid">{chips}</div>
      {taskType !== 'regression' && cm && <ConfusionMatrix cm={cm} />}
    </div>
  )
}

function StabilityBadge({ foldVariance }) {
  if (foldVariance == null || isNaN(parseFloat(foldVariance))) return null
  const v = parseFloat(foldVariance)
  let label, color, title
  if (v <= 0.005) { label = 'stable'; color = '#10b981'; title = `CV fold variance: ${v.toFixed(4)} — model is stable across folds` }
  else if (v <= 0.02) { label = 'moderate'; color = '#f59e0b'; title = `CV fold variance: ${v.toFixed(4)} — some variance across folds` }
  else { label = 'unstable'; color = '#ef4444'; title = `CV fold variance: ${v.toFixed(4)} — high variance, possible overfitting` }
  return (
    <span title={title} style={{
      marginLeft: 6, padding: '1px 6px', borderRadius: 4, fontSize: 11,
      background: color + '22', color, border: `1px solid ${color}55`, fontWeight: 600,
    }}>
      {label}
    </span>
  )
}

function EliminationAudit({ audit }) {
  if (!audit || !audit.eliminated_family) return null
  const correct = audit.elimination_correct
  const rank = audit.subsample_rank_of_eliminated
  const total = audit.total_candidates
  const icon = correct ? '✅' : '⚠️'
  const label = correct ? 'Correct elimination' : 'Questionable elimination'
  return (
    <div className="elim-audit">
      <span className="elim-audit-icon">{icon}</span>
      <span className="elim-audit-text">
        <strong>Progressive screening:</strong> <code>{audit.eliminated_family}</code> eliminated after Round 1
        (subsample rank {rank}/{total}) — <em>{label}</em>.
        {typeof audit.eliminated_subsample_score === 'number' && (
          <> Subsample score: <code>{audit.eliminated_subsample_score.toFixed(4)}</code>.</>
        )}
      </span>
    </div>
  )
}

export default function ModelTable({ data }) {
  const [expanded, setExpanded] = useState({})
  if (!data?.comparison_table?.length) return null

  const rows = data.comparison_table
  const interpretation = data.interpretation || ''
  const taskType = data.task_type || ''
  const metric = data.metric || ''
  const metricDisplay = data.metric_display || metric
  const lowerIsBetter = data.metric_is_lower_better === true
  const eliminationAudit = data.elimination_audit || null

  const toggleRow = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }))

  // Collect all test display scores for relative coloring of regression metrics
  const allTestScores = rows.map(r => {
    const v = r.test_display_score ?? r.test_score
    return v != null ? parseFloat(v) : null
  }).filter(v => v != null && !isNaN(v))

  return (
    <div className="model-table-wrap">
      {metricDisplay && (
        <div className="model-table-meta">
          Primary metric: <strong>{metricDisplay}</strong>
          {taskType && <> · Task: <strong>{taskType}</strong></>}
        </div>
      )}
      <EliminationAudit audit={eliminationAudit} />
      <div className="model-table-scroll">
        <table className="model-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>CV Score</th>
              <th>Test Score</th>
              <th>Time (s)</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isBest = row.is_best
              const isElim = (row.status ?? '').startsWith('elim') || (row.status ?? '').startsWith('skip')
              const isFramework = row.source === 'automl_framework'
              const hasDetail = row.detailed_metrics && Object.keys(row.detailed_metrics).length > 0
              const isExpanded = expanded[i]
              return (
                <React.Fragment key={i}>
                  <tr className={isBest ? 'row-best' : isElim ? 'row-elim' : isFramework ? 'row-framework' : ''}>
                    <td>
                      <div className="model-name-cell">
                        <span style={isElim ? { opacity: 0.5 } : {}}>
                          {(row.model_name ?? '').replace(/^\*BEST\*\s*/, '')}
                        </span>
                        {isBest && <span className="best-badge-pill">Best</span>}
                        <FrameworkBadge source={row.source} />
                        <StabilityBadge foldVariance={row.fold_variance} />
                      </div>
                    </td>
                    <td>{scoreCell(row.cv_display_score ?? row.cv_score, lowerIsBetter)}</td>
                    <td>{scoreCell(row.test_display_score ?? row.test_score, lowerIsBetter, allTestScores)}</td>
                    <td>
                      {typeof row.training_time === 'number'
                        ? row.training_time.toFixed(1)
                        : row.training_time ?? '—'}
                    </td>
                    <td><StatusBadge status={row.status} /></td>
                    <td>
                      {hasDetail && (
                        <button
                          className="metrics-toggle-btn"
                          onClick={() => toggleRow(i)}
                          title={isExpanded ? 'Hide metrics' : 'Show all metrics'}
                        >
                          {isExpanded ? '▲' : '▼'}
                        </button>
                      )}
                    </td>
                  </tr>
                  {hasDetail && isExpanded && (
                    <tr className="metrics-detail-row">
                      <td colSpan={6}>
                        <DetailedMetrics metrics={row.detailed_metrics} taskType={taskType} primaryMetric={metric} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>

      {data.leakage_warning_models?.length > 0 && (
        <div className="leakage-warning-banner">
          ⚠️ <strong>Potential data leakage:</strong> {data.leakage_warning_models.join(', ')} scored ≥{' '}
          {(data.near_perfect_sentinel ?? 0.999).toFixed(3)} — excluded from best-model selection.
          Check for features that encode the target.
        </div>
      )}
      {interpretation && (
        <div className="model-interpretation">
          <span className="model-interpretation-label">LLM Interpretation:</span>
          <span className="model-interpretation-text">{interpretation}</span>
        </div>
      )}
    </div>
  )
}
