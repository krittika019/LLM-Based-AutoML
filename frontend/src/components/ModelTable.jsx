import React, { useState } from 'react'

function scoreCell(v) {
  if (v == null || v === '' || v === 'N/A') return <span className="score-na">—</span>
  const num = parseFloat(v)
  if (isNaN(num)) return <span>{v}</span>
  const color =
    num >= 0.9  ? '#10b981' :
    num >= 0.75 ? '#3b82f6' :
    num >= 0.6  ? '#f59e0b' : '#ef4444'
  return <span style={{ color, fontWeight: 600 }}>{num.toFixed(4)}</span>
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

function metricColor(key, val) {
  const n = parseFloat(val)
  if (isNaN(n)) return '#94a3b8'
  if (LOWER_BETTER.has(key)) {
    return n <= 0.1 ? '#10b981' : n <= 0.3 ? '#3b82f6' : n <= 0.6 ? '#f59e0b' : '#ef4444'
  }
  return n >= 0.9 ? '#10b981' : n >= 0.75 ? '#3b82f6' : n >= 0.6 ? '#f59e0b' : '#ef4444'
}

function MetricChip({ label, value }) {
  const n = parseFloat(value)
  const display = isNaN(n) ? '—' : n.toFixed(4)
  return (
    <div className="metric-chip">
      <span className="metric-chip-label">{label}</span>
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

function DetailedMetrics({ metrics, taskType }) {
  if (!metrics || !Object.keys(metrics).length) return null
  const labelMap = taskType === 'regression' ? REG_LABELS : CLS_LABELS
  const cm = metrics.confusion_matrix
  const chips = Object.entries(labelMap)
    .filter(([k]) => metrics[k] != null)
    .map(([k, label]) => <MetricChip key={k} label={label} value={metrics[k]} />)

  return (
    <div className="metrics-detail">
      <div className="metrics-chip-grid">{chips}</div>
      {taskType !== 'regression' && cm && <ConfusionMatrix cm={cm} />}
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

  const toggleRow = (i) => setExpanded(prev => ({ ...prev, [i]: !prev[i] }))

  return (
    <div className="model-table-wrap">
      {metric && (
        <div className="model-table-meta">
          Primary metric: <strong>{metric}</strong>
          {taskType && <> · Task: <strong>{taskType}</strong></>}
        </div>
      )}
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
                      </div>
                    </td>
                    <td>{scoreCell(row.cv_score)}</td>
                    <td>{scoreCell(row.test_score)}</td>
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
                        <DetailedMetrics metrics={row.detailed_metrics} taskType={taskType} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              )
            })}
          </tbody>
        </table>
      </div>

      {interpretation && (
        <div className="model-interpretation">
          <span className="model-interpretation-label">LLM Interpretation:</span>
          <span className="model-interpretation-text">{interpretation}</span>
        </div>
      )}
    </div>
  )
}
