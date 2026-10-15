import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid,
  ScatterChart, Scatter, ReferenceLine,
} from 'recharts'
import ModelTable from './ModelTable.jsx'
import UnsupervisedResults from './UnsupervisedResults.jsx'

// ── Shared section wrapper ──────────────────────────────────────────────────

function AnalyticsSection({ title, subtitle, children }) {
  return (
    <section className="analytics-section">
      <div className="analytics-section-header">
        <h3 className="analytics-section-title">{title}</h3>
        {subtitle && <p className="analytics-section-subtitle">{subtitle}</p>}
      </div>
      {children}
    </section>
  )
}

// ── Decision Router breakdown ────────────────────────────────────────────────

function RouterBreakdown({ stats }) {
  if (!stats || stats.rule_decisions == null) return null
  const total = (stats.rule_decisions ?? 0) + (stats.stat_decisions ?? 0) + (stats.llm_decisions ?? 0)
  if (total === 0) return <p className="chart-none">No routing data available for this run.</p>

  const segments = [
    { key: 'rule', label: 'Rule-based', value: stats.rule_decisions ?? 0, color: '#10b981' },
    { key: 'stat', label: 'Statistical', value: stats.stat_decisions ?? 0, color: '#3b82f6' },
    { key: 'llm', label: 'LLM', value: stats.llm_decisions ?? 0, color: '#ef4444' },
  ]

  return (
    <div className="chart-section">
      <div className="router-stat-callout">
        <span className="router-stat-num">{stats.avoided_pct?.toFixed?.(1) ?? stats.avoided_pct}%</span>
        <span className="router-stat-label">of preprocessing decisions resolved without an LLM call</span>
      </div>
      <div className="router-bar-wrap">
        {segments.map(s => (
          <div
            key={s.key}
            className="router-bar-seg"
            style={{ width: `${total > 0 ? (s.value / total) * 100 : 0}%`, background: s.color }}
            title={`${s.label}: ${s.value} (${total > 0 ? ((s.value / total) * 100).toFixed(1) : 0}%)`}
          />
        ))}
      </div>
      <div className="router-legend">
        {segments.map(s => (
          <div key={s.key} className="router-legend-item">
            <span className="router-legend-dot" style={{ background: s.color }} />
            {s.label}: <strong>{s.value}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── SHAP feature impact ──────────────────────────────────────────────────────

function ShapChart({ shap }) {
  if (!shap) return null
  if (!shap.available) {
    return (
      <p className="chart-none">
        SHAP explanation unavailable{shap.reason ? ` (${shap.reason})` : ''}.
      </p>
    )
  }
  const entries = (shap.features || []).slice().reverse()
  if (entries.length === 0) return <p className="chart-none">No feature impact data.</p>

  return (
    <div className="chart-section">
      <ResponsiveContainer width="100%" height={Math.max(140, entries.length * 28)}>
        <BarChart data={entries} layout="vertical" margin={{ top: 4, right: 44, left: 6, bottom: 4 }}>
          <CartesianGrid horizontal={false} strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 11 }} interval={0} />
          <Tooltip
            formatter={(v, key, entry) => [
              `${Number(entry.payload.mean_abs_shap).toFixed(4)} (${entry.payload.mean_signed_shap >= 0 ? '+' : ''}${Number(entry.payload.mean_signed_shap).toFixed(4)} signed)`,
              'Mean |SHAP|',
            ]}
            contentStyle={{ fontSize: 12 }}
          />
          <Bar dataKey="mean_abs_shap" radius={[0, 4, 4, 0]}>
            {entries.map((e, i) => (
              <Cell key={i} fill={e.mean_signed_shap >= 0 ? '#ef4444' : '#3b82f6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="shap-legend">
        <span><span className="router-legend-dot" style={{ background: '#ef4444' }} /> pushes prediction up</span>
        <span><span className="router-legend-dot" style={{ background: '#3b82f6' }} /> pushes prediction down</span>
      </div>
      <p className="analytics-caption">
        {shap.method === 'tree' && `Exact SHAP values (TreeExplainer) on ${shap.n_samples_explained} test rows.`}
        {shap.method === 'linear' && `Exact SHAP values (LinearExplainer) on ${shap.n_samples_explained} test rows.`}
        {shap.method === 'kernel' && (shap.note || `Approximate SHAP values (KernelExplainer) on ${shap.n_samples_explained} test rows.`)}
      </p>
    </div>
  )
}

// ── Diagnostic plot: confusion matrix (classification) / scatter (regression) ──

function ConfusionMatrixPlot({ cm }) {
  if (!cm || !cm.length) return <p className="chart-none">No confusion matrix available.</p>
  const n = cm.length
  return (
    <div className="confusion-wrap">
      <div className="confusion-grid" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
        {cm.map((row, ri) =>
          row.map((cell, ci) => (
            <div key={`${ri}-${ci}`} className={`confusion-cell ${ri === ci ? 'confusion-cell-diag' : ''}`}>
              {cell}
            </div>
          ))
        )}
      </div>
      <div className="confusion-legend">Rows = Actual · Cols = Predicted</div>
    </div>
  )
}

function ScatterDiagnostic({ sample }) {
  if (!sample || sample.length === 0) return <p className="chart-none">No prediction sample available.</p>
  const vals = sample.flatMap(p => [p.actual, p.predicted])
  const lo = Math.min(...vals), hi = Math.max(...vals)
  return (
    <div>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
          <XAxis type="number" dataKey="actual" name="Actual" domain={[lo, hi]} tick={{ fontSize: 11 }} />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[lo, hi]} tick={{ fontSize: 11 }} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 12 }} />
          <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="#94a3b8" strokeDasharray="4 4" />
          <Scatter data={sample} fill="#6366f1" fillOpacity={0.55} />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="analytics-caption">Points on the dashed line are perfect predictions.</p>
    </div>
  )
}

function DiagnosticPlot({ taskType, metrics }) {
  if (!metrics) return <p className="chart-none">No diagnostic data available.</p>
  if (taskType === 'regression') {
    return <ScatterDiagnostic sample={metrics.scatter_sample} />
  }
  return <ConfusionMatrixPlot cm={metrics.confusion_matrix} />
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function AnalyticsPage({ session }) {
  if (!session) {
    return <div className="analytics-empty">Run a pipeline to see analytics.</div>
  }

  const execResults = session.execution_results || {}
  const unsupervisedResults = session.unsupervised_results || {}
  const isUnsupervised = !!unsupervisedResults.status
  const routerStats = session.preprocessing_meta?.router_stats || null
  const shap = session.shap_summary || null
  const taskType = execResults.task_type || ''
  const bestMetrics = execResults.best_model_metrics || {}
  const bestModelName = (execResults.best_model_name || '').replace(/^\*BEST\*\s*/, '')

  if (isUnsupervised) {
    const algoLabel = unsupervisedResults.algorithm === 'kmeans' ? 'K-Means clustering' : 'PCA dimensionality reduction'
    return (
      <div className="analytics-page">
        <div className="analytics-header">
          <h2>Analytics</h2>
          <p>Unsupervised run — no target column, no predicted-vs-actual comparison.</p>
        </div>

        <AnalyticsSection
          title="Decision Router"
          subtitle="How many preprocessing decisions were resolved deterministically vs. forwarded to the LLM."
        >
          <RouterBreakdown stats={routerStats} />
        </AnalyticsSection>

        <AnalyticsSection
          title="Unsupervised Results"
          subtitle={`${algoLabel} on the full dataset — validated with internal metrics, no held-out labels.`}
        >
          <UnsupervisedResults results={unsupervisedResults} />
        </AnalyticsSection>
      </div>
    )
  }

  return (
    <div className="analytics-page">
      <div className="analytics-header">
        <h2>Analytics</h2>
        <p>A closer look at how this run's decisions and predictions were made.</p>
      </div>

      <AnalyticsSection
        title="Decision Router"
        subtitle="How many preprocessing decisions were resolved deterministically vs. forwarded to the LLM."
      >
        <RouterBreakdown stats={routerStats} />
      </AnalyticsSection>

      <AnalyticsSection
        title="Model Comparison"
        subtitle="Every model that was trained and evaluated, ranked by test score."
      >
        {execResults.comparison_table?.length
          ? <ModelTable data={execResults} />
          : <p className="chart-none">No model results available.</p>}
      </AnalyticsSection>

      <AnalyticsSection
        title={taskType === 'regression' ? 'Predicted vs. Actual' : 'Confusion Matrix'}
        subtitle={bestModelName ? `Diagnostic for the winning model — ${bestModelName}.` : undefined}
      >
        <DiagnosticPlot taskType={taskType} metrics={bestMetrics} />
      </AnalyticsSection>

      <AnalyticsSection
        title="SHAP Feature Impact"
        subtitle={bestModelName ? `What drives ${bestModelName}'s predictions, ranked by mean |SHAP| value.` : undefined}
      >
        <ShapChart shap={shap} />
      </AnalyticsSection>
    </div>
  )
}
