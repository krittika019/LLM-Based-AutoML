import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid,
} from 'recharts'

// ── Theme-aware chart colors ──────────────────────────────────────────────────

function useChartColors() {
  const isDark = typeof document !== 'undefined' &&
    document.querySelector('.app-root')?.classList.contains('dark')
  return {
    axisTick: isDark ? '#a3a3a3' : '#9ca3af',
    gridLine: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
  }
}

// ── Insight block ─────────────────────────────────────────────────────────────

function InsightBlock({ text }) {
  if (!text) return null
  return (
    <div className="insight-block">
      <span className="insight-icon">💡</span>
      <span>{text}</span>
    </div>
  )
}

// ── Dynamic explanations ──────────────────────────────────────────────────────

function genDistExplanation(col, shape) {
  if (!col) return null
  const map = {
    approx_normal:     `${col} follows an approximately normal distribution — ideal for most models, no transformation needed.`,
    right_skewed:      `${col} is right-skewed with a long tail of high values. A log transformation may be applied to normalize it before training.`,
    left_skewed:       `${col} is left-skewed with values concentrated at the high end. A power transformation may improve model performance.`,
    insufficient_data: `${col} has too few unique values to reliably characterize its distribution shape.`,
  }
  return map[shape] ?? `${col} has been analyzed for skew and outliers prior to feature engineering.`
}

function genMissingExplanation(missing) {
  if (!missing) return null
  const sorted = Object.entries(missing).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
  if (sorted.length === 0)
    return 'No missing values detected — this dataset is complete and requires no imputation.'
  const high = sorted.filter(([, v]) => v > 50)
  const med  = sorted.filter(([, v]) => v > 0 && v <= 50)
  const parts = []
  if (high.length > 0) {
    const names = high.map(([c, v]) => `${c} (${v.toFixed(1)}%)`).join(', ')
    parts.push(`${names} ${high.length === 1 ? 'has' : 'have'} >50% missing and will be converted to a binary flag column.`)
  }
  if (med.length > 0) {
    const names = med.slice(0, 3).map(([c, v]) => `${c} (${v.toFixed(1)}%)`).join(', ')
    parts.push(`${names} will be imputed using median or mode based on the column dtype.`)
  }
  return parts.join(' ')
}

function genCorrExplanation(columns, matrix) {
  if (!columns || !matrix) return null
  const pairs = []
  for (let i = 0; i < columns.length; i++) {
    for (let j = i + 1; j < columns.length; j++) {
      const v = matrix[i]?.[j]
      if (v != null) pairs.push({ c1: columns[i], c2: columns[j], v })
    }
  }
  pairs.sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
  if (pairs.length === 0) return 'No significant correlations detected between numeric features.'
  const top = pairs.slice(0, 3).map(({ c1, c2, v }) =>
    `${c1}–${c2} (${v > 0 ? '+' : ''}${v.toFixed(2)})`).join(', ')
  const veryHigh = pairs.filter(p => Math.abs(p.v) > 0.9)
  const note = veryHigh.length > 0
    ? ` ${veryHigh[0].c1} and ${veryHigh[0].c2} are highly correlated — one may be dropped to avoid multicollinearity.`
    : ' No feature pairs show correlation high enough to trigger removal.'
  return `Strongest correlations: ${top}.${note}`
}

// ── Correlation heatmap (custom SVG) ─────────────────────────────────────────

function corrColor(v) {
  const c = Math.max(-1, Math.min(1, v))
  if (c >= 0) {
    const t = 1 - c
    return `rgb(${Math.round(255 * t)},${Math.round(255 * t)},255)`
  } else {
    const t = 1 + c
    return `rgb(255,${Math.round(255 * t)},${Math.round(255 * t)})`
  }
}

function CorrelationHeatmap({ columns, matrix }) {
  if (!columns || columns.length < 2) return null
  const n = columns.length
  const cell = Math.max(52, Math.min(72, Math.floor(680 / n)))
  const labelW = 100
  const fontSize = Math.max(9, Math.min(12, cell * 0.22))

  // Short column names fit horizontally above their cell — much more legible
  // than rotated text. Only rotate when a name would actually overflow.
  const approxCharW = fontSize * 0.62
  const displayNames = columns.map(c => (c.length > 15 ? c.slice(0, 14) + '…' : c))
  const maxLabelW = Math.max(...displayNames.map(c => c.length * approxCharW))
  const fitsHorizontal = maxLabelW <= cell - 6

  const topPad = fitsHorizontal
    ? fontSize + 20
    : Math.ceil(maxLabelW * Math.SQRT1_2) + 20   // room for the 45° diagonal extent
  const svgW = n * cell + labelW + 24
  const svgH = topPad + n * cell + 8

  return (
    <div className="chart-section">
      <div style={{ overflow: 'auto' }}>
        <svg
          width={svgW}
          height={svgH}
          style={{ display: 'block', fontFamily: 'inherit' }}
        >
          {/* Column headers — horizontal when they fit, rotated 45° otherwise */}
          {displayNames.map((label, ci) => {
            const cx = labelW + ci * cell + cell / 2
            if (fitsHorizontal) {
              return (
                <text
                  key={`col-${ci}`}
                  x={cx}
                  y={topPad - 10}
                  textAnchor="middle"
                  fontSize={fontSize}
                  fill="currentColor"
                  opacity={0.7}
                >
                  {label}
                </text>
              )
            }
            // Anchored at the baseline just above the grid, rotated so the label
            // reads upward-right and stays entirely within the padded header area.
            return (
              <g key={`col-${ci}`} transform={`translate(${cx},${topPad - 6}) rotate(-45)`}>
                <text textAnchor="start" fontSize={fontSize} fill="currentColor" opacity={0.7}>
                  {label}
                </text>
              </g>
            )
          })}
          {columns.map((row, ri) => (
            <g key={`row-${ri}`} transform={`translate(0, ${topPad + ri * cell})`}>
              <text
                x={labelW - 4}
                y={cell / 2 + 4}
                textAnchor="end"
                fontSize={Math.max(9, Math.min(13, cell * 0.24))}
                fill="currentColor"
                opacity={0.65}
              >
                {row.length > 12 ? row.slice(0, 11) + '…' : row}
              </text>
              {columns.map((_, ci) => {
                const v = (matrix[ri] && matrix[ri][ci] != null) ? matrix[ri][ci] : 0
                return (
                  <g key={ci}>
                    <rect
                      x={labelW + ci * cell}
                      y={0}
                      width={cell}
                      height={cell}
                      fill={corrColor(v)}
                      stroke="rgba(0,0,0,0.08)"
                      strokeWidth={0.5}
                      rx={2}
                    />
                    {cell >= 24 && (
                      <text
                        x={labelW + ci * cell + cell / 2}
                        y={cell / 2 + 4}
                        textAnchor="middle"
                        fontSize={Math.max(7, cell * 0.28)}
                        fill={Math.abs(v) > 0.6 ? '#fff' : '#333'}
                      >
                        {v.toFixed(2)}
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          ))}
        </svg>
      </div>
    </div>
  )
}

// ── Missing values bar chart ──────────────────────────────────────────────────

function MissingValuesChart({ missing }) {
  const { axisTick } = useChartColors()
  const entries = Object.entries(missing)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .map(([col, pct]) => ({ col, pct }))

  if (entries.length === 0) {
    return (
      <div className="chart-section">
        <p className="chart-none">No missing values detected ✓</p>
      </div>
    )
  }

  return (
    <div className="chart-section">
      <ResponsiveContainer width="100%" height={Math.min(entries.length * 20 + 12, 200)}>
        <BarChart
          data={entries}
          layout="vertical"
          margin={{ top: 4, right: 40, left: 6, bottom: 4 }}
        >
          <XAxis type="number" domain={[0, 100]}
            tick={{ fontSize: 11, fill: axisTick }} unit="%" />
          <YAxis type="category" dataKey="col" width={90}
            tick={{ fontSize: 11, fill: axisTick }} interval={0} />
          <Tooltip
            formatter={(v) => [`${v.toFixed(1)}%`, 'Missing']}
            contentStyle={{ fontSize: 12 }}
          />
          <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
            {entries.map((e, i) => (
              <Cell key={i} fill={e.pct > 50 ? '#ef4444' : e.pct > 20 ? '#f59e0b' : '#3b82f6'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Full-width distribution chart for selected feature ────────────────────────

function DistributionFull({ bins }) {
  const { axisTick, gridLine } = useChartColors()
  if (!bins || bins.length === 0) return null
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={bins} margin={{ top: 8, right: 16, left: 0, bottom: 28 }}>
        <CartesianGrid vertical={false} stroke={gridLine} strokeDasharray="3 3" />
        <XAxis dataKey="x" tick={{ fontSize: 11, fill: axisTick }} interval={0} angle={-35} textAnchor="end" />
        <YAxis tick={{ fontSize: 11, fill: axisTick }} width={40} />
        <Tooltip
          formatter={(v) => [v, 'Count']}
          contentStyle={{ fontSize: 12 }}
        />
        <Bar dataKey="count" fill="#f97316" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Distributions tab with feature pill selector ──────────────────────────────

function DistributionsTab({ distributions, distribution_shapes }) {
  const cols = Object.keys(distributions || {})
  const [selected, setSelected] = useState(cols[0] ?? null)

  if (cols.length === 0) {
    return (
      <div className="chart-section">
        <p className="chart-none">No distribution data available.</p>
      </div>
    )
  }

  const bins        = distributions[selected] ?? []
  const shape       = distribution_shapes?.[selected]
  const explanation = genDistExplanation(selected, shape)

  return (
    <div className="chart-section">
      {/* Feature pills */}
      <div className="feature-pills">
        {cols.map(col => (
          <button
            key={col}
            className={`feature-pill${selected === col ? ' active' : ''}`}
            onClick={() => setSelected(col)}
          >
            {col}
          </button>
        ))}
      </div>

      {/* Full-width chart */}
      {selected && <DistributionFull bins={bins} />}

      {/* Per-feature explanation */}
      <InsightBlock text={explanation} />
    </div>
  )
}

// ── Main ChartPanel ───────────────────────────────────────────────────────────

export default function ChartPanel({ data }) {
  const [activeTab, setActiveTab] = useState(null)

  if (!data) return null

  const { missing_values, correlation, distributions, distribution_shapes } = data

  const hasCorr = correlation?.columns?.length >= 2
  const hasDist = distributions && Object.keys(distributions).length > 0

  const defaultTab = hasDist ? 'dist' : 'missing'
  const currentTab = activeTab ?? defaultTab

  const tabs = [
    { id: 'dist',    label: 'Distributions'  },
    { id: 'missing', label: 'Missing Values' },
    { id: 'corr',    label: 'Correlation',   skip: !hasCorr },
  ].filter(t => !t.skip)

  return (
    <div className="chart-panel">
      <div className="chart-tabs">
        {tabs.map(t => (
          <button
            key={t.id}
            className={`chart-tab${currentTab === t.id ? ' active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {currentTab === 'dist' && (
        <DistributionsTab
          distributions={distributions ?? {}}
          distribution_shapes={distribution_shapes ?? {}}
        />
      )}

      {currentTab === 'missing' && (
        <>
          <MissingValuesChart missing={missing_values ?? {}} />
          <InsightBlock text={genMissingExplanation(missing_values)} />
        </>
      )}

      {currentTab === 'corr' && hasCorr && (
        <>
          <CorrelationHeatmap columns={correlation.columns} matrix={correlation.matrix} />
          <InsightBlock text={genCorrExplanation(correlation.columns, correlation.matrix)} />
        </>
      )}
    </div>
  )
}
