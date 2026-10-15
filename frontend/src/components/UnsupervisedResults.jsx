

import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, Cell, CartesianGrid,
  ScatterChart, Scatter,
} from 'recharts'
import { CLUSTER_COLORS } from '../constants.js'

// ── Plain-language metric interpretation (deterministic — no LLM call) ───────
// Thresholds follow the conventional ranges for these indices (Rousseeuw 1987
// for silhouette; lower-is-better, no fixed scale, for Davies-Bouldin).

function interpretSilhouette(score) {
  if (score == null) return null
  if (score >= 0.7) return 'Strong, clearly separated groups.'
  if (score >= 0.5) return 'Reasonably well-separated groups, with some overlap.'
  if (score >= 0.25) return 'Moderate separation — the groups overlap somewhat.'
  if (score >= 0) return 'Weak separation — the groups overlap substantially.'
  return 'Points are likely assigned to the wrong group — the split found here is not reliable.'
}

function interpretDaviesBouldin(score) {
  if (score == null) return null
  if (score <= 0.5) return 'Groups are compact and well-separated (lower is better).'
  if (score <= 1) return 'Groups are reasonably distinct (lower is better).'
  if (score <= 2) return 'Groups have noticeable overlap (lower is better).'
  return 'Groups overlap heavily — treat this split as a rough starting point, not a firm segmentation.'
}

function silhouetteLabel(score) {
  if (score == null) return { text: 'Unknown', color: '#94a3b8' }
  if (score >= 0.7) return { text: 'Strong', color: '#10b981' }
  if (score >= 0.5) return { text: 'Good', color: '#10b981' }
  if (score >= 0.25) return { text: 'Moderate', color: '#f59e0b' }
  if (score >= 0) return { text: 'Weak', color: '#ef4444' }
  return { text: 'Poor', color: '#ef4444' }
}

// ── Executive summary (deterministic, from the computed cluster_sizes only) ──

function ExecutiveSummary({ results, colorFor }) {
  const sizes = results.cluster_sizes || {}
  const entries = Object.entries(sizes)
    .filter(([k]) => k !== '-1')
    .map(([k, v]) => [Number(k), v])
    .sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return null

  const total = Object.values(sizes).reduce((a, b) => a + b, 0)
  const [largestId, largestN] = entries[0]
  const [smallestId, smallestN] = entries[entries.length - 1]
  const nClusters = entries.length
  const noise = sizes['-1']

  return (
    <div className="exec-summary">
      <p>
        <strong>{total.toLocaleString()}</strong> record{total === 1 ? '' : 's'} were grouped into{' '}
        <strong>{nClusters}</strong> segment{nClusters === 1 ? '' : 's'}
        {results.excluded_features?.length
          ? <> using {results.feature_columns.length} feature{results.feature_columns.length === 1 ? '' : 's'} ({results.feature_columns.join(', ')}) — {results.excluded_features.join(', ')} {results.excluded_features.length === 1 ? 'was' : 'were'} excluded as identifier column{results.excluded_features.length === 1 ? '' : 's'}</>
          : null}.
      </p>
      {nClusters > 1 && (
        <p>
          The largest segment (<span className="exec-summary-dot" style={{ background: colorFor(largestId) }} />
          Cluster {largestId}) has <strong>{largestN.toLocaleString()}</strong> records
          ({(largestN / total * 100).toFixed(1)}%); the smallest (
          <span className="exec-summary-dot" style={{ background: colorFor(smallestId) }} />
          Cluster {smallestId}) has <strong>{smallestN.toLocaleString()}</strong> ({(smallestN / total * 100).toFixed(1)}%).
        </p>
      )}
      {noise ? (
        <p className="analytics-caption">
          {noise.toLocaleString()} record{noise === 1 ? '' : 's'} could not be confidently assigned to any group (shown as "noise").
        </p>
      ) : null}
    </div>
  )
}

// ── Key results cards ─────────────────────────────────────────────────────────

function KeyResultsCards({ results }) {
  const sizes = results.cluster_sizes || {}
  const entries = Object.entries(sizes)
    .filter(([k]) => k !== '-1')
    .map(([k, v]) => [Number(k), v])
    .sort((a, b) => b[1] - a[1])
  if (entries.length === 0) return null

  const total = Object.values(sizes).reduce((a, b) => a + b, 0)
  const [, largestN] = entries[0]
  const [, smallestN] = entries[entries.length - 1]
  const sep = silhouetteLabel(results.silhouette_score)

  const cards = [
    { label: 'Total Records', value: total.toLocaleString() },
    { label: 'Segments Found', value: entries.length },
    { label: 'Largest Segment', value: `${largestN.toLocaleString()}`, sub: `${(largestN / total * 100).toFixed(1)}%` },
    { label: 'Smallest Segment', value: `${smallestN.toLocaleString()}`, sub: `${(smallestN / total * 100).toFixed(1)}%` },
    { label: 'Separation', value: sep.text, color: sep.color },
  ]

  return (
    <div className="key-results-grid">
      {cards.map(c => (
        <div className="key-results-card" key={c.label}>
          <div className="key-results-label">{c.label}</div>
          <div className="key-results-value" style={{ color: c.color }}>
            {c.value}{c.sub && <span className="key-results-sub"> ({c.sub})</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Segment size chart (sorted, horizontal) ───────────────────────────────────

function SegmentSizeChart({ sizes, colorFor }) {
  if (!sizes) return null
  const total = Object.values(sizes).reduce((a, b) => a + b, 0)
  const entries = Object.entries(sizes)
    .map(([k, v]) => ({ cluster: k, count: v, pct: total > 0 ? +(v / total * 100).toFixed(1) : 0 }))
    .sort((a, b) => b.count - a.count)
  if (entries.length < 2) return null

  return (
    <div className="chart-section" style={{ padding: '0 0 14px' }}>
      <div className="chart-title">Segment Sizes</div>
      <ResponsiveContainer width="100%" height={Math.max(100, entries.length * 30)}>
        <BarChart data={entries} layout="vertical" margin={{ top: 4, right: 44, left: 6, bottom: 4 }}>
          <CartesianGrid horizontal={false} strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="cluster" width={70}
            tickFormatter={(v) => v === '-1' ? 'Noise' : `Cluster ${v}`}
            tick={{ fontSize: 11 }} interval={0} />
          <Tooltip formatter={(v, k) => [k === 'count' ? v : `${v}%`, k === 'count' ? 'Records' : 'Share']} contentStyle={{ fontSize: 12 }} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {entries.map((e) => (
              <Cell key={e.cluster} fill={e.cluster === '-1' ? '#cbd5e1' : colorFor(Number(e.cluster))} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Metric chip row ──────────────────────────────────────────────────────────

function MetricChips({ items }) {
  return (
    <div className="metrics-chip-grid" style={{ marginBottom: 14 }}>
      {items.filter(it => it.value != null).map(it => (
        <div className="metric-chip" key={it.label}>
          <span className="metric-chip-label">{it.label}</span>
          <span className="metric-chip-value" style={{ color: it.color || '#6366f1' }}>{it.value}</span>
        </div>
      ))}
    </div>
  )
}

// ── Elbow chart (inertia per k, chosen k highlighted) ────────────────────────

function ElbowChart({ elbowData, bestK }) {
  if (!elbowData?.length) return null
  return (
    <div className="chart-section" style={{ padding: '0 0 14px' }}>
      <div className="chart-title">Elbow — Inertia by k</div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={elbowData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid vertical={false} stroke="rgba(99,102,241,0.08)" strokeDasharray="3 3" />
          <XAxis dataKey="k" tick={{ fontSize: 11 }} label={{ value: 'k', position: 'insideBottom', offset: -2, fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} width={50} />
          <Tooltip
            formatter={(v, key) => [v, key === 'inertia' ? 'Inertia' : key]}
            labelFormatter={(k) => `k = ${k}`}
            contentStyle={{ fontSize: 12 }}
          />
          <Bar dataKey="inertia" radius={[3, 3, 0, 0]}>
            {elbowData.map((e, i) => (
              <Cell key={i} fill={e.k === bestK ? '#6366f1' : '#cbd5e1'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="analytics-caption">k = {bestK} chosen by highest silhouette score (highlighted).</p>
    </div>
  )
}

// ── Cluster scatter (2D projection, colored by cluster) ──────────────────────

function ClusterScatter({ points }) {
  if (!points?.length) return <p className="chart-none">No projection data available.</p>
  const byCluster = {}
  for (const p of points) {
    const c = p.cluster ?? 0
    ;(byCluster[c] ??= []).push(p)
  }
  const clusterIds = Object.keys(byCluster).map(Number).sort((a, b) => a - b)

  return (
    <div>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
          <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 11 }} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {clusterIds.map(cid => (
            <Scatter
              key={cid}
              name={`Cluster ${cid}`}
              data={byCluster[cid]}
              fill={CLUSTER_COLORS[cid % CLUSTER_COLORS.length]}
              fillOpacity={0.65}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <p className="analytics-caption">2D PCA projection of the clustered features — axes have no inherent unit.</p>
    </div>
  )
}

// ── Cluster profile table ─────────────────────────────────────────────────────

function ZScorePill({ feature }) {
  const z = feature.z_score
  const color = z >= 0 ? '#ef4444' : '#3b82f6'
  return (
    <span
      title={`mean=${feature.mean}, z=${z}`}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 4, margin: '2px 4px 2px 0',
        padding: '2px 8px', borderRadius: 999, fontSize: 11, fontWeight: 600,
        background: color + '18', color, border: `1px solid ${color}40`,
      }}
    >
      {feature.name} {z >= 0 ? '+' : ''}{z.toFixed(2)}
    </span>
  )
}

function ClusterProfileTable({ profiles, colorFor }) {
  if (!profiles?.length) return <p className="chart-none">No cluster profile data available.</p>
  return (
    <div className="model-table-scroll">
      <table className="model-table">
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Size</th>
            <th>Distinguishing features (mean, z-score vs. overall)</th>
          </tr>
        </thead>
        <tbody>
          {profiles.map(p => (
            <tr key={p.cluster}>
              <td>
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 700,
                }}>
                  <span style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: colorFor(p.cluster), display: 'inline-block',
                  }} />
                  {p.cluster}
                </span>
              </td>
              <td>{p.size} <span style={{ color: 'var(--text-muted)' }}>({p.pct}%)</span></td>
              <td style={{ whiteSpace: 'normal' }}>
                {(p.top_features || []).map(f => <ZScorePill key={f.name} feature={f} />)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="analytics-caption">
        <span style={{ color: '#ef4444' }}>Red</span> = above the overall average, <span style={{ color: '#3b82f6' }}>blue</span> = below. Larger |z| = more distinctive to that cluster.
      </p>
    </div>
  )
}

// ── PCA-specific views ────────────────────────────────────────────────────────

function VarianceChart({ evr, cumulative }) {
  if (!evr?.length) return <p className="chart-none">No explained-variance data available.</p>
  const data = evr.map((v, i) => ({ component: `PC${i + 1}`, pct: +(v * 100).toFixed(1) }))
  return (
    <div className="chart-section" style={{ padding: '0 0 14px' }}>
      <div className="chart-title">Explained Variance per Component</div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid vertical={false} stroke="rgba(99,102,241,0.08)" strokeDasharray="3 3" />
          <XAxis dataKey="component" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} width={44} unit="%" />
          <Tooltip formatter={(v) => [`${v}%`, 'Variance']} contentStyle={{ fontSize: 12 }} />
          <Bar dataKey="pct" fill="#6366f1" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="analytics-caption">
        Cumulative: the first {evr.length} components capture{' '}
        <strong>{((cumulative?.[cumulative.length - 1] ?? 0) * 100).toFixed(1)}%</strong> of the original variance.
      </p>
    </div>
  )
}

function PlainScatter({ points }) {
  if (!points?.length) return <p className="chart-none">No projection data available.</p>
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
        <XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 11 }} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 12 }} />
        <Scatter data={points} fill="#6366f1" fillOpacity={0.55} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// ── Page-level component ──────────────────────────────────────────────────────

export default function UnsupervisedResults({ results }) {
  if (!results || !results.status) {
    return <p className="chart-none">No unsupervised results available.</p>
  }

  if (results.status !== 'ok') {
    return (
      <p className="chart-none">
        This run did not complete: {results.reason || 'unknown error'}.
      </p>
    )
  }

  const colorFor = (cid) => CLUSTER_COLORS[cid % CLUSTER_COLORS.length]

  if (results.algorithm === 'kmeans' || results.algorithm === 'dbscan' || results.algorithm === 'hdbscan'
      || results.algorithm === 'gmm' || results.algorithm === 'agglomerative') {
    const silNote = interpretSilhouette(results.silhouette_score)
    const dbNote = interpretDaviesBouldin(results.davies_bouldin_score)
    return (
      <div>
        <ExecutiveSummary results={results} colorFor={colorFor} />

        <KeyResultsCards results={results} />

        <SegmentSizeChart sizes={results.cluster_sizes} colorFor={colorFor} />

        <div className="chart-title" style={{ marginTop: 4 }}>Cluster Map</div>
        <p className="analytics-caption" style={{ marginTop: -4, marginBottom: 8 }}>
          Each point is one record. Records with similar characteristics sit closer together; color shows which group they were assigned to.
        </p>
        <ClusterScatter points={results.projection_2d} />

        <div className="chart-title" style={{ marginTop: 18 }}>What Distinguishes Each Group</div>
        <ClusterProfileTable profiles={results.cluster_profiles} colorFor={colorFor} />

        <div className="chart-title" style={{ marginTop: 18 }}>How Reliable Is This Grouping?</div>
        <div className="chart-section" style={{ padding: '0 0 14px' }}>
          {silNote && (
            <p style={{ margin: '0 0 6px' }}>
              <strong>Silhouette score {results.silhouette_score}</strong> — {silNote}
            </p>
          )}
          {dbNote && (
            <p style={{ margin: 0 }}>
              <strong>Davies-Bouldin index {results.davies_bouldin_score}</strong> — {dbNote}
            </p>
          )}
          {!silNote && !dbNote && <p className="chart-none">Quality metrics could not be computed for this run.</p>}
        </div>

        <details className="analytics-details">
          <summary>Technical details</summary>
          <MetricChips items={[
            { label: 'Algorithm', value: results.algorithm },
            { label: 'k / clusters found', value: results.best_k ?? results.n_clusters_found },
            { label: 'Silhouette', value: results.silhouette_score, color: '#10b981' },
            { label: 'Davies-Bouldin', value: results.davies_bouldin_score },
            { label: 'Inertia', value: results.inertia },
            { label: 'Noise points', value: results.n_noise },
            { label: 'Rows', value: results.n_rows },
          ]} />
          {results.excluded_features?.length > 0 && (
            <p className="analytics-caption">
              Excluded as identifier column(s), not used for clustering: {results.excluded_features.join(', ')}.
            </p>
          )}
          <ElbowChart elbowData={results.elbow_data} bestK={results.best_k} />
        </details>
      </div>
    )
  }

  // dimensionality_reduction: pca / tsne / umap
  const isLinear = results.algorithm === 'pca'
  return (
    <div>
      <p>
        <strong>{(results.n_rows ?? 0).toLocaleString()}</strong> records were reduced from{' '}
        <strong>{results.feature_columns?.length ?? '?'}</strong> feature{results.feature_columns?.length === 1 ? '' : 's'} down to{' '}
        <strong>{results.n_components}</strong> dimension{results.n_components === 1 ? '' : 's'} for visualization
        {results.excluded_features?.length
          ? <> ({results.excluded_features.join(', ')} excluded as identifier column{results.excluded_features.length === 1 ? '' : 's'})</>
          : null}.
      </p>
      {isLinear
        ? <VarianceChart evr={results.explained_variance_ratio} cumulative={results.cumulative_variance} />
        : <p className="analytics-caption">{results.note}</p>}
      <div className="chart-title">2D Projection</div>
      <PlainScatter points={results.projection_2d} />

      <details className="analytics-details">
        <summary>Technical details</summary>
        <MetricChips items={[
          { label: 'Algorithm', value: results.algorithm },
          { label: 'Components', value: results.n_components },
          { label: 'Rows', value: results.n_rows },
        ]} />
      </details>
    </div>
  )
}
