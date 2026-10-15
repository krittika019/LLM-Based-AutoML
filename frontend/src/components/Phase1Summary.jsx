import React, { useState } from 'react'

function StatTile({ label, value }) {
  return (
    <div className="p1-stat-tile">
      <div className="p1-stat-value">{value}</div>
      <div className="p1-stat-label">{label}</div>
    </div>
  )
}

function ValidationRow({ check }) {
  return (
    <div className={`p1-check-row${check.passed ? '' : ' p1-check-fail'}`}>
      <span className="p1-check-icon">{check.passed ? '✅' : '⚠️'}</span>
      <span>{check.label}</span>
    </div>
  )
}

function AssumptionRow({ item }) {
  const icon = item.level === 'warning' ? '⚠️' : '💡'
  return (
    <div className={`p1-assumption-row${item.level === 'warning' ? ' p1-assumption-warning' : ''}`}>
      <span className="p1-check-icon">{icon}</span>
      <span>{item.text}</span>
    </div>
  )
}

const ROLE_BADGE = {
  target: { label: 'target', cls: 'p1-role-target' },
  feature: { label: 'feature', cls: 'p1-role-feature' },
  identifier: { label: 'identifier', cls: 'p1-role-identifier' },
  unsupported: { label: 'unsupported', cls: 'p1-role-unsupported' },
}

export default function Phase1Summary({ data }) {
  const [showColumns, setShowColumns] = useState(false)
  if (!data) return null

  const { problem, dataset, validation, assumptions, what_i_understood } = data

  return (
    <div className="p1-summary">
      {/* What I understood */}
      <div className="p1-block">
        <div className="p1-block-title">Task Understanding</div>
        <p className="p1-understood-text">{what_i_understood}</p>
        <div className="p1-problem-grid">
          <div><span className="p1-field-label">Task</span><span className="p1-field-value">{problem.task_type}</span></div>
          <div><span className="p1-field-label">Target</span><span className="p1-field-value"><code>{problem.target}</code></span></div>
          <div><span className="p1-field-label">Primary metric</span><span className="p1-field-value">{problem.primary_metric_label}</span></div>
          {problem.secondary_metrics?.length > 0 && (
            <div>
              <span className="p1-field-label">Secondary metrics</span>
              <span className="p1-field-value">{problem.secondary_metrics.map(m => m.label).join(', ')}</span>
            </div>
          )}
        </div>
      </div>

      {/* Dataset */}
      <div className="p1-block">
        <div className="p1-block-title">
          Dataset{dataset.source_filename ? <span className="p1-block-title-sub"> — {dataset.source_filename}</span> : null}
        </div>
        <div className="p1-stat-grid">
          <StatTile label="Rows" value={dataset.rows.toLocaleString()} />
          <StatTile label="Columns" value={dataset.columns} />
          <StatTile label="Features" value={dataset.feature_count} />
          <StatTile label="Numerical" value={dataset.numerical_count} />
          <StatTile label="Categorical" value={dataset.categorical_count} />
          {dataset.datetime_count > 0 && <StatTile label="Datetime" value={dataset.datetime_count} />}
          {dataset.boolean_count > 0 && <StatTile label="Boolean" value={dataset.boolean_count} />}
          <StatTile label="Missing values" value={dataset.missing_values} />
          <StatTile label="Duplicate rows" value={dataset.duplicate_rows} />
        </div>
      </div>

      {/* Data Validation */}
      <div className="p1-block">
        <div className="p1-block-title">Data Validation</div>
        <div className="p1-check-list">
          {validation.checks.map(c => <ValidationRow key={c.id} check={c} />)}
        </div>
      </div>

      {/* Assumptions & Warnings */}
      {assumptions?.length > 0 && (
        <div className="p1-block">
          <div className="p1-block-title">Assumptions &amp; Warnings</div>
          <div className="p1-check-list">
            {assumptions.map((a, i) => <AssumptionRow key={i} item={a} />)}
          </div>
        </div>
      )}

      {/* Column Understanding — collapsed by default, technical detail */}
      <div className="p1-block">
        <button className="p1-toggle-btn" onClick={() => setShowColumns(s => !s)}>
          {showColumns ? '▲ Hide' : '▼ Show'} column details ({data.columns.table.length} columns)
        </button>
        {showColumns && (
          <div className="p1-col-table-scroll">
            <table className="p1-col-table">
              <thead>
                <tr><th>Column</th><th>Type</th><th>Role</th></tr>
              </thead>
              <tbody>
                {data.columns.table.map(c => {
                  const badge = ROLE_BADGE[c.role] || ROLE_BADGE.feature
                  return (
                    <tr key={c.name}>
                      <td><code>{c.name}</code></td>
                      <td>{c.dtype}</td>
                      <td><span className={`p1-role-badge ${badge.cls}`}>{badge.label}</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
