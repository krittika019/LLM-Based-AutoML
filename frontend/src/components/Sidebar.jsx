import React, { useRef, useState } from 'react'

export default function Sidebar({ open, onClose, fileInfo, onFileInfo }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const ALLOWED = ['.csv', '.tsv', '.parquet', '.xlsx', '.xls']

  async function uploadFile(file) {
    if (!file) return
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!ALLOWED.includes(ext)) {
      setError(`Unsupported format "${ext}". Use ${ALLOWED.join(', ')}.`)
      return
    }
    setUploading(true)
    setError('')
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch('/upload', { method: 'POST', body: form })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `Upload failed (${res.status})`)
      }
      const info = await res.json()
      onFileInfo(info)
    } catch (err) {
      setError(err.message ?? 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) uploadFile(file)
  }

  function handleChoose(e) {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
    e.target.value = ''
  }

  return (
    <aside className={`sidebar${open ? ' sidebar-open' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-header-title">
          <span className="sidebar-header-icon">📂</span>
          Dataset
        </div>
        <button className="sidebar-close" onClick={onClose} aria-label="Close sidebar">
          ×
        </button>
      </div>

      <div className="sidebar-body">
        {/* Hidden file input — used by both drop zone and re-upload button */}
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.tsv,.parquet,.xlsx,.xls"
          style={{ display: 'none' }}
          onChange={handleChoose}
        />

        {/* Drop zone — only shown when no file is loaded */}
        {!fileInfo && (
          <div
            className={`drop-zone${dragging ? ' drag-over' : ''}${uploading ? ' uploading' : ''}`}
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
          >
            {uploading ? (
              <span className="drop-zone-text">Uploading…</span>
            ) : (
              <>
                <span className="drop-zone-icon">📁</span>
                <span className="drop-zone-text">
                  Drop CSV / TSV / Parquet / Excel<br />
                  <em>or click to choose</em>
                </span>
              </>
            )}
          </div>
        )}

        {/* Re-upload button — shown when a file is already loaded */}
        {fileInfo && (
          <button
            className="upload-btn"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
          >
            ⬆ {uploading ? 'Uploading…' : 'Upload dataset'}
          </button>
        )}

        {error && <p className="sidebar-error">⚠️ {error}</p>}

        {/* File info card */}
        {fileInfo && (
          <div className="file-info-card">
            <div className="file-info-header">
              <div className="file-info-name-wrap">
                <span className="file-name">{fileInfo.filename}</span>
              </div>
              <span className="file-size-badge">{fileInfo.size_kb.toFixed(1)} KB</span>
              <button
                className="remove-file-btn"
                onClick={() => onFileInfo(null)}
                title="Remove file"
                aria-label="Remove file"
              >
                ×
              </button>
            </div>

            {/* Prominent rows × cols stat */}
            <div className="file-stats">
              <strong>{fileInfo.rows.toLocaleString()}</strong>
              <span className="file-stats-dim"> rows × </span>
              <strong>{fileInfo.cols}</strong>
              <span className="file-stats-dim"> columns</span>
              {'  '}
              <span className="loaded-badge">✓ Loaded</span>
            </div>

            {/* Column chips */}
            <div className="columns-label">Columns</div>
            <div className="column-chips">
              {fileInfo.columns.map(c => (
                <span key={c} className="chip">{c}</span>
              ))}
              {fileInfo.extra_cols > 0 && (
                <span className="chip chip-muted">+{fileInfo.extra_cols} more</span>
              )}
            </div>

            {/* Preview table */}
            {fileInfo.preview && (
              <>
                <div className="preview-label">
                  <span>Preview</span>
                  <span style={{ opacity: 0.5, fontWeight: 400 }}>first 5 rows</span>
                </div>
                <div className="preview-scroll">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        {fileInfo.preview.columns.map(c => (
                          <th key={c}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {fileInfo.preview.rows.map((row, ri) => (
                        <tr key={ri}>
                          {row.map((cell, ci) => (
                            <td key={ci}>{String(cell ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </aside>
  )
}
