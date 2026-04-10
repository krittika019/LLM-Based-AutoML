import React, { useState, useEffect } from 'react'

/**
 * Very minimal Python syntax highlighter.
 * Returns an array of React spans for a single line of code.
 */
function highlightLine(text) {
  // Patterns tested in order — first match wins
  const patterns = [
    // Comment
    { re: /^(#.*)$/, cls: 'tok-cmt' },
    // Decorator
    { re: /^(@\w+)/, cls: 'tok-dec' },
  ]

  // For full-line matches
  for (const { re, cls } of patterns) {
    const m = re.exec(text)
    if (m) {
      return [<span key="0" className={cls}>{text}</span>]
    }
  }

  // Token-by-token split for strings, keywords, numbers, function calls
  const KEYWORDS = /\b(def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|in|not|and|or|is|lambda|pass|break|continue|yield|raise|del|global|nonlocal|True|False|None|async|await)\b/g
  const STRING_RE = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g
  const NUMBER_RE = /\b(\d+\.?\d*)\b/g
  const FUNC_RE   = /\b([a-zA-Z_]\w*)\s*(?=\()/g

  // Tokenize by splitting on string boundaries first
  const parts = []
  let last = 0
  const allMatches = []

  let m
  const str = new RegExp(STRING_RE.source, 'g')
  while ((m = str.exec(text)) !== null) {
    allMatches.push({ start: m.index, end: m.index + m[0].length, cls: 'tok-str', val: m[0] })
  }
  allMatches.sort((a, b) => a.start - b.start)

  for (const token of allMatches) {
    if (token.start < last) continue
    const before = text.slice(last, token.start)
    if (before) parts.push({ val: before, cls: null })
    parts.push({ val: token.val, cls: token.cls })
    last = token.end
  }
  if (last < text.length) parts.push({ val: text.slice(last), cls: null })

  // For non-string parts, apply keyword / number / fn highlighting
  const result = []
  let key = 0
  for (const part of parts) {
    if (part.cls) {
      result.push(<span key={key++} className={part.cls}>{part.val}</span>)
      continue
    }
    // Sub-tokenize
    const sub = part.val
    const subParts = []
    let subLast = 0
    const subMatches = []

    let sm
    const kw = new RegExp(KEYWORDS.source, 'g')
    while ((sm = kw.exec(sub)) !== null) {
      subMatches.push({ start: sm.index, end: sm.index + sm[0].length, cls: 'tok-kw', val: sm[0] })
    }
    const fn = new RegExp(FUNC_RE.source, 'g')
    while ((sm = fn.exec(sub)) !== null) {
      subMatches.push({ start: sm.index, end: sm.index + sm[0].length, cls: 'tok-fn', val: sm[0] })
    }
    const num = new RegExp(NUMBER_RE.source, 'g')
    while ((sm = num.exec(sub)) !== null) {
      subMatches.push({ start: sm.index, end: sm.index + sm[0].length, cls: 'tok-num', val: sm[0] })
    }
    subMatches.sort((a, b) => a.start - b.start)

    let covered = new Array(sub.length).fill(false)
    const sorted = []
    for (const tok of subMatches) {
      if (covered.slice(tok.start, tok.end).some(Boolean)) continue
      for (let i = tok.start; i < tok.end; i++) covered[i] = true
      sorted.push(tok)
    }
    sorted.sort((a, b) => a.start - b.start)

    subLast = 0
    for (const tok of sorted) {
      if (tok.start > subLast) {
        subParts.push({ val: sub.slice(subLast, tok.start), cls: 'code-text' })
      }
      subParts.push({ val: tok.val, cls: tok.cls })
      subLast = tok.end
    }
    if (subLast < sub.length) {
      subParts.push({ val: sub.slice(subLast), cls: 'code-text' })
    }

    for (const sp of subParts) {
      result.push(<span key={key++} className={sp.cls}>{sp.val}</span>)
    }
  }
  return result
}

export default function ScriptPanel({ open, runId, onClose }) {
  const [code, setCode] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fetchError, setFetchError] = useState('')
  const [copied, setCopied] = useState(false)

  // Fetch script when panel opens
  useEffect(() => {
    if (!open || !runId) return
    if (code !== null) return    // already loaded for this run

    setLoading(true)
    setFetchError('')

    fetch(`/download/script/${runId}`)
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.text()
      })
      .then(text => setCode(text))
      .catch(err => setFetchError(err.message ?? 'Failed to fetch script'))
      .finally(() => setLoading(false))
  }, [open, runId, code])

  // Reset when runId changes (new pipeline run)
  useEffect(() => {
    setCode(null)
    setFetchError('')
  }, [runId])

  const handleCopy = () => {
    if (!code) return
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }

  const lines = code ? code.split('\n') : []

  return (
    <div className={`script-panel${open ? ' open' : ''}`} aria-label="Pipeline script panel">
      {/* Header */}
      <div className="script-panel-header">
        <div className="script-panel-title-row">
          <span className="script-panel-title">Pipeline Script</span>
          <div className="script-panel-actions">
            <button
              className={`icon-btn${copied ? ' copied' : ''}`}
              onClick={handleCopy}
              title="Copy to clipboard"
              aria-label="Copy script"
            >
              {copied ? '✓' : '⎘'}
            </button>
            {runId && (
              <a
                href={`/download/script/${runId}`}
                download="automl_pipeline.py"
                className="icon-btn"
                title="Download .py file"
                aria-label="Download script"
              >
                ⬇
              </a>
            )}
            <button
              className="icon-btn"
              onClick={onClose}
              title="Close panel"
              aria-label="Close script panel"
            >
              ×
            </button>
          </div>
        </div>
        <div className="script-panel-subtitle">
          Runs independently — no AutoML system required
        </div>
      </div>

      {/* Code viewer */}
      <div className="script-code-wrap">
        <div className="script-code-inner">
          {loading && (
            <div style={{ padding: '20px 16px', fontSize: '0.82rem', color: 'var(--text-dim)' }}>
              Loading script…
            </div>
          )}
          {fetchError && (
            <div style={{ padding: '20px 16px', fontSize: '0.82rem', color: '#ef4444' }}>
              ⚠ {fetchError}
            </div>
          )}
          {!loading && !fetchError && lines.map((line, i) => (
            <div className="code-line" key={i}>
              <span className="line-num">{i + 1}</span>
              <span>{highlightLine(line)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="script-panel-footer">
        {runId ? (
          <a
            href={`/download/script/${runId}`}
            download="automl_pipeline.py"
            className="download-full-btn"
          >
            ⬇ Download .py file
          </a>
        ) : (
          <div className="download-full-btn" style={{ opacity: 0.4, cursor: 'default' }}>
            ⬇ Download .py file
          </div>
        )}
      </div>
    </div>
  )
}
