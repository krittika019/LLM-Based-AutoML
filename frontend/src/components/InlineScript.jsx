import React, { useState, useEffect } from 'react'

// ── Minimal Python syntax highlighter (shared with ScriptPanel) ───────────────

function highlightLine(text) {
  const patterns = [
    { re: /^(#.*)$/, cls: 'tok-cmt' },
    { re: /^(@\w+)/, cls: 'tok-dec' },
  ]
  for (const { re, cls } of patterns) {
    const m = re.exec(text)
    if (m) return [<span key="0" className={cls}>{text}</span>]
  }

  const KEYWORDS = /\b(def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|in|not|and|or|is|lambda|pass|break|continue|yield|raise|del|global|nonlocal|True|False|None|async|await)\b/g
  const STRING_RE = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g
  const NUMBER_RE = /\b(\d+\.?\d*)\b/g
  const FUNC_RE   = /\b([a-zA-Z_]\w*)\s*(?=\()/g

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

  const result = []
  let key = 0
  for (const part of parts) {
    if (part.cls) {
      result.push(<span key={key++} className={part.cls}>{part.val}</span>)
      continue
    }
    const sub = part.val
    const subMatches = []
    let sm
    const kw = new RegExp(KEYWORDS.source, 'g')
    while ((sm = kw.exec(sub)) !== null)
      subMatches.push({ start: sm.index, end: sm.index + sm[0].length, cls: 'tok-kw', val: sm[0] })
    const fn = new RegExp(FUNC_RE.source, 'g')
    while ((sm = fn.exec(sub)) !== null)
      subMatches.push({ start: sm.index, end: sm.index + sm[0].length, cls: 'tok-fn', val: sm[0] })
    const num = new RegExp(NUMBER_RE.source, 'g')
    while ((sm = num.exec(sub)) !== null)
      subMatches.push({ start: sm.index, end: sm.index + sm[0].length, cls: 'tok-num', val: sm[0] })
    subMatches.sort((a, b) => a.start - b.start)

    const covered = new Array(sub.length).fill(false)
    const sorted = []
    for (const tok of subMatches) {
      if (covered.slice(tok.start, tok.end).some(Boolean)) continue
      for (let i = tok.start; i < tok.end; i++) covered[i] = true
      sorted.push(tok)
    }
    sorted.sort((a, b) => a.start - b.start)

    let subLast = 0
    const subParts = []
    for (const tok of sorted) {
      if (tok.start > subLast) subParts.push({ val: sub.slice(subLast, tok.start), cls: 'code-text' })
      subParts.push({ val: tok.val, cls: tok.cls })
      subLast = tok.end
    }
    if (subLast < sub.length) subParts.push({ val: sub.slice(subLast), cls: 'code-text' })
    for (const sp of subParts) result.push(<span key={key++} className={sp.cls}>{sp.val}</span>)
  }
  return result
}

// ── Sandbox verification badge ────────────────────────────────────────────────

const VERIFY_LABELS = {
  verified: '✓ Verified',
  score_mismatch: '⚠ Score differs',
  failed: '✕ Not runnable',
  skipped: 'Not verified',
}

function verifyTone(status) {
  if (status === 'verified') return 'ok'
  if (status === 'skipped') return 'muted'
  return status === 'failed' ? 'bad' : 'warn'
}

function VerifyBadge({ validation }) {
  if (!validation?.status) return null
  return (
    <span className={`script-verify-badge ${verifyTone(validation.status)}`} title={validation.message}>
      {VERIFY_LABELS[validation.status] ?? validation.status}
    </span>
  )
}

// ── InlineScript component ────────────────────────────────────────────────────

export default function InlineScript({ runId, visible, validation, fallbackCode }) {
  const [code, setCode]           = useState(null)
  const [loading, setLoading]     = useState(false)
  const [fetchError, setFetchError] = useState('')
  const [copied, setCopied]       = useState(false)

  useEffect(() => {
    if (!visible || !runId) return
    if (code !== null) return
    setLoading(true)
    setFetchError('')
    fetch(`/download/script/${runId}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.text()
      })
      .then(text => { setCode(text); setLoading(false) })
      .catch(err => {
        // The final event already carries the script — show it if the download failed
        // (e.g. backend restarted and lost its run registry).
        if (fallbackCode) setCode(fallbackCode)
        else setFetchError(err.message)
        setLoading(false)
      })
  }, [visible, runId, code, fallbackCode])

  // Reset when runId changes
  useEffect(() => { setCode(null); setFetchError('') }, [runId])

  if (!visible) return null

  const lines = code ? code.split('\n') : []

  const handleCopy = () => {
    if (!code) return
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    })
  }

  return (
    <div className="inline-script-wrap">
      {/* Header bar */}
      <div className="inline-script-header">
        <span className="inline-script-title">
          pipeline.py
          <VerifyBadge validation={validation} />
        </span>
        <div className="inline-script-actions">
          <button
            className={`icon-btn${copied ? ' copied' : ''}`}
            onClick={handleCopy}
            title="Copy to clipboard"
            disabled={!code}
          >
            {copied ? '✓' : '⎘'}
          </button>
          {runId && (
            <a
              href={`/download/script/${runId}`}
              download="automl_pipeline.py"
              className="icon-btn"
              title="Download .py"
            >
              ⬇
            </a>
          )}
          {runId && ['train', 'test'].map(part => (
            <a
              key={part}
              href={`/download/split/${runId}/${part}`}
              download={`${part}.csv`}
              className="split-download"
              title={`Download ${part}.csv — the exact split the script expects`}
            >
              {part}.csv
            </a>
          ))}
        </div>
      </div>

      {validation?.message && (
        <div className={`script-verify-note ${verifyTone(validation.status)}`}>
          {validation.message}
          {validation.source === 'assembled' && ' · built from the recorded pipeline steps'}
          {validation.backend && ` · ${validation.backend} sandbox`}
          {validation.attempts > 1 && ` · ${validation.attempts} attempts`}
        </div>
      )}

      {/* Code body */}
      <div className="inline-script-body">
        {loading && (
          <div className="inline-script-status">Loading script…</div>
        )}
        {fetchError && (
          <div className="inline-script-status" style={{ color: '#ef4444' }}>
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
  )
}
