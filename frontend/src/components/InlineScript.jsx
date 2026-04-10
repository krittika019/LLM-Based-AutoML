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

// ── InlineScript component ────────────────────────────────────────────────────

export default function InlineScript({ runId, visible }) {
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
      .catch(err => { setFetchError(err.message); setLoading(false) })
  }, [visible, runId, code])

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
        <span className="inline-script-title">pipeline.py</span>
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
        </div>
      </div>

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
