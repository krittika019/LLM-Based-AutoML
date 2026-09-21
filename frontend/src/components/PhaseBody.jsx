import React, { useMemo } from 'react'
import { marked } from 'marked'
import ChartPanel from './ChartPanel.jsx'
import ModelTable from './ModelTable.jsx'
import Phase1Summary from './Phase1Summary.jsx'

marked.use({ gfm: true, breaks: true })

/** Simple markdown → HTML conversion for display inside a phase panel. */
function renderMarkdown(text) {
  // The content comes from our own server, so dangerouslySetInnerHTML is acceptable.
  try {
    return marked.parse(String(text ?? ''))
  } catch {
    return String(text ?? '')
  }
}

/** Renders one phase's messages/charts/model results. Used as the active tab panel. */
export default function PhaseBody({ phase }) {
  const { status, messages, charts, modelResults, phase1Summary } = phase

  // Deduplicate consecutive identical messages (node labels can repeat on retry)
  const dedupedMessages = useMemo(() => {
    const out = []
    for (const m of messages) {
      if (out.length === 0 || out[out.length - 1] !== m) out.push(m)
    }
    return out
  }, [messages])

  return (
    <div className="phase-tab-body">
      {phase1Summary && <Phase1Summary data={phase1Summary} />}

      {dedupedMessages.length === 0 && !phase1Summary && status === 'running' && (
        <p className="phase-processing">Processing…</p>
      )}

      {dedupedMessages.length === 0 && status === 'pending' && (
        <p className="phase-processing">Not started yet.</p>
      )}

      {dedupedMessages.map((msg, i) => (
        <div key={i} className="agent-msg-row">
          <div className="agent-avatar-dot" title="Agent">✦</div>
          <div
            className="phase-message"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(msg) }}
          />
        </div>
      ))}

      {charts && <ChartPanel data={charts} />}

      {modelResults && <ModelTable data={modelResults} />}
    </div>
  )
}
