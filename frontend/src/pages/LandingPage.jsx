import React, { useEffect, useRef, useState } from 'react'
import '../run/run.css'
import './landing.css'

const REPO_URL = 'https://github.com/krittika019/LLM-Based-AutoML'

const load = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v) } catch { return d } }
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* storage unavailable */ } }

const Check = () => (
  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 8.5l3 3 7-7" /></svg>
)

const Logo = ({ size = 30 }) => (
  <svg width={size} height={size} viewBox="0 0 28 28" aria-hidden="true"><rect width="28" height="28" rx="8" fill="var(--primary)" /><circle cx="9" cy="9" r="2.6" fill="var(--on-primary)" /><circle cx="19" cy="14" r="2.6" fill="var(--accent)" /><circle cx="9" cy="19" r="2.6" fill="var(--on-primary)" /><path d="M11.4 10.2L16.6 12.8M11.4 17.8L16.6 15.2" stroke="var(--on-primary)" strokeWidth="1.6" strokeLinecap="round" /></svg>
)

/** Fades a block in the first time it scrolls into view. */
function Reveal({ as: Tag = 'div', className = '', delay = 0, children, ...rest }) {
  const ref = useRef(null)
  const [shown, setShown] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') { setShown(true); return }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setShown(true); io.disconnect() } }, { threshold: 0.15 })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return <Tag ref={ref} className={`ld-reveal${shown ? ' in' : ''} ${className}`} style={{ transitionDelay: `${delay}ms` }} {...rest}>{children}</Tag>
}

// Hero: a run playing out — the prompt types in, phases tick off, the script verifies.
const PROMPT = 'Predict which patients are at risk of heart disease. Catch every real case.'
const HERO_ROWS = [
  { text: <>Target <b>HeartDisease</b> · binary classification</>, time: 'Phase 1' },
  { text: 'Cleaned and encoded · features kept only if CV improves', time: 'Phase 3' },
  { text: 'Candidates tuned and compared on held-out data', time: 'Phase 5' },
]

function HeroDemo() {
  const [typed, setTyped] = useState(0)
  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) { setTyped(PROMPT.length); return }
    let i = 0
    const t = setInterval(() => { i += 1; setTyped(i); if (i >= PROMPT.length) clearInterval(t) }, 28)
    return () => clearInterval(t)
  }, [])
  return (
    <div className="ld-demo ld-fu ld-d3" aria-label="Example run">
      <div className="ld-bubble">{PROMPT.slice(0, typed)}<span className="ld-caret" aria-hidden="true" /></div>
      <div className="ld-flow"><span className="ld-flowdot" aria-hidden="true" /></div>
      <div className="ld-steps">
        {HERO_ROWS.map((r, i) => (
          <div key={i} className={`ld-row ld-r${i + 1}`}>
            <span className={`ld-tick ld-pop ld-r${i + 1}`}><Check /></span>
            <span className="ld-row-text">{r.text}</span>
            <span className="ld-row-time mono">{r.time}</span>
          </div>
        ))}
        <div className="ld-row ld-final">
          <span className="ld-tick ld-tick-accent ld-pop ld-r4">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 5L3 8l3 3M10 5l3 3-3 3" /></svg>
          </span>
          <span className="ld-row-text" style={{ fontWeight: 500 }}>pipeline.py reproduces the reported score</span>
          <span className="rw-pill ok">Verified</span>
        </div>
      </div>
    </div>
  )
}

const FEATURES = [
  { title: 'A pipeline you can watch', text: 'Seven phases run as a live graph. Click any phase to see what it found, how long it took and which decisions it made.',
    icon: <><circle cx="12" cy="13" r="3" fill="none" stroke="var(--text)" strokeWidth="1.6" /><circle cx="24" cy="18" r="3" fill="none" stroke="var(--accent)" strokeWidth="1.6" /><circle cx="12" cy="23" r="3" fill="none" stroke="var(--text)" strokeWidth="1.6" /><path d="M14.7 14.3l6.6 2.4M14.7 21.7l6.6-2.4" stroke="var(--text)" strokeWidth="1.6" /></> },
  { title: 'Code that is verified', text: 'The exported script runs in an isolated sandbox and has to reproduce the reported score before it is marked verified.',
    icon: <path d="M12 18.5l4 4 8-9" fill="none" stroke="var(--ok)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /> },
  { title: 'Decisions with a source', text: 'Every cleaning, encoding and model choice is logged with its reason and tagged as a rule, a statistical test or the LLM.',
    icon: <path d="M11 12h14M11 18h14M11 24h9" stroke="var(--text)" strokeWidth="1.6" strokeLinecap="round" /> },
  { title: 'Features that earn their place', text: 'Proposed features are kept only when they improve the cross-validated score by more than its noise.',
    icon: <><path d="M11 25V19M17 25V13M23 25V16" stroke="var(--text)" strokeWidth="2" strokeLinecap="round" /><path d="M10 11l5 3 4-3 6 3" fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></> },
  { title: 'Explainable results', text: 'Threshold explorer, ROC curve, confusion matrix, SHAP impact and model stability in one analytics view.',
    icon: <path d="M10 13h10M10 18h15M10 23h6" stroke="var(--text)" strokeWidth="2.4" strokeLinecap="round" /> },
  { title: 'Try the model', text: 'Predict a single row in the browser, score a CSV, or call the model from code — with the same preprocessing as training.',
    icon: <path d="M19 9l-7 10h5l-1 8 7-10h-5z" fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeLinejoin="round" /> },
]

const VERIFY_STEPS = [
  'Static checks against the installed library versions',
  'Executed in an isolated sandbox with no network access',
  'One prediction per test row, in the original labels',
  'Score must match the one the run reported',
]

export default function LandingPage() {
  const [dark, setDark] = useState(() => load('rw-theme', null) ?? !!window.matchMedia?.('(prefers-color-scheme: dark)').matches)
  useEffect(() => { save('rw-theme', dark) }, [dark])
  useEffect(() => {
    document.body.style.background = dark ? '#0c0c0e' : '#f6f6f4'
    return () => { document.body.style.background = '' }
  }, [dark])

  return (
    <div className={`rw ld${dark ? ' rw-dark' : ''}`}>
      <header className="ld-wrap ld-header">
        <a href="#/" className="ld-brand"><Logo />AutoML Assistant</a>
        <nav aria-label="Main" className="ld-nav">
          <a href="#how" onClick={e => { e.preventDefault(); document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' }) }}>How it works</a>
          <a href="#features" onClick={e => { e.preventDefault(); document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' }) }}>Features</a>
          <a href="#verified" onClick={e => { e.preventDefault(); document.getElementById('verified')?.scrollIntoView({ behavior: 'smooth' }) }}>Verified code</a>
        </nav>
        <div className="rw-grow" />
        <button className="rw-icon-btn" onClick={() => setDark(d => !d)} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}>
          {dark
            ? <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><circle cx="8" cy="8" r="3" /><path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" /></svg>
            : <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M13.5 9.5A5.5 5.5 0 016.5 2.5a5.5 5.5 0 107 7z" /></svg>}
        </button>
        <a href="#/app" className="rw-btn primary">Open the app</a>
      </header>

      <section className="ld-wrap ld-hero">
        <div className="ld-hero-copy">
          <span className="ld-badge ld-fu ld-d1"><span className="ld-pulse" />AutoML that shows its work</span>
          <h1 className="ld-fu ld-d2">From raw data to a verified model, in one sentence.</h1>
          <p className="ld-lead ld-fu ld-d3">Upload a dataset and describe your goal in plain English. Watch every phase run as a live pipeline — then download code that has already been executed and checked against the score it reports.</p>
          <div className="ld-ctas ld-fu ld-d4">
            <a href="#/app" className="ld-cta">
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M5 3l8 5-8 5z" /></svg>
              Start a run
              <svg className="ld-cta-arrow" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 8h10M9 4l4 4-4 4" /></svg>
            </a>
            <a href="#how" className="ld-cta ld-cta-ghost" onClick={e => { e.preventDefault(); document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' }) }}>See how it works</a>
          </div>
          <span className="ld-fine ld-fu ld-d5">Classification, regression and clustering on CSV, Excel or Parquet.</span>
        </div>
        <HeroDemo />
      </section>

      <section aria-label="Measured results" className="ld-band">
        <div className="ld-wrap ld-stats">
          <Reveal><div className="ld-stat mono">8 / 8</div><div className="ld-stat-label">benchmark datasets where the exported script reproduced the reported score</div></Reveal>
          <Reveal delay={100}><div className="ld-stat mono">0.0000</div><div className="ld-stat-label">score gap between each run and its exported code</div></Reveal>
          <Reveal delay={200}><div className="ld-stat mono">7</div><div className="ld-stat-label">phases, each with its decisions, timing and evidence on screen</div></Reveal>
        </div>
      </section>

      <section id="how" className="ld-wrap ld-section">
        <Reveal className="ld-head">
          <span className="ld-eyebrow">How it works</span>
          <h2>Three steps. Every decision visible.</h2>
        </Reveal>
        <ol className="ld-grid3">
          {[
            ['01', 'Upload your data', 'Drop in a CSV, Excel or Parquet file. Every column is profiled before anything is decided.'],
            ['02', 'Describe your goal', 'One sentence in plain English. The target column, task type and metric are worked out from it.'],
            ['03', 'Get a verified pipeline', 'Watch seven phases run, compare models, then download code that reproduces the result — or try the model right away.'],
          ].map(([n, t, d], i) => (
            <Reveal as="li" key={n} delay={i * 100} className="ld-card ld-lift">
              <span className="mono ld-num">{n}</span><h3>{t}</h3><p>{d}</p>
            </Reveal>
          ))}
        </ol>
      </section>

      <section id="features" className="ld-band ld-band-tall">
        <div className="ld-wrap ld-section">
          <Reveal className="ld-head">
            <span className="ld-eyebrow">Features</span>
            <h2>Built for results you can trust and explain.</h2>
          </Reveal>
          <div className="ld-grid3">
            {FEATURES.map((f, i) => (
              <Reveal as="article" key={f.title} delay={(i % 3) * 100} className="ld-card ld-card-alt ld-lift">
                <svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true"><rect width="36" height="36" rx="10" fill="var(--surface-3)" />{f.icon}</svg>
                <h3>{f.title}</h3><p>{f.text}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section id="verified" className="ld-wrap ld-section ld-split">
        <Reveal className="ld-split-copy">
          <span className="ld-eyebrow">Verified code</span>
          <h2>The code you download is the model you saw.</h2>
          <p className="ld-lead" style={{ fontSize: 17 }}>The script is assembled from the steps the pipeline actually ran, then checked in four stages.</p>
          <ol className="ld-checks">
            {VERIFY_STEPS.map((s, i) => <li key={s}><span className="mono">{i + 1}</span>{s}</li>)}
          </ol>
        </Reveal>
        <Reveal delay={150} className="ld-code">
          <div className="ld-code-head"><span className="mono">pipeline.py</span><span className="rw-pill ok">Verified</span><div className="rw-grow" /><span className="rw-hint">gap 0.0000</span></div>
          <pre className="mono"><span className="c"># DecLog Entry14: kept 9 columns after feature selection</span>{'\n'}
<span className="k">KEEP_COLUMNS</span> = [<span className="s">'Age'</span>, <span className="s">'Sex'</span>, <span className="s">'ChestPainType'</span>, …]{'\n\n'}
train, test = <span className="s">apply_deep_clean</span>(train, test, ASSIGNMENTS, …){'\n'}
train, test = <span className="s">add_raw_features</span>(train, test, RAW_FEATURES){'\n\n'}
<span className="c"># DecLog Entry31: CatBoost, best test score</span>{'\n'}
model = <span className="s">CatBoostClassifier</span>(depth=<span className="k">6</span>, iterations=<span className="k">1000</span>, …){'\n'}
<span className="s">fit_model</span>(model, X_train, y_fit, FIT, <span className="s">'CatBoost'</span>)</pre>
        </Reveal>
      </section>

      <section className="ld-wrap" style={{ paddingBottom: 96 }}>
        <Reveal className="ld-final-cta">
          <div>
            <h2>Ready to run your first pipeline?</h2>
            <p>Bring a dataset and a sentence. Leave with a model, an explanation and verified code.</p>
          </div>
          <a href="#/app" className="ld-cta ld-cta-accent">Start a run</a>
        </Reveal>
      </section>

      <footer className="ld-footer">
        <div className="ld-wrap ld-footer-in">
          <span style={{ fontWeight: 600, color: 'var(--text)' }}>AutoML Assistant</span>
          <span>Research project · MIT licensed</span>
          <div className="rw-grow" />
          <a href="#how" onClick={e => { e.preventDefault(); document.getElementById('how')?.scrollIntoView({ behavior: 'smooth' }) }}>How it works</a>
          <a href="#features" onClick={e => { e.preventDefault(); document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' }) }}>Features</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </footer>
    </div>
  )
}
