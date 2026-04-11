import React, { useState, useEffect, useRef } from 'react'

// ── Scroll reveal hook ───────────────────────────────────────────────────────
function useScrollReveal() {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add('reveal-visible')
          observer.disconnect()
        }
      },
      { threshold: 0.10 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return ref
}

// ── Nav link helper ──────────────────────────────────────────────────────────
function NavLink({ href, children, onClick }) {
  return (
    <a href={href} className="lp-nav-link" onClick={onClick}>{children}</a>
  )
}

// ── Section wrapper (scroll reveal) ─────────────────────────────────────────
function Section({ id, className = '', children }) {
  const ref = useScrollReveal()
  return (
    <section id={id} className={`lp-section reveal-init ${className}`} ref={ref}>
      {children}
    </section>
  )
}

// ── How It Works card ────────────────────────────────────────────────────────
function HowCard({ num, icon, title, desc }) {
  return (
    <div className="lp-how-card">
      <div className="lp-how-num">{num}</div>
      <div className="lp-how-icon" aria-hidden>{icon}</div>
      <h3 className="lp-how-title">{title}</h3>
      <p className="lp-how-desc">{desc}</p>
    </div>
  )
}

// ── Feature card ─────────────────────────────────────────────────────────────
function FeatureCard({ icon, title, desc }) {
  return (
    <div className="lp-feature-card">
      <div className="lp-feature-icon-wrap" aria-hidden>{icon}</div>
      <h3 className="lp-feature-title">{title}</h3>
      <p className="lp-feature-desc">{desc}</p>
    </div>
  )
}

// ── Advantage card ───────────────────────────────────────────────────────────
function AdvCard({ icon, title, desc }) {
  return (
    <div className="lp-adv-card">
      <div className="lp-adv-icon-wrap" aria-hidden>{icon}</div>
      <h3 className="lp-adv-title">{title}</h3>
      <p className="lp-adv-desc">{desc}</p>
    </div>
  )
}

// ── Persona card ─────────────────────────────────────────────────────────────
function PersonaCard({ icon, role, quote }) {
  return (
    <div className="lp-persona-card">
      <div className="lp-persona-header">
        <div className="lp-persona-icon-wrap" aria-hidden>{icon}</div>
        <span className="lp-persona-role">{role}</span>
      </div>
      <p className="lp-persona-quote">"{quote}"</p>
    </div>
  )
}

// ── Global fixed background (gradient + dot grid + blobs) ───────────────────
function GlobalBackground() {
  return (
    <div className="lp-global-bg" aria-hidden>
      <div className="lp-bg-dots" />
      <div className="lp-bg-blob lp-bg-blob-1 animate-blob-float" />
      <div className="lp-bg-blob lp-bg-blob-2 animate-blob-float-reverse animation-delay-2000" />
      <div className="lp-bg-blob lp-bg-blob-3 animate-blob-float animation-delay-4000" />
    </div>
  )
}

// ── Navbar (morphs to floating pill after 50 px scroll) ─────────────────────
function Navbar() {
  const [scrolled, setScrolled]     = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const closeMenu = () => setMobileOpen(false)

  return (
    <div className={`lp-nav-wrapper${scrolled ? ' lp-nav-scrolled' : ''}`}>
      <div className={`lp-navbar-container${scrolled ? ' lp-navbar-pill' : ''}`}>
        <div className="lp-navbar-inner">

          {/* Brand */}
          <a href="#/" className="lp-brand">
            <div className="lp-brand-icon-wrap">
              <div className="lp-brand-icon">✦</div>
              <span className="lp-brand-status-dot" />
            </div>
            <span className="lp-brand-name">AutoML Assistant</span>
          </a>

          {/* Desktop nav links */}
          <nav className="lp-nav-links">
            <NavLink href="#features">Features</NavLink>
            <NavLink href="#how">How It Works</NavLink>
            <NavLink href="#advantages">Advantages</NavLink>
          </nav>

          {/* Desktop CTAs */}
          <div className="lp-nav-actions">
            <a href="#/login" className="lp-btn-login">Log in</a>
            <a href="#/app"   className="lp-btn-get-started">Get Started</a>
          </div>

          {/* Mobile hamburger */}
          <button
            className={`lp-hamburger${mobileOpen ? ' lp-hamburger-open' : ''}`}
            onClick={() => setMobileOpen(o => !o)}
            aria-label="Toggle navigation"
            aria-expanded={mobileOpen}
          >
            <span /><span /><span />
          </button>
        </div>

        {/* Mobile dropdown */}
        <div className={`lp-mobile-menu${mobileOpen ? ' lp-mobile-menu-open' : ''}`}>
          <div className="lp-mobile-menu-inner">
            <NavLink href="#features"   onClick={closeMenu}>Features</NavLink>
            <NavLink href="#how"        onClick={closeMenu}>How It Works</NavLink>
            <NavLink href="#advantages" onClick={closeMenu}>Advantages</NavLink>
            <div className="lp-mobile-actions">
              <a href="#/login" className="lp-btn-login"       onClick={closeMenu}>Log in</a>
              <a href="#/app"   className="lp-btn-get-started" onClick={closeMenu}>Get Started</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main Landing Page ────────────────────────────────────────────────────────
export default function LandingPage() {
  return (
    <div className="lp-root">
      <GlobalBackground />
      <Navbar />

      <main className="lp-main">

        {/* ── Hero ── */}
        <section id="hero" className="lp-hero-section">
          {/* Background grid dots */}
          <div className="lp-hero-bg" aria-hidden>
            <div className="lp-hero-bg-dots" />
          </div>

          {/* Gradient Blobs */}
          <div className="absolute top-20 -left-40 w-80 h-80 bg-blue-300/30 rounded-full mix-blend-multiply filter blur-xl animate-blob-float"></div>
          <div className="absolute bottom-20 -right-40 w-96 h-96 bg-indigo-300/30 rounded-full mix-blend-multiply filter blur-xl animate-blob-float-reverse"></div>

          <div className="lp-hero-inner">
            <div className="lp-hero-content">

              {/* H1 — three styled lines */}
              <h1 className="lp-hero-h1 animate-fade-in-up">
                <span className="lp-hero-h1-dark">From Raw Data to</span>
                <span className="lp-hero-h1-blue">Production ML</span>
                <span className="lp-hero-h1-italic">in Minutes.</span>
              </h1>

              {/* Subtitle with highlighted phrases */}
              <p className="lp-hero-sub animate-fade-in-up animation-delay-200">
                Upload any dataset, describe your goal in plain English, and let our AI{' '}
                <span className="lp-hero-highlight">clean your data,</span>{' '}
                <span className="lp-hero-highlight">select the best model,</span> and{' '}
                <span className="lp-hero-highlight">export production code</span> — no ML expertise needed.
              </p>

              {/* CTA Buttons */}
              <div className="lp-hero-ctas animate-fade-in-up animation-delay-400">
                <a href="#/app" className="lp-btn-primary lp-btn-lg">Get Started Free</a>
                <a href="#how"  className="lp-btn-outline lp-btn-lg">
                  <svg className="lp-btn-play-icon" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                  Watch in Action
                </a>
              </div>

            </div>
          </div>

          {/* Bottom fade */}
          <div className="lp-hero-bottom-fade" aria-hidden="true" />
        </section>

        {/* ── How It Works ── */}
        <Section id="how" className="lp-how-section">
          <div className="lp-section-inner">
            <h2 className="lp-section-h2">How It Works</h2>
            <p className="lp-section-sub">Three simple steps from data to deployed model.</p>
            <div className="lp-how-grid">
              <HowCard num="1" icon="⬆"   title="Upload Your Data"   desc="Drop in a CSV or connect to your data source. We handle the rest." />
              <HowCard num="2" icon="💬"  title="Describe Your Goal"  desc="Tell us what you want in plain English — no code needed." />
              <HowCard num="3" icon="</>" title="Get Your Pipeline"   desc="Receive a production-ready ML pipeline with full explanations." />
            </div>
          </div>
        </Section>

        {/* ── Powerful Features ── */}
        <section id="features" className="lp-features-section">
          <div className="lp-section-inner">
            <p className="lp-features-quote reveal-init" ref={useScrollReveal()}>
              "From raw data to production model — even when you have no ML background"
            </p>
            <div className="lp-feature-stack">
              {[
                {
                  title: 'Auto Preprocessing',
                  desc: 'Missing values, outliers, encoding, and scaling — all handled automatically with a full explanation of every decision made.',
                  img: new URL('../assets/images/auto_preprocessing.png', import.meta.url).href,
                },
                {
                  title: 'Smart Model Selection',
                  desc: 'Evaluates dozens of algorithms and picks the best fit for your data type, size, and goal — no guesswork required.',
                  img: new URL('../assets/images/Smart_Model_Selection.png', import.meta.url).href,
                },
                {
                  title: 'Explainable AI',
                  desc: 'Every preprocessing step and model choice is documented in plain English so you always understand what happened and why.',
                  img: new URL('../assets/images/explainable_AI.png', import.meta.url).href,
                },
                {
                  title: 'Production-Ready Code',
                  desc: 'Download a clean, self-contained Python script ready to deploy. No refactoring, no cleanup — just run it.',
                  img: new URL('../assets/images/code_pipeline.png', import.meta.url).href,
                },
                {
                  title: 'Interactive EDA',
                  desc: 'Histograms, correlation heatmaps, missing-value analysis and distribution plots generated automatically for your dataset.',
                  img: new URL('../assets/images/interactive_EDA.png', import.meta.url).href,
                },
                {
                  title: 'Stakeholder-Ready PDF Reports',
                  desc: 'Automatically generates a polished, non-technical PDF report summarising your dataset, the models evaluated, the best result, and actionable recommendations — written in plain language your manager or client can read without any ML background.',
                  bullets: [
                    'Executive summary with key findings',
                    'Visual model comparison and performance charts',
                    'Plain-English explanation of every decision',
                    'Deployment checklist and next-step recommendations',
                  ],
                  img: new URL('../assets/images/stackholder_pdf.png', import.meta.url).href,
                  cta: true,
                },
              ].map((f, i) => (
                <div
                  key={i}
                  className="lp-feature-sticky-card"
                  style={{ top: `${80 + i * 20}px`, zIndex: i + 1, transitionDelay: `${0.1 + i * 0.1}s` }}
                >
                  {/* Text side */}
                  <div className="lp-fsc-text">
                    <h3 className="lp-fsc-title">{f.title}</h3>
                    <p className="lp-fsc-desc">{f.desc}</p>
                    {f.bullets && (
                      <ul className="lp-fsc-bullets">
                        {f.bullets.map((b, bi) => (
                          <li key={bi} className="lp-fsc-bullet-item">
                            <span className="lp-fsc-bullet-dot">✦</span>{b}
                          </li>
                        ))}
                      </ul>
                    )}
                    {f.cta && (
                      <a href="#/app" className="lp-btn-primary lp-btn-lg" style={{ marginTop: '24px', display: 'inline-flex' }}>
                        Try It Free
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" style={{ width: '20px', height: '20px', marginLeft: '8px' }}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 12H6.75m6 6 6-6-6-6" />
                        </svg>
                      </a>
                    )}
                  </div>
                  {/* Image side */}
                  <div className={`lp-fsc-image ${i % 2 === 1 ? 'lp-fsc-image--left' : ''}`}>
                    {f.img
                      ? <img src={f.img} alt={f.title} className="lp-fsc-img" />
                      : <div className="lp-fsc-image-placeholder" />
                    }
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Advantages ── */}
        <Section id="advantages" className="lp-adv-section">
          <div className="lp-section-inner">
            <h2 className="lp-section-h2">Why AutoML Assistant?</h2>
            <p className="lp-section-sub">Skip the boilerplate. Focus on insights.</p>
            <div className="lp-adv-grid">
              <AdvCard icon="⏱" title="Hours → Minutes"         desc="What takes days of manual work finishes in minutes." />
              <AdvCard icon="🛡" title="No ML Expertise Needed" desc="Business analysts and domain experts can build real models." />
              <AdvCard icon="⚡" title="Transparent Decisions"  desc="See why each preprocessing step and model was chosen." />
            </div>
          </div>
        </Section>

        {/* ── Built for Everyone ── */}
        <Section className="lp-persona-section">
          <div className="lp-section-inner">
            <h2 className="lp-section-h2">Built for Everyone</h2>
            <p className="lp-section-sub">From analysts to scientists — AutoML adapts to your workflow.</p>
            <div className="lp-persona-grid">
              <PersonaCard icon="📊" role="Business Analyst" quote="I uploaded our sales CSV and asked 'predict next quarter revenue.' Got a working model in 3 minutes." />
              <PersonaCard icon="🧪" role="Data Scientist"   quote="Perfect for quick baselines. I skip the boilerplate and jump straight to tuning." />
              <PersonaCard icon="👤" role="Domain Expert"    quote="I have years of clinical data. Now I can explore ML hypotheses without writing code." />
            </div>
          </div>
        </Section>

        {/* ── CTA ── */}
        <Section className="lp-cta-section">
          <div className="lp-section-inner lp-cta-inner">
            <h2 className="lp-cta-h2">Ready to automate your ML workflow?</h2>
            <p className="lp-cta-sub">Start building production models in minutes — completely free.</p>
            <a href="#/app" className="lp-btn-primary lp-btn-lg lp-cta-btn">Get Started Free →</a>
            <div className="lp-cta-badges">
              <span className="lp-cta-badge">⊙ No credit card</span>
              <span className="lp-cta-badge">⊙ Free tier forever</span>
            </div>
          </div>
        </Section>

      </main>

      {/* ── Footer ── */}
      <footer className="lp-footer">
        <div className="lp-footer-inner">
          <div className="lp-footer-brand-col">
            <a href="#/" className="lp-brand lp-footer-brand">
              <div className="lp-brand-icon-wrap">
                <div className="lp-brand-icon">✦</div>
              </div>
              <span className="lp-brand-name">AutoML Assistant</span>
            </a>
            <p className="lp-footer-tagline">From raw data to production ML —<br />powered by AI, built for everyone.</p>
          </div>

          <div className="lp-footer-nav-col">
            <p className="lp-footer-col-title">Product</p>
            <a href="#features" className="lp-footer-link">Features</a>
            <a href="#how"      className="lp-footer-link">How It Works</a>
            <a href="#/"        className="lp-footer-link">Pricing</a>
            <a href="#/"        className="lp-footer-link">Changelog</a>
          </div>

          <div className="lp-footer-nav-col">
            <p className="lp-footer-col-title">Resources</p>
            <a href="#/" className="lp-footer-link">Documentation</a>
            <a href="#/" className="lp-footer-link">API Reference</a>
            <a href="#/" className="lp-footer-link">Blog</a>
            <a href="#/" className="lp-footer-link">Support</a>
          </div>

          <div className="lp-footer-nav-col">
            <p className="lp-footer-col-title">Company</p>
            <a href="#/" className="lp-footer-link">About</a>
            <a href="#/" className="lp-footer-link">Privacy</a>
            <a href="#/" className="lp-footer-link">Terms</a>
            <a href="#/" className="lp-footer-link">Contact</a>
          </div>
        </div>
        <div className="lp-footer-bottom">
          <p>© 2026 AutoML Assistant. All rights reserved.</p>
        </div>
      </footer>

    </div>
  )
}
