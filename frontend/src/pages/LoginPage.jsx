import React, { useState } from 'react'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    // Navigate to app on form submit (placeholder — real auth would validate here)
    window.location.hash = '#/app'
  }

  return (
    <div className="auth-root">
      {/* Minimal navbar */}
      <header className="auth-navbar">
        <a href="#/" className="lp-brand">
          <div className="lp-brand-icon">✦</div>
          <span>AutoML Assistant</span>
        </a>
      </header>

      <main className="auth-main">
        <div className="auth-card">
          <h1 className="auth-title">Welcome back</h1>
          <p className="auth-sub">Log in to continue to AutoML Assistant</p>

          {/* OAuth buttons */}
          <div className="auth-oauth-row">
            <button className="auth-oauth-btn" type="button">
              <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
                <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.2l6.7-6.7C35.8 2.5 30.3 0 24 0 14.6 0 6.6 5.4 2.7 13.3l7.8 6C12.3 13.2 17.6 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4 6.9-10 6.9-17z"/>
                <path fill="#FBBC05" d="M10.5 28.8A14.6 14.6 0 0 1 9.5 24c0-1.7.3-3.3.8-4.8l-7.8-6A24 24 0 0 0 0 24c0 3.9.9 7.5 2.5 10.8l8-6z"/>
                <path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.5-5.8c-2.1 1.4-4.8 2.3-8.4 2.3-6.4 0-11.8-4.3-13.7-10l-8 6.1C6.5 42.6 14.6 48 24 48z"/>
              </svg>
              Google
            </button>
            <button className="auth-oauth-btn" type="button">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.2 11.38.6.11.82-.26.82-.58v-2.03c-3.34.72-4.04-1.61-4.04-1.61-.54-1.38-1.33-1.75-1.33-1.75-1.09-.74.08-.73.08-.73 1.2.08 1.84 1.23 1.84 1.23 1.07 1.83 2.8 1.3 3.49 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.005 2.04.14 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.64 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58A12.01 12.01 0 0 0 24 12C24 5.37 18.63 0 12 0z"/>
              </svg>
              GitHub
            </button>
          </div>

          <div className="auth-divider"><span>or continue with email</span></div>

          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="auth-field">
              <label className="auth-label" htmlFor="login-email">Email</label>
              <input
                id="login-email"
                type="email"
                className="auth-input"
                placeholder="you@example.com"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>

            <div className="auth-field">
              <div className="auth-label-row">
                <label className="auth-label" htmlFor="login-password">Password</label>
                <a href="#/" className="auth-forgot">Forgot password?</a>
              </div>
              <input
                id="login-password"
                type="password"
                className="auth-input"
                placeholder="••••••••"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                autoComplete="current-password"
              />
            </div>

            <button type="submit" className="auth-submit-btn">Log in</button>
          </form>

          <p className="auth-switch">
            Don't have an account? <a href="#/signup" className="auth-switch-link">Sign up</a>
          </p>
        </div>
      </main>
    </div>
  )
}
