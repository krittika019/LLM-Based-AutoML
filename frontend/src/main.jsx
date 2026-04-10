import React, { useState, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import LandingPage from './pages/LandingPage.jsx'
import LoginPage from './pages/LoginPage.jsx'
import SignupPage from './pages/SignupPage.jsx'
import './index.css'

function Router() {
  const getRoute = () => {
    const h = window.location.hash
    if (h === '#/app') return 'app'
    if (h === '#/login') return 'login'
    if (h === '#/signup') return 'signup'
    return 'landing'
  }

  const [route, setRoute] = useState(getRoute)

  useEffect(() => {
    const onHash = () => setRoute(getRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  if (route === 'app')    return <App />
  if (route === 'login')  return <LoginPage />
  if (route === 'signup') return <SignupPage />
  return <LandingPage />
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Router />
  </React.StrictMode>,
)
