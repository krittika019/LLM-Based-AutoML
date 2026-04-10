import { useEffect, useRef } from 'react'

export default function Orb({
  hue = 220,
  hoverIntensity = 0.2,
  rotateOnHover = true,
  forceHoverState = false,
}) {
  const canvasRef = useRef(null)
  const hoveredRef = useRef(forceHoverState)
  const tRef = useRef(0)
  const animRef = useRef(null)

  useEffect(() => {
    hoveredRef.current = forceHoverState
  }, [forceHoverState])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    /* ── resize: setTransform avoids accumulation bug ── */
    function setSize() {
      const dpr = window.devicePixelRatio || 1
      const w = canvas.offsetWidth
      const h = canvas.offsetHeight
      if (w === 0 || h === 0) return
      canvas.width  = w * dpr
      canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    setSize()
    const ro = new ResizeObserver(setSize)
    ro.observe(canvas)

    /* ── hover ── */
    function onEnter() { hoveredRef.current = true }
    function onLeave() { hoveredRef.current = forceHoverState }
    if (rotateOnHover) {
      canvas.addEventListener('mouseenter', onEnter)
      canvas.addEventListener('mouseleave', onLeave)
    }

    /* ── draw loop ── */
    function draw() {
      const W = canvas.offsetWidth
      const H = canvas.offsetHeight
      if (W === 0 || H === 0) { animRef.current = requestAnimationFrame(draw); return }

      const isHov = hoveredRef.current
      tRef.current += isHov ? 0.012 * (1 + hoverIntensity * 5) : 0.012
      const t = tRef.current

      const cx = W / 2
      const cy = H / 2
      const r  = Math.min(W, H) / 2 * 0.96

      ctx.clearRect(0, 0, W, H)

      /* rotating hue triplet — 80× multiplier for fast visible shift */
      const h1 = (hue + t * 80)         % 360
      const h2 = (hue + 120 + t * 80)   % 360
      const h3 = (hue + 240 + t * 80)   % 360

      /* layer 1: sphere base fill */
      const fill = ctx.createRadialGradient(cx - r * 0.28, cy - r * 0.30, 0, cx, cy, r)
      fill.addColorStop(0,    'rgba(255,255,255,0.92)')
      fill.addColorStop(0.20, 'rgba(225,240,255,0.78)')
      fill.addColorStop(0.42, 'rgba(200,225,255,0.52)')
      fill.addColorStop(0.62, `hsla(${h3},72%,78%,0.24)`)
      fill.addColorStop(0.80, `hsla(${h1},82%,65%,0.38)`)
      fill.addColorStop(0.92, `hsla(${h2},88%,60%,0.50)`)
      fill.addColorStop(1,    `hsla(${h1},72%,55%,0.25)`)
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = fill; ctx.fill()

      /* layer 2: orbiting colour spotlight (visibly rotates around interior) */
      const ox = cx + r * 0.55 * Math.cos(t)
      const oy = cy + r * 0.55 * Math.sin(t)
      const spot = ctx.createRadialGradient(ox, oy, 0, ox, oy, r * 0.75)
      spot.addColorStop(0,   `hsla(${h2}, 92%, 70%, 0.32)`)
      spot.addColorStop(0.5, `hsla(${h3}, 85%, 65%, 0.16)`)
      spot.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = spot; ctx.fill()

      /* layer 3: counter-rotating second spotlight */
      const ox2 = cx + r * 0.48 * Math.cos(-t * 1.3 + 2.1)
      const oy2 = cy + r * 0.48 * Math.sin(-t * 1.3 + 2.1)
      const spot2 = ctx.createRadialGradient(ox2, oy2, 0, ox2, oy2, r * 0.60)
      spot2.addColorStop(0,   `hsla(${h1}, 90%, 72%, 0.28)`)
      spot2.addColorStop(0.5, `hsla(${h2}, 80%, 68%, 0.12)`)
      spot2.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = spot2; ctx.fill()

      /* layer 4: iridescent rim ring */
      const rim = ctx.createRadialGradient(cx, cy, r * 0.70, cx, cy, r)
      rim.addColorStop(0,   'rgba(0,0,0,0)')
      rim.addColorStop(0.5, `hsla(${h2}, 90%, 68%, 0.28)`)
      rim.addColorStop(0.8, `hsla(${h3}, 85%, 62%, 0.35)`)
      rim.addColorStop(1,   `hsla(${h1}, 80%, 60%, 0.22)`)
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = rim; ctx.fill()

      /* layer 5: static top-left specular sheen */
      const sheen = ctx.createRadialGradient(
        cx - r * 0.22, cy - r * 0.28, 0,
        cx - r * 0.05, cy - r * 0.10, r * 0.58,
      )
      sheen.addColorStop(0,    'rgba(255,255,255,0.72)')
      sheen.addColorStop(0.45, 'rgba(255,255,255,0.24)')
      sheen.addColorStop(1,    'rgba(255,255,255,0.00)')
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = sheen; ctx.fill()

      /* layer 6: small bright specular dot */
      const dot = ctx.createRadialGradient(
        cx - r * 0.28, cy - r * 0.28, 0,
        cx - r * 0.28, cy - r * 0.28, r * 0.13,
      )
      dot.addColorStop(0, 'rgba(255,255,255,0.92)')
      dot.addColorStop(1, 'rgba(255,255,255,0.00)')
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = dot; ctx.fill()

      /* layer 7: thin iridescent border */
      ctx.beginPath()
      ctx.arc(cx, cy, r - 1, 0, Math.PI * 2)
      ctx.strokeStyle = `hsla(${h1}, 82%, 76%, 0.55)`
      ctx.lineWidth = 1.5
      ctx.stroke()

      animRef.current = requestAnimationFrame(draw)
    }

    animRef.current = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(animRef.current)
      ro.disconnect()
      canvas.removeEventListener('mouseenter', onEnter)
      canvas.removeEventListener('mouseleave', onLeave)
    }
  }, [hue, hoverIntensity, rotateOnHover])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}
