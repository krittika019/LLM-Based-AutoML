import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Proxy API calls to the FastAPI server during development.
// In production (after `vite build`), serve the dist/ folder via FastAPI's StaticFiles.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/upload': { target: 'http://localhost:8000', changeOrigin: true },
      '/run':    { target: 'http://localhost:8000', changeOrigin: true },
      '/stream': { target: 'http://localhost:8000', changeOrigin: true },
      '/download': { target: 'http://localhost:8000', changeOrigin: true },
      '/predict': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
