import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: `npm run dev` serves on :5173 and proxies /api to the Flask backend.
// Build: `npm run build` emits ./dist, which Flask serves as static assets.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:5000',
    },
  },
})
