import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The frontend never talks to the simulator directly: everything goes through
// /api, which is proxied to the backend in dev and can point anywhere later.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.PLANT_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
