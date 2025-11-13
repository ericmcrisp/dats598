import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // CRITICAL: Listen on all interfaces for Docker
    port: 5173,
    watch: {
      usePolling: true  // Better file watching in Docker
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // Use Docker service name, not localhost!
        changeOrigin: true,
        secure: false,
      }
    }
  }
})