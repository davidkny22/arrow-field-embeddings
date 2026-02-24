import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import type { Plugin } from 'vite'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'

/**
 * Serve .json.gz files as opaque binary (application/gzip) so the browser
 * does NOT auto-decompress them via Content-Encoding. The client uses pako
 * to decompress explicitly.
 */
function serveGzipPresets(): Plugin {
  return {
    name: 'serve-gzip-presets',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.endsWith('.json.gz')) {
          const filePath = join(__dirname, 'public', req.url)
          if (existsSync(filePath)) {
            const data = readFileSync(filePath)
            res.setHeader('Content-Type', 'application/gzip')
            res.setHeader('Content-Length', data.length.toString())
            res.setHeader('Cache-Control', 'no-cache')
            res.end(data)
            return
          }
        }
        next()
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [serveGzipPresets(), react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three', '@react-three/fiber', '@react-three/drei', '@react-three/postprocessing'],
        },
      },
    },
  },
})
