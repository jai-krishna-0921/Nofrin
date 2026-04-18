import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: '../static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/research': 'http://localhost:8000',
      '/stream':   'http://localhost:8000',
      '/status':   'http://localhost:8000',
    },
  },
})
