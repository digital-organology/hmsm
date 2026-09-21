import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      // Keep the browser's Host so Python can validate its Origin as well.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false },
    },
  },
});
