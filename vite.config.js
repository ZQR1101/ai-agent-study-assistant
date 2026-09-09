import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  root: "frontend",
  // Build into backend/static so the FastAPI app can serve the SPA directly.
  // This makes `pip install` / Docker single-container deployments self-contained.
  build: {
    outDir: "../backend/static",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5500,
  },
  preview: {
    host: "127.0.0.1",
    port: 5500,
  },
})
