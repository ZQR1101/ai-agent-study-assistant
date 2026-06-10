import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  root: "frontend",
  server: {
    host: "127.0.0.1",
    port: 5500,
  },
  preview: {
    host: "127.0.0.1",
    port: 5500,
  },
})
