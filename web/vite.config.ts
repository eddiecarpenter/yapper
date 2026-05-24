/**
 * Vite configuration for the Yapper SPA.
 *
 * The SPA is built into `web/dist/` so the Go relay's
 * `//go:embed all:web/dist` directive (Task 2) can pick it up and
 * serve it from the single binary. The dev server proxies `/ws` to
 * `localhost:8080` so a developer running `npm run dev` against a
 * separately-running relay sees the same WebSocket origin the
 * production build sees from the embedded SPA.
 *
 * - `outDir: "dist"` matches what `internal/api/static.go` embeds.
 * - `base: "/"` keeps URLs root-relative, which lets the
 *   SPA-fallback handler return `index.html` for unknown deep links
 *   without rewriting asset paths.
 * - The `/ws` proxy is `ws: true` so the WebSocket upgrade is
 *   forwarded correctly to the Go relay during `npm run dev`.
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/ws": {
        target: "ws://localhost:8080",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
