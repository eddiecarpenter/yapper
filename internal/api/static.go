package api

import (
	"io/fs"
	"log"
	"net/http"
	"path"
	"strings"

	"github.com/eddiecarpenter/yapper/web"
)

// spaIndexFile is the SPA's bootstrap document. The static handler
// returns it as the fallback response for any path that does not
// resolve to an embedded file (per AD-6 and design plan KD-2 — SPA
// fallback to index.html). The constant exists so the warning path
// and the handler agree on the name.
const spaIndexFile = "index.html"

// spaFS returns the embedded SPA filesystem rooted at the dist
// directory (so URL paths line up with file paths inside the SPA
// build output). The underlying bytes come from the `web` package,
// whose //go:embed directive captures the Vite build output. We
// re-root here so callers (the handler factory and the
// startup-warning probe) see the same view of the SPA tree.
func spaFS() fs.FS {
	sub, err := fs.Sub(web.FS, "dist")
	if err != nil {
		// fs.Sub only errors when the path is invalid — "dist" is
		// always a valid path component, so this branch is purely
		// defensive. Surface as the unrooted FS rather than panic
		// so a future refactor cannot crash the relay at startup.
		log.Printf("spa: failed to sub embedded fs: %v (serving unrooted SPA)", err)
		return web.FS
	}
	return sub
}

// newSPAHandler returns an http.Handler that serves files from the
// given filesystem and falls back to index.html for unknown HTML
// requests. The contract:
//
//   - Known embedded file (exact match, after stripping the leading
//     "/") → served via http.FileServer; content-type derived from
//     the extension, status 200.
//   - Unknown path with no file extension (e.g. "/settings", a
//     deep-linked client route) → falls back to index.html with
//     status 200 so the SPA's client-side router can render it.
//   - Unknown path with a file extension (e.g. "/missing.css") →
//     404. We do not fall back here because a missing asset is a
//     real error, not a deep-link.
//
// The fallback heuristic uses "has an extension" rather than
// reading the Accept header — design plan KD-2 notes this is the
// simpler rule and is sufficient for the spike. A future Feature
// can refine if strict content-negotiation becomes important.
//
// If `spaIndexFile` is absent in the embedded FS (the developer
// forgot to run `npm run build` before `go build`), all routes —
// including the fallback path — return 404. The startup-time
// warning emitted by warnIfSPAMissing surfaces this misconfiguration
// in the server log; we do NOT crash on startup because
// backend-only contributors and CI must still be able to build and
// run the relay's HTTP surface.
func newSPAHandler(filesystem fs.FS) http.Handler {
	fileServer := http.FileServer(http.FS(filesystem))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		urlPath := r.URL.Path
		// Trim the leading "/" so we can stat against the embedded
		// fs (whose root has no leading slash). "/" itself maps to
		// the index file.
		clean := strings.TrimPrefix(urlPath, "/")
		if clean == "" {
			clean = spaIndexFile
		}
		if _, err := fs.Stat(filesystem, clean); err == nil {
			fileServer.ServeHTTP(w, r)
			return
		}
		// Miss — fall back to index.html only when the path looks
		// like a client-route (no file extension). Anything with an
		// extension is asset-shaped; a miss is a real 404.
		if path.Ext(clean) == "" {
			if _, err := fs.Stat(filesystem, spaIndexFile); err == nil {
				// Serve index.html directly via ServeFileFS. We do
				// NOT clone-and-rewrite the request because
				// http.FileServer's helpful /index.html → ./
				// relative redirect would resolve against the
				// original request URL and loop. ServeFileFS sets
				// the right Content-Type and ETag headers without
				// the redirect dance.
				http.ServeFileFS(w, r, filesystem, spaIndexFile)
				return
			}
		}
		http.NotFound(w, r)
	})
}

// warnIfSPAMissing logs a single startup-time WARN if index.html
// is not present in the embedded SPA filesystem. The log line is
// deliberately actionable — it names the build command the
// developer most likely forgot. The function does NOT block
// startup (per the design plan: empty dist must not break the
// build path).
func warnIfSPAMissing(filesystem fs.FS) {
	if _, err := fs.Stat(filesystem, spaIndexFile); err != nil {
		log.Printf("WARN: SPA index.html not embedded (%v); /ws will work but / will 404. Run `npm run build` in web/.", err)
	}
}
