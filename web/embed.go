// Package web exposes the bundled Yapper SPA produced by
// `npm run build` so the Go relay can serve it from a single,
// self-contained binary via the static handler in internal/api/.
//
// The package lives next to the TypeScript SPA sources because
// Go's //go:embed directive resolves paths relative to the file
// that declares it, and `..` is forbidden — placing the embed at
// the module root or inside internal/api would require a path that
// reaches outside the package directory. Hosting it here lets the
// directive use the natural `dist` relative path, matching the
// Vite build output one directory below.
//
// AD-6 (single-binary deployment via go:embed) is what this
// package implements; internal/api/static.go is the consumer.
package web

import "embed"

// FS is the embedded Vite-bundled SPA. The `all:` prefix has two
// effects: (1) it includes files whose names start with `.` or
// `_` (Vite emits hashed asset names that occasionally pick up
// leading underscores), and (2) it lets the build succeed when
// `dist/` is empty — a clean checkout without a TypeScript build
// must still compile for backend-only contributors and CI. The
// static handler logs a startup WARN when index.html is missing
// so the misconfiguration is surfaced at run time without
// breaking the build path.
//
//go:embed all:dist
var FS embed.FS
