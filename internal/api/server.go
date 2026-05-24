package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
	"github.com/eddiecarpenter/yapper/internal/session"
)

const (
	// readHeaderTimeout caps how long the server waits for request
	// headers — a sane default for any net/http server exposed
	// beyond localhost. The WebSocket upgrade itself happens after
	// the header phase so this does not constrain in-flight
	// completions.
	readHeaderTimeout = 10 * time.Second
)

// NewServer constructs the http.Server for the Yapper relay.
//
// The mux carries two routes:
//
//   - GET /ws — WebSocket upgrade, one turn per connection.
//   - GET /  — embedded SPA served from `web/dist` (AD-6,
//     docs/ARCHITECTURE.md §6.6). Unknown deep-link paths fall
//     back to index.html so client-side routing works without a
//     reverse proxy. If the SPA was not built before `go build`
//     ran, a startup-time WARN names the missing index.html and
//     `/` serves a 404 — but the WebSocket relay continues to
//     work, which is what backend-only contributors need.
//
// The returned server has not been started — callers run
// ListenAndServe themselves so signal handling and shutdown remain
// in the caller's scope.
func NewServer(cfg *config.Config, client llm.LLMClient, store session.Store) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/ws", NewWebSocketHandler(cfg, client, store))
	spa := spaFS()
	warnIfSPAMissing(spa)
	mux.Handle("/", newSPAHandler(spa))

	return &http.Server{
		Addr:              ":" + strconv.Itoa(cfg.Server.Port),
		Handler:           mux,
		ReadHeaderTimeout: readHeaderTimeout,
	}
}
