package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
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
//   - GET /  — 204 No Content. Useful for reverse-proxy health
//     probes and confirming the binary is up. The embedded SPA
//     (docs/ARCHITECTURE.md §6.6 `go:embed web/dist`) will live here
//     in a later Feature; the current Feature ships only the relay
//     endpoint because the SPA build artefact does not yet exist.
//
// The returned server has not been started — callers run
// ListenAndServe themselves so signal handling and shutdown remain
// in the caller's scope.
func NewServer(cfg *config.Config, client llm.LLMClient) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/ws", NewWebSocketHandler(cfg, client))
	mux.HandleFunc("/", stubRootHandler)

	return &http.Server{
		Addr:              ":" + strconv.Itoa(cfg.Server.Port),
		Handler:           mux,
		ReadHeaderTimeout: readHeaderTimeout,
	}
}

// stubRootHandler returns 204 No Content for any path that isn't
// matched by a more specific route on the mux. It exists so reverse
// proxies and human curls don't see a 404 on the relay's root.
func stubRootHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}
