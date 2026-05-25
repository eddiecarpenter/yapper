// Package api implements the Yapper relay's HTTP and WebSocket
// surface — the `/ws` upgrade endpoint that proxies turns from the
// browser dialogue loop to a configured LLMClient, plus the stub `/`
// health endpoint.
//
// The wire protocol is declared in docs/ARCHITECTURE.md §5.2:
// inbound `{type:"turn", text, history}` frames; outbound
// `{type:"token", text}` frames followed by `{type:"done", usage}`,
// or a single `{type:"error", message}` on the failure path.
package api

import (
	"context"
	"errors"
	"log"
	"net/http"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"

	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
	"github.com/eddiecarpenter/yapper/internal/session"
)

// sessionCookieName is the cookie that round-trips the
// session.Store key across WebSocket upgrades. Cookies are
// session-only at the browser (no Expires / Max-Age) — the
// canonical lifecycle lives server-side in the Store's TTL
// eviction (per design plan KD-1 and Feature 17 Task 4 notes).
const sessionCookieName = "yapper_session"

// Frame type discriminators on the WebSocket wire protocol. Defined
// once so tests, the handler, and any future consumer reference the
// same strings.
const (
	turnFrameType  = "turn"
	tokenFrameType = "token"
	doneFrameType  = "done"
	errorFrameType = "error"
)

// turnFrame is the single inbound frame the relay accepts on each
// connection.
//
// Feature 17 Task 4 moved conversation history server-side (per
// design plan KD-1 — cookie-keyed session store as canonical
// owner). The History field is preserved on the wire for one
// release of graceful migration: the browser MAY still send it,
// but the server NEVER reads it (the session.Store is the source
// of truth). A future cleanup cycle will remove the field once
// every deployed client has migrated.
type turnFrame struct {
	Type string `json:"type"`
	Text string `json:"text"`
	// Deprecated: history is now server-canonical via
	// internal/session/.Store. The field is retained on the wire
	// for one-release backwards compatibility so old browser
	// builds do not break, but the server's WebSocket handler
	// ignores any value supplied here. Remove in a follow-on
	// Feature once the migration window closes.
	History []llm.Message `json:"history,omitempty"`

	// NoThinking, when true, passes enable_thinking:false to the LLM
	// so chain-of-thought reasoning is suppressed. Useful for voice
	// assistants where thinking tokens add latency with no benefit.
	// Omitting the field (or sending false) leaves the server default
	// in effect (currently: thinking disabled by default).
	NoThinking *bool `json:"no_thinking,omitempty"`
}

// tokenFrame is an outbound streaming text chunk.
type tokenFrame struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// doneFrame is the terminator on the success path; Usage carries the
// final token-count record reported by the upstream provider (zero
// values when the provider does not supply usage).
type doneFrame struct {
	Type  string    `json:"type"`
	Usage llm.Usage `json:"usage"`
}

// errorFrame is the terminator on the failure path. The Message
// field is sanitised by classifyError — API keys and raw upstream
// stack traces never appear here.
type errorFrame struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// WebSocketHandler implements net/http.Handler against the relay's
// WebSocket protocol.
//
// One turn per connection is the MVP shape — every browser turn
// opens a fresh socket, drives one CompletionRequest, and closes.
// Multi-turn streaming over a single socket is reserved for a later
// Feature (see docs/ARCHITECTURE.md §10 "Evolution Seams").
//
// The handler is safe for concurrent use; each connection runs an
// independent goroutine with its own derived context. The
// session.Store is the canonical owner of per-conversation history
// (Feature 17 Task 4) — the handler resolves the session via the
// `yapper_session` cookie on each upgrade, then reads / writes the
// session's history rather than trusting browser-supplied state.
type WebSocketHandler struct {
	cfg     *config.Config
	client  llm.LLMClient
	store   session.Store
	resolve sessionResolver
}

// sessionResolver is the function shape the handler uses to map
// an inbound HTTP request to a session.Session and decide whether
// a Set-Cookie response header is required. Carved out as an
// indirection because the upgrade-time Set-Cookie path is the
// only place test code needs to assert behaviour without driving
// the full WebSocket dance.
type sessionResolver func(r *http.Request) (*session.Session, *http.Cookie)

// NewWebSocketHandler returns a handler bound to cfg, client, and
// the session store. The store is required (callers must pass a
// non-nil value); a nil store would defeat the whole point of
// server-canonical history and would panic on first use.
func NewWebSocketHandler(cfg *config.Config, client llm.LLMClient, store session.Store) *WebSocketHandler {
	if store == nil {
		panic("api: NewWebSocketHandler called with nil session.Store")
	}
	h := &WebSocketHandler{cfg: cfg, client: client, store: store}
	h.resolve = h.defaultResolveSession
	return h
}

// defaultResolveSession reads `yapper_session` from the request's
// cookies. If the cookie is present, the corresponding session is
// returned (re-created empty if it was evicted) and no Set-Cookie
// is returned. If the cookie is absent, a new session is minted
// and a fresh Set-Cookie value is returned for the response
// writer to emit before the WebSocket upgrade completes.
//
// The cookie is HttpOnly + SameSite=Lax + Path=/ + session-only
// (no Expires / Max-Age) — the server-side TTL is the real
// lifecycle (design plan §3).
func (h *WebSocketHandler) defaultResolveSession(r *http.Request) (*session.Session, *http.Cookie) {
	if c, err := r.Cookie(sessionCookieName); err == nil && c.Value != "" {
		return h.store.GetOrCreate(c.Value), nil
	}
	sess := h.store.GetOrCreate("")
	return sess, &http.Cookie{
		Name:     sessionCookieName,
		Value:    sess.ID,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	}
}

// ServeHTTP performs the WebSocket upgrade and drives a single turn
// per connection. Errors during the turn are reported back to the
// browser as a sanitised error frame; the server never panics out
// of this method even on hostile inputs.
//
// The cookie is resolved BEFORE websocket.Accept because
// coder/websocket writes upgrade response headers eagerly during
// Accept — once Accept returns, the underlying ResponseWriter has
// been hijacked and Set-Cookie additions are dropped on the floor.
func (h *WebSocketHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	sess, setCookie := h.resolve(r)
	if setCookie != nil {
		http.SetCookie(w, setCookie)
	}

	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		// The MVP runs on localhost; the browser's Origin header is
		// not a useful authentication signal at this stage. A future
		// deployment Feature will set OriginPatterns explicitly and
		// remove this skip when the relay leaves localhost.
		InsecureSkipVerify: true,
	})
	if err != nil {
		log.Printf("ws: accept failed: %v", err)
		return
	}
	// CloseNow is the idempotent teardown — if conn.Close has
	// already been called via the normal path the second close is a
	// no-op. Combined with the explicit Close call at the end of
	// ServeHTTP this guarantees the socket is released on every exit
	// path including panics.
	defer func() { _ = conn.CloseNow() }()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// Watch r.Context() for cancellation triggered by the
	// server's BaseContext (which main.go derives from the
	// SIGINT-cancellable context). When the server is shutting
	// down, this goroutine wakes any in-flight wsjson.Read /
	// wsjson.Write by closing the connection with a
	// StatusGoingAway frame. The per-turn LLM context, derived
	// from r.Context(), is cancelled at the same moment so the
	// adapter aborts its upstream stream inside the bounded
	// shutdown drain window (Feature 17 Task 6, AC-2).
	//
	// The select races r.Context().Done() against ctx.Done() —
	// ctx is the per-turn context that the defer cancel() above
	// fires when ServeHTTP returns normally. Without that race,
	// the goroutine would leak past every successful turn,
	// waiting forever for r.Context to cancel.
	go func() {
		select {
		case <-r.Context().Done():
			_ = conn.Close(websocket.StatusGoingAway, "server shutting down")
		case <-ctx.Done():
			// Normal exit — the per-turn cancel fired. Nothing
			// to do; the deferred CloseNow / Close calls in
			// the outer ServeHTTP body will tear down the conn.
		}
	}()

	if err := h.handleTurn(ctx, conn, sess); err != nil {
		log.Printf("ws: turn aborted: %v", err)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// handleTurn reads the inbound turn frame, drives CompleteStream,
// and writes token / done frames on success or a single error frame
// on failure. Returning a non-nil error means the turn did not
// complete successfully — the caller uses this only for log
// telemetry, not for further protocol action.
//
// History is read from sess (the resolved session.Session for this
// connection's cookie); inbound.History is IGNORED. On success the
// user turn and the full assistant reply are appended back to the
// session. Touch is called before the LLM call so a long-running
// stream cannot race with the eviction goroutine.
func (h *WebSocketHandler) handleTurn(ctx context.Context, conn *websocket.Conn, sess *session.Session) error {
	var inbound turnFrame
	if err := wsjson.Read(ctx, conn, &inbound); err != nil {
		// Read failed — the client likely disconnected before
		// sending a frame. There is nothing to reply to.
		return err
	}
	if inbound.Type != turnFrameType {
		msg := "expected `turn` frame, got `" + inbound.Type + "`"
		_ = wsjson.Write(ctx, conn, errorFrame{Type: errorFrameType, Message: msg})
		return errors.New(msg)
	}

	// Keep the session alive across the LLM call. The eviction
	// goroutine runs on its own ticker; without this, a long
	// stream could outlast the TTL and the post-call Append below
	// would silently no-op against a reaped session.
	h.store.Touch(sess.ID)

	// Resolve thinking preference: browser toggle takes precedence;
	// fall back to server default (thinking off for voice assistant).
	noThinking := true // server default: thinking disabled
	if inbound.NoThinking != nil {
		noThinking = *inbound.NoThinking
	}
	enableThinking := !noThinking
	req := llm.CompletionRequest{
		Model:          h.cfg.LLM.Model,
		Messages:       buildMessages(h.cfg.LLM.SystemPrompt, sess.History, inbound.Text),
		EnableThinking: &enableThinking,
	}

	// thinkFilter strips Qwen3-style <think>…</think> reasoning blocks
	// from the token stream before forwarding to the browser. Models that
	// don't emit these tags pass through unchanged. The filter is
	// stateful across tokens because a tag boundary may split across
	// multiple delta chunks.
	filter := newThinkFilter()

	resp, err := h.client.CompleteStream(ctx, req,
		func(tok string) {
			visible := filter.Write(tok)
			if visible == "" {
				return
			}
			// Token writes are best-effort — if the browser
			// disconnects mid-stream, the next write fails and the
			// adapter's ctx-cancellation path takes over. We do not
			// surface the error here because the handler still needs
			// to drive the stream to completion (the adapter is
			// guaranteed to return promptly on ctx.Done()).
			_ = wsjson.Write(ctx, conn, tokenFrame{Type: tokenFrameType, Text: visible})
		},
		nil,
	)
	if err != nil {
		msg := classifyError(err, h.cfg)
		_ = wsjson.Write(ctx, conn, errorFrame{Type: errorFrameType, Message: msg})
		return err
	}

	// Persist the successful turn back into the session. We only
	// append on success — a failed turn (e.g. provider 401) must
	// not leak the user's question into the recorded history,
	// because the next turn's LLM call would replay it.
	h.store.Append(sess.ID, llm.Message{Role: "user", Content: inbound.Text})
	if resp != nil && resp.Content != "" {
		h.store.Append(sess.ID, llm.Message{Role: "assistant", Content: resp.Content})
	}

	usage := llm.Usage{}
	if resp != nil {
		usage = resp.Usage
	}
	_ = wsjson.Write(ctx, conn, doneFrame{Type: doneFrameType, Usage: usage})
	return nil
}

// buildMessages composes the message list sent to the LLM: optional
// system prompt, the session's accumulated history (in order), and
// the new user turn.
//
// The system prompt is sourced from cfg.LLM.SystemPrompt; the
// Anthropic adapter lifts system-role messages into its top-level
// `system` field automatically (see internal/llm/anthropic.go).
//
// History MUST come from the server-side session.Store (Feature 17
// Task 4 / design plan KD-1); the WebSocket handler no longer
// trusts inbound.History.
func buildMessages(systemPrompt string, history []llm.Message, userText string) []llm.Message {
	out := make([]llm.Message, 0, len(history)+2)
	if systemPrompt != "" {
		out = append(out, llm.Message{Role: "system", Content: systemPrompt})
	}
	out = append(out, history...)
	out = append(out, llm.Message{Role: "user", Content: userText})
	return out
}
