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
)

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
// connection. History is the conversation so far; the browser is the
// canonical owner of session state for the MVP (server-side session
// state is an Evolution Seam — see docs/ARCHITECTURE.md §10).
type turnFrame struct {
	Type    string        `json:"type"`
	Text    string        `json:"text"`
	History []llm.Message `json:"history,omitempty"`
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
// independent goroutine with its own derived context.
type WebSocketHandler struct {
	cfg    *config.Config
	client llm.LLMClient
}

// NewWebSocketHandler returns a handler bound to cfg and client.
// Callers are expected to construct cfg via internal/config and
// client via llm.NewLLMClient at startup.
func NewWebSocketHandler(cfg *config.Config, client llm.LLMClient) *WebSocketHandler {
	return &WebSocketHandler{cfg: cfg, client: client}
}

// ServeHTTP performs the WebSocket upgrade and drives a single turn
// per connection. Errors during the turn are reported back to the
// browser as a sanitised error frame; the server never panics out
// of this method even on hostile inputs.
func (h *WebSocketHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
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

	if err := h.handleTurn(ctx, conn); err != nil {
		log.Printf("ws: turn aborted: %v", err)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// handleTurn reads the inbound turn frame, drives CompleteStream,
// and writes token / done frames on success or a single error frame
// on failure. Returning a non-nil error means the turn did not
// complete successfully — the caller uses this only for log
// telemetry, not for further protocol action.
func (h *WebSocketHandler) handleTurn(ctx context.Context, conn *websocket.Conn) error {
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

	req := llm.CompletionRequest{
		Model:    h.cfg.LLM.Model,
		Messages: buildMessages(h.cfg.LLM.SystemPrompt, inbound.History, inbound.Text),
	}

	resp, err := h.client.CompleteStream(ctx, req,
		func(tok string) {
			// Token writes are best-effort — if the browser
			// disconnects mid-stream, the next write fails and the
			// adapter's ctx-cancellation path takes over. We do not
			// surface the error here because the handler still needs
			// to drive the stream to completion (the adapter is
			// guaranteed to return promptly on ctx.Done()).
			_ = wsjson.Write(ctx, conn, tokenFrame{Type: tokenFrameType, Text: tok})
		},
		nil,
	)
	if err != nil {
		msg := classifyError(err, h.cfg)
		_ = wsjson.Write(ctx, conn, errorFrame{Type: errorFrameType, Message: msg})
		return err
	}

	usage := llm.Usage{}
	if resp != nil {
		usage = resp.Usage
	}
	_ = wsjson.Write(ctx, conn, doneFrame{Type: doneFrameType, Usage: usage})
	return nil
}

// buildMessages composes the message list sent to the LLM: optional
// system prompt, the history from the browser (in order), and the
// new user turn.
//
// The system prompt is sourced from cfg.LLM.SystemPrompt; the
// Anthropic adapter lifts system-role messages into its top-level
// `system` field automatically (see internal/llm/anthropic.go).
func buildMessages(systemPrompt string, history []llm.Message, userText string) []llm.Message {
	out := make([]llm.Message, 0, len(history)+2)
	if systemPrompt != "" {
		out = append(out, llm.Message{Role: "system", Content: systemPrompt})
	}
	out = append(out, history...)
	out = append(out, llm.Message{Role: "user", Content: userText})
	return out
}
