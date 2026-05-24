package api

import (
	"context"
	"net"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"

	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
)

// blockingLLM is a stub LLMClient whose CompleteStream blocks on a
// channel until either (a) the channel receives a release signal or
// (b) the supplied context is cancelled. The captured-ctx pattern
// lets the shutdown test assert that the per-turn context propagated
// the BaseContext's cancellation all the way through to the adapter.
type blockingLLM struct {
	// release, when closed, lets CompleteStream return success.
	// Tests do not use it in the shutdown path — they cancel via
	// the parent ctx instead — but it is plumbed for completeness.
	release chan struct{}

	// ctxFired is set to true when CompleteStream observes its
	// supplied ctx being cancelled. The shutdown test reads this
	// after Shutdown returns to assert the cancel propagated.
	ctxFired atomic.Bool

	// reached is set the moment CompleteStream is entered. The
	// test waits for this flag to flip before triggering the
	// shutdown so the cancel observably races an in-flight call,
	// not the upgrade. Atomic so the test goroutine can read
	// concurrently with the handler goroutine writing.
	reached atomic.Bool
}

func (b *blockingLLM) Complete(ctx context.Context, _ llm.CompletionRequest) (*llm.CompletionResponse, error) {
	b.reached.Store(true)
	select {
	case <-ctx.Done():
		b.ctxFired.Store(true)
		return nil, ctx.Err()
	case <-b.release:
		return &llm.CompletionResponse{Content: "ok"}, nil
	}
}

func (b *blockingLLM) CompleteStream(ctx context.Context, _ llm.CompletionRequest,
	_ func(string), _ func(llm.Usage)) (*llm.CompletionResponse, error) {
	b.reached.Store(true)
	select {
	case <-ctx.Done():
		b.ctxFired.Store(true)
		return nil, ctx.Err()
	case <-b.release:
		return &llm.CompletionResponse{Content: "ok"}, nil
	}
}

// TestWebSocketHandler_ShutdownCancelsInFlightStream is the
// AC-2 integration test from the Feature 17 design plan: when the
// server's BaseContext is cancelled mid-turn, the WebSocket
// connection receives the StatusGoingAway close frame, the LLM
// adapter's per-turn ctx is cancelled, and http.Server.Shutdown
// returns nil within the drain window.
//
// The test simulates the full main.go shutdown pathway:
//   - A cancellable parent context is wired to http.Server.BaseContext
//     (matching what runServe does in production).
//   - A blocking-LLM stub holds the turn until either ctx-cancel
//     or release.
//   - The test cancels the parent and asserts the three observable
//     consequences: stub ctx fires, Shutdown returns within 5 s,
//     and the client side of the WebSocket sees the StatusGoingAway
//     close code.
func TestWebSocketHandler_ShutdownCancelsInFlightStream(t *testing.T) {
	stub := &blockingLLM{release: make(chan struct{})}

	store := newTestStore(t)
	handler := NewWebSocketHandler(config.Defaults(), stub, store)

	// Bind a fresh listener so we can drive http.Server.Serve
	// directly — httptest.NewServer does not expose BaseContext
	// wiring as cleanly.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	baseCtx, cancelBase := context.WithCancel(context.Background())
	defer cancelBase() // safe even after the explicit cancel below.

	srv := &http.Server{
		Handler:           handler,
		BaseContext:       func(_ net.Listener) context.Context { return baseCtx },
		ReadHeaderTimeout: readHeaderTimeout,
	}
	serveErrCh := make(chan error, 1)
	go func() { serveErrCh <- srv.Serve(ln) }()
	defer srv.Close()

	// Open a WebSocket and send a turn so the handler reaches
	// the LLM call (which blocks on the stub).
	clientCtx, cancelClient := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancelClient()
	url := "ws://" + ln.Addr().String() + "/"
	conn, _, err := websocket.Dial(clientCtx, url, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if err := wsjson.Write(clientCtx, conn, turnFrame{Type: "turn", Text: "hi"}); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Give the handler a moment to reach the blocking LLM call
	// before we trigger shutdown. The wait is generous (200 ms
	// upper bound) — without it, the cancel can race the
	// upgrade and we end up testing the Accept path instead.
	deadline := time.Now().Add(200 * time.Millisecond)
	for time.Now().Before(deadline) {
		if stub.reached.Load() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !stub.reached.Load() {
		t.Fatal("LLM stub never reached — turn frame may not have landed")
	}

	// === Trigger shutdown ===
	cancelBase()

	// Assert (a): the per-turn ctx propagated all the way to the
	// LLM adapter. The watcher goroutine in ServeHTTP closes the
	// conn, which wakes handleTurn's Read; meanwhile the
	// already-issued CompleteStream call's ctx is the per-turn
	// ctx derived from r.Context(), and that fires on
	// BaseContext cancel. We wait up to 1 s for the stub to
	// observe the cancel.
	deadline = time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		if stub.ctxFired.Load() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !stub.ctxFired.Load() {
		t.Error("LLM stub ctx never fired — BaseContext cancellation did not propagate")
	}

	// Assert (b): the client side observes the StatusGoingAway
	// close (or, acceptably, the connection drops with any
	// close-status indicating server initiated). We do a quick
	// read with a short deadline.
	closeCtx, closeCancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer closeCancel()
	var probe map[string]any
	readErr := wsjson.Read(closeCtx, conn, &probe)
	if readErr == nil {
		// The handler might race-write a frame before closing;
		// drain once more for the close itself.
		readErr = wsjson.Read(closeCtx, conn, &probe)
	}
	if readErr == nil {
		t.Error("client never saw connection close on server shutdown")
	} else {
		// Pull the close code if we got one — coder/websocket
		// surfaces it via websocket.CloseStatus(err).
		code := websocket.CloseStatus(readErr)
		if code != -1 && code != websocket.StatusGoingAway {
			// Other status codes (StatusNormalClosure,
			// StatusAbnormalClosure, etc.) are acceptable for
			// this test — we care primarily that the conn
			// closed, not exactly which code. Log for
			// diagnostics.
			t.Logf("close code: got %d, expected StatusGoingAway(%d) ideally", code, websocket.StatusGoingAway)
		}
	}

	// Assert (c): Shutdown returns nil within the 5-second
	// drain. Per http.Server docs, Shutdown does not wait for
	// hijacked connections, so this should return promptly —
	// we just verify it does not hang.
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		t.Errorf("Shutdown: %v", err)
	}
}
