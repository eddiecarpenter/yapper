package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"

	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
	"github.com/eddiecarpenter/yapper/internal/session"
)

// newTestStore returns a fresh in-memory session store with a
// generous TTL so eviction never interferes with the
// per-test-case assertions. Tests that need to exercise eviction
// instantiate NewMemoryStore directly.
func newTestStore(t *testing.T) session.Store {
	t.Helper()
	s := session.NewMemoryStore(1*time.Hour, 1*time.Hour)
	t.Cleanup(s.Close)
	return s
}

// stubLLMClient is a deterministic LLMClient used by the WebSocket
// E2E tests. It records the request it received and either streams a
// configured token sequence (and optional usage) or returns a fixed
// error. The mutex makes the recorded request safe to inspect from
// the test goroutine after the handler returns.
type stubLLMClient struct {
	mu         sync.Mutex
	tokens     []string
	usage      llm.Usage
	err        error
	gotRequest llm.CompletionRequest
}

func (s *stubLLMClient) Complete(_ context.Context, req llm.CompletionRequest) (*llm.CompletionResponse, error) {
	s.mu.Lock()
	s.gotRequest = req
	err := s.err
	tokens := s.tokens
	usage := s.usage
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return &llm.CompletionResponse{Content: strings.Join(tokens, ""), Usage: usage}, nil
}

func (s *stubLLMClient) CompleteStream(_ context.Context, req llm.CompletionRequest,
	onToken func(string), onUsage func(llm.Usage)) (*llm.CompletionResponse, error) {
	s.mu.Lock()
	s.gotRequest = req
	err := s.err
	tokens := s.tokens
	usage := s.usage
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	for _, t := range tokens {
		if onToken != nil {
			onToken(t)
		}
	}
	if onUsage != nil && usage != (llm.Usage{}) {
		onUsage(usage)
	}
	return &llm.CompletionResponse{Content: strings.Join(tokens, ""), Usage: usage}, nil
}

func (s *stubLLMClient) lastRequest() llm.CompletionRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gotRequest
}

// framePayload is the union shape of every outbound frame. Token /
// done / error are distinguished by Type; the other fields are
// populated as needed.
type framePayload struct {
	Type    string    `json:"type"`
	Text    string    `json:"text,omitempty"`
	Message string    `json:"message,omitempty"`
	Usage   llm.Usage `json:"usage"`
}

func wsScheme(httpURL string) string {
	return "ws" + strings.TrimPrefix(httpURL, "http") + "/"
}

// readFrames drains the connection until a `done` or `error` frame
// arrives (or the read fails). Helpful for keeping per-test logic
// short — the assertions follow the returned slice.
func readFrames(ctx context.Context, t *testing.T, conn *websocket.Conn) []framePayload {
	t.Helper()
	var out []framePayload
	for {
		var f framePayload
		if err := wsjson.Read(ctx, conn, &f); err != nil {
			return out
		}
		out = append(out, f)
		if f.Type == doneFrameType || f.Type == errorFrameType {
			return out
		}
	}
}

func TestWebSocketHandler_HappyPath_StreamsTokensThenDone(t *testing.T) {
	stub := &stubLLMClient{
		tokens: []string{"Hel", "lo, ", "world!"},
		usage:  llm.Usage{Input: 4, Output: 6},
	}
	cfg := config.Defaults()
	cfg.LLM.SystemPrompt = "be terse"
	cfg.LLM.Model = "test-model"

	store := newTestStore(t)
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, store))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	// Browser-supplied History is now IGNORED (Feature 17 Task 4)
	// — we send it to prove the server discards it, then assert
	// the LLM only saw the session's (empty) history plus the new
	// user turn.
	if err := wsjson.Write(ctx, conn, turnFrame{
		Type: "turn",
		Text: "hi there",
		History: []llm.Message{
			{Role: "user", Content: "browser-supplied-IGNORED"},
			{Role: "assistant", Content: "browser-supplied-IGNORED"},
		},
	}); err != nil {
		t.Fatalf("write turn: %v", err)
	}

	frames := readFrames(ctx, t, conn)

	var tokens []string
	var doneFrames int
	var usage llm.Usage
	for _, f := range frames {
		switch f.Type {
		case tokenFrameType:
			tokens = append(tokens, f.Text)
		case doneFrameType:
			doneFrames++
			usage = f.Usage
		case errorFrameType:
			t.Fatalf("unexpected error frame: %q", f.Message)
		}
	}
	if !reflect.DeepEqual(tokens, stub.tokens) {
		t.Errorf("tokens: got %v, want %v", tokens, stub.tokens)
	}
	if doneFrames != 1 {
		t.Errorf("done frames: got %d, want exactly 1", doneFrames)
	}
	if usage != stub.usage {
		t.Errorf("usage: got %+v, want %+v", usage, stub.usage)
	}

	// Verify the LLM saw the system prompt + EMPTY session history
	// + the new user turn — NOT the browser-supplied history.
	wantMsgs := []llm.Message{
		{Role: "system", Content: "be terse"},
		{Role: "user", Content: "hi there"},
	}
	got := stub.lastRequest()
	if !reflect.DeepEqual(got.Messages, wantMsgs) {
		t.Errorf("LLM messages: got %+v, want %+v", got.Messages, wantMsgs)
	}
	if got.Model != "test-model" {
		t.Errorf("model: got %q, want test-model", got.Model)
	}
	// Confirm any browser-supplied history string never made it
	// into the LLM request — the dedicated marker would surface
	// here if the server had passed inbound.History through.
	for _, m := range got.Messages {
		if strings.Contains(m.Content, "IGNORED") {
			t.Errorf("server leaked browser-supplied history into LLM request: %+v", m)
		}
	}
}

func TestWebSocketHandler_NoSystemPrompt_OmitsSystemMessage(t *testing.T) {
	stub := &stubLLMClient{tokens: []string{"x"}}
	cfg := config.Defaults()
	cfg.LLM.SystemPrompt = ""
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, newTestStore(t)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()
	_ = wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "hi"})
	_ = readFrames(ctx, t, conn)

	got := stub.lastRequest()
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Errorf("expected exactly one user message, got %+v", got.Messages)
	}
}

func TestWebSocketHandler_OllamaUnreachable_SendsActionableErrorAndStaysAlive(t *testing.T) {
	stub := &stubLLMClient{
		err: errors.New(`openai: do stream request: Post "http://localhost:11434/v1/chat/completions": dial tcp 127.0.0.1:11434: connect: connection refused`),
	}
	cfg := config.Defaults()
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, newTestStore(t)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if err := wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "hi"}); err != nil {
		t.Fatalf("write: %v", err)
	}
	frames := readFrames(ctx, t, conn)
	if len(frames) != 1 || frames[0].Type != errorFrameType {
		t.Fatalf("expected exactly one error frame, got %+v", frames)
	}
	if !strings.Contains(frames[0].Message, "Ollama unreachable") ||
		!strings.Contains(frames[0].Message, "ollama serve") {
		t.Errorf("error message not actionable: %q", frames[0].Message)
	}

	// Verify the server still accepts new connections (i.e. the
	// previous turn's failure did not crash the listener).
	conn2, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("second dial failed — server may have crashed: %v", err)
	}
	defer func() { _ = conn2.CloseNow() }()
	if err := wsjson.Write(ctx, conn2, turnFrame{Type: "turn", Text: "again"}); err != nil {
		t.Fatalf("second write: %v", err)
	}
	frames2 := readFrames(ctx, t, conn2)
	if len(frames2) == 0 {
		t.Errorf("second connection got no frames — server is not responsive")
	}
}

func TestWebSocketHandler_HTTP401_SendsAPIKeyActionableError(t *testing.T) {
	stub := &stubLLMClient{
		err: errors.New("openai: http 401: Incorrect API key"),
	}
	cfg := config.Defaults()
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, newTestStore(t)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()
	_ = wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "hi"})

	frames := readFrames(ctx, t, conn)
	if len(frames) != 1 || frames[0].Type != errorFrameType {
		t.Fatalf("frames: %+v", frames)
	}
	if !strings.Contains(frames[0].Message, "Missing or invalid API key") {
		t.Errorf("expected actionable 401 message, got %q", frames[0].Message)
	}
}

func TestWebSocketHandler_HTTP404ModelMissing_SendsModelPullMessage(t *testing.T) {
	stub := &stubLLMClient{
		err: errors.New("openai: http 404: model not found"),
	}
	cfg := config.Defaults()
	cfg.LLM.Model = "missing-model"
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, newTestStore(t)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()
	_ = wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "hi"})

	frames := readFrames(ctx, t, conn)
	if len(frames) != 1 || frames[0].Type != errorFrameType {
		t.Fatalf("frames: %+v", frames)
	}
	if !strings.Contains(frames[0].Message, "missing-model") ||
		!strings.Contains(frames[0].Message, "ollama pull") {
		t.Errorf("expected actionable model-not-found message, got %q", frames[0].Message)
	}
}

func TestWebSocketHandler_BadFrameType_SendsErrorFrame(t *testing.T) {
	stub := &stubLLMClient{tokens: []string{"x"}}
	srv := httptest.NewServer(NewWebSocketHandler(config.Defaults(), stub, newTestStore(t)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()
	_ = wsjson.Write(ctx, conn, map[string]any{"type": "garbage", "text": "x"})

	frames := readFrames(ctx, t, conn)
	if len(frames) != 1 || frames[0].Type != errorFrameType {
		t.Fatalf("frames: %+v", frames)
	}
	if !strings.Contains(frames[0].Message, "turn") {
		t.Errorf("error message should mention the expected frame type, got %q", frames[0].Message)
	}
}

func TestWebSocketHandler_APIKeyNotLeakedToBrowser(t *testing.T) {
	const secret = "sk-this-should-never-leak-987654321"
	stub := &stubLLMClient{
		err: errors.New("openai: http 500: opaque-failure including the bare key " + secret + " in upstream body"),
	}
	cfg := config.Defaults()
	cfg.LLM.APIKey = secret
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, newTestStore(t)))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()
	_ = wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "hi"})

	frames := readFrames(ctx, t, conn)
	if len(frames) != 1 {
		t.Fatalf("expected exactly one error frame, got %+v", frames)
	}
	if strings.Contains(frames[0].Message, secret) {
		t.Errorf("API key leaked to browser inside error frame: %q", frames[0].Message)
	}
}

func TestWebSocketHandler_TimeoutError_SendsTimeoutMessage(t *testing.T) {
	stub := &stubLLMClient{
		err: context.DeadlineExceeded,
	}
	cfg := config.Defaults()
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, newTestStore(t)))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()
	_ = wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "hi"})
	frames := readFrames(ctx, t, conn)
	if len(frames) != 1 || frames[0].Type != errorFrameType {
		t.Fatalf("frames: %+v", frames)
	}
	if !strings.Contains(frames[0].Message, "timed out") {
		t.Errorf("expected timeout message, got %q", frames[0].Message)
	}
}

func TestNewServer_MountsWSAndRoot(t *testing.T) {
	stub := &stubLLMClient{}
	cfg := config.Defaults()
	srv := NewServer(cfg, stub, newTestStore(t))

	wantAddr := ":" + strconv.Itoa(config.DefaultServerPort)
	if srv.Addr != wantAddr {
		t.Errorf("Addr: got %q, want %q", srv.Addr, wantAddr)
	}
	if srv.ReadHeaderTimeout != readHeaderTimeout {
		t.Errorf("ReadHeaderTimeout: got %v", srv.ReadHeaderTimeout)
	}

	httpSrv := httptest.NewServer(srv.Handler)
	defer httpSrv.Close()

	// `/` is now backed by the embedded SPA handler. When the SPA
	// is built (web/dist/index.html present in the embed), this
	// returns 200; in a CI environment where the TS bundler has
	// not run, the embed is empty and the handler returns 404.
	// Both shapes are acceptable here — what this assertion guards
	// is that the route is REGISTERED (not 5xx, not panicking).
	resp, err := http.Get(httpSrv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNotFound {
		t.Errorf("GET /: status %d, want 200 (SPA built) or 404 (empty embed)", resp.StatusCode)
	}
	resp.Body.Close()

	// `/ws` is routed to the WebSocket handler. A plain HTTP GET
	// without upgrade headers returns a non-2xx (the handler rejects
	// the missing upgrade) — what we verify is that it is NOT 404,
	// which would mean the route isn't registered.
	resp, err = http.Get(httpSrv.URL + "/ws")
	if err != nil {
		t.Fatalf("GET /ws: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		t.Errorf("/ws is not routed; got 404")
	}
}

// TestWebSocketHandler_SessionCookie_MintedWhenAbsent verifies that
// the first WebSocket upgrade with no `yapper_session` cookie
// receives a Set-Cookie carrying the freshly minted session id, and
// that subsequent connections without a cookie mint DIFFERENT ids.
func TestWebSocketHandler_SessionCookie_MintedWhenAbsent(t *testing.T) {
	stub := &stubLLMClient{tokens: []string{"x"}}
	srv := httptest.NewServer(NewWebSocketHandler(config.Defaults(), stub, newTestStore(t)))
	defer srv.Close()

	mintedID := func() string {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		// Use websocket.Dial with HTTPClient defaulting (no
		// cookie jar) so each call starts from a clean slate.
		conn, resp, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
		if err != nil {
			t.Fatalf("dial: %v", err)
		}
		defer func() { _ = conn.CloseNow() }()
		// The Set-Cookie header lands on the upgrade response.
		var got string
		for _, c := range resp.Cookies() {
			if c.Name == sessionCookieName {
				got = c.Value
				break
			}
		}
		if got == "" {
			t.Fatalf("no %s cookie on upgrade response (cookies=%v)", sessionCookieName, resp.Cookies())
		}
		// Drive one turn so the handler exits cleanly without
		// log spam from the unread frame.
		_ = wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: "x"})
		_ = readFrames(ctx, t, conn)
		return got
	}

	id1 := mintedID()
	id2 := mintedID()
	if id1 == id2 {
		t.Errorf("expected distinct session ids on two cookie-less connections, both got %q", id1)
	}
	// IDs are the 256-bit base64-url-no-pad shape (43 chars).
	for _, id := range []string{id1, id2} {
		if len(id) != 43 || strings.ContainsAny(id, "+/=") {
			t.Errorf("unexpected session id shape: %q (want 43-char base64-url-no-pad)", id)
		}
	}
}

// TestWebSocketHandler_SessionCookie_MultiTurnHistory verifies that
// two turns on the SAME cookie produce a server-recorded history,
// and a third turn (re-using the cookie) sees that history reflected
// in the LLM call's Messages — proving the server is the canonical
// owner of conversation state and that the cookie is the lookup
// key.
func TestWebSocketHandler_SessionCookie_MultiTurnHistory(t *testing.T) {
	stub := &stubLLMClient{
		tokens: []string{"reply"},
	}
	cfg := config.Defaults()
	cfg.LLM.SystemPrompt = ""
	store := newTestStore(t)
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub, store))
	defer srv.Close()

	jar, _ := cookiejar.New(nil)
	httpClient := &http.Client{Jar: jar}

	// driveTurn opens a WebSocket carrying any cookies the jar
	// already holds for srv.URL, sends one turn, and waits for
	// done. The jar accumulates the Set-Cookie that lands on the
	// first upgrade, so the second and third dial reuse it.
	driveTurn := func(text string) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), &websocket.DialOptions{
			HTTPClient: httpClient,
		})
		if err != nil {
			t.Fatalf("dial(%s): %v", text, err)
		}
		defer func() { _ = conn.CloseNow() }()
		if err := wsjson.Write(ctx, conn, turnFrame{Type: "turn", Text: text}); err != nil {
			t.Fatalf("write(%s): %v", text, err)
		}
		_ = readFrames(ctx, t, conn)
	}

	driveTurn("first user")
	driveTurn("second user")
	driveTurn("third user")

	// The cookie jar accumulated exactly one yapper_session
	// cookie, so all three turns hit the same Session in the
	// store.
	httpURL, _ := url.Parse(srv.URL)
	cookies := jar.Cookies(httpURL)
	var sid string
	for _, c := range cookies {
		if c.Name == sessionCookieName {
			sid = c.Value
			break
		}
	}
	if sid == "" {
		t.Fatalf("no %s in cookie jar after three turns", sessionCookieName)
	}

	got, ok := store.Get(sid)
	if !ok {
		t.Fatalf("session %s missing from store after three turns", sid)
	}
	// Three turns × (user + assistant) = 6 messages.
	if len(got.History) != 6 {
		t.Fatalf("history length: got %d, want 6 (3 turns × 2 messages)", len(got.History))
	}
	wantRoles := []string{"user", "assistant", "user", "assistant", "user", "assistant"}
	wantUserContents := []string{"first user", "second user", "third user"}
	userIdx := 0
	for i, m := range got.History {
		if m.Role != wantRoles[i] {
			t.Errorf("history[%d].Role: got %q, want %q", i, m.Role, wantRoles[i])
		}
		if m.Role == "user" {
			if m.Content != wantUserContents[userIdx] {
				t.Errorf("user-turn %d: got %q, want %q", userIdx, m.Content, wantUserContents[userIdx])
			}
			userIdx++
		}
	}

	// The LAST LLM call should have seen the history accumulated
	// across the first two turns (4 messages) plus the third
	// user turn — five total, since cfg.LLM.SystemPrompt is "".
	lastReq := stub.lastRequest()
	if len(lastReq.Messages) != 5 {
		t.Errorf("third-turn LLM messages: got %d, want 5 (2 turns of history + new user)",
			len(lastReq.Messages))
	}
	if lastReq.Messages[len(lastReq.Messages)-1].Content != "third user" {
		t.Errorf("third-turn LLM should end with the new user turn, got %+v",
			lastReq.Messages[len(lastReq.Messages)-1])
	}
}

func TestClassifyError_TableDriven(t *testing.T) {
	cfg := config.Defaults()
	cfg.LLM.Model = "llama3.2:3b"
	cfg.LLM.BaseURL = "http://localhost:11434/v1"

	cases := []struct {
		name    string
		err     error
		contain []string
	}{
		{
			"connection refused → ollama unreachable",
			errors.New("dial tcp 127.0.0.1:11434: connect: connection refused"),
			[]string{"Ollama unreachable", "ollama serve", cfg.LLM.BaseURL},
		},
		{
			"no such host → ollama unreachable",
			errors.New(`Get "http://noresolve.local/": dial tcp: lookup noresolve.local: no such host`),
			[]string{"Ollama unreachable"},
		},
		{
			"http 404 → model not pulled",
			errors.New("openai: http 404: model not found"),
			[]string{"llama3.2:3b", "ollama pull"},
		},
		{
			"http 401 → API key",
			errors.New("openai: http 401: Incorrect API key"),
			[]string{"YAPPER_LLM_API_KEY"},
		},
		{
			"context.DeadlineExceeded → timeout",
			context.DeadlineExceeded,
			[]string{"timed out"},
		},
		{
			"context.Canceled → cancelled",
			context.Canceled,
			[]string{"cancelled"},
		},
		{
			"fallback path quotes sanitised error",
			errors.New("opaque: failure"),
			[]string{"LLM request failed", "opaque: failure"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := classifyError(tc.err, cfg)
			for _, want := range tc.contain {
				if !strings.Contains(got, want) {
					t.Errorf("missing %q in result %q", want, got)
				}
			}
		})
	}
}

func TestClassifyError_NilError_ReturnsEmpty(t *testing.T) {
	if got := classifyError(nil, config.Defaults()); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

func TestSanitiseErrorText_RedactsKey(t *testing.T) {
	const key = "sk-ant-leak-12345"
	in := "upstream said " + key + " is wrong"
	out := sanitiseErrorText(in, key)
	if strings.Contains(out, key) {
		t.Errorf("key not redacted: %q", out)
	}
}

func TestSanitiseErrorText_EmptyKey_NoOp(t *testing.T) {
	in := "the message"
	if got := sanitiseErrorText(in, ""); got != in {
		t.Errorf("got %q, want %q", got, in)
	}
}
