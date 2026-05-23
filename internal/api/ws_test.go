package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
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
)

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

	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, wsScheme(srv.URL), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.CloseNow() }()

	if err := wsjson.Write(ctx, conn, turnFrame{
		Type: "turn",
		Text: "hi there",
		History: []llm.Message{
			{Role: "user", Content: "previous user"},
			{Role: "assistant", Content: "previous reply"},
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

	// Verify the LLM saw the system prompt + history + new user turn.
	wantMsgs := []llm.Message{
		{Role: "system", Content: "be terse"},
		{Role: "user", Content: "previous user"},
		{Role: "assistant", Content: "previous reply"},
		{Role: "user", Content: "hi there"},
	}
	got := stub.lastRequest()
	if !reflect.DeepEqual(got.Messages, wantMsgs) {
		t.Errorf("LLM messages: got %+v, want %+v", got.Messages, wantMsgs)
	}
	if got.Model != "test-model" {
		t.Errorf("model: got %q, want test-model", got.Model)
	}
}

func TestWebSocketHandler_NoSystemPrompt_OmitsSystemMessage(t *testing.T) {
	stub := &stubLLMClient{tokens: []string{"x"}}
	cfg := config.Defaults()
	cfg.LLM.SystemPrompt = ""
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
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
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
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
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
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
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
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
	srv := httptest.NewServer(NewWebSocketHandler(config.Defaults(), stub))
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
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
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
	srv := httptest.NewServer(NewWebSocketHandler(cfg, stub))
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
	srv := NewServer(cfg, stub)

	wantAddr := ":" + strconv.Itoa(config.DefaultServerPort)
	if srv.Addr != wantAddr {
		t.Errorf("Addr: got %q, want %q", srv.Addr, wantAddr)
	}
	if srv.ReadHeaderTimeout != readHeaderTimeout {
		t.Errorf("ReadHeaderTimeout: got %v", srv.ReadHeaderTimeout)
	}

	httpSrv := httptest.NewServer(srv.Handler)
	defer httpSrv.Close()

	// `/` should 204.
	resp, err := http.Get(httpSrv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("GET /: status %d, want 204", resp.StatusCode)
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
