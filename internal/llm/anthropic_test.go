package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/eddiecarpenter/yapper/internal/config"
)

func TestNewLLMClient_AnthropicHost_ReturnsAnthropicClient(t *testing.T) {
	cases := []string{
		"https://api.anthropic.com",
		"https://api.anthropic.com/v1",
		"https://staging.anthropic.com/v1",
		"https://API.ANTHROPIC.COM", // case-insensitive
	}
	for _, u := range cases {
		t.Run(u, func(t *testing.T) {
			cli, err := NewLLMClient(config.LLMConfig{BaseURL: u, APIKey: "sk-ant-x"})
			if err != nil {
				t.Fatalf("NewLLMClient: %v", err)
			}
			if _, ok := cli.(*AnthropicLLMClient); !ok {
				t.Errorf("got %T, want *AnthropicLLMClient", cli)
			}
		})
	}
}

func TestNewLLMClient_NonAnthropic_ReturnsOpenAIClient(t *testing.T) {
	cases := []string{
		"http://localhost:11434/v1",
		"https://api.openai.com/v1",
		"https://api.groq.com/openai/v1",
		"https://openrouter.ai/api/v1",
		"http://127.0.0.1:11434/v1",
	}
	for _, u := range cases {
		t.Run(u, func(t *testing.T) {
			cli, err := NewLLMClient(config.LLMConfig{BaseURL: u})
			if err != nil {
				t.Fatalf("NewLLMClient: %v", err)
			}
			if _, ok := cli.(*OpenAILLMClient); !ok {
				t.Errorf("got %T, want *OpenAILLMClient", cli)
			}
		})
	}
}

func TestNewLLMClient_EmptyBaseURL_ReturnsError(t *testing.T) {
	_, err := NewLLMClient(config.LLMConfig{})
	if err == nil {
		t.Fatal("expected error for empty base URL")
	}
}

func TestNewLLMClient_PropagatesAPIKey(t *testing.T) {
	cli, err := NewLLMClient(config.LLMConfig{
		BaseURL: "https://api.anthropic.com",
		APIKey:  "sk-ant-secretvalue",
	})
	if err != nil {
		t.Fatalf("NewLLMClient: %v", err)
	}
	a, ok := cli.(*AnthropicLLMClient)
	if !ok {
		t.Fatalf("got %T", cli)
	}
	if a.APIKey != "sk-ant-secretvalue" {
		t.Errorf("APIKey: got %q", a.APIKey)
	}
}

// anthropicSSEWriter is a small helper that serialises SSE events
// the way Anthropic's /v1/messages stream actually does.
type anthropicSSEWriter struct {
	w http.ResponseWriter
	f http.Flusher
}

func newAnthropicSSE(t *testing.T, w http.ResponseWriter) *anthropicSSEWriter {
	t.Helper()
	w.Header().Set("Content-Type", "text/event-stream")
	flusher, ok := w.(http.Flusher)
	if !ok {
		t.Fatalf("response writer is not a Flusher")
	}
	return &anthropicSSEWriter{w: w, f: flusher}
}

func (s *anthropicSSEWriter) event(name string, payload any) {
	data, _ := json.Marshal(payload)
	fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", name, data)
	s.f.Flush()
}

func TestAnthropicCompleteStream_HappyPath_DeltasAndUsage(t *testing.T) {
	deltas := []string{"Hello", ", ", "world"}

	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request shape.
		if got := r.Header.Get("anthropic-version"); got != anthropicVersion {
			t.Errorf("anthropic-version: got %q, want %q", got, anthropicVersion)
		}
		if got := r.Header.Get("x-api-key"); got != "sk-ant-test" {
			t.Errorf("x-api-key: got %q", got)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization must be empty when x-api-key is set, got %q", got)
		}
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("Accept: got %q", got)
		}

		var got anthropicMessagesRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode: %v", err)
		}
		if got.System != "be terse" {
			t.Errorf("System: got %q, want %q (system-role message must be lifted to top-level System)", got.System, "be terse")
		}
		for _, m := range got.Messages {
			if m.Role == "system" {
				t.Errorf("system message leaked into Messages array: %+v", m)
			}
		}
		if !got.Stream {
			t.Errorf("Stream: want true")
		}
		if got.MaxTokens <= 0 {
			t.Errorf("MaxTokens must be set, got %d", got.MaxTokens)
		}

		sse := newAnthropicSSE(t, w)
		sse.event("message_start", map[string]any{"type": "message_start"})
		sse.event("content_block_start", map[string]any{"type": "content_block_start", "index": 0})
		for _, d := range deltas {
			sse.event("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{"type": "text_delta", "text": d},
			})
		}
		sse.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		sse.event("message_delta", map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn"},
			"usage": map[string]any{"input_tokens": 11, "output_tokens": 8},
		})
		sse.event("message_stop", map[string]any{"type": "message_stop"})
	}))
	defer stub.Close()

	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-test")
	var (
		tokens []string
		usage  Usage
	)
	resp, err := cli.CompleteStream(context.Background(), CompletionRequest{
		Model: "claude-3-haiku-20240307",
		Messages: []Message{
			{Role: "system", Content: "be terse"},
			{Role: "user", Content: "hi"},
		},
	}, func(t string) { tokens = append(tokens, t) }, func(u Usage) { usage = u })
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	if !equalStrings(tokens, deltas) {
		t.Errorf("tokens: got %v, want %v", tokens, deltas)
	}
	if usage.Input != 11 || usage.Output != 8 {
		t.Errorf("Usage: got %+v, want {11 8}", usage)
	}
	if resp.Content != strings.Join(deltas, "") {
		t.Errorf("Content: got %q, want %q", resp.Content, strings.Join(deltas, ""))
	}
}

func TestAnthropicCompleteStream_OAuthToken_UsesBearerHeader(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "" {
			t.Errorf("x-api-key must be empty for OAuth flow, got %q", got)
		}
		if got, want := r.Header.Get("Authorization"), "Bearer oauth-jwt-token"; got != want {
			t.Errorf("Authorization: got %q, want %q", got, want)
		}
		sse := newAnthropicSSE(t, w)
		sse.event("message_stop", map[string]any{"type": "message_stop"})
	}))
	defer stub.Close()

	cli := NewAnthropicLLMClient(stub.URL, "oauth-jwt-token")
	_, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "claude", Messages: []Message{{Role: "user", Content: "hi"}}},
		nil, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
}

func TestAnthropicCompleteStream_NoAuth_WhenKeyEmpty(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "" {
			t.Errorf("x-api-key must be empty, got %q", got)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization must be empty, got %q", got)
		}
		sse := newAnthropicSSE(t, w)
		sse.event("message_stop", map[string]any{"type": "message_stop"})
	}))
	defer stub.Close()

	cli := NewAnthropicLLMClient(stub.URL, "")
	if _, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "claude"}, nil, nil); err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
}

func TestAnthropicCompleteStream_IgnoresUnknownEvents(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		sse := newAnthropicSSE(t, w)
		sse.event("ping", map[string]any{})
		sse.event("message_start", map[string]any{"type": "message_start"})
		sse.event("content_block_start", map[string]any{"type": "content_block_start"})
		sse.event("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{"type": "text_delta", "text": "hi"},
		})
		sse.event("totally_unknown", map[string]any{"weird": "future event"})
		sse.event("content_block_stop", map[string]any{"type": "content_block_stop"})
		sse.event("message_stop", map[string]any{"type": "message_stop"})
	}))
	defer stub.Close()

	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-x")
	var tokens []string
	resp, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "x"}, func(t string) { tokens = append(tokens, t) }, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if !equalStrings(tokens, []string{"hi"}) {
		t.Errorf("tokens: got %v, want [hi] (unknown events must be dropped silently)", tokens)
	}
	if resp.Content != "hi" {
		t.Errorf("Content: got %q, want %q", resp.Content, "hi")
	}
}

func TestAnthropicCompleteStream_HTTPError_QuotesBody(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"type":"invalid_request_error","message":"max_tokens missing"}}`))
	}))
	defer stub.Close()

	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-x")
	_, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "x"}, nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error must include HTTP status: %v", err)
	}
	if !strings.Contains(err.Error(), "max_tokens missing") {
		t.Errorf("error must echo upstream body: %v", err)
	}
}

func TestAnthropicCompleteStream_ContextCancellation_ReturnsCtxErr(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse := newAnthropicSSE(t, w)
		for i := 0; i < 200; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			sse.event("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": 0,
				"delta": map[string]any{"type": "text_delta", "text": "x"},
			})
			time.Sleep(15 * time.Millisecond)
		}
	}))
	defer stub.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-x")
	_, err := cli.CompleteStream(ctx, CompletionRequest{Model: "x"}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestAnthropicCompleteStream_MalformedDelta_ReturnsParseError(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		fmt.Fprint(w, "event: content_block_delta\ndata: {not-valid-json}\n\n")
		f.Flush()
		fmt.Fprint(w, "event: message_stop\ndata: {}\n\n")
		f.Flush()
	}))
	defer stub.Close()
	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-x")
	_, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "x"}, nil, nil)
	if err == nil {
		t.Fatal("expected parse error")
	}
	if !strings.Contains(err.Error(), "parse content_block_delta") {
		t.Errorf("error must signal parse failure: %v", err)
	}
}

func TestSetAnthropicAuthHeaders_TableDriven(t *testing.T) {
	cases := []struct {
		name       string
		key        string
		wantHeader string
		wantValue  string
	}{
		{"empty omits both headers", "", "", ""},
		{"sk-ant- key uses x-api-key", "sk-ant-1234", "x-api-key", "sk-ant-1234"},
		{"oauth jwt uses Authorization Bearer", "eyJ0eXAi.oauth.token", "Authorization", "Bearer eyJ0eXAi.oauth.token"},
		{"opaque key uses Authorization Bearer", "opaque-thing", "Authorization", "Bearer opaque-thing"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "http://x", nil)
			SetAnthropicAuthHeaders(req, tc.key)
			if tc.wantHeader == "" {
				if got := req.Header.Get("x-api-key"); got != "" {
					t.Errorf("x-api-key should be empty, got %q", got)
				}
				if got := req.Header.Get("Authorization"); got != "" {
					t.Errorf("Authorization should be empty, got %q", got)
				}
				return
			}
			if got := req.Header.Get(tc.wantHeader); got != tc.wantValue {
				t.Errorf("%s: got %q, want %q", tc.wantHeader, got, tc.wantValue)
			}
		})
	}
}

func TestAnthropicComplete_NonStreaming_PicksUpContentAndUsage(t *testing.T) {
	const want = "Bonjour"
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got anthropicMessagesRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode: %v", err)
		}
		if got.Stream {
			t.Errorf("Complete must set Stream=false; got Stream=true")
		}
		body := anthropicMessagesResponse{
			Content: []anthropicContentBlock{{Type: "text", Text: want}},
			Usage:   anthropicUsage{InputTokens: 5, OutputTokens: 3},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	defer stub.Close()
	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-x")
	resp, err := cli.Complete(context.Background(), CompletionRequest{
		Model:    "claude-3",
		Messages: []Message{{Role: "user", Content: "Hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != want {
		t.Errorf("Content: got %q, want %q", resp.Content, want)
	}
	if resp.Usage.Input != 5 || resp.Usage.Output != 3 {
		t.Errorf("Usage: got %+v, want {5 3}", resp.Usage)
	}
}

func TestAnthropicCompleteStream_AggregatesMultipleSystemMessages(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got anthropicMessagesRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode: %v", err)
		}
		if got.System != "rule one\nrule two" {
			t.Errorf("System: got %q, want %q (multiple system messages must be concatenated)",
				got.System, "rule one\nrule two")
		}
		sse := newAnthropicSSE(t, w)
		sse.event("message_stop", map[string]any{"type": "message_stop"})
	}))
	defer stub.Close()
	cli := NewAnthropicLLMClient(stub.URL, "sk-ant-x")
	_, err := cli.CompleteStream(context.Background(), CompletionRequest{
		Model: "x",
		Messages: []Message{
			{Role: "system", Content: "rule one"},
			{Role: "system", Content: "rule two"},
			{Role: "user", Content: "hi"},
		},
	}, nil, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
}
