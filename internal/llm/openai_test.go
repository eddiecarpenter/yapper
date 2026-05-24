package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestOpenAIComplete_NonStreaming_PicksUpContentAndUsage(t *testing.T) {
	const want = "Hello world"

	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if got.Stream {
			t.Errorf("Complete must set Stream=false on the outgoing payload; got Stream=true")
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Content-Type: got %q, want application/json", r.Header.Get("Content-Type"))
		}
		body := openAIResponse{
			Choices: []openAIChoice{{Message: Message{Role: "assistant", Content: want}}},
			Usage:   &openAIUsage{PromptTokens: 3, CompletionTokens: 2},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, "")
	resp, err := cli.Complete(context.Background(), CompletionRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != want {
		t.Errorf("Content: got %q, want %q", resp.Content, want)
	}
	if resp.Usage.Input != 3 || resp.Usage.Output != 2 {
		t.Errorf("Usage: got %+v, want {3 2}", resp.Usage)
	}
}

func TestOpenAICompleteStream_RequestShapeAndDeltasAndUsage(t *testing.T) {
	deltas := []string{"Hello", ", ", "world", "!"}

	var (
		seenModel    string
		seenMessages []Message
		seenStream   bool
		seenAuth     string
		seenAccept   string
	)
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got CompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode: %v", err)
		}
		seenModel = got.Model
		seenMessages = got.Messages
		seenStream = got.Stream
		seenAuth = r.Header.Get("Authorization")
		seenAccept = r.Header.Get("Accept")

		w.Header().Set("Content-Type", "text/event-stream")
		f, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("stub response writer does not support Flusher")
		}
		for _, d := range deltas {
			chunk := openAIStreamChunk{
				Choices: []openAIChoice{{Delta: Message{Content: d}}},
			}
			out, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", out)
			f.Flush()
		}
		final := openAIStreamChunk{
			Choices: []openAIChoice{{FinishReason: "stop"}},
			Usage:   &openAIUsage{PromptTokens: 4, CompletionTokens: 7},
		}
		out, _ := json.Marshal(final)
		fmt.Fprintf(w, "data: %s\n\n", out)
		f.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		f.Flush()
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, "sk-test-12345")

	var (
		tokens []string
		usage  Usage
	)
	resp, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "llama3.2:3b", Messages: []Message{{Role: "user", Content: "hi"}}},
		func(t string) { tokens = append(tokens, t) },
		func(u Usage) { usage = u },
	)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	// (a) request body shape
	if seenModel != "llama3.2:3b" {
		t.Errorf("Model: got %q", seenModel)
	}
	if len(seenMessages) != 1 || seenMessages[0].Content != "hi" {
		t.Errorf("Messages: got %+v", seenMessages)
	}
	if !seenStream {
		t.Errorf("Stream: got false, want true")
	}
	if seenAuth != "Bearer sk-test-12345" {
		t.Errorf("Authorization: got %q, want \"Bearer sk-test-12345\"", seenAuth)
	}
	if seenAccept != "text/event-stream" {
		t.Errorf("Accept: got %q, want text/event-stream", seenAccept)
	}

	// (b) onToken called in order, exact set of deltas
	if !equalStrings(tokens, deltas) {
		t.Errorf("tokens: got %v, want %v", tokens, deltas)
	}

	// (c) onUsage called with final values
	if usage.Input != 4 || usage.Output != 7 {
		t.Errorf("Usage: got %+v, want {Input:4 Output:7}", usage)
	}

	// (d) returned CompletionResponse aggregates content + usage
	if got, want := resp.Content, strings.Join(deltas, ""); got != want {
		t.Errorf("Content: got %q, want %q", got, want)
	}
	if resp.Usage != usage {
		t.Errorf("response Usage: got %+v, want %+v", resp.Usage, usage)
	}
}

func TestOpenAICompleteStream_NoAuthHeader_ForOllamaPath(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization: must be empty for unauthenticated provider, got %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, "")
	if _, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "llama3.2:3b"}, nil, nil); err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
}

func TestOpenAICompleteStream_AbsentUsage_TreatedAsZero(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, c := range []openAIStreamChunk{
			{Choices: []openAIChoice{{Delta: Message{Content: "a"}}}},
			{Choices: []openAIChoice{{Delta: Message{Content: "b"}}}},
		} {
			out, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", out)
			f.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, "")
	called := false
	resp, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "x"}, nil, func(Usage) { called = true })
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if called {
		t.Errorf("onUsage must not be called when the provider emits no usage chunk")
	}
	if resp.Usage != (Usage{}) {
		t.Errorf("Usage: got %+v, want zero value", resp.Usage)
	}
	if resp.Content != "ab" {
		t.Errorf("Content: got %q, want %q", resp.Content, "ab")
	}
}

func TestOpenAICompleteStream_HTTPError_QuotesBody(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"Incorrect API key"}}`)
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, "bad-key")
	_, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "x"}, nil, nil)
	if err == nil {
		t.Fatal("expected error from CompleteStream, got nil")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error must include HTTP status: %v", err)
	}
	if !strings.Contains(err.Error(), "Incorrect API key") {
		t.Errorf("error must echo upstream body: %v", err)
	}
}

func TestOpenAICompleteStream_ContextCancellation_ReturnsCtxErr(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for i := 0; i < 200; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			chunk := openAIStreamChunk{Choices: []openAIChoice{{Delta: Message{Content: "x"}}}}
			out, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", out)
			f.Flush()
			time.Sleep(15 * time.Millisecond)
		}
	}))
	defer stub.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	cli := NewOpenAILLMClient(stub.URL, "")
	_, err := cli.CompleteStream(ctx, CompletionRequest{Model: "x"}, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestOpenAICompleteStream_MalformedJSON_ReturnsParseError(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		fmt.Fprint(w, "data: {not-valid-json}\n\n")
		f.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		f.Flush()
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, "")
	_, err := cli.CompleteStream(context.Background(), CompletionRequest{Model: "x"}, nil, nil)
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
	if !strings.Contains(err.Error(), "parse stream chunk") {
		t.Errorf("error must signal a parse failure, got: %v", err)
	}
}

func TestOpenAIComplete_BaseURLTrimmedAtConstruction(t *testing.T) {
	c := NewOpenAILLMClient("http://x/v1/", "")
	if got, want := c.BaseURL, "http://x/v1"; got != want {
		t.Errorf("BaseURL: got %q, want %q", got, want)
	}
}

func TestOpenAIComplete_BaseURLEmpty_ReturnsError(t *testing.T) {
	c := NewOpenAILLMClient("", "")
	_, err := c.Complete(context.Background(), CompletionRequest{Model: "x"})
	if err == nil {
		t.Fatal("expected error for empty base URL")
	}
}

func TestSetAuthHeaders_TableDriven(t *testing.T) {
	cases := []struct {
		name string
		key  string
		want string
	}{
		{"empty key omits header", "", ""},
		{"sk- key produces Bearer", "sk-abc", "Bearer sk-abc"},
		{"opaque key produces Bearer", "anything-goes", "Bearer anything-goes"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodPost, "http://x", nil)
			SetAuthHeaders(req, tc.key)
			if got := req.Header.Get("Authorization"); got != tc.want {
				t.Errorf("Authorization: got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestGetMaskedAPIKey_TableDriven(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"4 chars fully masked", "abcd", "****"},
		{"sk-style key", "sk-secretvalue-9999", "***************9999"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := GetMaskedAPIKey(tc.in); got != tc.want {
				t.Errorf("GetMaskedAPIKey(%q): got %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestAPIKey_NeverLoggedByOpenAILLMClient pins the AD-8 invariant —
// the adapter must not log raw API keys. The test runs a full
// CompleteStream cycle with a known secret while capturing the
// process's default log output; the captured text must not contain
// the raw key under any code path.
func TestAPIKey_NeverLoggedByOpenAILLMClient(t *testing.T) {
	const secret = "sk-please-do-not-log-me-12345"
	var sink safeBuffer
	prev := log.Writer()
	log.SetOutput(&sink)
	t.Cleanup(func() { log.SetOutput(prev) })

	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer stub.Close()

	cli := NewOpenAILLMClient(stub.URL, secret)
	if _, err := cli.CompleteStream(context.Background(),
		CompletionRequest{Model: "x"}, nil, nil); err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}

	if got := sink.String(); strings.Contains(got, secret) {
		t.Errorf("raw API key leaked to log output: %q", got)
	}
}

// safeBuffer is a goroutine-safe bytes.Buffer used as the sink for
// the captured log output in TestAPIKey_NeverLoggedByOpenAILLMClient.
// The standard log package's writer is shared across goroutines so a
// mutex is required even though the test is single-threaded — the
// http client's transport may write transport-level errors from a
// background goroutine.
type safeBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (s *safeBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *safeBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
