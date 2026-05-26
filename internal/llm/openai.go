package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	// openAIChatCompletionsPath is the canonical OpenAI Chat
	// Completions endpoint. Every OpenAI-compatible provider
	// (Ollama, Groq, OpenRouter, OpenAI) serves this path mounted
	// under the configured base URL.
	openAIChatCompletionsPath = "/chat/completions"

	// sseDataPrefix is the canonical SSE event prefix. Bytes after
	// this prefix on a single line are the JSON payload for one
	// stream chunk.
	sseDataPrefix = "data: "

	// sseDoneToken signals the end of an OpenAI-compatible stream.
	// After receiving it the adapter stops parsing and returns the
	// aggregated response.
	sseDoneToken = "[DONE]"

	// defaultHTTPTimeout caps non-streaming requests. Streaming
	// requests do NOT use this client-side timeout — they rely on
	// ctx instead so long generations aren't artificially capped.
	defaultHTTPTimeout = 60 * time.Second

	// maxErrorBodyBytes caps how much of an upstream error body the
	// adapter quotes back when surfacing an HTTP error. Anything
	// larger is truncated so a misbehaving upstream cannot produce
	// unbounded error messages.
	maxErrorBodyBytes = 64 * 1024

	// maxSSELineBytes is the largest single SSE line the scanner
	// accepts. Default bufio.Scanner caps at 64 KiB which is too
	// small for some providers; 1 MiB is more than enough headroom
	// while still bounding memory.
	maxSSELineBytes = 1 * 1024 * 1024
)

// OpenAILLMClient is the OpenAI-compatible LLM adapter. The same
// struct drives any provider speaking the OpenAI Chat Completions
// wire format — Ollama (the AD-4 default), OpenAI itself, Groq,
// OpenRouter, etc. Distinct providers are selected by swapping
// BaseURL (and APIKey, which is left empty for Ollama).
type OpenAILLMClient struct {
	// BaseURL is the upstream API root, e.g. http://localhost:11434/v1
	// for Ollama or https://api.openai.com/v1 for OpenAI. Trailing
	// slashes are trimmed at construction so the joined path is
	// well-formed.
	BaseURL string

	// APIKey is the bearer token sent in the Authorization header.
	// Leave empty for unauthenticated providers (Ollama).
	APIKey string

	// HTTPClient is the transport used for both streaming and
	// non-streaming requests. A nil value is replaced with a default
	// http.Client that uses defaultHTTPTimeout — set explicitly to
	// inject custom transports (proxy, retry, instrumentation).
	HTTPClient *http.Client
}

// NewOpenAILLMClient returns an OpenAILLMClient with the supplied
// base URL and API key, plus a default http.Client. Trailing slashes
// in baseURL are trimmed so the joined chat-completions path is
// always exactly one slash from the base.
func NewOpenAILLMClient(baseURL, apiKey string) *OpenAILLMClient {
	return &OpenAILLMClient{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: defaultHTTPTimeout},
	}
}

func (c *OpenAILLMClient) httpClient() *http.Client {
	if c.HTTPClient == nil {
		return &http.Client{Timeout: defaultHTTPTimeout}
	}
	return c.HTTPClient
}

// Complete performs a single non-streaming Chat Completions call.
// Stream is forced to false on the outgoing payload; the response
// JSON is parsed into a CompletionResponse with token usage populated
// when the upstream supplies it.
func (c *OpenAILLMClient) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	req.Stream = false
	httpReq, err := c.buildRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient().Do(httpReq)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, fmt.Errorf("openai: do request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, httpError("openai", resp)
	}

	var payload openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}
	return payload.toCompletion(), nil
}

// CompleteStream performs a streaming Chat Completions call.
//
// Each text delta from the upstream is delivered to onToken in
// arrival order; the final usage record (when the upstream reports
// one) is delivered to onUsage exactly once. Either callback may be
// nil — the method still drives the stream to completion and
// aggregates the assistant text into the returned response.
//
// ctx cancellation aborts the upstream connection (the request was
// constructed with http.NewRequestWithContext) and the method
// returns ctx.Err() rather than the lower-level "use of closed
// connection" error.
func (c *OpenAILLMClient) CompleteStream(ctx context.Context, req CompletionRequest,
	onToken func(string), onUsage func(Usage)) (*CompletionResponse, error) {

	req.Stream = true
	httpReq, err := c.buildRequest(ctx, req)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient().Do(httpReq)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, fmt.Errorf("openai: do stream request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, httpError("openai", resp)
	}

	var (
		buf   strings.Builder
		usage Usage
	)
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineBytes)
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		line := scanner.Text()
		if !strings.HasPrefix(line, sseDataPrefix) {
			continue
		}
		payload := strings.TrimPrefix(line, sseDataPrefix)
		if payload == sseDoneToken {
			break
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return nil, fmt.Errorf("openai: parse stream chunk: %w", err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content == "" {
				continue
			}
			buf.WriteString(choice.Delta.Content)
			if onToken != nil {
				onToken(choice.Delta.Content)
			}
		}
		if chunk.Usage != nil {
			usage = Usage{
				Input:  chunk.Usage.PromptTokens,
				Output: chunk.Usage.CompletionTokens,
			}
			if onUsage != nil {
				onUsage(usage)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, fmt.Errorf("openai: read stream: %w", err)
	}

	return &CompletionResponse{Content: buf.String(), Usage: usage}, nil
}

func (c *OpenAILLMClient) buildRequest(ctx context.Context, req CompletionRequest) (*http.Request, error) {
	if c.BaseURL == "" {
		return nil, errors.New("openai: base URL is empty")
	}

	// For streaming requests, request usage in the final SSE chunk.
	// OpenAI-compatible providers (including LM Studio) require the
	// stream_options.include_usage flag — without it the [DONE] chunk
	// carries no token counts and the t/s metric stays at zero.
	var body []byte
	var err error
	if req.Stream {
		type streamOptions struct {
			IncludeUsage bool `json:"include_usage"`
		}
		type requestWithStreamOpts struct {
			CompletionRequest
			StreamOptions streamOptions `json:"stream_options"`
		}
		body, err = json.Marshal(requestWithStreamOpts{
			CompletionRequest: req,
			StreamOptions:     streamOptions{IncludeUsage: true},
		})
	} else {
		body, err = json.Marshal(req)
	}
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+openAIChatCompletionsPath, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	SetAuthHeaders(httpReq, c.APIKey)
	return httpReq, nil
}

// httpError builds a human-readable error from a non-2xx response,
// trimming the body to maxErrorBodyBytes so a misbehaving upstream
// cannot blow up the error message.
func httpError(prefix string, resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
	return fmt.Errorf("%s: http %d: %s", prefix, resp.StatusCode, strings.TrimSpace(string(body)))
}

// openAIResponse mirrors the non-streamed Chat Completions response.
type openAIResponse struct {
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

type openAIChoice struct {
	Message      Message `json:"message"`
	Delta        Message `json:"delta"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

type openAIStreamChunk struct {
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

func (r *openAIResponse) toCompletion() *CompletionResponse {
	var b strings.Builder
	for _, c := range r.Choices {
		if c.Message.Content != "" {
			b.WriteString(c.Message.Content)
		}
	}
	out := &CompletionResponse{Content: b.String()}
	if r.Usage != nil {
		out.Usage = Usage{
			Input:  r.Usage.PromptTokens,
			Output: r.Usage.CompletionTokens,
		}
	}
	return out
}
