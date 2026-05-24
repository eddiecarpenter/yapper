package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

const (
	// anthropicMessagesPath is the canonical Anthropic Messages API
	// endpoint, served under the configured base URL.
	anthropicMessagesPath = "/v1/messages"

	// anthropicVersion pins the API version header. Bumping this
	// requires re-verifying the SSE schema and wire-format match.
	anthropicVersion = "2023-06-01"

	// anthropicDefaultMaxTokens is the fallback MaxTokens value.
	// The Anthropic Messages API requires this field to be set; the
	// MVP relay does not yet expose it through CompletionRequest, so
	// the adapter supplies a sensible default that keeps the relay
	// usable without forcing every caller to specify a limit.
	anthropicDefaultMaxTokens = 4096

	// SSE event names the adapter recognises. Everything else
	// (message_start, content_block_start, content_block_stop, ping,
	// unknown future events) is dropped silently per the MVP scope
	// declared in the Design Plan.
	anthropicEventContentBlockDelta = "content_block_delta"
	anthropicEventMessageDelta      = "message_delta"
	anthropicEventMessageStop       = "message_stop"

	// anthropicAPIKeyPrefix is the heuristic used to tell Anthropic
	// API keys apart from OAuth tokens. Keys created in the
	// Anthropic console begin with `sk-ant-`; OAuth-flow tokens
	// (Workbench / claude.ai) are JWTs and never carry this prefix.
	anthropicAPIKeyPrefix = "sk-ant-"
)

// AnthropicLLMClient implements LLMClient against Anthropic's native
// /v1/messages API. It speaks the SSE event protocol (events:
// message_start, content_block_*, message_delta, message_stop) and
// supports both API-key (x-api-key) and OAuth (Authorization: Bearer)
// flows — the choice is made per-request by SetAnthropicAuthHeaders
// from the shape of the configured key.
type AnthropicLLMClient struct {
	// BaseURL is the upstream API root, e.g. https://api.anthropic.com.
	// Trailing slashes are trimmed at construction so the joined
	// path is always exactly one slash from the base.
	BaseURL string

	// APIKey carries either an Anthropic API key (starts with
	// sk-ant-) or an OAuth bearer token. The adapter selects the
	// correct header based on the prefix; see SetAnthropicAuthHeaders.
	APIKey string

	// HTTPClient is the transport used for both streaming and
	// non-streaming requests. nil falls back to a default client
	// using defaultHTTPTimeout (the same default the OpenAI adapter
	// uses).
	HTTPClient *http.Client

	// MaxTokens overrides anthropicDefaultMaxTokens when > 0. Set
	// explicitly when the caller needs longer or shorter
	// completions; the relay's MVP uses the default.
	MaxTokens int
}

// NewAnthropicLLMClient returns an AnthropicLLMClient configured for
// baseURL + apiKey with the package's default HTTP timeout and the
// default MaxTokens value. Trailing slashes on baseURL are trimmed so
// the joined path is well-formed.
func NewAnthropicLLMClient(baseURL, apiKey string) *AnthropicLLMClient {
	return &AnthropicLLMClient{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: defaultHTTPTimeout},
		MaxTokens:  anthropicDefaultMaxTokens,
	}
}

func (c *AnthropicLLMClient) httpClient() *http.Client {
	if c.HTTPClient == nil {
		return &http.Client{Timeout: defaultHTTPTimeout}
	}
	return c.HTTPClient
}

func (c *AnthropicLLMClient) maxTokens() int {
	if c.MaxTokens <= 0 {
		return anthropicDefaultMaxTokens
	}
	return c.MaxTokens
}

// Complete implements LLMClient.Complete via a single non-streaming
// POST against /v1/messages.
func (c *AnthropicLLMClient) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
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
		return nil, fmt.Errorf("anthropic: do request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, httpError("anthropic", resp)
	}
	var payload anthropicMessagesResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("anthropic: decode response: %w", err)
	}
	return payload.toCompletion(), nil
}

// CompleteStream implements LLMClient.CompleteStream against the
// Anthropic native SSE protocol.
//
// Each content_block_delta event's text payload is delivered via
// onToken in arrival order; the final usage value (carried by
// message_delta and confirmed by message_stop) is delivered via
// onUsage exactly once. message_stop terminates the loop. All other
// event types (message_start, content_block_start, content_block_stop,
// ping, unknown future events) are silently dropped.
//
// ctx cancellation aborts the upstream connection and returns
// ctx.Err() — same contract as OpenAILLMClient.CompleteStream.
func (c *AnthropicLLMClient) CompleteStream(ctx context.Context, req CompletionRequest,
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
		return nil, fmt.Errorf("anthropic: do stream request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, httpError("anthropic", resp)
	}

	var (
		buf   strings.Builder
		usage Usage
		event string
	)
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineBytes)

streamLoop:
	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		line := scanner.Text()
		switch {
		case line == "":
			// Blank line terminates the current SSE event scope. The
			// next event name must be re-declared explicitly — the
			// adapter does not assume a default event type.
			event = ""
		case strings.HasPrefix(line, "event: "):
			event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			payload := strings.TrimPrefix(line, "data: ")
			switch event {
			case anthropicEventContentBlockDelta:
				var ev anthropicContentBlockDeltaEvent
				if err := json.Unmarshal([]byte(payload), &ev); err != nil {
					return nil, fmt.Errorf("anthropic: parse content_block_delta: %w", err)
				}
				if ev.Delta.Text != "" {
					buf.WriteString(ev.Delta.Text)
					if onToken != nil {
						onToken(ev.Delta.Text)
					}
				}
			case anthropicEventMessageDelta:
				var ev anthropicMessageDeltaEvent
				if err := json.Unmarshal([]byte(payload), &ev); err != nil {
					return nil, fmt.Errorf("anthropic: parse message_delta: %w", err)
				}
				if ev.Usage.OutputTokens > 0 || ev.Usage.InputTokens > 0 {
					usage = Usage{Input: ev.Usage.InputTokens, Output: ev.Usage.OutputTokens}
				}
			case anthropicEventMessageStop:
				if onUsage != nil && usage != (Usage{}) {
					onUsage(usage)
				}
				break streamLoop
			default:
				// message_start, content_block_start,
				// content_block_stop, ping, unknown future events —
				// dropped silently per MVP scope.
			}
		}
	}
	if err := scanner.Err(); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, fmt.Errorf("anthropic: read stream: %w", err)
	}

	return &CompletionResponse{Content: buf.String(), Usage: usage}, nil
}

// buildRequest converts a CompletionRequest into the
// Anthropic-specific wire shape: system-role messages are lifted
// into the top-level System field (Anthropic does not accept
// role:"system" in the Messages array), and MaxTokens is populated
// from the client's configured limit.
func (c *AnthropicLLMClient) buildRequest(ctx context.Context, req CompletionRequest) (*http.Request, error) {
	if c.BaseURL == "" {
		return nil, errors.New("anthropic: base URL is empty")
	}
	wire := anthropicMessagesRequest{
		Model:     req.Model,
		MaxTokens: c.maxTokens(),
		Stream:    req.Stream,
	}
	for _, m := range req.Messages {
		if m.Role == "system" {
			if wire.System == "" {
				wire.System = m.Content
			} else {
				wire.System += "\n" + m.Content
			}
			continue
		}
		wire.Messages = append(wire.Messages, Message{Role: m.Role, Content: m.Content})
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+anthropicMessagesPath, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	SetAnthropicAuthHeaders(httpReq, c.APIKey)
	return httpReq, nil
}

// SetAnthropicAuthHeaders selects between Anthropic's two auth flows
// based on a key-shape heuristic:
//
//   - apiKey starts with `sk-ant-` → treated as an Anthropic API key
//     and sent via the `x-api-key` header.
//   - Anything else → treated as an OAuth bearer token and sent via
//     `Authorization: Bearer ...`.
//
// An empty apiKey leaves the request unauthenticated, which suits
// local Anthropic-compatible proxies that don't require credentials
// (analogous to the OpenAI adapter's Ollama path).
func SetAnthropicAuthHeaders(req *http.Request, apiKey string) {
	if apiKey == "" {
		return
	}
	if strings.HasPrefix(apiKey, anthropicAPIKeyPrefix) {
		req.Header.Set("x-api-key", apiKey)
		return
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
}

// anthropicMessagesRequest is the wire shape Anthropic's /v1/messages
// expects: a top-level System field (instead of a system-role
// message), an explicit MaxTokens, and the standard Messages array.
type anthropicMessagesRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens"`
	Stream    bool      `json:"stream,omitempty"`
	System    string    `json:"system,omitempty"`
}

// anthropicMessagesResponse mirrors the non-streamed /v1/messages
// response shape — `content` is a list of typed content blocks.
type anthropicMessagesResponse struct {
	Content []anthropicContentBlock `json:"content"`
	Usage   anthropicUsage          `json:"usage"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// anthropicContentBlockDeltaEvent is the SSE payload for
// `event: content_block_delta` — the only event the streaming
// adapter consumes for token text.
type anthropicContentBlockDeltaEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

// anthropicMessageDeltaEvent is the SSE payload for
// `event: message_delta` — carries the running usage record. The
// adapter snapshots usage from this event and emits it on
// message_stop.
type anthropicMessageDeltaEvent struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason string `json:"stop_reason"`
	} `json:"delta"`
	Usage anthropicUsage `json:"usage"`
}

func (r *anthropicMessagesResponse) toCompletion() *CompletionResponse {
	var b strings.Builder
	for _, blk := range r.Content {
		if blk.Type == "text" && blk.Text != "" {
			b.WriteString(blk.Text)
		}
	}
	return &CompletionResponse{
		Content: b.String(),
		Usage:   Usage{Input: r.Usage.InputTokens, Output: r.Usage.OutputTokens},
	}
}
