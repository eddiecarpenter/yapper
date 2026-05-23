package llm

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"

	"github.com/eddiecarpenter/yapper/internal/config"
)

// LLMClient is the provider-agnostic interface the WebSocket relay
// uses to talk to whichever model backend is configured. Adapters
// for OpenAI-compatible providers (Ollama, OpenAI, Groq, OpenRouter)
// and Anthropic implement this interface; the relay selects between
// them via NewLLMClient at startup.
//
// Both methods must respect ctx cancellation — a closed browser
// connection in the WebSocket handler propagates a cancel through
// ctx and the adapter is expected to abort its upstream call and
// return ctx.Err().
type LLMClient interface {
	// Complete performs a non-streaming completion. The returned
	// CompletionResponse holds the full assistant text and (when
	// the provider reports it) the usage counts.
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)

	// CompleteStream performs a streaming completion.
	//
	// Each text delta from the upstream is delivered to onToken in
	// arrival order. The final usage record (when the provider
	// supplies one) is delivered to onUsage exactly once before the
	// method returns. Either callback may be nil — the adapter must
	// still drive the stream to completion and aggregate the result
	// into the returned CompletionResponse for callers that want the
	// final value in a single shot.
	CompleteStream(ctx context.Context, req CompletionRequest,
		onToken func(string), onUsage func(Usage)) (*CompletionResponse, error)
}

// anthropicHostMarker is the hostname substring that selects the
// AnthropicLLMClient adapter from NewLLMClient. The match is
// case-insensitive and substring-based so both production
// (api.anthropic.com) and staging hosts route correctly. The choice
// reflects AC-4 on Feature #15: "endpoint contains anthropic.com →
// AnthropicLLMClient".
const anthropicHostMarker = "anthropic.com"

// NewLLMClient returns the LLMClient adapter that matches cfg.BaseURL.
//
//   - cfg.BaseURL host contains "anthropic.com" (case-insensitive) →
//     AnthropicLLMClient with the cfg.APIKey wired through.
//   - Anything else (Ollama, OpenAI, Groq, OpenRouter, custom local
//     proxies) → OpenAILLMClient.
//
// The single hostname heuristic lets one config field (base_url)
// cover every provider the spike supports — a config change between
// dev and prod is one line.
func NewLLMClient(cfg config.LLMConfig) (LLMClient, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("llm: base URL is empty")
	}
	parsed, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("llm: parse base URL %q: %w", cfg.BaseURL, err)
	}
	host := strings.ToLower(parsed.Hostname())
	if strings.Contains(host, anthropicHostMarker) {
		return NewAnthropicLLMClient(cfg.BaseURL, cfg.APIKey), nil
	}
	return NewOpenAILLMClient(cfg.BaseURL, cfg.APIKey), nil
}
