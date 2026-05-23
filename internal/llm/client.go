package llm

import "context"

// LLMClient is the provider-agnostic interface the WebSocket relay
// uses to talk to whichever model backend is configured. Adapters
// for OpenAI-compatible providers (Ollama, OpenAI, Groq, OpenRouter)
// and Anthropic implement this interface; the relay selects between
// them via NewLLMClient at startup (added in Task 3).
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
