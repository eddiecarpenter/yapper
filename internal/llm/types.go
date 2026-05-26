// Package llm contains the provider-agnostic LLM client interface,
// concrete adapter implementations (OpenAI-compatible, Anthropic),
// and the shared wire types used to communicate with both the
// upstream providers and the WebSocket relay's browser clients.
//
// Types declared in this file mirror docs/ARCHITECTURE.md §5.1 and
// §5.2 verbatim — JSON tags included — so the same struct can be
// serialised to the browser without translation.
package llm

// Message represents a single turn in a conversation as exchanged
// with an LLM provider. Roles follow the OpenAI / Anthropic
// convention: "system" | "user" | "assistant".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest is the inbound request shape for the LLM
// adapters. The Stream field is set by the adapter — callers set
// only Model and Messages — but the JSON tag is preserved so the
// struct can be re-marshalled to upstream providers as is.
type CompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`

	// EnableThinking controls chain-of-thought reasoning on models that
	// support it (e.g. Qwen3 via LM Studio). A nil pointer omits the
	// field entirely so providers that don't recognise it (Ollama,
	// OpenAI) receive a clean request. Set to false for voice assistants
	// where thinking tokens add latency with no conversational benefit.
	EnableThinking *bool `json:"enable_thinking,omitempty"`

	// Stop is an optional list of sequences at which the model should
	// stop generating. Useful for models that emit special tokens (e.g.
	// "<turn|>") that signal a new conversational turn. Omitted when
	// empty so providers that don't support the field receive a clean
	// request.
	Stop []string `json:"stop,omitempty"`
}

// Usage tracks token consumption per completion. The Input / Output
// field names match the relay's WebSocket wire protocol
// (docs/ARCHITECTURE.md §5.2) so the same value can be forwarded
// straight to the browser inside the `done` frame.
//
// Providers that do not report usage (e.g. OpenAI without
// `stream_options.include_usage:true`) leave both fields at zero —
// callers must treat absent usage as zero, not an error.
type Usage struct {
	Input  int `json:"input"`
	Output int `json:"output"`
}

// CompletionResponse is the aggregated result of a single (streamed
// or non-streamed) Complete call.
//
// Content holds the full assistant text concatenated across every
// delta the upstream emitted; Usage holds the token counts (zero
// when absent).
type CompletionResponse struct {
	Content string `json:"content"`
	Usage   Usage  `json:"usage"`
}
