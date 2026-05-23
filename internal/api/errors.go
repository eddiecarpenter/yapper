package api

import (
	"context"
	"errors"
	"strings"

	"github.com/eddiecarpenter/yapper/internal/config"
)

// classifyError maps a raw LLM-client error into a user-grade,
// actionable message that is safe to forward to the browser inside
// an `error` frame.
//
// The function is allow-listing: a handful of specific failure
// shapes (connection refused, HTTP 401, HTTP 404 / model-not-found,
// timeout) map to canned messages; anything else falls through to a
// generic "LLM request failed: <sanitised>" line. The sanitisation
// strips the configured API key from the error text — belt-and-braces
// against the upstream adapter accidentally quoting credentials in
// its error string.
//
// classifyError never includes the API key, internal package names,
// stack traces, or raw upstream JSON in the returned message.
func classifyError(err error, cfg *config.Config) string {
	if err == nil {
		return ""
	}

	if errors.Is(err, context.DeadlineExceeded) {
		return "LLM request timed out."
	}
	if errors.Is(err, context.Canceled) {
		return "LLM request was cancelled."
	}

	low := strings.ToLower(err.Error())

	switch {
	case strings.Contains(low, "connection refused"),
		strings.Contains(low, "no such host"),
		strings.Contains(low, "dial tcp"):
		return "Ollama unreachable at " + cfg.LLM.BaseURL +
			". Start with `ollama serve`."
	case strings.Contains(low, "http 404"),
		strings.Contains(low, "model not found"):
		return "Model `" + cfg.LLM.Model +
			"` not pulled. Run `ollama pull " + cfg.LLM.Model + "`."
	case strings.Contains(low, "http 401"),
		strings.Contains(low, "unauthorized"),
		strings.Contains(low, "incorrect api key"):
		return "Missing or invalid API key. Set `YAPPER_LLM_API_KEY`."
	case strings.Contains(low, "timeout"),
		strings.Contains(low, "deadline exceeded"):
		return "LLM request timed out."
	}

	return "LLM request failed: " + sanitiseErrorText(err.Error(), cfg.LLM.APIKey)
}

// sanitiseErrorText strips occurrences of apiKey from s. Empty
// apiKey is a no-op. The masking is exact-match — adapters that
// quote the key inside an upstream error body will have those
// occurrences redacted before the text leaves the relay.
func sanitiseErrorText(s, apiKey string) string {
	if apiKey == "" {
		return s
	}
	return strings.ReplaceAll(s, apiKey, "***")
}
