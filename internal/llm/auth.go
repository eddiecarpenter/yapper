package llm

import (
	"net/http"

	"github.com/eddiecarpenter/yapper/internal/config"
)

// SetAuthHeaders attaches the Authorization header required by
// OpenAI-compatible providers (Ollama, OpenAI, Groq, OpenRouter).
//
// An empty apiKey leaves the request unauthenticated, which is the
// Ollama path — the local server requires no credentials and any
// stray header would either be ignored or cause a 401, depending on
// the provider. Cloud providers (OpenAI et al.) require the header
// and would reject the request without it.
func SetAuthHeaders(req *http.Request, apiKey string) {
	if apiKey == "" {
		return
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
}

// GetMaskedAPIKey returns the log-safe form of apiKey: all but the
// trailing four characters replaced by asterisks. Empty input
// returns the empty string.
//
// The masking algorithm lives in internal/config as MaskAPIKey so it
// has one audit point for AD-8 compliance; this wrapper exists so
// callers in internal/llm have a local symbol to import alongside
// SetAuthHeaders.
func GetMaskedAPIKey(apiKey string) string {
	return config.MaskAPIKey(apiKey)
}
