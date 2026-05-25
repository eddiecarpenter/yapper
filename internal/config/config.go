// Package config loads the Yapper relay's runtime configuration from
// YAML on disk and applies environment-variable overrides on top.
//
// The schema mirrors docs/ARCHITECTURE.md §6.7. Defaults match AD-4 —
// fully offline Ollama on localhost with the llama3.2:3b model — so a
// freshly built binary runs without any config file present.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Default values for the Yapper relay configuration. They match AD-4
// in docs/ARCHITECTURE.md: zero-config Ollama on localhost.
const (
	DefaultServerPort        = 8080
	DefaultSessionTTLMinutes = 30
	DefaultLLMProvider       = "openai_compat"
	DefaultLLMBaseURL        = "http://localhost:1234/v1"
	DefaultLLMModel          = "google/gemma-3-12b"
	DefaultLLMStream         = true
	DefaultContextBudget     = 4096
	DefaultSystemPrompt      = "You are Yapper, a helpful voice assistant. Be concise."
)

// Environment variable names used to override YAML-derived values.
// These are stable identifiers — see docs/ARCHITECTURE.md §6.7.
const (
	EnvLLMBaseURL = "YAPPER_LLM_BASE_URL"
	EnvLLMModel   = "YAPPER_LLM_MODEL"
	EnvLLMAPIKey  = "YAPPER_LLM_API_KEY"
	EnvServerPort = "YAPPER_SERVER_PORT"
)

// ServerConfig models the `server:` section of the YAML schema.
type ServerConfig struct {
	Port              int `yaml:"port"`
	SessionTTLMinutes int `yaml:"session_ttl_minutes"`
}

// LLMConfig models the `llm:` section of the YAML schema.
//
// APIKey is never written to YAML — it is supplied via the
// YAPPER_LLM_API_KEY environment variable only (AD-8 credentials
// rule). The yaml:"-" tag enforces this at parse time.
type LLMConfig struct {
	Provider      string `yaml:"provider"`
	BaseURL       string `yaml:"base_url"`
	Model         string `yaml:"model"`
	Stream        bool   `yaml:"stream"`
	ContextBudget int    `yaml:"context_budget"`
	SystemPrompt  string `yaml:"system_prompt"`

	APIKey string `yaml:"-"`
}

// Config is the parsed configuration for the Yapper relay.
type Config struct {
	Server ServerConfig `yaml:"server"`
	LLM    LLMConfig    `yaml:"llm"`
}

// Defaults returns a Config populated with the AD-4 default values.
// A freshly-built binary with no config file and no env vars uses
// these directly.
func Defaults() *Config {
	return &Config{
		Server: ServerConfig{
			Port:              DefaultServerPort,
			SessionTTLMinutes: DefaultSessionTTLMinutes,
		},
		LLM: LLMConfig{
			Provider:      DefaultLLMProvider,
			BaseURL:       DefaultLLMBaseURL,
			Model:         DefaultLLMModel,
			Stream:        DefaultLLMStream,
			ContextBudget: DefaultContextBudget,
			SystemPrompt:  DefaultSystemPrompt,
		},
	}
}

// candidatePaths is the implicit search order used by Load when no
// explicit path is supplied. Exposed as a package variable so tests
// can override it without manipulating the working directory.
var candidatePaths = []string{"yapper.yaml", "config/yapper.yaml"}

// Load reads configuration from disk and applies environment-variable
// overrides on top.
//
// Search order:
//
//  1. If path != "", read that file (returns an error if the file
//     does not exist — an explicit path is a hard requirement).
//  2. Else try the candidatePaths in order and load the first match.
//  3. Else use the AD-4 defaults from Defaults() unchanged.
//
// After the YAML-derived configuration is in place the env-var
// overrides listed in docs/ARCHITECTURE.md §6.7 are applied. This
// lets a single binary be reused across dev (YAML on disk) and
// production (env vars only) without redeploying.
func Load(path string) (*Config, error) {
	cfg := Defaults()

	resolved, err := resolveConfigPath(path)
	if err != nil {
		return nil, err
	}
	if resolved != "" {
		if err := loadYAML(resolved, cfg); err != nil {
			return nil, fmt.Errorf("load config %q: %w", resolved, err)
		}
	}

	if err := applyEnvOverrides(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// resolveConfigPath maps the caller-supplied path to a concrete file
// to read. An empty path triggers the candidate search; a non-empty
// path is required to exist.
func resolveConfigPath(explicit string) (string, error) {
	if explicit != "" {
		if _, err := os.Stat(explicit); err != nil {
			return "", fmt.Errorf("config file %q: %w", explicit, err)
		}
		return explicit, nil
	}
	for _, candidate := range candidatePaths {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", nil
}

func loadYAML(path string, cfg *Config) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("parse yaml: %w", err)
	}
	return nil
}

// applyEnvOverrides mutates cfg in place, overlaying each environment
// variable that is set to a non-empty value. Empty / unset variables
// leave the YAML / default value untouched.
func applyEnvOverrides(cfg *Config) error {
	if v := os.Getenv(EnvLLMBaseURL); v != "" {
		cfg.LLM.BaseURL = v
	}
	if v := os.Getenv(EnvLLMModel); v != "" {
		cfg.LLM.Model = v
	}
	if v := os.Getenv(EnvLLMAPIKey); v != "" {
		cfg.LLM.APIKey = v
	}
	if v := os.Getenv(EnvServerPort); v != "" {
		port, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("env %s=%q: not a valid integer: %w", EnvServerPort, v, err)
		}
		if port <= 0 || port > 65535 {
			return fmt.Errorf("env %s=%d: port out of range (1..65535)", EnvServerPort, port)
		}
		cfg.Server.Port = port
	}
	return nil
}

// GetMaskedAPIKey returns the configured LLM API key with every
// character except the trailing four replaced by asterisks. An empty
// key returns an empty string; keys of four characters or fewer are
// fully masked.
//
// This is the canonical safe representation for log lines (AD-8) —
// never write APIKey directly to a log.
func (c *LLMConfig) GetMaskedAPIKey() string {
	return MaskAPIKey(c.APIKey)
}

// MaskAPIKey is the package-level form of (*LLMConfig).GetMaskedAPIKey
// for callers that hold a key string directly (e.g., adapter code in
// internal/llm). Keeping the masking algorithm in one place ensures a
// single audit point for AD-8 compliance.
func MaskAPIKey(key string) string {
	if key == "" {
		return ""
	}
	if len(key) <= 4 {
		return strings.Repeat("*", len(key))
	}
	return strings.Repeat("*", len(key)-4) + key[len(key)-4:]
}
