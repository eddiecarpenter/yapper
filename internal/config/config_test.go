package config

import (
	"os"
	"path/filepath"
	"testing"
)

// chdir switches the test process into dir and restores the original
// working directory at cleanup time. Tests use this rather than
// running every Load() through an explicit path so the search-path
// behaviour (candidatePaths) is exercised honestly.
func chdir(t *testing.T, dir string) {
	t.Helper()
	old, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir %q: %v", dir, err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
}

// clearEnv blanks every YAPPER_* env var the loader consults, with
// t.Setenv arranging cleanup. An empty string is treated as unset by
// applyEnvOverrides, so this is the simplest portable way to isolate
// each test from the runner's environment.
func clearEnv(t *testing.T) {
	t.Helper()
	t.Setenv(EnvLLMBaseURL, "")
	t.Setenv(EnvLLMModel, "")
	t.Setenv(EnvLLMAPIKey, "")
	t.Setenv(EnvServerPort, "")
}

func TestLoad_NoFileNoEnv_ReturnsDefaults(t *testing.T) {
	chdir(t, t.TempDir())
	clearEnv(t)

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: unexpected error %v", err)
	}

	if cfg.Server.Port != DefaultServerPort {
		t.Errorf("Server.Port: got %d, want %d", cfg.Server.Port, DefaultServerPort)
	}
	if cfg.Server.SessionTTLMinutes != DefaultSessionTTLMinutes {
		t.Errorf("Server.SessionTTLMinutes: got %d, want %d", cfg.Server.SessionTTLMinutes, DefaultSessionTTLMinutes)
	}
	if cfg.LLM.Provider != DefaultLLMProvider {
		t.Errorf("LLM.Provider: got %q, want %q", cfg.LLM.Provider, DefaultLLMProvider)
	}
	if cfg.LLM.BaseURL != DefaultLLMBaseURL {
		t.Errorf("LLM.BaseURL: got %q, want %q", cfg.LLM.BaseURL, DefaultLLMBaseURL)
	}
	if cfg.LLM.Model != DefaultLLMModel {
		t.Errorf("LLM.Model: got %q, want %q", cfg.LLM.Model, DefaultLLMModel)
	}
	if cfg.LLM.Stream != DefaultLLMStream {
		t.Errorf("LLM.Stream: got %v, want %v", cfg.LLM.Stream, DefaultLLMStream)
	}
	if cfg.LLM.ContextBudget != DefaultContextBudget {
		t.Errorf("LLM.ContextBudget: got %d, want %d", cfg.LLM.ContextBudget, DefaultContextBudget)
	}
	if cfg.LLM.SystemPrompt != DefaultSystemPrompt {
		t.Errorf("LLM.SystemPrompt: got %q, want %q", cfg.LLM.SystemPrompt, DefaultSystemPrompt)
	}
	if cfg.LLM.APIKey != "" {
		t.Errorf("LLM.APIKey: got %q, want empty", cfg.LLM.APIKey)
	}
}

func TestLoad_YAMLOnly_OverridesEveryField(t *testing.T) {
	chdir(t, t.TempDir())
	clearEnv(t)

	body := []byte(`
server:
  port: 9999
  session_ttl_minutes: 5
llm:
  provider: anthropic
  base_url: https://api.anthropic.com
  model: claude-3-opus
  stream: false
  context_budget: 8192
  system_prompt: "Be terse."
`)
	cfgPath := filepath.Join(t.TempDir(), "explicit.yaml")
	if err := os.WriteFile(cfgPath, body, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	checks := []struct {
		name string
		got  any
		want any
	}{
		{"Server.Port", cfg.Server.Port, 9999},
		{"Server.SessionTTLMinutes", cfg.Server.SessionTTLMinutes, 5},
		{"LLM.Provider", cfg.LLM.Provider, "anthropic"},
		{"LLM.BaseURL", cfg.LLM.BaseURL, "https://api.anthropic.com"},
		{"LLM.Model", cfg.LLM.Model, "claude-3-opus"},
		{"LLM.Stream", cfg.LLM.Stream, false},
		{"LLM.ContextBudget", cfg.LLM.ContextBudget, 8192},
		{"LLM.SystemPrompt", cfg.LLM.SystemPrompt, "Be terse."},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestLoad_SearchPaths_PicksFirstMatch(t *testing.T) {
	dir := t.TempDir()
	chdir(t, dir)
	clearEnv(t)

	// Write yapper.yaml (the first candidate) — Load should pick it
	// over config/yapper.yaml.
	body := []byte("server:\n  port: 7000\n")
	if err := os.WriteFile(filepath.Join(dir, "yapper.yaml"), body, 0o600); err != nil {
		t.Fatalf("write yapper.yaml: %v", err)
	}
	// Write a second candidate that would conflict if picked.
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatalf("mkdir config: %v", err)
	}
	conflict := []byte("server:\n  port: 9000\n")
	if err := os.WriteFile(filepath.Join(dir, "config", "yapper.yaml"), conflict, 0o600); err != nil {
		t.Fatalf("write config/yapper.yaml: %v", err)
	}

	cfg, err := Load("")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Server.Port != 7000 {
		t.Errorf("Server.Port: got %d, want 7000 (first candidate)", cfg.Server.Port)
	}
}

func TestLoad_EnvOverrides_BeatYAMLAndDefaults(t *testing.T) {
	chdir(t, t.TempDir())

	cases := []struct {
		name  string
		env   map[string]string
		check func(t *testing.T, cfg *Config)
	}{
		{
			name: "BaseURL only",
			env:  map[string]string{EnvLLMBaseURL: "https://api.openai.com/v1"},
			check: func(t *testing.T, cfg *Config) {
				if cfg.LLM.BaseURL != "https://api.openai.com/v1" {
					t.Errorf("BaseURL: got %q", cfg.LLM.BaseURL)
				}
			},
		},
		{
			name: "Model only",
			env:  map[string]string{EnvLLMModel: "gpt-4o-mini"},
			check: func(t *testing.T, cfg *Config) {
				if cfg.LLM.Model != "gpt-4o-mini" {
					t.Errorf("Model: got %q", cfg.LLM.Model)
				}
			},
		},
		{
			name: "APIKey only",
			env:  map[string]string{EnvLLMAPIKey: "sk-test-1234"},
			check: func(t *testing.T, cfg *Config) {
				if cfg.LLM.APIKey != "sk-test-1234" {
					t.Errorf("APIKey: got %q", cfg.LLM.APIKey)
				}
			},
		},
		{
			name: "ServerPort only",
			env:  map[string]string{EnvServerPort: "9090"},
			check: func(t *testing.T, cfg *Config) {
				if cfg.Server.Port != 9090 {
					t.Errorf("Server.Port: got %d", cfg.Server.Port)
				}
			},
		},
		{
			name: "All four set together",
			env: map[string]string{
				EnvLLMBaseURL: "https://groq.example/v1",
				EnvLLMModel:   "llama-3.1-70b-versatile",
				EnvLLMAPIKey:  "gsk_secret_value",
				EnvServerPort: "5555",
			},
			check: func(t *testing.T, cfg *Config) {
				if cfg.LLM.BaseURL != "https://groq.example/v1" {
					t.Errorf("BaseURL: got %q", cfg.LLM.BaseURL)
				}
				if cfg.LLM.Model != "llama-3.1-70b-versatile" {
					t.Errorf("Model: got %q", cfg.LLM.Model)
				}
				if cfg.LLM.APIKey != "gsk_secret_value" {
					t.Errorf("APIKey: got %q", cfg.LLM.APIKey)
				}
				if cfg.Server.Port != 5555 {
					t.Errorf("Server.Port: got %d", cfg.Server.Port)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			clearEnv(t)
			for k, v := range tc.env {
				t.Setenv(k, v)
			}
			cfg, err := Load("")
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			tc.check(t, cfg)
		})
	}
}

func TestLoad_EnvOverridesYAML(t *testing.T) {
	chdir(t, t.TempDir())
	clearEnv(t)

	cfgPath := filepath.Join(t.TempDir(), "cfg.yaml")
	body := []byte("llm:\n  base_url: https://yaml.example/v1\n  model: yaml-model\n")
	if err := os.WriteFile(cfgPath, body, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Setenv(EnvLLMBaseURL, "https://env.example/v1")

	cfg, err := Load(cfgPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.LLM.BaseURL != "https://env.example/v1" {
		t.Errorf("BaseURL: env override lost, got %q", cfg.LLM.BaseURL)
	}
	if cfg.LLM.Model != "yaml-model" {
		t.Errorf("Model: YAML value lost, got %q", cfg.LLM.Model)
	}
}

func TestLoad_ExplicitPathMissing_ReturnsError(t *testing.T) {
	chdir(t, t.TempDir())
	clearEnv(t)

	cfg, err := Load(filepath.Join(t.TempDir(), "nonexistent.yaml"))
	if err == nil {
		t.Fatalf("expected error, got cfg=%+v", cfg)
	}
}

func TestLoad_MalformedYAML_ReturnsError(t *testing.T) {
	chdir(t, t.TempDir())
	clearEnv(t)

	cfgPath := filepath.Join(t.TempDir(), "broken.yaml")
	if err := os.WriteFile(cfgPath, []byte("server: {port: : :}\n"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	cfg, err := Load(cfgPath)
	if err == nil {
		t.Fatalf("expected error, got cfg=%+v", cfg)
	}
}

func TestLoad_PortEnvInvalid_ReturnsError(t *testing.T) {
	chdir(t, t.TempDir())
	clearEnv(t)

	cases := []string{"not-a-number", "-1", "0", "70000"}
	for _, v := range cases {
		t.Run(v, func(t *testing.T) {
			t.Setenv(EnvServerPort, v)
			_, err := Load("")
			if err == nil {
				t.Fatalf("expected error for port=%q", v)
			}
		})
	}
}

func TestMaskAPIKey_TableDriven(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"empty", "", ""},
		{"single char fully masked", "a", "*"},
		{"4 chars fully masked", "abcd", "****"},
		{"5 chars masks 1 + keeps last 4", "abcde", "*bcde"},
		{"sk-style key", "sk-abcdef0123456789xyz", "******************9xyz"},
		{"22-char key keeps last 4", "1234567890abcdef987654", "******************7654"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MaskAPIKey(tc.in)
			if got != tc.want {
				t.Errorf("MaskAPIKey(%q): got %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestLLMConfigGetMaskedAPIKey_UsesMaskAPIKey(t *testing.T) {
	c := &LLMConfig{APIKey: "sk-secretvalue-9999"}
	if got, want := c.GetMaskedAPIKey(), "***************9999"; got != want {
		t.Errorf("GetMaskedAPIKey: got %q, want %q", got, want)
	}
	empty := &LLMConfig{}
	if got := empty.GetMaskedAPIKey(); got != "" {
		t.Errorf("GetMaskedAPIKey(empty): got %q, want empty", got)
	}
}
