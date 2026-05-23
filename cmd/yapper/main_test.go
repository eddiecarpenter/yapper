package main

import (
	"bytes"
	"context"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/eddiecarpenter/yapper/internal/config"
)

func TestRun_NoArgs_ReturnsError(t *testing.T) {
	var stderr bytes.Buffer
	err := run(context.Background(), nil, &stderr)
	if err == nil {
		t.Fatalf("expected error on empty args, got nil")
	}
	if !strings.Contains(stderr.String(), "usage:") {
		t.Errorf("expected usage banner in stderr, got %q", stderr.String())
	}
}

func TestRun_UnknownSubcommand_ReturnsError(t *testing.T) {
	var stderr bytes.Buffer
	err := run(context.Background(), []string{"bogus"}, &stderr)
	if err == nil {
		t.Fatalf("expected error for unknown subcommand")
	}
	if !strings.Contains(stderr.String(), "bogus") {
		t.Errorf("expected stderr to mention the bad subcommand, got %q", stderr.String())
	}
}

func TestRun_HelpFlag_ReturnsNil(t *testing.T) {
	var stderr bytes.Buffer
	if err := run(context.Background(), []string{"--help"}, &stderr); err != nil {
		t.Fatalf("--help: unexpected error %v", err)
	}
	if !strings.Contains(stderr.String(), "usage:") {
		t.Errorf("expected usage banner, got %q", stderr.String())
	}
}

// TestRunServe_StartsAndShutsDown is the closest thing to an
// integration test for the binary — it spins up the serve loop on a
// fixed high port via the YAPPER_SERVER_PORT env override, hits the
// stub `/` endpoint, then cancels the context to verify graceful
// shutdown returns nil. The LLM adapter is constructed but never
// called, so this test does not depend on a real Ollama instance.
func TestRunServe_StartsAndShutsDown(t *testing.T) {
	t.Setenv(config.EnvServerPort, "18877")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- runServe(ctx, nil)
	}()

	deadline := time.Now().Add(2 * time.Second)
	var resp *http.Response
	var err error
	for time.Now().Before(deadline) {
		resp, err = http.Get("http://127.0.0.1:18877/")
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		cancel()
		<-done
		t.Fatalf("server never became reachable: %v", err)
	}
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("status: got %d, want %d", resp.StatusCode, http.StatusNoContent)
	}
	resp.Body.Close()

	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("runServe returned unexpected error after shutdown: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("runServe did not return within 3s of context cancel")
	}
}
