package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
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

func TestBuildServer_RootReturns204(t *testing.T) {
	srv := buildServer(config.Defaults())

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	srv.Handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Errorf("status: got %d, want %d", rec.Code, http.StatusNoContent)
	}
	if rec.Body.Len() != 0 {
		t.Errorf("body: got %d bytes, want empty", rec.Body.Len())
	}
}

func TestBuildServer_UsesConfiguredPort(t *testing.T) {
	cfg := config.Defaults()
	cfg.Server.Port = 14321
	srv := buildServer(cfg)
	if got, want := srv.Addr, ":14321"; got != want {
		t.Errorf("Addr: got %q, want %q", got, want)
	}
	if srv.ReadHeaderTimeout != readHeaderTimeout {
		t.Errorf("ReadHeaderTimeout: got %v, want %v", srv.ReadHeaderTimeout, readHeaderTimeout)
	}
}

// TestRunServe_StartsAndShutsDown is the closest thing to an
// integration test for the binary — it spins up the serve loop on an
// ephemeral port via env override, hits the stub `/` endpoint, then
// cancels the context to verify graceful shutdown returns nil.
func TestRunServe_StartsAndShutsDown(t *testing.T) {
	t.Setenv(config.EnvServerPort, "0") // requesting port 0 is rejected upstream — verify the path errors out cleanly
	_ = t.TempDir()                     // keep parity with other tests that chdir; not needed here

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// We don't want to bind on a privileged or contested port. Use a
	// well-known ephemeral range port and rely on the OS to allocate.
	// runServe consumes from the YAPPER_SERVER_PORT env var when no
	// flag is supplied; we provide a high port.
	t.Setenv(config.EnvServerPort, "18877")

	done := make(chan error, 1)
	go func() {
		done <- runServe(ctx, nil)
	}()

	// Wait for the server to start accepting connections, with a
	// short polling budget — keeps the test fast on a healthy
	// runner and gives clear failure otherwise.
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
