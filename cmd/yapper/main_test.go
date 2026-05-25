package main

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"os/exec"
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
		// --open=false so CI does not attempt to spawn a
		// browser (which would log a WARN about xdg-open on
		// any container without a desktop session and would
		// fork a real child process). Functionality test, not
		// browser-open test.
		done <- runServe(ctx, []string{"--open=false"})
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
	// Root path is now backed by the embedded SPA handler (Feature
	// #17 Task 2). When the TS bundle is present in web/dist/ at
	// compile time the response is 200 (index.html); when CI built
	// the binary before running `npm run build`, the embed is empty
	// and the SPA handler returns 404. Both shapes prove the server
	// is up and routing — neither is a 5xx.
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNotFound {
		t.Errorf("status: got %d, want 200 (SPA built) or 404 (empty embed)", resp.StatusCode)
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

// TestBrowserCommand_TableDriven verifies the per-platform dispatch
// of `browserCommand`. The test overrides the package-level `goos`
// indirection (set by the helper at package scope) so the assertion
// runs on every CI architecture rather than only the host's OS.
func TestBrowserCommand_TableDriven(t *testing.T) {
	const url = "http://localhost:8080"
	prev := goos
	t.Cleanup(func() { goos = prev })

	cases := []struct {
		os       string
		wantProg string
		wantArgs []string
	}{
		{"darwin", "open", []string{url}},
		{"linux", "xdg-open", []string{url}},
		{"windows", "cmd", []string{"/c", "start", "", url}},
		{"plan9", "", nil},   // unsupported → empty program
		{"freebsd", "", nil}, // unsupported → empty program
	}
	for _, tc := range cases {
		t.Run(tc.os, func(t *testing.T) {
			goos = tc.os
			gotProg, gotArgs := browserCommand(url)
			if gotProg != tc.wantProg {
				t.Errorf("program: got %q, want %q", gotProg, tc.wantProg)
			}
			if len(gotArgs) != len(tc.wantArgs) {
				t.Fatalf("args length: got %d (%v), want %d (%v)",
					len(gotArgs), gotArgs, len(tc.wantArgs), tc.wantArgs)
			}
			for i := range gotArgs {
				if gotArgs[i] != tc.wantArgs[i] {
					t.Errorf("args[%d]: got %q, want %q", i, gotArgs[i], tc.wantArgs[i])
				}
			}
		})
	}
}

// TestOpenBrowser_UnsupportedPlatform_ReturnsError verifies that
// openBrowser surfaces an error rather than panicking on a platform
// for which we have no command — the caller logs WARN and the
// server continues. Pinning `goos = "plan9"` triggers the empty-
// command branch.
func TestOpenBrowser_UnsupportedPlatform_ReturnsError(t *testing.T) {
	prev := goos
	t.Cleanup(func() { goos = prev })
	goos = "plan9"

	if err := openBrowser("http://localhost:8080"); err == nil {
		t.Fatal("openBrowser on unsupported platform: expected error, got nil")
	}
}

// TestOpenBrowser_SuccessfulSpawn verifies that openBrowser delegates
// to execCommand and returns nil on a successful Start. The test
// overrides execCommand with `/bin/true` (a no-op program present on
// every Unix CI runner). We do NOT exercise the macOS / Linux /
// Windows real-commands here — browserCommand's table test covers
// the dispatch — what this test exercises is the spawn path itself.
func TestOpenBrowser_SuccessfulSpawn(t *testing.T) {
	if _, err := exec.LookPath("true"); err != nil {
		t.Skip("/usr/bin/true not available; skipping spawn test")
	}
	prev := execCommand
	prevGoos := goos
	t.Cleanup(func() {
		execCommand = prev
		goos = prevGoos
	})
	// Force the darwin branch so browserCommand returns a non-
	// empty program — we then intercept the actual spawn.
	goos = "darwin"
	var seenName string
	var seenArgs []string
	execCommand = func(name string, args ...string) *exec.Cmd {
		seenName = name
		seenArgs = args
		return exec.Command("true")
	}

	if err := openBrowser("http://localhost:8080"); err != nil {
		t.Fatalf("openBrowser: unexpected error %v", err)
	}
	if seenName != "open" {
		t.Errorf("execCommand name: got %q, want \"open\"", seenName)
	}
	if len(seenArgs) != 1 || seenArgs[0] != "http://localhost:8080" {
		t.Errorf("execCommand args: got %v, want [\"http://localhost:8080\"]", seenArgs)
	}
}

// TestOpenBrowser_SpawnFailure_ReturnsError verifies that a failing
// exec.Cmd.Start surfaces as an error from openBrowser (so the
// scheduler logs WARN). We trigger the failure by handing back a
// Cmd targeting a non-existent absolute path.
func TestOpenBrowser_SpawnFailure_ReturnsError(t *testing.T) {
	prev := execCommand
	prevGoos := goos
	t.Cleanup(func() {
		execCommand = prev
		goos = prevGoos
	})
	goos = "linux"
	execCommand = func(_ string, _ ...string) *exec.Cmd {
		return exec.Command("/nonexistent/binary-that-cannot-possibly-exist-1234567890")
	}

	err := openBrowser("http://localhost:8080")
	if err == nil {
		t.Fatal("expected spawn failure, got nil")
	}
	// Sanity: the error mentions the command name so a log
	// reader can identify what failed.
	if !strings.Contains(err.Error(), "xdg-open") {
		t.Errorf("error should mention the program; got %q", err.Error())
	}
	// Defensive: errors.Is against a known sentinel keeps the
	// signature explicit for a future code-path tightening; for
	// now we just confirm the error is non-nil and informative.
	_ = errors.Is(err, exec.ErrNotFound)
}
