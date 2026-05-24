// Command yapper is the Go relay server for the Yapper local-first
// voice assistant spike. It serves a WebSocket relay endpoint (`/ws`)
// that proxies turns from the browser dialogue loop to a configurable
// LLM backend, plus a small stub HTTP surface for health probes.
//
// The binary supports a single subcommand:
//
//	yapper serve [--config path]
//
// which loads configuration via the internal/config package,
// constructs the LLM adapter via llm.NewLLMClient (Ollama by default,
// per AD-4), wires both into the api.NewServer mux, and runs an HTTP
// server on the configured port. Shutdown on SIGINT/SIGTERM is
// graceful with a bounded drain window.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/eddiecarpenter/yapper/internal/api"
	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
	"github.com/eddiecarpenter/yapper/internal/session"
)

// goos is a package-level indirection over runtime.GOOS so tests
// can pin the platform branch without recompiling. Production code
// reads this exclusively (never runtime.GOOS directly) so the test
// override has effect.
var goos = runtime.GOOS

// execCommand is a package-level indirection over exec.Command so
// tests can intercept the spawn. Production code reads this
// exclusively (never exec.Command directly) so the test override
// has effect. The signature matches exec.Command exactly.
var execCommand = exec.Command

// sessionEvictInterval is how often the in-memory session.Store
// scans for expired sessions. One minute is the spike default —
// well below the configured session TTL (30 minutes) so the
// upper bound on a session living past its expiry is one
// interval (60 s), while the goroutine overhead is negligible.
const sessionEvictInterval = time.Minute

// shutdownTimeout caps the time the server is given to drain
// in-flight requests after a shutdown signal arrives.
const shutdownTimeout = 5 * time.Second

// browserOpenDelay is the wait between the server starting its
// listener goroutine and the browser-open helper firing. 100 ms is
// more than enough for the listener to bind on any sane host but
// short enough that the human perceives "yapper serve" as
// instant. Per Task 5 notes the upper bound is 250 ms.
const browserOpenDelay = 100 * time.Millisecond

func main() {
	if err := run(context.Background(), os.Args[1:], os.Stderr); err != nil {
		log.Fatalf("yapper: %v", err)
	}
}

// run is the argv-driven entry point, broken out from main so tests
// can drive subcommand dispatch without spawning the process.
func run(ctx context.Context, args []string, stderr io.Writer) error {
	if len(args) == 0 {
		printUsage(stderr)
		return errors.New("missing subcommand")
	}
	switch args[0] {
	case "serve":
		return runServe(ctx, args[1:])
	case "-h", "--help", "help":
		printUsage(stderr)
		return nil
	default:
		fmt.Fprintf(stderr, "yapper: unknown subcommand %q\n", args[0])
		printUsage(stderr)
		return fmt.Errorf("unknown subcommand %q", args[0])
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: yapper serve [--config path] [--open=true|false]")
}

// browserCommand returns the (program, args) tuple that opens the
// user's default browser at url for the current platform. The
// helper is split out from openBrowser so a table-driven test can
// assert the dispatch logic without running exec.Command.
//
// Unsupported platforms (anything other than darwin / linux /
// windows) return ("", nil) — the caller treats that as "no-op
// with a WARN log" rather than an error.
func browserCommand(url string) (string, []string) {
	switch goos {
	case "darwin":
		// `open <url>` — do NOT use `open -a`, which would force
		// a specific application; the user's default browser is
		// what we want (per Task 5 notes).
		return "open", []string{url}
	case "linux":
		// xdg-open is the cross-DE standard on Linux; missing on
		// some minimal containers but that is a config issue, not
		// a yapper bug. The WARN on spawn failure surfaces it.
		return "xdg-open", []string{url}
	case "windows":
		// `cmd /c start "" <url>` — the empty title is critical;
		// without it `start` interprets the first quoted arg as
		// the window title. Quoting via the slice means no shell
		// escaping is needed on a URL.
		return "cmd", []string{"/c", "start", "", url}
	default:
		return "", nil
	}
}

// openBrowser spawns the platform-appropriate browser-open
// command for url. Returns nil on a successful spawn; an error
// when no command is available for the current platform or when
// exec.Command.Start fails.
//
// The function does NOT block waiting for the spawned process —
// we want the server's listener up and accepting connections
// regardless of how long the OS takes to open a browser window.
func openBrowser(url string) error {
	name, args := browserCommand(url)
	if name == "" {
		return fmt.Errorf("openBrowser: unsupported platform %q", goos)
	}
	cmd := execCommand(name, args...)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("openBrowser: %s: %w", name, err)
	}
	// Reap the child asynchronously so we do not leak a zombie
	// on Linux/macOS. Wait blocks until the browser process
	// exits, but the spawn already returned — running this in a
	// goroutine keeps the binary's foreground loop unblocked.
	go func() { _ = cmd.Wait() }()
	return nil
}

// scheduleBrowserOpen kicks off the browser-open helper from a
// goroutine after browserOpenDelay so the listener has a chance
// to bind first. A spawn failure is logged at WARN-prefix and is
// NEVER fatal — per AC, headless CI must be able to set
// --open=false to disable the helper outright, and a missing
// xdg-open on a CI box must not change server exit status.
//
// The function returns immediately — it does NOT wait for the
// browser to actually open. This is the "do not block startup"
// rule from the Task 5 notes.
func scheduleBrowserOpen(url string) {
	go func() {
		time.Sleep(browserOpenDelay)
		if err := openBrowser(url); err != nil {
			log.Printf("WARN: browser auto-open failed (%v); navigate to %s manually.", err, url)
		}
	}()
}

// runServe parses the `serve` subcommand flags, loads configuration,
// constructs the LLM adapter and the api.Server, and runs the HTTP
// server until the supplied context is cancelled or SIGINT/SIGTERM
// arrives.
func runServe(parent context.Context, args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to YAML config file (optional)")
	openBrowserFlag := fs.Bool("open", true, "open the default browser at the SPA URL on startup")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		return err
	}

	llmClient, err := llm.NewLLMClient(cfg.LLM)
	if err != nil {
		return fmt.Errorf("init llm client: %w", err)
	}

	// Server-side, cookie-keyed conversation history (Feature 17
	// design plan KD-1). Closed at shutdown so the eviction
	// goroutine exits cleanly.
	store := session.NewMemoryStore(
		time.Duration(cfg.Server.SessionTTLMinutes)*time.Minute,
		sessionEvictInterval,
	)
	defer store.Close()

	srv := api.NewServer(cfg, llmClient, store)

	log.Printf("server starting on %s (llm=%s/%s base_url=%s api_key=%s)",
		srv.Addr, cfg.LLM.Provider, cfg.LLM.Model, cfg.LLM.BaseURL,
		cfg.LLM.GetMaskedAPIKey())

	ctx, cancel := signal.NotifyContext(parent, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	errCh := make(chan error, 1)
	go func() {
		err := srv.ListenAndServe()
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		errCh <- err
	}()

	// Fire the browser-open helper AFTER the listener goroutine
	// is started but BEFORE the select. The helper sleeps
	// briefly so the listener wins the race in practice; a
	// failure inside the helper logs WARN and never affects
	// the server's lifecycle.
	if *openBrowserFlag {
		scheduleBrowserOpen(fmt.Sprintf("http://localhost:%d", cfg.Server.Port))
	}

	select {
	case <-ctx.Done():
		log.Printf("server shutting down")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer shutdownCancel()
		return srv.Shutdown(shutdownCtx)
	case err := <-errCh:
		return err
	}
}
