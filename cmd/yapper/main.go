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
	"os/signal"
	"syscall"
	"time"

	"github.com/eddiecarpenter/yapper/internal/api"
	"github.com/eddiecarpenter/yapper/internal/config"
	"github.com/eddiecarpenter/yapper/internal/llm"
)

// shutdownTimeout caps the time the server is given to drain
// in-flight requests after a shutdown signal arrives.
const shutdownTimeout = 5 * time.Second

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
	fmt.Fprintln(w, "usage: yapper serve [--config path]")
}

// runServe parses the `serve` subcommand flags, loads configuration,
// constructs the LLM adapter and the api.Server, and runs the HTTP
// server until the supplied context is cancelled or SIGINT/SIGTERM
// arrives.
func runServe(parent context.Context, args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	cfgPath := fs.String("config", "", "path to YAML config file (optional)")
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

	srv := api.NewServer(cfg, llmClient)

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
