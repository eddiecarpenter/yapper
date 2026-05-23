// Command yapper is the Go relay server for the Yapper local-first
// voice assistant spike. It serves a small HTTP surface and (once
// Task 4 lands) a WebSocket upgrade endpoint that proxies turns from
// the browser dialogue loop to a configurable LLM backend.
//
// At this stage the binary supports a single subcommand:
//
//	yapper serve [--config path]
//
// which loads configuration via the internal/config package and
// starts an HTTP server on the configured port. The default
// configuration (AD-4) targets a local Ollama instance at
// localhost:11434/v1 with the llama3.2:3b model.
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
	"strconv"
	"syscall"
	"time"

	"github.com/eddiecarpenter/yapper/internal/config"
)

// shutdownTimeout caps the time the server is given to drain in-flight
// requests after a shutdown signal arrives.
const shutdownTimeout = 5 * time.Second

// readHeaderTimeout caps how long the server waits for request
// headers — a sane default for any net/http server exposed beyond
// localhost.
const readHeaderTimeout = 10 * time.Second

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
// and runs the HTTP server until the supplied context is cancelled
// or a signal (SIGINT / SIGTERM) arrives.
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

	srv := buildServer(cfg)

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

// buildServer constructs the http.Server and its routing mux. Task 4
// will mount the real `/ws` upgrade endpoint and adjust the `/`
// behaviour; until then `/` returns 204 No Content so reverse-proxy
// health probes succeed and any browser hit shows the relay is up
// without leaking error pages.
func buildServer(cfg *config.Config) *http.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("/", stubRootHandler)
	return &http.Server{
		Addr:              ":" + strconv.Itoa(cfg.Server.Port),
		Handler:           mux,
		ReadHeaderTimeout: readHeaderTimeout,
	}
}

func stubRootHandler(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}
