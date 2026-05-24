package api

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

// makeSPAFS returns a test-only fs.FS shaped like the production
// SPA: index.html at the root, plus an assets/ subdir with a JS and
// a CSS file. testing/fstest.MapFS satisfies fs.FS and fs.StatFS,
// which is what newSPAHandler reads through.
func makeSPAFS() fstest.MapFS {
	return fstest.MapFS{
		"index.html":        {Data: []byte("<!doctype html><div id=\"root\"></div>")},
		"assets/app.js":     {Data: []byte("console.log('yapper')")},
		"assets/styles.css": {Data: []byte("body { font-family: sans-serif }")},
	}
}

// makeEmptySPAFS models the "developer forgot to npm run build"
// state — the embed succeeded but there is no index.html.
func makeEmptySPAFS() fstest.MapFS {
	return fstest.MapFS{}
}

func TestSPAHandler_KnownFile_ServesEmbeddedBytes(t *testing.T) {
	handler := newSPAHandler(makeSPAFS())
	srv := httptest.NewServer(handler)
	defer srv.Close()

	cases := []struct {
		path       string
		wantBody   string
		wantPrefix string // content-type prefix (full type includes charset)
	}{
		{"/index.html", "<!doctype html>", "text/html"},
		{"/assets/app.js", "console.log", "text/javascript"},
		{"/assets/styles.css", "body { font-family", "text/css"},
	}
	for _, tc := range cases {
		t.Run(tc.path, func(t *testing.T) {
			resp, err := http.Get(srv.URL + tc.path)
			if err != nil {
				t.Fatalf("GET %s: %v", tc.path, err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("GET %s: status %d, want 200", tc.path, resp.StatusCode)
			}
			body, _ := io.ReadAll(resp.Body)
			if !strings.Contains(string(body), tc.wantBody) {
				t.Errorf("GET %s: body %q does not contain %q", tc.path, body, tc.wantBody)
			}
			ct := resp.Header.Get("Content-Type")
			if !strings.HasPrefix(ct, tc.wantPrefix) {
				t.Errorf("GET %s: content-type %q does not start with %q", tc.path, ct, tc.wantPrefix)
			}
		})
	}
}

func TestSPAHandler_RootPath_ServesIndex(t *testing.T) {
	srv := httptest.NewServer(newSPAHandler(makeSPAFS()))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("GET /: status %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "<!doctype html>") {
		t.Errorf("GET /: body does not look like index.html: %q", body)
	}
}

func TestSPAHandler_UnknownExtensionlessPath_FallsBackToIndex(t *testing.T) {
	// Client-route deep links — e.g. /settings or /conversations/42 —
	// must resolve to index.html so the SPA's client-side router
	// renders them.
	srv := httptest.NewServer(newSPAHandler(makeSPAFS()))
	defer srv.Close()

	for _, path := range []string{"/settings", "/conversations/42", "/deep/nested/route"} {
		t.Run(path, func(t *testing.T) {
			resp, err := http.Get(srv.URL + path)
			if err != nil {
				t.Fatalf("GET %s: %v", path, err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Errorf("GET %s: status %d, want 200 (SPA fallback)", path, resp.StatusCode)
			}
			body, _ := io.ReadAll(resp.Body)
			if !strings.Contains(string(body), "<!doctype html>") {
				t.Errorf("GET %s: fallback did not return index.html: %q", path, body)
			}
		})
	}
}

func TestSPAHandler_UnknownAssetPath_Returns404(t *testing.T) {
	// A path with an extension is asset-shaped; a miss is a real
	// 404, not a SPA deep-link. We do not fall back so that a typo
	// or a missing chunk reports honestly rather than masquerading
	// as the bootstrap HTML.
	srv := httptest.NewServer(newSPAHandler(makeSPAFS()))
	defer srv.Close()

	for _, path := range []string{"/missing.css", "/assets/nope.js", "/favicon.ico"} {
		t.Run(path, func(t *testing.T) {
			resp, err := http.Get(srv.URL + path)
			if err != nil {
				t.Fatalf("GET %s: %v", path, err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusNotFound {
				t.Errorf("GET %s: status %d, want 404", path, resp.StatusCode)
			}
		})
	}
}

func TestSPAHandler_EmptyFS_AllRoutes404(t *testing.T) {
	// The "forgot to npm run build" path. The handler must not
	// crash, and every route returns 404 — including the SPA
	// fallback (which has nothing to fall back to).
	srv := httptest.NewServer(newSPAHandler(makeEmptySPAFS()))
	defer srv.Close()

	for _, path := range []string{"/", "/index.html", "/settings", "/missing.css"} {
		t.Run(path, func(t *testing.T) {
			resp, err := http.Get(srv.URL + path)
			if err != nil {
				t.Fatalf("GET %s: %v", path, err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusNotFound {
				t.Errorf("GET %s: status %d, want 404 (empty embed)", path, resp.StatusCode)
			}
		})
	}
}

func TestSpaFS_PointsAtDist(t *testing.T) {
	// Smoke test: the package-level spaFS() resolves to a non-nil
	// fs.FS. Whether index.html is present depends on whether the
	// SPA was built before `go test` ran (see makeSPAFS for the
	// hermetic test path). We assert only that the wrapper does
	// not return nil and that read attempts don't panic — the
	// behavioural assertions live in the tests above against
	// fstest.MapFS, which is hermetic.
	fsys := spaFS()
	if fsys == nil {
		t.Fatal("spaFS returned nil")
	}
	// Open is permitted to error (empty embed) but must not panic.
	if f, err := fsys.Open("nonexistent-probe.txt"); err == nil {
		_ = f.Close()
	}
}
