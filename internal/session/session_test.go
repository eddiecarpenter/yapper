package session

import (
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/eddiecarpenter/yapper/internal/llm"
)

// newTestStore returns a store with a 1-second TTL and a 1-second
// eviction tick. Tests that need to exercise eviction override
// these values via NewMemoryStore directly; the helper exists so
// the "doesn't care about eviction" tests do not have to think
// about timings.
func newTestStore(t *testing.T) *memoryStore {
	t.Helper()
	s := NewMemoryStore(1*time.Second, 1*time.Second)
	t.Cleanup(s.Close)
	return s
}

func TestGetOrCreate_EmptyID_MintsFreshSession(t *testing.T) {
	s := newTestStore(t)

	sess := s.GetOrCreate("")

	if sess == nil {
		t.Fatal("GetOrCreate(\"\") returned nil")
	}
	if sess.ID == "" {
		t.Error("minted session has empty ID")
	}
	// 256 bits in base64-url-no-pad = 43 chars (32 bytes × 4/3,
	// rounded up, minus padding).
	if got := len(sess.ID); got != 43 {
		t.Errorf("session ID length: got %d, want 43", got)
	}
	// base64-url alphabet — no `+` or `/` or `=`.
	if strings.ContainsAny(sess.ID, "+/=") {
		t.Errorf("session ID %q contains non-url-safe characters", sess.ID)
	}
	if len(sess.History) != 0 {
		t.Errorf("fresh session has %d messages, want 0", len(sess.History))
	}
	if sess.LastSeen.IsZero() {
		t.Error("fresh session has zero LastSeen")
	}
}

func TestGetOrCreate_EmptyID_MintsUniqueIDs(t *testing.T) {
	s := newTestStore(t)

	const n = 100
	seen := make(map[string]struct{}, n)
	for i := 0; i < n; i++ {
		sess := s.GetOrCreate("")
		if _, dup := seen[sess.ID]; dup {
			t.Fatalf("duplicate session ID minted at iteration %d: %s", i, sess.ID)
		}
		seen[sess.ID] = struct{}{}
	}
}

func TestGetOrCreate_ExistingID_ReturnsSameSession(t *testing.T) {
	s := newTestStore(t)

	first := s.GetOrCreate("")
	second := s.GetOrCreate(first.ID)

	if first != second {
		t.Errorf("GetOrCreate(existing) returned a different pointer: first=%p second=%p", first, second)
	}
	if first.ID != second.ID {
		t.Errorf("IDs differ: %q vs %q", first.ID, second.ID)
	}
}

func TestGet_PresentAndAbsent(t *testing.T) {
	s := newTestStore(t)
	sess := s.GetOrCreate("")

	got, ok := s.Get(sess.ID)
	if !ok {
		t.Fatal("Get(existing) returned ok=false")
	}
	if got != sess {
		t.Errorf("Get(existing) returned different pointer")
	}

	_, ok = s.Get("nonexistent-session-id")
	if ok {
		t.Error("Get(unknown) returned ok=true")
	}
}

func TestAppend_ThenGet_ReturnsAppendedMessages(t *testing.T) {
	s := newTestStore(t)
	sess := s.GetOrCreate("")

	s.Append(sess.ID, llm.Message{Role: "user", Content: "hello"})
	s.Append(sess.ID, llm.Message{Role: "assistant", Content: "hi back"})
	s.Append(sess.ID, llm.Message{Role: "user", Content: "follow up"})

	got, ok := s.Get(sess.ID)
	if !ok {
		t.Fatal("Get failed")
	}
	if len(got.History) != 3 {
		t.Fatalf("history length: got %d, want 3", len(got.History))
	}
	wantRoles := []string{"user", "assistant", "user"}
	for i, m := range got.History {
		if m.Role != wantRoles[i] {
			t.Errorf("history[%d].Role: got %q, want %q", i, m.Role, wantRoles[i])
		}
	}
}

func TestAppend_UnknownID_IsNoop(t *testing.T) {
	s := newTestStore(t)

	// Should not panic, should not auto-create.
	s.Append("never-created", llm.Message{Role: "user", Content: "x"})

	if _, ok := s.Get("never-created"); ok {
		t.Error("Append(unknown) auto-created a session")
	}
}

func TestTouch_UpdatesLastSeen(t *testing.T) {
	s := newTestStore(t)
	sess := s.GetOrCreate("")

	before := sess.LastSeen
	// Sleep long enough that the wall clock advances reliably
	// (time.Now precision varies by platform).
	time.Sleep(2 * time.Millisecond)
	s.Touch(sess.ID)

	got, _ := s.Get(sess.ID)
	if !got.LastSeen.After(before) {
		t.Errorf("Touch did not advance LastSeen: before=%v after=%v", before, got.LastSeen)
	}
}

func TestTouch_UnknownID_IsNoop(t *testing.T) {
	s := newTestStore(t)
	// Should not panic, should not create.
	s.Touch("nonexistent")
	if _, ok := s.Get("nonexistent"); ok {
		t.Error("Touch(unknown) created a session")
	}
}

func TestEviction_RemovesExpiredSessions(t *testing.T) {
	// 10 ms TTL, 5 ms eviction interval — well below any test
	// runner's flake budget but enough that the background sweep
	// fires before the assertion.
	s := NewMemoryStore(10*time.Millisecond, 5*time.Millisecond)
	defer s.Close()

	sess := s.GetOrCreate("")
	id := sess.ID

	// Wait long enough that LastSeen + TTL has elapsed AND the
	// eviction goroutine has had a chance to run.
	time.Sleep(50 * time.Millisecond)

	if _, ok := s.Get(id); ok {
		t.Errorf("session %s not evicted after TTL + interval", id)
	}
}

func TestEviction_LeavesFreshSessions(t *testing.T) {
	// Long TTL, frequent eviction — we expect NO evictions.
	s := NewMemoryStore(10*time.Second, 5*time.Millisecond)
	defer s.Close()

	sess := s.GetOrCreate("")
	id := sess.ID

	// Wait for several eviction ticks.
	time.Sleep(30 * time.Millisecond)

	if _, ok := s.Get(id); !ok {
		t.Error("fresh session evicted incorrectly")
	}
}

func TestClose_StopsEvictionGoroutine(t *testing.T) {
	// We cannot directly observe goroutine death from the test
	// runtime without runtime.NumGoroutine plumbing, but we can
	// observe that Close is safe to call twice and that the
	// store remains usable for Get afterwards.
	s := NewMemoryStore(1*time.Second, 1*time.Millisecond)
	sess := s.GetOrCreate("")

	s.Close()
	s.Close() // idempotent — must not panic

	// Store remains accessible after close.
	if _, ok := s.Get(sess.ID); !ok {
		t.Error("Get failed after Close — store should still serve from memory")
	}
}

func TestStoreSatisfiesInterface(t *testing.T) {
	// Compile-time check that *memoryStore satisfies Store. If a
	// future refactor breaks this, the test will fail to compile.
	var _ Store = (*memoryStore)(nil)
	_ = NewMemoryStore(time.Second, time.Second)
}

func TestConcurrentAppend_NoRaceOnSameSession(t *testing.T) {
	// `go test -race` is the canonical guard here — this test
	// simply gives the race detector something to chew on. The
	// mutex inside memoryStore must serialise concurrent Appends
	// on the same session without deadlock or data corruption.
	s := newTestStore(t)
	sess := s.GetOrCreate("")

	const writers = 8
	const each = 100

	var wg sync.WaitGroup
	wg.Add(writers)
	for i := 0; i < writers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < each; j++ {
				s.Append(sess.ID, llm.Message{Role: "user", Content: "x"})
			}
		}()
	}
	wg.Wait()

	got, _ := s.Get(sess.ID)
	if len(got.History) != writers*each {
		t.Errorf("history length after %d×%d concurrent appends: got %d, want %d",
			writers, each, len(got.History), writers*each)
	}
}
