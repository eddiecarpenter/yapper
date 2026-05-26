// Package session provides an in-memory, TTL-evicting session store
// for the Yapper relay's per-connection conversation history.
//
// Sessions are keyed by an opaque, cryptographically random ID
// (typically minted on first WebSocket upgrade and round-tripped via
// the `yapper_session` cookie). Each Session carries the LLM
// conversation history as an ordered list of llm.Message; the
// WebSocket handler (internal/api/ws.go, Feature 17 Task 4) appends
// the new user turn and the assistant reply on every interaction.
//
// The in-memory implementation is the only shape this package
// ships in the spike — see docs/ARCHITECTURE.md §10 (Evolution
// Seams) for the Redis-backed alternative the same Store interface
// will admit when multi-process scaling is needed. Keeping the
// store behind an interface here is the seam.
//
// Lifecycle:
//
//  1. NewMemoryStore(ttl, evictInterval) starts a background
//     goroutine that scans for expired sessions every evictInterval
//     and removes any whose LastSeen + ttl has elapsed.
//  2. The store is safe for concurrent use — Get / GetOrCreate /
//     Touch / Append run from many goroutines without external
//     locking by the caller.
//  3. Close() stops the eviction goroutine and releases the store.
//     Calling Close() more than once is a safe no-op. Callers
//     typically `defer store.Close()` immediately after construction.
//
// The package depends on internal/llm only for the Message type and
// has no HTTP, persistence, or YAML dependencies — that is what
// makes its unit tests fast and hermetic.
package session

import (
	"crypto/rand"
	"encoding/base64"
	"sync"
	"time"

	"github.com/eddiecarpenter/yapper/internal/llm"
)

// Session is the per-conversation record the store holds. The
// fields are exported so the WebSocket handler can read History
// directly when building the LLM request, and so tests can
// construct expected values without going through Store mutators.
//
// A Session is owned by the Store that produced it — direct
// mutation by callers is not part of the contract (use Append /
// Touch). The Store does not deep-copy Session values returned
// from Get; concurrent reads of History are safe because Append
// only ever appends (never re-slices the underlying array in
// place) but callers that intend to mutate must copy first.
type Session struct {
	// ID is the opaque, cryptographically random identifier
	// minted at session creation. Treat as opaque — no encoding
	// of timestamps, user info, or session state is permitted in
	// the ID (per design plan KD-5: ID is purely a lookup key).
	ID string

	// History is the conversation so far, oldest first. The
	// WebSocket handler appends user and assistant messages here
	// via Append; the LLM adapter consumes the slice as the
	// `messages` field of the upstream request.
	History []llm.Message

	// LastSeen is the wall-clock time of the most recent
	// interaction. The eviction goroutine compares this against
	// `time.Now() - ttl` to decide whether a session is expired.
	// Updated by Touch (and implicitly by GetOrCreate on a hit).
	LastSeen time.Time
}

// Store is the abstract session-store interface the WebSocket
// handler depends on. The in-memory implementation in this package
// (memoryStore) is the only production shape today; a Redis-backed
// implementation can be plugged in without touching call sites by
// satisfying this same interface.
//
// Semantics:
//
//   - Get returns the session for the given ID if it exists. The
//     boolean is false when no such session is known (either it was
//     never created or it has been evicted). Callers must handle
//     the absent case — there is no implicit Touch on Get.
//
//   - GetOrCreate(id) is the "session resolution" entry point for
//     the WebSocket handler. Passing an empty id mints a new
//     session with a freshly generated ID and an empty history;
//     passing an existing id returns the existing session and
//     updates LastSeen. Always returns a non-nil *Session.
//
//   - Touch updates LastSeen on an existing session and is a
//     no-op when the id is unknown. The handler calls this at the
//     start of every turn so a long-running stream does not race
//     with the eviction goroutine.
//
//   - Append appends a message to the named session's history and
//     updates LastSeen. A no-op when the id is unknown (the
//     caller should never Append against a session it has not
//     resolved via GetOrCreate first).
type Store interface {
	Get(id string) (*Session, bool)
	GetOrCreate(id string) *Session
	Touch(id string)
	Append(id string, msg llm.Message)
	// Clear removes all accumulated conversation history from the
	// named session while keeping the session alive (LastSeen is
	// updated). A no-op when id is unknown.
	Clear(id string)
}

// sessionIDByteLen is the entropy of a freshly-minted session ID
// in bytes (32 bytes = 256 bits). Base64-url encoding without
// padding emits 43 characters for 32 input bytes — short enough to
// fit in a cookie comfortably, wide enough that collisions are not
// a practical concern.
const sessionIDByteLen = 32

// generateSessionID returns a cryptographically random 256-bit ID
// encoded as URL-safe base64 without padding. crypto/rand is used
// (never math/rand) so the IDs are unguessable; the URL-safe
// encoding makes the ID safe to round-trip through a cookie
// without further escaping.
//
// A read failure from crypto/rand is treated as fatal because the
// platform's entropy source being unavailable indicates a deeply
// broken host — fabricating an ID with weak entropy would create
// a much worse security hole than a panic at startup. In practice
// rand.Read on Linux never fails.
func generateSessionID() string {
	buf := make([]byte, sessionIDByteLen)
	if _, err := rand.Read(buf); err != nil {
		panic("session: crypto/rand failed: " + err.Error())
	}
	return base64.RawURLEncoding.EncodeToString(buf)
}

// memoryStore is the in-memory Store. It is unexported because
// callers should depend on the Store interface; the concrete type
// is reached only through NewMemoryStore for construction and the
// returned *memoryStore for Close().
type memoryStore struct {
	mu       sync.Mutex
	sessions map[string]*Session
	ttl      time.Duration

	// stop is closed by Close to signal the eviction goroutine to
	// exit. closeOnce guards against multiple Close calls.
	stop      chan struct{}
	closeOnce sync.Once
}

// NewMemoryStore returns a freshly-constructed in-memory Store
// with a started background eviction goroutine. The goroutine
// wakes every evictInterval and removes any session whose
// LastSeen + ttl has elapsed.
//
// Caller MUST call Close() to stop the goroutine — typically as a
// `defer store.Close()` immediately after construction. Forgetting
// to close leaks the goroutine for the lifetime of the process.
//
// Parameter shapes:
//   - ttl ≤ 0 is permitted but means "evict immediately on first
//     scan after LastSeen" — useful for tests, not for production.
//   - evictInterval ≤ 0 disables the background goroutine entirely
//     (the goroutine is still started but exits immediately); the
//     store still works for Get/Append/Touch but never evicts.
//     The handler should always pass a positive value in production.
func NewMemoryStore(ttl, evictInterval time.Duration) *memoryStore {
	s := &memoryStore{
		sessions: make(map[string]*Session),
		ttl:      ttl,
		stop:     make(chan struct{}),
	}
	go s.evictLoop(evictInterval)
	return s
}

// Close stops the eviction goroutine. Idempotent — calling Close
// twice does not panic. After Close, Get / GetOrCreate / Touch /
// Append still work (the in-memory map is still accessible), but
// no new evictions happen. In practice the relay's main.go calls
// Close on shutdown and the store goes out of scope shortly after.
func (s *memoryStore) Close() {
	s.closeOnce.Do(func() {
		close(s.stop)
	})
}

// Get returns the session for id, or (nil, false) if not present.
// Does NOT update LastSeen — Touch is a separate operation so the
// caller can decide whether a Get counts as an interaction. The
// WebSocket handler does Touch + Append on every turn, but a hypothetical
// admin "inspect this session" path should be able to read without
// extending the session's life.
func (s *memoryStore) Get(id string) (*Session, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	return sess, ok
}

// GetOrCreate is the canonical entry point for the WebSocket
// handler: pass the cookie value (empty string when the cookie is
// absent) and receive a *Session. An empty id always mints a new
// session with a fresh ID; a non-empty id returns the existing
// session if present, or — defensively — mints a new session with
// the SUPPLIED id if it does not exist.
//
// The "mint with supplied id" branch is deliberately defensive:
// if the eviction goroutine reaped a session between the cookie
// being set and the next WebSocket upgrade, we re-create the
// (empty) session under the same id so the cookie does not have
// to be re-issued. The history is empty in that case, which is
// the only sensible reset — we cannot reconstruct evicted state.
//
// Updates LastSeen on the returned session before returning.
func (s *memoryStore) GetOrCreate(id string) *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id == "" {
		id = generateSessionID()
	}
	if existing, ok := s.sessions[id]; ok {
		existing.LastSeen = time.Now()
		return existing
	}
	fresh := &Session{
		ID:       id,
		History:  nil,
		LastSeen: time.Now(),
	}
	s.sessions[id] = fresh
	return fresh
}

// Touch updates the session's LastSeen to now. No-op when id is
// unknown. Called by the WebSocket handler at the start of every
// turn to keep an active session alive even when the turn itself
// takes longer than the TTL.
func (s *memoryStore) Touch(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sess, ok := s.sessions[id]; ok {
		sess.LastSeen = time.Now()
	}
}

// Append appends msg to the named session's history and updates
// LastSeen. No-op when id is unknown — Append against an evicted
// or non-existent session silently drops the message rather than
// auto-creating, because auto-creation here would mask a real bug
// in the caller (the WebSocket handler always resolves the session
// via GetOrCreate before any Append).
func (s *memoryStore) Append(id string, msg llm.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess, ok := s.sessions[id]
	if !ok {
		return
	}
	sess.History = append(sess.History, msg)
	sess.LastSeen = time.Now()
}

// Clear removes all accumulated history from the named session and
// updates LastSeen. A no-op when id is unknown — callers that hold
// the session cookie can always resolve the session via GetOrCreate
// first, but Clear is defined to be safe to call on an unknown id so
// callers do not have to guard it themselves.
func (s *memoryStore) Clear(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sess, ok := s.sessions[id]; ok {
		sess.History = nil
		sess.LastSeen = time.Now()
	}
}

// evictLoop is the background goroutine started by NewMemoryStore.
// Wakes every interval, scans for expired sessions, and deletes
// them. Exits when s.stop is closed (via Close()).
//
// interval ≤ 0 is treated as "do not run" — the loop exits
// immediately. This is a defensive concession to test code that
// might want a store with no background activity at all.
func (s *memoryStore) evictLoop(interval time.Duration) {
	if interval <= 0 {
		return
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-s.stop:
			return
		case now := <-t.C:
			s.evictOnce(now)
		}
	}
}

// evictOnce performs one eviction sweep. Holds the lock for the
// duration of the sweep — for a relay with at most a handful of
// active sessions this is trivial; if the spike grows past
// thousands of concurrent sessions, swap to a periodic snapshot
// + per-id delete. Kept simple for now per AD-1 (spike scope).
func (s *memoryStore) evictOnce(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, sess := range s.sessions {
		if now.Sub(sess.LastSeen) >= s.ttl {
			delete(s.sessions, id)
		}
	}
}
