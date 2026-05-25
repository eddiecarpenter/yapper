package api

import "strings"

// thinkFilter strips <think>…</think> reasoning blocks from a
// streaming token sequence. Qwen3 (and similar chain-of-thought
// models) may emit these blocks even when enable_thinking:false is
// set — the filter is a safety net so reasoning tokens never reach
// the browser or get spoken aloud.
//
// The filter is stateful: a single tag can span multiple delta chunks
// (e.g. "<thi" in one chunk, "nk>" in the next), so a simple
// strings.Replace per chunk would miss split boundaries.
//
// Usage:
//
//	f := newThinkFilter()
//	for each streaming token tok {
//	    visible := f.Write(tok)
//	    if visible != "" { send(visible) }
//	}
type thinkFilter struct {
	// buf accumulates characters when we're inside a potential or
	// confirmed <think> tag or inside a confirmed block.
	buf string
	// inBlock is true once we have seen a complete <think> opener and
	// are swallowing content until </think>.
	inBlock bool
}

func newThinkFilter() *thinkFilter { return &thinkFilter{} }

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// Write accepts the next delta token and returns the portion that
// should be forwarded to the browser (empty string = swallow it).
func (f *thinkFilter) Write(tok string) string {
	f.buf += tok
	var out strings.Builder

	for len(f.buf) > 0 {
		if f.inBlock {
			// Swallow everything until we see </think>.
			idx := strings.Index(f.buf, thinkClose)
			if idx == -1 {
				// Close tag not yet arrived — keep buffering.
				return out.String()
			}
			// Consume through the end of </think> and exit block mode.
			f.buf = f.buf[idx+len(thinkClose):]
			f.inBlock = false
			continue
		}

		// Not in a block — look for the next <think> opener.
		idx := strings.Index(f.buf, thinkOpen)
		if idx == -1 {
			// No opener in buf. But the tail might be a partial opener
			// (e.g. buf ends with "<thi") — hold back enough chars to
			// detect it on the next Write, and flush the rest.
			safe := safeFlushLen(f.buf, thinkOpen)
			out.WriteString(f.buf[:safe])
			f.buf = f.buf[safe:]
			return out.String()
		}

		// Flush everything before the opener, then enter block mode.
		out.WriteString(f.buf[:idx])
		f.buf = f.buf[idx+len(thinkOpen):]
		f.inBlock = true
	}

	return out.String()
}

// safeFlushLen returns the largest prefix of s that cannot be the
// start of prefix. Characters beyond that index are safe to emit;
// characters from that index onward must be held in the buffer in
// case the next Write completes the tag.
func safeFlushLen(s, prefix string) int {
	// Check whether any suffix of s is a prefix of the tag opener.
	for n := len(prefix) - 1; n > 0; n-- {
		if strings.HasSuffix(s, prefix[:n]) {
			return len(s) - n
		}
	}
	return len(s)
}
