import type { Message } from "../llm/types";

/**
 * Token-count heuristic — design plan Risk R3.
 *
 * Splits every message's `content` on whitespace, concatenates the
 * resulting word arrays, multiplies the total count by 1.3, and rounds up.
 *
 * Faithful to the spec wording in Task 3 (`Math.ceil(history.flatMap(m =>
 * m.content.split(/\s+/)).length * 1.3)`). Empty strings produced by
 * leading/trailing whitespace count as words — accepted as a documented
 * imprecision for the spike; a proper tokenizer (gpt-tokenizer) is parked
 * for a follow-on phase.
 */
export function estimateTokenCount(history: ReadonlyArray<Message>): number {
  return Math.ceil(history.flatMap((m) => m.content.split(/\s+/)).length * 1.3);
}

/**
 * Sliding-window truncation. Drops oldest non-system messages until the
 * estimated token count is within `contextBudget`. A leading
 * `role:"system"` message (if present) is never removed — the system
 * prompt always primes the LLM regardless of how long the conversation
 * grows.
 *
 * Returns a new array — the input is treated as immutable so React state
 * updates remain safe.
 */
export function truncateHistory(
  history: ReadonlyArray<Message>,
  contextBudget: number,
): ReadonlyArray<Message> {
  if (contextBudget <= 0) {
    // Defensive: a non-positive budget would loop forever; keep only the
    // system prompt (if any) and discard everything else.
    const first = history[0];
    return first && first.role === "system" ? [first] : [];
  }

  const working: Message[] = [...history];
  // Index 0 is the system prompt's protected slot if and only if the
  // existing index-0 message has role "system". This matches the design
  // plan's KD-4: the caller seeds history with the system message at
  // index 0; the hook never inserts one itself.
  const systemHeld = working.length > 0 && working[0]?.role === "system";
  const removableStart = systemHeld ? 1 : 0;

  while (estimateTokenCount(working) > contextBudget) {
    if (working.length <= removableStart) {
      // Only the system prompt is left (or nothing) — further truncation
      // would violate the "always retain system" rule. Stop.
      break;
    }
    working.splice(removableStart, 1);
  }

  return working;
}

/**
 * Append a user + assistant turn to the history, then truncate. Pure
 * function — used by the reducer's COMPLETE_TURN handler.
 */
export function appendTurn(
  history: ReadonlyArray<Message>,
  userText: string,
  assistantReply: string,
  contextBudget: number,
): ReadonlyArray<Message> {
  const next: Message[] = [
    ...history,
    { role: "user", content: userText },
    { role: "assistant", content: assistantReply },
  ];
  return truncateHistory(next, contextBudget);
}
