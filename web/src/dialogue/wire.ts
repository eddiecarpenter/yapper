/**
 * WebSocket wire-format types — the relay's protocol per
 * `docs/ARCHITECTURE.md` §5.2.
 *
 * Browser sends a `TurnRequest`; the relay streams zero-or-more
 * `TokenFrame`s followed by exactly one `DoneFrame`. An `ErrorFrame`
 * may be emitted at any time and terminates the in-flight turn.
 */
import type { Message } from "../llm/types";

/** Sample rate of the audio segment passed to `Transcriber.transcribe()`. */
export const STT_SAMPLE_RATE_HZ = 16000;

/**
 * Tagged relay-side error. `source === "connection"` means the WebSocket
 * itself failed (send threw, or the socket closed mid-turn) — the hook
 * surfaces this as a generic "Relay unreachable" message. `source ===
 * "frame"` means the relay sent a `{type:"error"}` payload — the hook
 * surfaces that exact `message` to the user, since the relay's framing
 * encodes upstream diagnostics (e.g. "Ollama unreachable at ...").
 */
export class RelayError extends Error {
  public readonly source: "connection" | "frame";

  constructor(message: string, source: "connection" | "frame") {
    super(message);
    this.name = "RelayError";
    this.source = source;
  }
}

export interface TurnRequest {
  type: "turn";
  text: string;
  history: ReadonlyArray<Message>;
}

export interface TokenFrame {
  type: "token";
  text: string;
}

export interface DoneFrame {
  type: "done";
  usage?: { input: number; output: number };
}

export interface ErrorFrame {
  type: "error";
  message: string;
}

export type RelayFrame = TokenFrame | DoneFrame | ErrorFrame;

/**
 * Parse an incoming WebSocket message. Returns `null` if the payload is
 * malformed (non-JSON, missing `type`, or unrecognised `type`). The hook
 * treats a `null` parse result as a silently-discarded frame in Task 2;
 * Task 4 (error handling) wraps the call so malformed frames are surfaced
 * as an actionable error.
 */
export function parseRelayFrame(raw: unknown): RelayFrame | null {
  if (typeof raw !== "string") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== "object") return null;
  const obj = parsed as Record<string, unknown>;
  const type = obj.type;
  if (type === "token" && typeof obj.text === "string") {
    return { type: "token", text: obj.text };
  }
  if (type === "done") {
    const usage = obj.usage;
    if (usage !== null && typeof usage === "object") {
      const u = usage as Record<string, unknown>;
      if (typeof u.input === "number" && typeof u.output === "number") {
        return { type: "done", usage: { input: u.input, output: u.output } };
      }
    }
    return { type: "done" };
  }
  if (type === "error" && typeof obj.message === "string") {
    return { type: "error", message: obj.message };
  }
  return null;
}
