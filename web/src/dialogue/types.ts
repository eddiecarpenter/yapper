import type { Message } from "../llm/types";
import type { Transcriber } from "../stt/types";
import type { Speaker } from "../tts/types";
import type { VAD } from "../vad/types";

/**
 * Discriminated union covering every legal state of the dialogue loop.
 *
 * The order below matches the natural turn progression:
 *
 *   idle → listening → transcribing → relaying → speaking → listening
 *
 * `error` is an out-of-band sink the hook transitions into when any stage
 * throws or the relay emits `{type:"error"}`. Auto-recovery from `error`
 * back to `listening` is implemented in Task 4.
 *
 * Using a string-literal union (rather than an enum) keeps the type
 * narrow at the `state.stage === "speaking"` check sites and avoids the
 * numeric-enum pitfalls called out in `standards/typescript.md`.
 */
export type DialogueStage =
  | "idle"
  | "listening"
  | "transcribing"
  | "relaying"
  | "speaking"
  | "error";

/**
 * Per-turn latency record — all values are milliseconds measured with
 * `performance.now()`. Captured in Task 2 and emitted via `console.table`
 * after every completed turn as part of the spike's latency report.
 */
export interface TimingRecord {
  /** Time spent inside the VAD silence-window before `onSpeechEnd` fires. */
  vad_ms: number;
  /** Wall-clock time from `onSpeechEnd` to STT returning the transcript. */
  stt_ms: number;
  /** Wall-clock time from sending the turn to receiving the first token. */
  llm_first_token_ms: number;
  /** Wall-clock time from sending the turn to receiving `{type:"done"}`. */
  llm_total_ms: number;
  /** Wall-clock time spent inside `Speaker.speak()` until playback ends. */
  tts_ms: number;
  /** End-to-end latency: VAD-end to playback-end. */
  total_ms: number;
}

/**
 * Constructor-style options passed to `useDialogue`. The hook takes
 * implementations of the three module contracts plus the relay URL — this
 * is how Risk R4 (relayUrl injection) is addressed: the URL is a hook
 * parameter, not a build-time constant.
 *
 * Dependency injection here also makes the hook unit-testable (KD-6): tests
 * pass stub `Transcriber` / `Speaker` / `VAD` implementations.
 */
export interface DialogueOptions {
  transcriber: Transcriber;
  speaker: Speaker;
  vad: VAD;
  /** WebSocket URL of the Go relay, e.g. `"ws://localhost:8080/ws"`. */
  relayUrl: string;
  /**
   * Sliding-window context budget — the maximum estimated token count of
   * the conversation history sent to the relay. Defaults to 4096 if
   * unspecified. Token estimation uses the word-count × 1.3 heuristic
   * documented in the design plan (Risk R3) and applied in Task 3.
   */
  contextBudget?: number;
}

/** Default sliding-window context budget when none is supplied. */
export const DEFAULT_CONTEXT_BUDGET = 4096;

/**
 * Externally observable state of the dialogue loop. React components
 * consuming the hook re-render whenever this object changes.
 */
export interface DialogueState {
  /** Current stage of the turn pipeline. */
  stage: DialogueStage;
  /**
   * Accumulated conversation history (oldest first). Includes both user
   * and assistant turns. A leading `system` message — if present — is
   * always retained by the sliding-window truncation logic.
   */
  history: ReadonlyArray<Message>;
  /** Timing record for the most recently completed turn, or `null`. */
  lastTiming: TimingRecord | null;
  /**
   * Actionable error message when `stage === "error"`. Cleared on the
   * next successful transition back to `listening`.
   */
  error: string | null;
}

/** Initial state used by the hook on mount and by tests as the baseline. */
export const INITIAL_DIALOGUE_STATE: DialogueState = {
  stage: "idle",
  history: [],
  lastTiming: null,
  error: null,
};
