import type { DialogueOptions, DialogueState } from "./types";
import { INITIAL_DIALOGUE_STATE } from "./types";

/**
 * `useDialogue` — turn-based browser voice loop (Feature #12).
 *
 * Task 1 of 4 — this is the scaffold stub. The hook accepts the full
 * `DialogueOptions` shape and returns the initial idle state. No runtime
 * logic (no `useEffect`, no WebSocket, no VAD wiring) is present yet —
 * Tasks 2–4 layer that in:
 *
 *   - Task 2 — voice turn loop + per-stage timing
 *   - Task 3 — conversation history + sliding-window truncation
 *   - Task 4 — error handling + lifecycle cleanup
 *
 * The signature is final at this point so consumers can begin wiring
 * `useDialogue` into the SPA scaffold (Feature #13) without waiting for
 * Tasks 2–4 to land.
 *
 * @param opts  Hook configuration — see `DialogueOptions`.
 * @returns     `DialogueState` representing the current loop status.
 */
export function useDialogue(opts: DialogueOptions): DialogueState {
  // The options are accepted but intentionally unused in this scaffold —
  // reading a field here silences the "unused parameter" warning without
  // pretending to do work. Tasks 2–4 will consume `opts` in earnest.
  void opts.transcriber;
  void opts.speaker;
  void opts.vad;
  void opts.relayUrl;
  void opts.contextBudget;

  return INITIAL_DIALOGUE_STATE;
}
