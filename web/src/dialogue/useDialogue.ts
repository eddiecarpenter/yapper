import { useEffect, useReducer, useRef } from "react";

import type { Message } from "../llm/types";
import { appendTurn } from "./history";
import type { DialogueOptions, DialogueStage, DialogueState, TimingRecord } from "./types";
import { DEFAULT_CONTEXT_BUDGET, INITIAL_DIALOGUE_STATE } from "./types";
import type { RelayFrame } from "./wire";
import { RelayError, STT_SAMPLE_RATE_HZ, parseRelayFrame } from "./wire";

/**
 * Delay between entering the `"error"` stage and auto-recovering to
 * `"listening"`. Exported for tests (so they can advance fake timers by
 * exactly this much) and so a follow-on tuning task can adjust the value
 * in one place.
 */
export const ERROR_RECOVERY_DELAY_MS = 500;

/**
 * `useDialogue` — turn-based browser voice loop (Feature #12).
 *
 * Task 2 of 4 layers the sequential turn pipeline on top of Task 1's
 * scaffold. The hook:
 *
 *   1. Opens a WebSocket to `relayUrl` on mount.
 *   2. Registers `vad.onSpeechEnd` as the speech-segment callback.
 *   3. Drives the pipeline VAD → STT → relay → TTS → listening on each
 *      utterance, dispatching stage transitions at every boundary.
 *   4. Emits a `TimingRecord` via `console.table` per turn.
 *   5. Closes the WebSocket and detaches the VAD callback on unmount.
 *
 * History accumulation, sliding-window truncation, error handling, and
 * resource teardown beyond `ws.close()` arrive in Tasks 3 and 4.
 */

type Action =
  | { type: "STAGE"; stage: DialogueStage }
  | { type: "ERROR"; message: string }
  | {
      type: "COMPLETE_TURN";
      timing: TimingRecord;
      userText: string;
      assistantReply: string;
      contextBudget: number;
    };

function reducer(state: DialogueState, action: Action): DialogueState {
  switch (action.type) {
    case "STAGE":
      return { ...state, stage: action.stage, error: null };
    case "ERROR":
      // Preserve history and lastTiming — only the stage + error string
      // change. This is what lets the UI display the message while still
      // showing the prior conversation.
      return { ...state, stage: "error", error: action.message };
    case "COMPLETE_TURN": {
      const nextHistory = appendTurn(
        state.history,
        action.userText,
        action.assistantReply,
        action.contextBudget,
      );
      return {
        ...state,
        stage: "listening",
        lastTiming: action.timing,
        error: null,
        history: nextHistory,
      };
    }
    default: {
      // Exhaustiveness check — TS will flag if a new Action variant is added
      // without a corresponding case.
      const _exhaustive: never = action;
      void _exhaustive;
      return state;
    }
  }
}

/**
 * Stage-specific actionable error messages — the source of truth for what
 * the UI sees in `state.error`. Defined as a const map so a future
 * localisation pass or message-format refresh edits one place, and so
 * tests can assert against the canonical strings.
 */
const STAGE_ERROR_MESSAGES = {
  stt: "Speech recognition failed — check WebGPU/WASM availability",
  tts: "Speech synthesis failed — check WebGPU/WASM availability",
  relayUnreachable: (relayUrl: string): string =>
    `Relay unreachable — check that the Go server is running at ${relayUrl}`,
} as const;

/**
 * Internal record returned by the per-turn token-collector promise.
 * Kept private so Task 3/4 can extend the shape (e.g. with `usage`)
 * without churning the public API.
 */
interface RelayTurnResult {
  reply: string;
  /** `performance.now()` at the moment the first `token` frame arrived. */
  tFirstMs: number;
  /** `performance.now()` at the moment the `done` frame arrived. */
  tDoneMs: number;
}

export function useDialogue(opts: DialogueOptions): DialogueState {
  // Lazy initial state — seeded with `opts.initialHistory` on first render
  // only. The lazy form prevents re-rendering with a new `initialHistory`
  // from clobbering an in-progress conversation; the system prompt seed
  // is a mount-time concern, not a re-render concern.
  const [state, dispatch] = useReducer<typeof reducer, DialogueOptions>(
    reducer,
    opts,
    (o: DialogueOptions): DialogueState => ({
      ...INITIAL_DIALOGUE_STATE,
      history: o.initialHistory ?? [],
    }),
  );

  // Latest-opts ref pattern: the effect runs once on mount, but the async
  // turn handler must see the *current* dependency references in case the
  // caller passes new instances on re-render. We don't tear down the
  // WebSocket on every prop change — that's the point of the pattern.
  const optsRef = useRef<DialogueOptions>(opts);
  optsRef.current = opts;

  // History ref — points at the current accumulated history so the async
  // turn handler can send the *pre-append* snapshot to the relay (per
  // Task 3: "history passed to the relay ... includes all current history
  // before appending the new user message").
  const historyRef = useRef<ReadonlyArray<Message>>(state.history);
  historyRef.current = state.history;

  // Track the current stage in a ref so the useEffect cleanup closure can
  // observe the live value (e.g. "is the loop currently speaking?") without
  // having to depend on `state` and re-run.
  const stageRef = useRef<DialogueStage>(state.stage);
  stageRef.current = state.stage;

  // Holds the auto-recovery timer ID so it can be cleared on unmount or on
  // a subsequent error before the previous one had time to recover.
  const recoveryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const { transcriber, speaker, vad, relayUrl } = optsRef.current;

    // `unmounted` short-circuits any in-flight async work so we never
    // dispatch into an unmounted hook (React would warn).
    let unmounted = false;

    const ws = new WebSocket(relayUrl);

    const handleOpen = (): void => {
      if (unmounted) return;
      dispatch({ type: "STAGE", stage: "listening" });
    };
    ws.addEventListener("open", handleOpen);

    /**
     * Surfaces an actionable error and schedules auto-recovery back to
     * `"listening"` after `ERROR_RECOVERY_DELAY_MS`. Idempotent: a fresh
     * error during the recovery window restarts the timer.
     */
    const raiseError = (message: string): void => {
      if (unmounted) return;
      dispatch({ type: "ERROR", message });

      if (recoveryTimerRef.current !== null) {
        clearTimeout(recoveryTimerRef.current);
      }
      recoveryTimerRef.current = setTimeout(() => {
        recoveryTimerRef.current = null;
        if (unmounted) return;
        dispatch({ type: "STAGE", stage: "listening" });
      }, ERROR_RECOVERY_DELAY_MS);
    };

    /**
     * Drives one full turn end-to-end. Bound to `vad.onSpeechEnd` below.
     * Each invocation is independent — the WebSocket is shared, but
     * per-turn state (accumulated tokens, t_first) lives inside this
     * closure.
     *
     * Each pipeline stage is wrapped in `try/catch` so a failure surfaces
     * an actionable message and triggers auto-recovery. A `return` after
     * `raiseError` halts the turn — the next VAD trigger starts cleanly.
     */
    const handleSpeechEnd = async (segment: Float32Array): Promise<void> => {
      if (unmounted) return;

      // ── 1. VAD end → transcribing ─────────────────────────────────────
      const t0 = performance.now();
      dispatch({ type: "STAGE", stage: "transcribing" });

      let text: string;
      try {
        text = await transcriber.transcribe(segment, STT_SAMPLE_RATE_HZ);
      } catch {
        raiseError(STAGE_ERROR_MESSAGES.stt);
        return;
      }
      const tStt = performance.now();
      if (unmounted) return;

      // ── 2. transcribing → relaying ────────────────────────────────────
      dispatch({ type: "STAGE", stage: "relaying" });

      let turnResult: RelayTurnResult;
      try {
        turnResult = await collectRelayTurn(ws, text, historyRef.current);
      } catch (e) {
        if (e instanceof RelayError) {
          if (e.source === "frame") {
            // Surface the relay's own diagnostic verbatim — it encodes
            // upstream context ("Ollama unreachable at …") that a generic
            // message would lose.
            raiseError(e.message);
          } else {
            raiseError(STAGE_ERROR_MESSAGES.relayUnreachable(optsRef.current.relayUrl));
          }
        } else {
          raiseError(STAGE_ERROR_MESSAGES.relayUnreachable(optsRef.current.relayUrl));
        }
        return;
      }
      const { reply, tFirstMs, tDoneMs } = turnResult;
      if (unmounted) return;

      // ── 3. relaying → speaking ───────────────────────────────────────
      dispatch({ type: "STAGE", stage: "speaking" });

      try {
        await speaker.speak(reply);
      } catch {
        raiseError(STAGE_ERROR_MESSAGES.tts);
        return;
      }
      const tTts = performance.now();
      if (unmounted) return;

      // ── 4. speaking → listening + emit timing ───────────────────────
      const timing: TimingRecord = {
        vad_ms: 0,
        stt_ms: tStt - t0,
        llm_first_token_ms: tFirstMs - tStt,
        llm_total_ms: tDoneMs - tStt,
        tts_ms: tTts - tDoneMs,
        total_ms: tTts - t0,
      };
      // Console-table is the spike's per-turn latency report — see
      // PROJECT_BRIEF.md §"Deliverables / Latency report".
      console.table(timing);

      dispatch({
        type: "COMPLETE_TURN",
        timing,
        userText: text,
        assistantReply: reply,
        // Read budget from the latest opts so a caller can tighten or
        // loosen the window across renders.
        contextBudget: optsRef.current.contextBudget ?? DEFAULT_CONTEXT_BUDGET,
      });
    };

    vad.onSpeechEnd = handleSpeechEnd;

    return () => {
      unmounted = true;

      // Cancel any pending auto-recovery timer so it cannot fire after
      // unmount.
      if (recoveryTimerRef.current !== null) {
        clearTimeout(recoveryTimerRef.current);
        recoveryTimerRef.current = null;
      }

      // If the loop is mid-utterance, stop playback first so the user's
      // ears are not held hostage by a teardown.
      if (stageRef.current === "speaking") {
        try {
          speaker.cancel();
        } catch {
          // best-effort — never throw out of a cleanup function
        }
      }

      // Release every long-lived browser resource the hook holds.
      // Each is wrapped so a buggy implementation cannot block the next
      // call (RTL surfaces unhandled throws in cleanup as act warnings).
      try {
        transcriber.dispose();
      } catch {
        /* best-effort */
      }
      try {
        vad.dispose();
      } catch {
        /* best-effort */
      }

      // Detach our callback so a re-registered VAD instance is not held
      // by a stale closure. We only remove if the slot still points at
      // *our* handler — never trample a fresh assignment.
      if (vad.onSpeechEnd === handleSpeechEnd) {
        vad.onSpeechEnd = undefined;
      }
      ws.removeEventListener("open", handleOpen);
      ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // The empty deps array is deliberate: we use the latest-opts ref to
    // see fresh transcriber/speaker/vad/relayUrl values without tearing
    // down the WebSocket on every render. See the optsRef declaration above.
  }, []);

  return state;
}

/**
 * Per-turn helper: sends the turn payload on `ws`, listens for `token` /
 * `done` / `error` frames, and resolves with the accumulated reply plus
 * the timing milestones. The listener is removed before the promise
 * settles, so a subsequent turn starts with a clean slate.
 *
 * Failure modes:
 *   - `ws.send` throws (socket not open) → `RelayError("...","connection")`.
 *   - The relay closes mid-turn before `{done}` → same.
 *   - The relay emits `{type:"error", message}` → `RelayError(message,"frame")`
 *     so the caller can surface the relay's own diagnostic to the user.
 */
function collectRelayTurn(
  ws: WebSocket,
  text: string,
  history: ReadonlyArray<Message>,
): Promise<RelayTurnResult> {
  return new Promise<RelayTurnResult>((resolve, reject) => {
    let reply = "";
    let tFirstMs = 0;
    let settled = false;

    const cleanup = (): void => {
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
    };

    const settleResolve = (value: RelayTurnResult): void => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };
    const settleReject = (err: RelayError): void => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    };

    const onMessage = (ev: MessageEvent): void => {
      const frame: RelayFrame | null = parseRelayFrame(ev.data);
      if (frame === null) return;
      if (frame.type === "token") {
        if (tFirstMs === 0) tFirstMs = performance.now();
        reply += frame.text;
        return;
      }
      if (frame.type === "done") {
        settleResolve({ reply, tFirstMs, tDoneMs: performance.now() });
        return;
      }
      // frame.type === "error"
      settleReject(new RelayError(frame.message, "frame"));
    };

    const onClose = (): void => {
      settleReject(new RelayError("WebSocket closed before turn completed", "connection"));
    };

    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose);

    try {
      ws.send(JSON.stringify({ type: "turn", text, history }));
    } catch (e) {
      const reason = e instanceof Error ? e.message : "send failed";
      settleReject(new RelayError(reason, "connection"));
    }
  });
}
