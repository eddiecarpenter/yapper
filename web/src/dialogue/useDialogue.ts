import { useEffect, useReducer, useRef } from "react";

import type { Message } from "../llm/types";
import { appendTurn } from "./history";
import type { DialogueOptions, DialogueStage, DialogueState, TimingRecord } from "./types";
import { DEFAULT_CONTEXT_BUDGET, INITIAL_DIALOGUE_STATE } from "./types";
import type { RelayFrame } from "./wire";
import { STT_SAMPLE_RATE_HZ, parseRelayFrame } from "./wire";

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

  useEffect(() => {
    const { transcriber, speaker, vad, relayUrl } = optsRef.current;

    // unmountedFlag short-circuits any in-flight async work so we never
    // dispatch into an unmounted hook (React would warn). Task 4 reuses
    // this flag for cleanup of mid-utterance teardown.
    let unmounted = false;

    const ws = new WebSocket(relayUrl);

    const handleOpen = (): void => {
      if (unmounted) return;
      dispatch({ type: "STAGE", stage: "listening" });
    };
    ws.addEventListener("open", handleOpen);

    /**
     * Drives one full turn end-to-end. Bound to `vad.onSpeechEnd` below.
     * Each invocation is independent — the WebSocket is shared, but per-turn
     * state (accumulated tokens, t_first) lives inside this closure.
     */
    const handleSpeechEnd = async (segment: Float32Array): Promise<void> => {
      if (unmounted) return;

      // ── 1. VAD end → transcribing ─────────────────────────────────────
      const t0 = performance.now();
      dispatch({ type: "STAGE", stage: "transcribing" });

      const text = await transcriber.transcribe(segment, STT_SAMPLE_RATE_HZ);
      const tStt = performance.now();

      if (unmounted) return;

      // ── 2. transcribing → relaying ────────────────────────────────────
      dispatch({ type: "STAGE", stage: "relaying" });

      const { reply, tFirstMs, tDoneMs } = await collectRelayTurn(ws, text, historyRef.current);

      if (unmounted) return;

      // ── 3. relaying → speaking ───────────────────────────────────────
      dispatch({ type: "STAGE", stage: "speaking" });

      await speaker.speak(reply);
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
 * An `error` frame rejects the promise — Task 4 wires the rejection into
 * the hook's error stage. Task 2 simply propagates the rejection so the
 * Vitest test for error frames has something assertable.
 */
function collectRelayTurn(
  ws: WebSocket,
  text: string,
  history: ReadonlyArray<Message>,
): Promise<RelayTurnResult> {
  return new Promise<RelayTurnResult>((resolve, reject) => {
    let reply = "";
    let tFirstMs = 0;

    const onMessage = (ev: MessageEvent): void => {
      const frame: RelayFrame | null = parseRelayFrame(ev.data);
      if (frame === null) return;
      if (frame.type === "token") {
        if (tFirstMs === 0) tFirstMs = performance.now();
        reply += frame.text;
        return;
      }
      if (frame.type === "done") {
        ws.removeEventListener("message", onMessage);
        resolve({ reply, tFirstMs, tDoneMs: performance.now() });
        return;
      }
      // frame.type === "error"
      ws.removeEventListener("message", onMessage);
      reject(new Error(frame.message));
    };

    ws.addEventListener("message", onMessage);
    ws.send(JSON.stringify({ type: "turn", text, history }));
  });
}
