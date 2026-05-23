/**
 * `KokoroSpeaker` — browser-side `Speaker` backed by Kokoro running
 * via Transformers.js + ONNX Web.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.4 + AD-2 / AD-10.
 *
 * Layered across the four tasks of Feature #14:
 *   - Task 1 — backend selection + provider logging (AC-4 logging
 *     clause).
 *   - Task 2 (this commit) — lazy Kokoro pipeline loading +
 *     loading-state observable (AC-4 fallback clause). Pipeline is
 *     built on the first `speak()` call; load failures are classified
 *     best-effort. The `speak()` body still throws "not yet
 *     implemented" after the load succeeds — Task 3 wires the real
 *     synthesis + Web Audio playback path on top.
 *   - Task 3 — `speak()` synthesis + Web Audio playback + voice option
 *     (AC-1, AC-3).
 *   - Task 4 — `cancel()` coordination + `dispose()` lifecycle (AC-2).
 *
 * The class is exported through `./index.ts` alongside the existing
 * `Speaker` type so consumers import from a single barrel — same shape
 * as the `web/src/stt/` module.
 */
import { pipeline } from "@huggingface/transformers";

import type { Speaker } from "./types";

/**
 * The ONNX Web execution provider this Speaker selected at
 * construction. Surfaced via `getProvider()` so tests can assert the
 * selection branch without grepping console output, and so a future UI
 * shell can display the active backend. Mirrors the `Provider` type
 * exported by `web/src/stt/WhisperTranscriber` so a future
 * cross-module surface can use one shared shape.
 */
export type Provider = "webgpu" | "wasm";

/**
 * Lifecycle state of the underlying Kokoro pipeline.
 *
 *   - `idle`    — no construction attempt yet (initial state) and the
 *                 state `dispose()` returns to.
 *   - `loading` — `pipeline(...)` is in flight.
 *   - `ready`   — pipeline constructed successfully and cached;
 *                 subsequent `speak()` calls reuse it.
 *   - `error`   — construction failed; `speak()` is rejecting
 *                 with the (classified) load error.
 *
 * UI code subscribes via `subscribe()` and renders the loading
 * indicator AC-4 calls for. The lightweight callback shape matches
 * the rest of `web/src/` (and `WhisperTranscriber`) — no RxJS or
 * external observable library is pulled in for one field.
 */
export type LoadingState = "idle" | "loading" | "ready" | "error";

/**
 * Pipeline shape this class actually uses — a callable that takes the
 * input text and returns the synthesised audio plus a `.dispose()`
 * for resource release. The real Transformers.js type is wider
 * (additional pipeline options, batch inputs, etc.); narrowing here
 * keeps the Task-2 code small and lets Task 3 add overloads/options
 * (notably the `voice` arg) as needed.
 */
type TtsPipeline = Awaited<ReturnType<typeof pipeline<"text-to-speech">>>;

/** Model id pinned by AD-10. Kept module-local so tests do not have to
 *  hard-code the same string in two places. */
export const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

/**
 * Classify a thrown error from `pipeline(...)` into one of three coarse
 * buckets so the UI can render something better than "an error occurred".
 *
 *   - `quota`   — IndexedDB / browser storage exhaustion. Browsers
 *                 typically surface this as `QuotaExceededError` (DOM)
 *                 or an error whose message mentions "quota".
 *   - `network` — fetch failed. Browsers surface this as a `TypeError`
 *                 whose message contains "Failed to fetch" / "Network
 *                 request failed"; node-fetch errors use similar wording.
 *   - `unknown` — anything else; surface the original message verbatim.
 *
 * The classification is best-effort, as the design plan notes — the
 * point is to give UX hooks something to branch on, not to be a
 * full taxonomy. Identical to the matching helper in
 * `WhisperTranscriber` — both modules share the same model-load
 * failure modes.
 */
export type LoadErrorCause = "quota" | "network" | "unknown";

function classifyLoadError(err: unknown): LoadErrorCause {
  if (err instanceof Error) {
    const name = err.name;
    const message = err.message ?? "";
    if (name === "QuotaExceededError" || /quota/i.test(message)) {
      return "quota";
    }
    if (name === "TypeError" && /(failed to fetch|network|networkerror)/i.test(message)) {
      return "network";
    }
  }
  return "unknown";
}

function loadErrorMessage(cause: LoadErrorCause, err: unknown): string {
  const original = err instanceof Error ? err.message : String(err);
  switch (cause) {
    case "quota":
      return `Failed to load Kokoro model: browser storage quota exceeded (${original})`;
    case "network":
      return `Failed to load Kokoro model: network unreachable (${original})`;
    case "unknown":
      return `Failed to load Kokoro model: ${original}`;
  }
}

export class KokoroSpeaker implements Speaker {
  private readonly provider: Provider;
  /**
   * Cached construction promise. Null until the first call to
   * `speak()`. Holding the promise (rather than the resolved
   * pipeline) means a second call that arrives while the first load
   * is still in flight piggy-backs on the same fetch — no double
   * download of the ~80 MB Kokoro bundle.
   */
  private pipelinePromise: Promise<TtsPipeline> | null = null;
  private loadingState: LoadingState = "idle";
  /**
   * Loading-state subscribers. A `Set` so a listener registered twice
   * notifies twice (matches Node's EventEmitter shape) and so removal
   * is O(1) without scanning. The returned unsubscribe is the only
   * way to detach a listener from outside the class.
   */
  private readonly listeners: Set<(state: LoadingState) => void> = new Set();

  /**
   * Probes `navigator.gpu` synchronously and pins the provider for the
   * lifetime of the instance. A truthy `navigator.gpu` selects WebGPU;
   * an undefined or null `navigator.gpu` falls back to WASM without
   * throwing — WASM is a first-class supported path per §6.4 / AD-2.
   *
   * The chosen provider is logged exactly once via `console.log` in the
   * canonical `provider: <name>` format required by AC-4. The log line
   * is identical to `WhisperTranscriber`'s so dialogue-loop log
   * scraping stays uniform across STT and TTS.
   *
   * The Kokoro pipeline and the `AudioContext` are deliberately NOT
   * instantiated here — see `loadPipeline()` (Task 2) for the lazy
   * pipeline gate and Task 3 for the lazy `AudioContext` gate.
   */
  constructor() {
    // `navigator.gpu` is the WebGPU entry point per the W3C WebGPU spec.
    // It is undefined in browsers without WebGPU and in jsdom (the test
    // environment), so the wasm-fallback branch is the natural default
    // for unit tests and the webgpu branch must be exercised with a
    // truthy stub.
    const hasWebGPU =
      typeof navigator !== "undefined" &&
      // The presence check uses `?? null` to coerce `undefined` to a
      // falsy value the linter can reason about.
      ((navigator as unknown as { gpu?: unknown }).gpu ?? null) !== null;

    this.provider = hasWebGPU ? "webgpu" : "wasm";
    // Deterministic single-line log so AC-4 can be verified from the
    // browser console without inspecting internal fields.
    console.log(`provider: ${this.provider}`);
  }

  /**
   * Return the execution provider chosen at construction. Used by tests
   * to assert the selection branch and by any future UI surface that
   * wants to display the active backend.
   */
  getProvider(): Provider {
    return this.provider;
  }

  /** Return the current pipeline-loading state. */
  getLoadingState(): LoadingState {
    return this.loadingState;
  }

  /**
   * Subscribe to loading-state transitions. The listener is called with
   * the new state every time it changes (not on the initial state — the
   * caller already has that via `getLoadingState()` and a synchronous
   * fire-on-subscribe is easy to layer on top from the call site).
   *
   * Returns an unsubscribe function — the canonical observable shape so
   * a React `useEffect` cleanup or any other resource-management
   * pattern can drop the subscription cleanly.
   */
  subscribe(listener: (state: LoadingState) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Apply a state transition: store the new value and notify every
   * subscriber. Subscriber callbacks are wrapped in try/catch so a
   * single buggy listener cannot prevent the rest from being notified;
   * this matches the resilience pattern in `useDialogue`'s
   * cleanup-best-effort handlers (and `WhisperTranscriber`).
   */
  private setState(next: LoadingState): void {
    if (this.loadingState === next) return;
    this.loadingState = next;
    for (const listener of this.listeners) {
      try {
        listener(next);
      } catch {
        /* best-effort — never let one bad listener block the others */
      }
    }
  }

  /**
   * Lazy-load the Kokoro pipeline. The first call kicks off the
   * `pipeline(...)` factory call (which fetches and caches the model
   * in IndexedDB); subsequent calls reuse the cached promise. A failure
   * during construction transitions state to `"error"`, clears the
   * cached promise (so a retry can be attempted later), and re-throws
   * a classified error.
   *
   * Exposed via `protected`-equivalent access for Task 3 — the
   * synthesis path will call this and then run the result through the
   * AudioContext. Task 2 itself only exercises this via `speak()`.
   */
  private loadPipeline(): Promise<TtsPipeline> {
    if (this.pipelinePromise !== null) {
      return this.pipelinePromise;
    }
    this.setState("loading");
    this.pipelinePromise = pipeline("text-to-speech", KOKORO_MODEL_ID, {
      device: this.provider,
    })
      .then((p) => {
        this.setState("ready");
        return p;
      })
      .catch((err: unknown) => {
        // On failure we drop the cached promise — without this, a retry
        // would resolve immediately to the rejection, with no way to
        // attempt the network/storage operation again.
        this.pipelinePromise = null;
        this.setState("error");
        const cause = classifyLoadError(err);
        throw new Error(loadErrorMessage(cause, err));
      });
    return this.pipelinePromise;
  }

  /**
   * Synthesise `text` to speech.
   *
   * Task 2 layer: triggers the lazy pipeline load and drives the
   * `idle → loading → ready` (or `→ error`) transitions observers
   * subscribed for. Beyond the load, the body throws "not yet
   * implemented" — Task 3 wires the real Kokoro invocation + Web
   * Audio playback path on top, and Task 4 coordinates `cancel()`.
   *
   * Load failures are surfaced as the classified Error from
   * `loadPipeline()`; runtime synthesis failures (Task 3) will
   * propagate unchanged so the dialogue hook's generic
   * "Speech synthesis failed" branch can pick them up.
   */
  async speak(_text: string): Promise<void> {
    await this.loadPipeline();
    // Task-2 stub: load done, but synthesis + playback wiring is
    // Task 3's concern. Throwing rather than silently no-oping
    // surfaces a Task-3 regression that forgets to replace this
    // line, instead of letting a partial implementation ship that
    // looks like a silent dialogue (Promise resolves but nothing
    // plays).
    throw new Error("KokoroSpeaker.speak() not yet implemented — pending Task 3");
  }

  /**
   * Cancel any in-flight playback. Task-2 stub: no-op. Task 4 wires
   * the real coordination with the active source node + in-flight
   * `speak()` resolver.
   */
  cancel(): void {
    // Intentionally empty in Task 2.
  }

  /**
   * Release the underlying model + audio resources. Task-2 stub: no-op.
   * Task 4 wires the full teardown (pipeline release, AudioContext
   * close, subscriber drop, state reset).
   */
  dispose(): void {
    // Intentionally empty in Task 2.
  }
}
