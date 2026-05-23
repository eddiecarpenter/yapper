/**
 * `KokoroSpeaker` — browser-side `Speaker` backed by Kokoro running
 * via Transformers.js + ONNX Web.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.4 + AD-2 / AD-10.
 *
 * Layered across the four tasks of Feature #14:
 *   - Task 1 — backend selection + provider logging (AC-4 logging
 *     clause).
 *   - Task 2 — lazy Kokoro pipeline loading + loading-state observable
 *     (AC-4 fallback clause).
 *   - Task 3 — `speak()` synthesis + Web Audio playback + voice option
 *     (AC-1, AC-3).
 *   - Task 4 (this commit) — `cancel()` coordination + `dispose()`
 *     lifecycle (AC-2). `cancel()` resolves the in-flight `speak()`
 *     Promise without error and stops the active source; `dispose()`
 *     cancels any in-flight playback, fire-and-forget releases the
 *     loaded pipeline, closes the owned `AudioContext`, detaches
 *     subscribers, and resets state to `"idle"`.
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
 * keeps the Task-2 code small.
 */
type TtsPipeline = Awaited<ReturnType<typeof pipeline<"text-to-speech">>>;

/** Model id pinned by AD-10. Kept module-local so tests do not have to
 *  hard-code the same string in two places. */
export const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

/** Default voice pinned by AD-10. Constructor `voice` option overrides. */
export const DEFAULT_VOICE = "bf_emma";

/**
 * Shape of a single Kokoro synthesis output. The Transformers.js v3
 * text-to-speech pipeline returns `{ audio: Float32Array,
 * sampling_rate: number }`; we parse defensively so a minor library
 * version drift that wraps or renames either field does not silently
 * pass through.
 */
type SynthesisOutput = {
  audio: Float32Array;
  sampling_rate: number;
};

/**
 * Reduce the structured output of the Kokoro pipeline to the
 * `{ audio, sampling_rate }` shape `speak()` needs to wrap in an
 * `AudioBuffer`. Throws an actionable error if the pipeline returned
 * something that does not look like a TTS output — this guards against
 * a future Transformers.js shape change masking as silent dialogue.
 *
 * Exported so Task 3's tests can pin the parser contract directly
 * without round-tripping through the full pipeline mock.
 */
export function extractSynthesisOutput(result: unknown): SynthesisOutput {
  if (
    result !== null &&
    typeof result === "object" &&
    "audio" in result &&
    "sampling_rate" in result &&
    (result as { audio: unknown }).audio instanceof Float32Array &&
    typeof (result as { sampling_rate: unknown }).sampling_rate === "number"
  ) {
    const r = result as { audio: Float32Array; sampling_rate: number };
    return { audio: r.audio, sampling_rate: r.sampling_rate };
  }
  throw new Error(
    "Kokoro pipeline returned an unexpected output shape (expected { audio: Float32Array, sampling_rate: number })",
  );
}

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

/**
 * Constructor options for `KokoroSpeaker`. The `voice` option is
 * the only configurable field today; per-call voice override is
 * Parking Lot (Feature #14 body).
 */
export interface KokoroSpeakerOptions {
  /**
   * Voice id forwarded to the Kokoro pipeline on every `speak()` call.
   * Defaults to `DEFAULT_VOICE` (`"bf_emma"`) per AD-10. Override at
   * construction time; switching voices mid-Speaker is intentionally
   * not supported in this Feature.
   */
  voice?: string;
}

export class KokoroSpeaker implements Speaker {
  private readonly provider: Provider;
  private readonly voice: string;
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
   * The owned `AudioContext`. Lazily constructed on the first
   * non-empty `speak()` call (constructor in some browsers requires a
   * user gesture, so deferring keeps the constructor side-effect free).
   * Set to null after `dispose()` (Task 4).
   */
  private audioContext: AudioContext | null = null;
  /**
   * The `AudioBufferSourceNode` currently scheduled for playback, or
   * `null` if no playback is in flight. Task 3 writes this so Task 4's
   * `cancel()` can read it and call `source.stop()` without having to
   * touch `speak()` again. The field is also cleared on natural
   * `onended` so cancel after natural completion is a safe no-op.
   */
  private activeSource: AudioBufferSourceNode | null = null;
  /**
   * Resolver for the in-flight `speak()` Promise. Pinned by the
   * Promise constructor closure inside `speak()` so `cancel()` can
   * resolve the awaiter without going through the `source.onended`
   * path. AC-2 requires the in-flight `speak()` to resolve (not
   * reject) when `cancel()` is called — resolving directly via this
   * field is the cleanest way to honour that even if `source.stop()`
   * later throws.
   */
  private activeResolver: (() => void) | null = null;
  /**
   * Abort flag set by `cancel()` so the cancel-during-synthesis path
   * (cancel called while the pipeline call is still in flight) can
   * discard the late-arriving synthesis result and resolve `speak()`
   * without ever scheduling playback. AC-2's "cancel while synthesis
   * is in flight" branch relies on this. Reset at the top of every
   * `speak()` so a cancel that arrives before any speak() does not
   * latently abort the next call.
   */
  private activeAborted = false;

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
   * pipeline gate and `ensureAudioContext()` (Task 3) for the lazy
   * AudioContext gate.
   *
   * Optional `options.voice` overrides the AD-10 default `"bf_emma"`.
   */
  constructor(options: KokoroSpeakerOptions = {}) {
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
    this.voice = options.voice ?? DEFAULT_VOICE;
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

  /**
   * Return the configured voice. Exposed so the dialogue hook or a
   * future UI can render the active voice without inspecting private
   * fields.
   */
  getVoice(): string {
    return this.voice;
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
   * Lazily construct the owned `AudioContext`. The constructor for
   * `AudioContext` can throw before a user gesture in some browsers,
   * so we defer until the first non-empty `speak()` call. The context
   * lives for the lifetime of the Speaker — Task 4 closes it in
   * `dispose()`.
   */
  private ensureAudioContext(): AudioContext {
    if (this.audioContext === null) {
      // `AudioContext` is provided by the Web Audio API in browsers.
      // jsdom does not implement it; tests stub the global constructor
      // before constructing the Speaker.
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  /**
   * Synthesise `text` to speech and play it through the Web Audio API.
   *
   * Behaviour:
   *
   *   - Empty / whitespace-only text resolves immediately, does not
   *     invoke the pipeline, and does not construct an `AudioContext`.
   *     Matches `WhisperTranscriber`'s empty-segment semantics: a
   *     no-content turn is a successful no-op.
   *   - Non-empty text: awaits the lazy pipeline load (which drives
   *     the `idle → loading → ready` transitions observers
   *     subscribed for), invokes Kokoro with the configured voice,
   *     wraps the resulting samples in an `AudioBuffer` on the owned
   *     `AudioContext`, schedules them on an `AudioBufferSourceNode`,
   *     starts playback, and resolves when the source's `onended`
   *     fires.
   *   - The resolver is pinned in the Promise constructor closure so
   *     Task 4's `cancel()` can pre-fire it without going through the
   *     `onended` path (the active source's `.stop()` will also fire
   *     `onended`, but resolving first guarantees the caller is
   *     unblocked even if `.stop()` throws).
   *
   * Load failures are surfaced as the classified Error from
   * `loadPipeline()`; runtime synthesis or playback failures propagate
   * unchanged so the dialogue hook's generic "Speech synthesis failed"
   * branch can pick them up.
   */
  async speak(text: string): Promise<void> {
    // Empty / whitespace-only input: resolve immediately. This matches
    // the empty-segment contract on the sibling `Transcriber`
    // interface and lets the dialogue hook treat a no-speech reply as
    // a successful no-op turn without special-casing the consumer.
    if (text.trim() === "") {
      return;
    }

    // Reset the abort flag at the entry to every speak() so a stray
    // `cancel()` that arrived before any speak() (legal no-op per AC)
    // does not latently abort this call.
    this.activeAborted = false;

    const tts = await this.loadPipeline();

    // Call Kokoro with the configured voice. Transformers.js v3's
    // public `TextToAudioPipelineOptions` type does not declare a
    // `voice` field (it predates Kokoro's model-specific options),
    // but the runtime pipeline accepts and forwards it to the
    // Kokoro model — narrow the type at the call site so the rest
    // of the file stays type-clean.
    const callWithVoice = tts as unknown as (
      text: string,
      options: { voice: string },
    ) => Promise<unknown>;
    const rawResult = await callWithVoice(text, { voice: this.voice });

    // Cancel-during-synthesis path (AC-2): if `cancel()` set the abort
    // flag while the pipeline call was in flight, discard the
    // late-arriving synthesis result and resolve `speak()` without
    // ever scheduling playback. No `AudioBufferSourceNode` is
    // constructed; no audio is played. The Speaker is fully reusable
    // — a subsequent `speak()` resets the flag at entry.
    if (this.activeAborted) {
      this.activeAborted = false;
      return;
    }

    const { audio, sampling_rate } = extractSynthesisOutput(rawResult);

    const ctx = this.ensureAudioContext();
    // Mono buffer at the pipeline's reported sample rate (24 kHz for
    // Kokoro). The Web Audio API automatically resamples to the
    // context's output rate (typically 48 kHz on macOS) — no explicit
    // resampling needed in the Speaker.
    const buffer = ctx.createBuffer(1, audio.length, sampling_rate);
    // `copyToChannel` is the canonical Web Audio path; it handles the
    // detached-ArrayBuffer / SAB-cloning edge cases the underlying
    // implementation cares about. The first arg is the source data,
    // the second the channel index. The cast narrows from
    // `Float32Array<ArrayBufferLike>` (the type Transformers.js exposes
    // for forward-compat with SharedArrayBuffer) to
    // `Float32Array<ArrayBuffer>` (what lib.dom.d.ts's copyToChannel
    // accepts under TS 5.7+). Kokoro never returns SAB-backed audio,
    // so the cast is safe at runtime.
    buffer.copyToChannel(audio as Float32Array<ArrayBuffer>, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    // Track the active source so Task 4's `cancel()` can call
    // `source.stop()` on it. Cleared on natural `onended` so a cancel
    // arriving after natural completion is a safe no-op.
    this.activeSource = source;

    // Resolve-on-playback-end: the Promise constructor closure pins
    // the resolver so `cancel()` can also call it. The `onended`
    // event fires for both natural end-of-buffer AND explicit
    // `source.stop()` — so this single path serves both the
    // success branch (AC-1) and the cancel branch (AC-2) cleanly.
    await new Promise<void>((resolve) => {
      this.activeResolver = resolve;
      source.onended = () => {
        // Clear the active-source pointer first so a cancel() that
        // raced with natural completion sees null and no-ops.
        if (this.activeSource === source) {
          this.activeSource = null;
        }
        if (this.activeResolver === resolve) {
          this.activeResolver = null;
        }
        resolve();
      };
      source.start();
    });
  }

  /**
   * Cancel any in-flight `speak()`. AC-2:
   *
   *   - During playback: stops the active `AudioBufferSourceNode`
   *     and resolves the in-flight `speak()` Promise *without error*.
   *     The resolver is invoked BEFORE `source.stop()` so the
   *     caller's awaiter is unblocked even if `.stop()` throws
   *     (defence-in-depth — `.stop()` should not, but Web Audio
   *     implementations have surprised us before).
   *   - During synthesis (pipeline call still in flight): sets the
   *     `activeAborted` flag so the abort check after the pipeline
   *     call returns short-circuits before constructing a source.
   *   - With no active `speak()`: safe no-op. The `activeAborted`
   *     flag may be set transiently but the next `speak()` resets
   *     it at entry, so no latent abort leaks into a later turn.
   *
   * `cancel()` is reusable: after a cancel, a follow-up `speak()`
   * call still works — the loaded pipeline and the owned
   * `AudioContext` are not released here (`dispose()` does that).
   */
  cancel(): void {
    // Set the abort flag first so a cancel-during-synthesis is
    // observed by the post-synthesis check inside `speak()`.
    this.activeAborted = true;

    // Resolve the pending Promise BEFORE calling `source.stop()` so
    // the awaiter is unblocked even if `.stop()` throws.
    const resolver = this.activeResolver;
    if (resolver !== null) {
      this.activeResolver = null;
      resolver();
    }

    // Stop the active source. Wrapping in try/catch because
    // `.stop()` can throw `InvalidStateError` if the source was
    // never started or has already been stopped — both safe to
    // ignore in this code path.
    const source = this.activeSource;
    if (source !== null) {
      this.activeSource = null;
      try {
        source.stop();
      } catch {
        /* defensive — see above */
      }
    }
  }

  /**
   * Release the underlying model + audio resources and reset the
   * Speaker to its initial state.
   *
   *   - Cancel any in-flight playback (same code path as `cancel()`).
   *   - Release the cached Kokoro pipeline via its `.dispose()`
   *     (fire-and-forget — `dispose()` is synchronous-by-interface
   *     per the `Speaker` type, and the pipeline's release is
   *     independent of the JS-side teardown).
   *   - Close the owned `AudioContext` (also fire-and-forget — the
   *     returned `Promise<void>` is not awaited; rejections are
   *     swallowed).
   *   - Detach all loading-state subscribers.
   *   - Reset state to `"idle"`.
   *
   * `dispose()` is idempotent: a second call finds every field
   * already null / cleared and exits cleanly without throwing. It
   * also leaves the Speaker in a state where a subsequent `speak()`
   * would lazily re-load the pipeline and re-create the
   * `AudioContext` — the class behaves as if freshly constructed.
   */
  dispose(): void {
    // First cancel any in-flight speak so the caller's Promise is
    // unblocked and the source is stopped.
    this.cancel();
    // Reset the abort flag set by the cancel() call above — leaving
    // it true would latently abort the very next speak() if the
    // disposed Speaker is reused.
    this.activeAborted = false;

    // Fire-and-forget pipeline dispose. Matches the
    // `WhisperTranscriber.dispose()` shape: chain `.then(p =>
    // p.dispose().catch(()=>undefined)).catch(()=>undefined)` so a
    // rejection in either link is swallowed and no UnhandledRejection
    // surfaces.
    if (this.pipelinePromise !== null) {
      const promise = this.pipelinePromise;
      promise
        .then((p) => {
          // `.dispose()` is the Transformers.js `Disposable` typedef;
          // returns `Promise<void>`. We do not await here because
          // dispose() is synchronous-by-interface.
          p.dispose().catch(() => undefined);
        })
        .catch(() => undefined);
      this.pipelinePromise = null;
    }

    // Close the owned AudioContext (returns a Promise — do not await;
    // dispose() is synchronous-by-interface). Catch any rejection
    // silently so a teardown race in some implementations does not
    // surface as a noisy console error.
    if (this.audioContext !== null) {
      const ctx = this.audioContext;
      this.audioContext = null;
      try {
        const closeResult = ctx.close();
        // `close()` returns a Promise in Web Audio; jsdom mocks may
        // return undefined. Guard before chaining.
        if (closeResult && typeof closeResult.catch === "function") {
          closeResult.catch(() => undefined);
        }
      } catch {
        /* defensive */
      }
    }

    // Drop subscribers and reset to the initial state. A future
    // `speak()` re-loads the pipeline lazily (which drives a fresh
    // idle → loading → ready transition that the post-dispose
    // re-subscribers would observe).
    this.listeners.clear();
    this.loadingState = "idle";
  }
}
