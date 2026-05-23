/**
 * `WhisperTranscriber` — browser-side `Transcriber` backed by Whisper running
 * via Transformers.js + ONNX Web.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.3 + AD-2 / AD-11.
 *
 * Layered across the three tasks of Feature #13:
 *   - Task 1 — backend selection + provider logging (AC-2 / AC-3).
 *   - Task 2 — lazy ASR-pipeline loading + loading-state observable
 *     (AC-4). Pipeline is built on the first transcribe() call;
 *     load failures are classified best-effort.
 *   - Task 3 (this commit) — transcription wiring (AC-1). `transcribe()`
 *     validates the sample rate (the audio module is responsible for
 *     resampling upstream per §6.3), runs the audio through the cached
 *     pipeline, and reduces the structured output to the plain
 *     transcript string the `Transcriber` interface contract requires.
 *     Empty / whitespace-only output is normalised to `""` per the
 *     existing JSDoc on `web/src/stt/types.ts`.
 *
 * The class is exported through `./index.ts` alongside the existing
 * `Transcriber` type so consumers import from a single barrel.
 */
import { pipeline } from "@huggingface/transformers";

import type { Transcriber } from "./types";

/**
 * The ONNX Web execution provider this Transcriber selected at
 * construction. Surfaced via `getProvider()` so tests can assert the
 * selection branch without grepping console output, and so a future UI
 * shell can display the active backend.
 */
export type Provider = "webgpu" | "wasm";

/**
 * Lifecycle state of the underlying ASR pipeline.
 *
 *   - `idle`    — no construction attempt yet (initial state) and the
 *                 state `dispose()` returns to.
 *   - `loading` — `pipeline(...)` is in flight.
 *   - `ready`   — pipeline constructed successfully and cached;
 *                 subsequent `transcribe()` calls reuse it.
 *   - `error`   — construction failed; `transcribe()` is rejecting
 *                 with the (classified) load error.
 *
 * UI code subscribes via `subscribe()` and renders the loading
 * indicator AC-4 calls for. The lightweight callback shape matches
 * the rest of `web/src/` — no RxJS or external observable library is
 * pulled in for one field.
 */
export type LoadingState = "idle" | "loading" | "ready" | "error";

/**
 * Pipeline shape this class actually uses — a callable that takes the
 * raw waveform and returns the transcription output plus a `.dispose()`
 * for resource release. The real Transformers.js type is wider (overloads
 * for URL inputs, timestamps options, etc.); narrowing here keeps the
 * Task-2 code small and lets Task 3 add overloads/options as needed.
 */
type AsrPipeline = Awaited<ReturnType<typeof pipeline<"automatic-speech-recognition">>>;

/** Model id pinned by AD-11. Kept module-local so tests do not have to
 *  hard-code the same string in two places. */
export const WHISPER_MODEL_ID = "Xenova/whisper-base.en";

/** Sample rate the Transcriber accepts. The audio module is responsible
 *  for resampling upstream per `docs/ARCHITECTURE.md` §6.3, so a
 *  mismatch is a programmer error and surfaces as an actionable Error
 *  rather than a silent re-resample (which would either duplicate
 *  upstream work or drift from the VAD's rate). The value matches
 *  `STT_SAMPLE_RATE_HZ` in `web/src/dialogue/wire.ts`. */
export const REQUIRED_SAMPLE_RATE_HZ = 16000;

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
 * full taxonomy.
 */
export type LoadErrorCause = "quota" | "network" | "unknown";

function classifyLoadError(err: unknown): LoadErrorCause {
  if (err instanceof Error) {
    const name = err.name;
    const message = err.message ?? "";
    if (name === "QuotaExceededError" || /quota/i.test(message)) {
      return "quota";
    }
    if (
      name === "TypeError" &&
      /(failed to fetch|network|networkerror)/i.test(message)
    ) {
      return "network";
    }
  }
  return "unknown";
}

/**
 * Reduce the structured output of the Transformers.js ASR pipeline to
 * the plain transcript string the `Transcriber` interface contract
 * requires. The pipeline returns `{ text: string, chunks?: Chunk[] }`
 * by default; when `chunk_length_s` is set it can return an array of
 * such objects. We never enable chunking, but we accept both shapes
 * so a future tuning task that turns chunking on for long-segment
 * support does not silently regress the public return type.
 *
 * Whitespace-only output is normalised to `""` per the JSDoc on
 * `web/src/stt/types.ts`: "Promise resolves with `""` when the
 * segment contained no recognised speech".
 *
 * Exported so Task 3's tests can pin the reducer contract directly
 * without round-tripping through the full pipeline mock.
 */
export function extractTranscript(
  result: unknown,
): string {
  // Array shape (chunked) — concatenate the per-chunk `text` fields in
  // order, then normalise. We do not insert separators because the
  // chunks overlap and the model emits its own spacing.
  if (Array.isArray(result)) {
    const joined = result
      .map((r) =>
        r !== null && typeof r === "object" && "text" in r && typeof (r as { text: unknown }).text === "string"
          ? (r as { text: string }).text
          : "",
      )
      .join("");
    return joined.trim();
  }
  if (
    result !== null &&
    typeof result === "object" &&
    "text" in result &&
    typeof (result as { text: unknown }).text === "string"
  ) {
    return (result as { text: string }).text.trim();
  }
  // Anything else (null, missing/invalid `text`) → empty transcript.
  // The interface contract permits this and the dialogue hook treats
  // an empty transcript as a no-speech segment to be ignored.
  return "";
}

function loadErrorMessage(cause: LoadErrorCause, err: unknown): string {
  const original = err instanceof Error ? err.message : String(err);
  switch (cause) {
    case "quota":
      return `Failed to load Whisper model: browser storage quota exceeded (${original})`;
    case "network":
      return `Failed to load Whisper model: network unreachable (${original})`;
    case "unknown":
      return `Failed to load Whisper model: ${original}`;
  }
}

export class WhisperTranscriber implements Transcriber {
  private readonly provider: Provider;
  /**
   * Cached construction promise. Null until the first call to
   * `transcribe()`. Holding the promise (rather than the resolved
   * pipeline) means a second call that arrives while the first load
   * is still in flight piggy-backs on the same fetch — no double
   * download of the 145 MB model.
   */
  private pipelinePromise: Promise<AsrPipeline> | null = null;
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
   * throwing — WASM is a first-class supported path per §6.3 / AD-2.
   *
   * The chosen provider is logged exactly once via `console.log` in the
   * canonical `provider: <name>` format required by AC-2 / AC-3.
   *
   * The ASR pipeline is deliberately NOT instantiated here — see
   * `transcribe()` for the lazy-load gate.
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
    // Deterministic single-line log so AC-2 / AC-3 can be verified from
    // the browser console without inspecting internal fields.
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
   * cleanup-best-effort handlers.
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
   * Lazy-load the ASR pipeline. The first call kicks off the
   * `pipeline(...)` factory call (which fetches and caches the model
   * in IndexedDB); subsequent calls reuse the cached promise. A failure
   * during construction transitions state to `"error"`, clears the
   * cached promise (so a retry can be attempted later), and re-throws
   * a classified error.
   */
  private loadPipeline(): Promise<AsrPipeline> {
    if (this.pipelinePromise !== null) {
      return this.pipelinePromise;
    }
    this.setState("loading");
    this.pipelinePromise = pipeline("automatic-speech-recognition", WHISPER_MODEL_ID, {
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
   * Transcribe a 16 kHz mono PCM segment to text.
   *
   * - Validates the sample rate before touching the pipeline so a
   *   wrong rate fails fast without ever paying the 145 MB model
   *   download cost.
   * - Awaits the lazy load (Task 2) — on the first call this also
   *   drives the `idle → loading → ready` transitions observers
   *   subscribed for.
   * - Calls the cached pipeline as a function (Transformers.js
   *   pipelines are callable + disposable) and reduces the
   *   structured output to the plain transcript string the
   *   `Transcriber` interface contract requires. `pipeline()` can
   *   return either a single output object or an array of them
   *   depending on whether `chunk_length_s` is set; here we never
   *   set it, so the single-output branch is the expected shape —
   *   but the reducer accepts both for robustness.
   * - Normalises empty / whitespace-only output to `""` per the
   *   existing JSDoc on the interface — "Promise resolves with `""`
   *   when the segment contained no recognised speech".
   *
   * Errors during the pipeline call itself are propagated unchanged;
   * the caller (e.g. `useDialogue`) wraps them in its own stage-error
   * machinery. The Task-2 error-taxonomy applies only to model-load
   * failures, not to inference failures (a different shape — and
   * the dialogue hook already has a generic "Speech recognition
   * failed" branch for runtime issues).
   */
  async transcribe(audio: Float32Array, sampleRate: number): Promise<string> {
    if (sampleRate !== REQUIRED_SAMPLE_RATE_HZ) {
      throw new Error(
        `WhisperTranscriber requires ${REQUIRED_SAMPLE_RATE_HZ} Hz audio, got ${sampleRate} Hz`,
      );
    }
    const asr = await this.loadPipeline();
    const result = await asr(audio);
    return extractTranscript(result);
  }

  /**
   * Release the cached pipeline (via the Transformers.js `.dispose()`
   * which is async, fired-and-forgotten — the model's resource
   * release is independent of the JS-side teardown), drop all
   * subscribers, and reset state to `"idle"`.
   *
   * Calling `dispose()` before any transcribe attempt is a safe no-op:
   * no pipeline to release, no subscribers (or just the ones the
   * caller registered).
   */
  dispose(): void {
    if (this.pipelinePromise !== null) {
      // Fire the underlying dispose if it resolved; ignore the rejection
      // case (we are already tearing down, no useful action on top).
      const promise = this.pipelinePromise;
      promise
        .then((p) => {
          // .dispose returns a Promise<void> per the Transformers.js
          // Disposable typedef; we do not await here because dispose()
          // is synchronous-by-interface (matches the Transcriber type)
          // and the caller should not block on async cleanup.
          p.dispose().catch(() => undefined);
        })
        .catch(() => undefined);
      this.pipelinePromise = null;
    }
    this.listeners.clear();
    this.loadingState = "idle";
  }
}
