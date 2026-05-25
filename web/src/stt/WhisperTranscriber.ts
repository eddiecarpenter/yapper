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
export const WHISPER_MODEL_ID = "Xenova/whisper-medium.en";

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

/**
 * Per-file download progress tracked by the Transformers.js progress
 * callback. Keyed by file name so we can compute an aggregate percentage
 * across all model artefacts as they arrive sequentially.
 */
type FileProgress = { loaded: number; total: number };

export class WhisperTranscriber implements Transcriber {
  private readonly provider: Provider;
  /** The model ID this instance was constructed with. */
  private readonly modelId: string;
  /**
   * Cached construction promise. Null until the first call to
   * `transcribe()` or `preload()`. Holding the promise (rather than
   * the resolved pipeline) means a second call that arrives while the
   * first load is still in flight piggy-backs on the same fetch — no
   * double download of the 145 MB model.
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
   * Per-file download progress, updated from the Transformers.js
   * progress_callback. Used to derive `downloadProgress` (0–100).
   */
  private readonly fileProgress = new Map<string, FileProgress>();
  /** Aggregate download progress (0–100). Notified to subscribers. */
  private downloadProgress = 0;
  private readonly progressListeners: Set<(pct: number) => void> = new Set();

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
  /**
   * @param modelId - HuggingFace model ID to use (defaults to
   *   `WHISPER_MODEL_ID`). Pass a different value to let the user
   *   choose a smaller/larger model at runtime.
   */
  constructor(modelId: string = WHISPER_MODEL_ID) {
    this.modelId = modelId;
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
    console.log(`provider: ${this.provider} model: ${this.modelId}`);
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
   * Current aggregate download progress, 0–100. Meaningful only when
   * `getLoadingState()` is `"loading"`.
   */
  getProgress(): number {
    return this.downloadProgress;
  }

  /**
   * Subscribe to download-progress updates (0–100). Fires on every chunk
   * received from the Transformers.js progress callback — typically many
   * times per second during the initial model fetch.
   *
   * Returns an unsubscribe function compatible with `useEffect` cleanup.
   */
  subscribeProgress(listener: (pct: number) => void): () => void {
    this.progressListeners.add(listener);
    return () => {
      this.progressListeners.delete(listener);
    };
  }

  /**
   * Eagerly trigger the model download and compilation so the pipeline
   * is warm before the first `transcribe()` call. Safe to call multiple
   * times — the underlying promise is cached, so concurrent / repeated
   * calls all piggy-back on the same load.
   *
   * Errors are swallowed here (they will surface again on the next
   * `transcribe()` call via the same rejected promise logic).
   */
  preload(): void {
    void this.loadPipeline().catch(() => undefined);
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
   * Update aggregate download progress from a single file's progress
   * report and notify subscribers. We keep a running sum of bytes
   * across all files seen so far — this means the percentage climbs
   * monotonically even as new files start downloading.
   */
  private handleProgress(file: string, loaded: number, total: number): void {
    this.fileProgress.set(file, { loaded, total });
    let totalLoaded = 0;
    let totalBytes = 0;
    for (const fp of this.fileProgress.values()) {
      totalLoaded += fp.loaded;
      totalBytes += fp.total;
    }
    const pct = totalBytes > 0 ? Math.round((totalLoaded / totalBytes) * 100) : 0;
    if (pct === this.downloadProgress) return;
    this.downloadProgress = pct;
    for (const listener of this.progressListeners) {
      try {
        listener(pct);
      } catch {
        /* best-effort */
      }
    }
  }

  /**
   * Internal helper that actually calls the Transformers.js `pipeline()`
   * factory with the selected device, with a transparent WebGPU → WASM
   * fallback when the browser's WebGPU implementation rejects the model
   * (e.g. Safari Metal backend missing a required op).
   *
   * On total failure: clears `pipelinePromise`, sets state to `"error"`,
   * and throws a classified error.
   */
  private async _buildPipeline(): Promise<AsrPipeline> {
    // Transformers.js progress_callback shape — we only care about the
    // "download" / "progress" statuses that carry loaded/total bytes.
    type ProgressEvent = {
      status: string;
      file?: string;
      loaded?: number;
      total?: number;
    };

    const progressCallback = (event: ProgressEvent): void => {
      if (
        (event.status === "download" || event.status === "progress") &&
        event.file !== undefined &&
        typeof event.loaded === "number" &&
        typeof event.total === "number" &&
        event.total > 0
      ) {
        this.handleProgress(event.file, event.loaded, event.total);
      }
    };

    // Dtype selection:
    //   WebGPU — fp16 (half-precision) is natively fast on Metal/Vulkan
    //            and halves the memory footprint vs fp32. Using fp16 for
    //            both encoder and the merged decoder avoids the
    //            "dtype not specified, using fp32" warning and gives
    //            a real speed improvement on Apple Silicon.
    //   WASM   — q8 (8-bit quantised) cuts model size and is fast on
    //            CPU; fp32 on WASM would be both large and slow.
    const webgpuDtype = { encoder_model: "fp16", decoder_model_merged: "fp16" } as const;
    const wasmDtype   = { encoder_model: "q8",   decoder_model_merged: "q8"   } as const;

    try {
      const p = await pipeline("automatic-speech-recognition", this.modelId, {
        device: this.provider,
        dtype: this.provider === "webgpu" ? webgpuDtype : wasmDtype,
        progress_callback: progressCallback,
      });
      this.setState("ready");
      return p;
    } catch (webgpuErr: unknown) {
      if (this.provider === "webgpu") {
        // Transparent fallback: Safari (and other partial WebGPU
        // implementations) may expose navigator.gpu but fail on individual
        // ONNX ops at model-load time. Retry with WASM before surfacing an
        // error so the app still works even if WebGPU is broken.
        console.warn(
          `[WhisperTranscriber] WebGPU pipeline failed (${webgpuErr instanceof Error ? webgpuErr.message : String(webgpuErr)}); falling back to WASM`,
        );
        // Downgrade the in-memory provider so getProvider() and logs reflect
        // the actual active backend for the rest of the session.
        (this as unknown as { provider: Provider }).provider = "wasm";
        console.log("provider: wasm (WebGPU fallback)");
        // Reset progress state so the WASM download is tracked cleanly.
        this.fileProgress.clear();
        this.downloadProgress = 0;
        try {
          const p = await pipeline("automatic-speech-recognition", this.modelId, {
            device: "wasm",
            dtype: wasmDtype,
            progress_callback: progressCallback,
          });
          this.setState("ready");
          return p;
        } catch (wasmErr: unknown) {
          // Both providers failed — fall through to the classified error.
          this.pipelinePromise = null;
          this.setState("error");
          const cause = classifyLoadError(wasmErr);
          throw new Error(loadErrorMessage(cause, wasmErr));
        }
      }
      // Non-WebGPU failure (WASM selected from the start).
      this.pipelinePromise = null;
      this.setState("error");
      const cause = classifyLoadError(webgpuErr);
      throw new Error(loadErrorMessage(cause, webgpuErr));
    }
  }

  /**
   * Lazy-load the ASR pipeline. The first call kicks off the
   * `pipeline(...)` factory call (which fetches and caches the model
   * in IndexedDB); subsequent calls reuse the cached promise. A failure
   * during construction transitions state to `"error"`, clears the
   * cached promise (so a retry can be attempted later), and re-throws
   * a classified error.
   *
   * WebGPU → WASM fallback: if the selected provider is `"webgpu"` and
   * the pipeline construction fails (e.g. unsupported op on Safari's
   * Metal backend, shader compile error), we transparently retry with
   * `"wasm"` and log a warning. This lets Safari users try WebGPU first
   * without ending up in a broken state when the browser's WebGPU
   * implementation doesn't yet support all the ops Whisper needs.
   */
  private loadPipeline(): Promise<AsrPipeline> {
    if (this.pipelinePromise !== null) {
      return this.pipelinePromise;
    }
    this.setState("loading");
    this.pipelinePromise = this._buildPipeline();
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
    this.progressListeners.clear();
    this.fileProgress.clear();
    this.downloadProgress = 0;
    this.loadingState = "idle";
  }
}
