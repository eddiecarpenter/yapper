/**
 * `SileroVAD` — browser-side `VAD` implementation backed by Silero
 * Voice Activity Detection v5, run via `@ricky0123/vad-web`'s `SileroV5`
 * model wrapper over ONNX Web.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.2 + AD-2 + the
 * Feature #16 Design Plan.
 *
 * ## Provider selection — KD-1 / R2 honest reporting
 *
 * The Design Plan calls for per-instance Provider selection
 * (WebGPU when available, WASM otherwise) mirroring
 * `WhisperTranscriber`/`KokoroSpeaker`. After integration we
 * confirmed that `@ricky0123/vad-web` binds to the `onnxruntime-web/wasm`
 * subpath — the library cannot run on WebGPU regardless of caller
 * preference. R2 in the Design Plan anticipates this: "do NOT silently
 * lie about which provider is active".
 *
 * Concretely:
 *   - The constructor still accepts an optional `preferredProvider`
 *     and probes `navigator.gpu` so future migrations to a
 *     WebGPU-capable backend do not require an API change.
 *   - `getProvider()` returns the ACTUAL runtime in use — currently
 *     always `"wasm"`. A console.log emits the standard
 *     `provider: wasm` line so the dialogue-loop log convention from
 *     `WhisperTranscriber` and `KokoroSpeaker` is preserved.
 *   - `getPreferredProvider()` returns what the constructor selected
 *     so a future UI surface can show "WebGPU requested, WASM in use"
 *     when the gap matters to the user.
 *
 * ## Sync `process()` over an async backend
 *
 * `VAD.process(frame): boolean` is synchronous, but ONNX inference is
 * inherently async. `process()` kicks off classification in the
 * background and returns the LAST settled `speaking` flag. The
 * one-frame staleness (~32 ms at 16 kHz / 512-sample frames) is well
 * within the latency budget for the "speaking now" UI indicator the
 * boolean drives. The aggregated-segment surface
 * (`onSpeechEnd(segment)`) does NOT suffer staleness — it is fired
 * from the inference resolution path, so it is exactly synchronised
 * with the hysteresis state machine.
 *
 * Callers that need to drain in-flight inferences before tearing down
 * (e.g. tests asserting on the final aggregated segment) can `await`
 * `flushPending()` — a documented public method, not just a test
 * hook. `dispose()` calls it internally with best-effort error
 * suppression so model release does not race with in-flight `process`.
 */
import { defaultModelFetcher } from "@ricky0123/vad-web";
import { SileroV5 } from "@ricky0123/vad-web/dist/models/v5";
import * as ort from "onnxruntime-web/wasm";

import type { VAD } from "./types";

/**
 * Provider taxonomy — same shape as `WhisperTranscriber` /
 * `KokoroSpeaker` so a future cross-module UI surface can render the
 * active backend uniformly.
 */
export type Provider = "webgpu" | "wasm";

/**
 * Lifecycle state of the underlying Silero model + ONNX session.
 * Identical state machine to the sibling modules so subscribers do
 * not have to special-case the VAD.
 */
export type LoadingState = "idle" | "loading" | "ready" | "error";

/**
 * Coarse classification of model-load failures. Mirrors
 * `WhisperTranscriber`/`KokoroSpeaker` so UI code branching on the
 * cause does not have to know which module produced the error.
 *
 *   - `network` — `fetch` for the model bytes failed (offline,
 *      404, CORS).
 *   - `model`   — model bytes loaded but the ONNX runtime rejected
 *      them (corruption, version mismatch, unsupported op).
 *   - `unknown` — anything else; surface the original message
 *      verbatim.
 */
export type LoadErrorCause = "network" | "model" | "unknown";

/**
 * The Silero v5 ONNX model file name. Defaults to a root-relative URL
 * so the SPA shell can serve it from the static asset root — that
 * matches the convention `@ricky0123/vad-web` uses for its own
 * default `baseAssetPath`. Constructor options override.
 */
export const SILERO_MODEL_ID = "silero_vad_v5.onnx";

/**
 * Default fetch URL — root-relative so the host SPA's static-asset
 * pipeline serves the model bytes alongside the rest of the bundle.
 * Constructor `modelUrl` option overrides; tests inject a stub
 * `modelFetcher` so they do not network.
 */
export const DEFAULT_MODEL_URL = `/${SILERO_MODEL_ID}`;

/**
 * Frame size Silero v5 expects — 512 samples at 16 kHz = 32 ms per
 * frame. Module-local constant so consumers can import a single
 * source of truth rather than hard-coding 512 in N call sites; also
 * matches `FRAME_SAMPLE_COUNT` exported by `web/src/audio/`.
 */
export const FRAME_SAMPLE_COUNT = 512;

/** Sample rate Silero v5 expects. */
export const TARGET_SAMPLE_RATE_HZ = 16000;

/** Default hysteresis thresholds and window sizes per the Design Plan. */
export const DEFAULT_SPEECH_THRESHOLD = 0.5;
export const DEFAULT_SILENCE_THRESHOLD = 0.35;
/** ~768 ms of silence at 16 kHz / 512-sample frames before end-of-segment. */
export const DEFAULT_MIN_SILENCE_FRAMES = 24;
/** ~256 ms of speech at 16 kHz / 512-sample frames before "speaking" latches. */
export const DEFAULT_MIN_SPEECH_FRAMES = 8;
/** ~256 ms of pre-detection audio prepended to each segment (KD-4). */
export const DEFAULT_PRE_ROLL_FRAMES = 8;

/**
 * Type the deep-imported `SileroV5` exposes for the loaded model.
 * Narrowed inline here so the rest of the class stays type-clean and
 * does not depend on the library's internal `Model` interface name.
 */
type SileroModel = Awaited<ReturnType<typeof SileroV5.new>>;

/** Loader function used by the production code and stubbable by tests. */
type ModelFetcher = (url: string) => Promise<ArrayBuffer>;

/**
 * Constructor options for `SileroVAD`. All fields are optional;
 * defaults match the Design Plan's published values so the no-arg
 * `new SileroVAD()` path is the common-case API.
 */
export interface SileroVADOptions {
  /**
   * Caller-preferred Provider. `"webgpu"` when omitted and
   * `navigator.gpu` is present; `"wasm"` otherwise. NOTE: the actual
   * runtime currently always uses WASM regardless of preference (see
   * the "Provider selection" header comment). `getProvider()` returns
   * the actual runtime; `getPreferredProvider()` returns this field.
   */
  preferredProvider?: Provider;
  /** Probability above which the frame counts as speech (default 0.5). */
  speechThreshold?: number;
  /** Probability below which the frame counts as silence (default 0.35). */
  silenceThreshold?: number;
  /** Number of consecutive silent frames before `onSpeechEnd` fires (default 24). */
  minSilenceFrames?: number;
  /** Number of speech frames before the "speaking" flag latches (default 8). */
  minSpeechFrames?: number;
  /** Number of pre-detection frames prepended to each emitted segment (default 8). */
  preRollFrames?: number;
  /** Model URL; injection seam for tests + non-default hosting paths. */
  modelUrl?: string;
  /** Model bytes fetcher; injected for tests so no network is hit. */
  modelFetcher?: ModelFetcher;
}

/**
 * Classify a load error into the coarse taxonomy `LoadErrorCause`.
 * Best-effort — the assertion is the same as the sibling modules: give
 * the UI something useful to branch on, not a full error taxonomy.
 */
function classifyLoadError(err: unknown): LoadErrorCause {
  if (err instanceof Error) {
    const name = err.name;
    const message = err.message ?? "";
    if (name === "TypeError" && /(failed to fetch|network|networkerror)/i.test(message)) {
      return "network";
    }
    if (/onnx|model|tensor|graph|operator/i.test(message)) {
      return "model";
    }
  }
  return "unknown";
}

function loadErrorMessage(cause: LoadErrorCause, err: unknown): string {
  const original = err instanceof Error ? err.message : String(err);
  switch (cause) {
    case "network":
      return `Failed to load Silero VAD model: network unreachable (${original})`;
    case "model":
      return `Failed to load Silero VAD model: ONNX runtime rejected the model (${original})`;
    case "unknown":
      return `Failed to load Silero VAD model: ${original}`;
  }
}

/**
 * Concatenate a list of equal-length Float32Arrays into a single
 * Float32Array. Exported so tests can pin the helper contract — the
 * order of the input list is preserved verbatim, no defensive copy,
 * no normalisation.
 */
export function concatFloat32(parts: Float32Array[]): Float32Array {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Float32Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

export class SileroVAD implements VAD {
  // ── VAD interface fields ────────────────────────────────────────
  /**
   * Public per-segment callback. Set by the consumer (typically the
   * dialogue hook via `createAudioPipeline`); the implementation
   * invokes it from the inference-resolution path each time a silence
   * window ends a speech segment.
   */
  onSpeechEnd?: (segment: Float32Array) => void;

  // ── Configuration ───────────────────────────────────────────────
  private readonly preferredProvider: Provider;
  private readonly provider: Provider; // Actual runtime — see header comment.
  private readonly speechThreshold: number;
  private readonly silenceThreshold: number;
  private readonly minSilenceFrames: number;
  private readonly minSpeechFrames: number;
  private readonly preRollFrames: number;
  private readonly modelUrl: string;
  private readonly modelFetcher: ModelFetcher;

  // ── Lifecycle ───────────────────────────────────────────────────
  /**
   * Cached construction promise. Null until the first `process()`
   * call. Holding the promise (not the model) means a `process()`
   * that arrives mid-load piggy-backs on the same fetch — no double
   * download of the ~2 MB Silero v5 model bytes.
   */
  private modelPromise: Promise<SileroModel> | null = null;
  private loadingState: LoadingState = "idle";
  private readonly listeners: Set<(state: LoadingState) => void> = new Set();
  private disposed = false;

  // ── State machine ───────────────────────────────────────────────
  /**
   * Cached "speaking" flag returned by `process()`. Updated from the
   * inference-resolution path each time a classification settles.
   * Starts false and never goes truthy until the first speech run of
   * length `minSpeechFrames` has been observed.
   */
  private speakingFlag = false;
  private speechFrameCount = 0;
  private silenceFrameCount = 0;
  /**
   * Rolling history of the most recent
   * `preRollFrames + minSpeechFrames` frames. When the state machine
   * transitions into "speaking", this buffer is copied as the initial
   * segment contents — the head of the buffer is the pre-detection
   * pre-roll (KD-4); the tail is the frames that contributed to the
   * detection threshold being crossed. Maintained as a plain array
   * (not a typed ring buffer) because the frame count is tiny (≤ 16
   * frames) and per-frame copying is cheap relative to ONNX inference.
   */
  private history: Float32Array[] = [];
  /**
   * Speech segment accumulator while `speakingFlag === true`. Drained
   * and concatenated into the `onSpeechEnd` argument when the silence
   * window elapses.
   */
  private speechSegment: Float32Array[] = [];

  // ── In-flight tracking ──────────────────────────────────────────
  /**
   * Chain of in-flight inference promises. `flushPending()` and
   * `dispose()` await this so model release does not race a
   * classification still running on the GPU/WASM thread. Each
   * `process()` invocation appends a new link to the chain (replacing
   * the field) so the chain is always pinned by the latest call.
   */
  private inflightTail: Promise<void> = Promise.resolve();

  constructor(options: SileroVADOptions = {}) {
    // Provider preference — same probe as Whisper / Kokoro. We do NOT
    // ALWAYS just trust the caller-supplied preference: if they ask
    // for webgpu but the page has no navigator.gpu, we'd be in a
    // mismatched state. Default behaviour: prefer webgpu when
    // available; fall back to wasm.
    const hasWebGPU =
      typeof navigator !== "undefined" &&
      ((navigator as unknown as { gpu?: unknown }).gpu ?? null) !== null;
    this.preferredProvider = options.preferredProvider ?? (hasWebGPU ? "webgpu" : "wasm");

    // R2 (Design Plan) — surface the ACTUAL runtime. The library binds
    // to onnxruntime-web/wasm, so wasm is the only runtime currently
    // available regardless of preference.
    this.provider = "wasm";

    // Single canonical log line, matching the sibling modules. Format:
    //   `provider: <actual>` — same string Whisper/Kokoro emit.
    console.log(`provider: ${this.provider}`);

    this.speechThreshold = options.speechThreshold ?? DEFAULT_SPEECH_THRESHOLD;
    this.silenceThreshold = options.silenceThreshold ?? DEFAULT_SILENCE_THRESHOLD;
    this.minSilenceFrames = options.minSilenceFrames ?? DEFAULT_MIN_SILENCE_FRAMES;
    this.minSpeechFrames = options.minSpeechFrames ?? DEFAULT_MIN_SPEECH_FRAMES;
    this.preRollFrames = options.preRollFrames ?? DEFAULT_PRE_ROLL_FRAMES;
    this.modelUrl = options.modelUrl ?? DEFAULT_MODEL_URL;
    this.modelFetcher = options.modelFetcher ?? defaultModelFetcher;
  }

  /** Actual runtime in use. May differ from the preferred provider. */
  getProvider(): Provider {
    return this.provider;
  }

  /** Caller-preferred provider — see KD-1 / R2. */
  getPreferredProvider(): Provider {
    return this.preferredProvider;
  }

  /** Current loading state of the underlying model. */
  getLoadingState(): LoadingState {
    return this.loadingState;
  }

  /**
   * Subscribe to loading-state transitions. Returns an unsubscribe
   * function — same shape as the sibling modules so a future UI
   * surface can show all three modules' loading bars uniformly.
   */
  subscribe(listener: (state: LoadingState) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private setState(next: LoadingState): void {
    if (this.loadingState === next) return;
    this.loadingState = next;
    for (const listener of this.listeners) {
      try {
        listener(next);
      } catch {
        /* best-effort — never let one buggy listener block the others */
      }
    }
  }

  /**
   * Lazy-load the Silero model. The first `process()` call starts the
   * fetch + ONNX session construction; subsequent calls reuse the
   * cached promise. Failures drop the cached promise so a retry is
   * actually retried (without that, the rejection would short-circuit
   * forever).
   */
  private loadModel(): Promise<SileroModel> {
    if (this.modelPromise !== null) {
      return this.modelPromise;
    }
    this.setState("loading");
    this.modelPromise = (async () => {
      try {
        const model = await SileroV5.new(ort, () => this.modelFetcher(this.modelUrl));
        this.setState("ready");
        return model;
      } catch (err: unknown) {
        this.modelPromise = null;
        this.setState("error");
        const cause = classifyLoadError(err);
        throw new Error(loadErrorMessage(cause, err));
      }
    })();
    return this.modelPromise;
  }

  /**
   * Apply a settled SpeechProbabilities to the hysteresis state
   * machine and (when a segment ends) fire `onSpeechEnd`.
   *
   * Algorithm:
   *   1. Push the frame into the rolling history (length =
   *      preRollFrames + minSpeechFrames). The history's contents at
   *      the speaking-transition instant become the segment start.
   *   2. Classify by threshold: speech / silence / dead-zone.
   *   3. If currently speaking: append every frame to the segment.
   *      Speech frames reset the silence counter. Silence frames
   *      increment it; on reaching `minSilenceFrames`, fire
   *      `onSpeechEnd` with the concatenated segment and reset state.
   *   4. If not currently speaking: speech frames bump the speech
   *      counter; on reaching `minSpeechFrames`, latch
   *      `speakingFlag = true` and seed the segment from history.
   *      Silence frames reset the speech counter; dead-zone frames
   *      leave counters unchanged.
   *
   * Order matters — pushing into history BEFORE branching means the
   * history snapshot taken at the transition instant already includes
   * the latest frame.
   */
  private applyClassification(frame: Float32Array, isSpeechProb: number): void {
    // Maintain rolling history.
    this.history.push(frame);
    const historyCapacity = this.preRollFrames + this.minSpeechFrames;
    while (this.history.length > historyCapacity) {
      this.history.shift();
    }

    const isSpeech = isSpeechProb >= this.speechThreshold;
    const isSilence = isSpeechProb < this.silenceThreshold;

    if (this.speakingFlag) {
      // Already in a segment — every frame goes in.
      this.speechSegment.push(frame);
      if (isSpeech) {
        this.silenceFrameCount = 0;
      } else if (isSilence) {
        this.silenceFrameCount++;
        if (this.silenceFrameCount >= this.minSilenceFrames) {
          // End of segment. Snapshot before the reset so a listener
          // that re-enters process() does not see a stale buffer.
          const segment = concatFloat32(this.speechSegment);
          this.speakingFlag = false;
          this.speechFrameCount = 0;
          this.silenceFrameCount = 0;
          this.speechSegment = [];
          const cb = this.onSpeechEnd;
          if (cb) {
            try {
              cb(segment);
            } catch {
              /* best-effort — buggy consumer must not break the VAD */
            }
          }
        }
      }
      // Dead-zone (between silenceThreshold and speechThreshold) —
      // do not touch counters; the frame is already in the segment.
    } else {
      // Not yet in a segment — counting up to minSpeechFrames.
      if (isSpeech) {
        this.speechFrameCount++;
        if (this.speechFrameCount >= this.minSpeechFrames) {
          // Latch speaking. Seed the segment from the history snapshot —
          // includes the pre-roll + the last `minSpeechFrames` frames
          // that contributed to the threshold being crossed.
          this.speakingFlag = true;
          this.speechSegment = this.history.slice();
          this.silenceFrameCount = 0;
        }
      } else if (isSilence) {
        // A single silent frame resets the run.
        this.speechFrameCount = 0;
      }
      // Dead-zone — leave counters unchanged.
    }
  }

  /**
   * Per-frame speech-detection probe. Returns the cached "speaking"
   * flag (updated from the inference-resolution path — see the
   * "Sync `process()` over an async backend" header comment). The
   * frame must be exactly `FRAME_SAMPLE_COUNT` samples at 16 kHz —
   * the Silero model is hard-wired to this size. A wrong length is a
   * programmer error and surfaces as a thrown `Error` rather than a
   * silent re-pad (which would either confuse the model or skew the
   * pre-roll buffer).
   *
   * Post-`dispose()` calls throw — the model is gone, the audio graph
   * upstream should not be still pushing frames. The pipeline layer
   * (Task 4) tears down its `onFrame` wiring on dispose precisely so
   * this contract is observed.
   */
  process(frame: Float32Array): boolean {
    if (this.disposed) {
      throw new Error("SileroVAD: cannot process() after dispose()");
    }
    if (frame.length !== FRAME_SAMPLE_COUNT) {
      throw new Error(
        `SileroVAD: expected ${FRAME_SAMPLE_COUNT}-sample frames at 16 kHz, got ${frame.length}`,
      );
    }

    // Chain this frame's inference behind the existing tail so frames
    // are processed in the order they arrive. The `await` between the
    // load-promise and the model.process() also serialises the very
    // first frames behind the model load.
    const prevTail = this.inflightTail;
    const next = (async () => {
      // Wait for the prior frame's inference to complete so the state
      // machine observes frames in order.
      await prevTail;
      // If dispose() raced — skip the inference but resolve cleanly.
      if (this.disposed) return;
      const model = await this.loadModel();
      // Recheck dispose after the (possibly long) load promise.
      if (this.disposed) return;
      const probs = await model.process(frame);
      if (this.disposed) return;
      this.applyClassification(frame, probs.isSpeech);
      // Update the speaking flag observers of process() will see on
      // the NEXT call.
      this.speakingFlag = this.speakingFlag; // explicit no-op to anchor the comment.
    })().catch(() => {
      // Inference errors during the steady-state path are swallowed
      // best-effort here — the load-error path already classifies and
      // surfaces model-load failures via loadingState. A runtime
      // inference failure on a single frame should not tear down the
      // whole VAD; the next frame will retry.
    });
    this.inflightTail = next;

    return this.speakingFlag;
  }

  /**
   * Resolve when every `process()` invocation up to this point has
   * completed (or failed). Used by tests to deterministically observe
   * the post-frame state; also called internally by `dispose()` so
   * model release does not race a classification still on the GPU.
   */
  async flushPending(): Promise<void> {
    await this.inflightTail;
  }

  /**
   * Release the ONNX session and any in-class buffers. After
   * `dispose()`:
   *   - Subsequent `process()` calls throw.
   *   - In-flight inferences resolve to no-ops (the dispose check
   *     inside the inference body short-circuits).
   *   - The `onSpeechEnd` callback is detached so a late-arriving
   *     segment does not surprise the caller.
   *   - The loading state resets to `"idle"`.
   *
   * Idempotent: a second call exits cleanly.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    // Release the model in the background after pending inferences
    // drain — fire-and-forget, matches the Whisper/Kokoro pattern.
    const modelPromise = this.modelPromise;
    this.modelPromise = null;
    if (modelPromise !== null) {
      this.inflightTail
        .then(() => modelPromise)
        .then((m) => m.release().catch(() => undefined))
        .catch(() => undefined);
    }

    this.onSpeechEnd = undefined;
    this.listeners.clear();
    this.loadingState = "idle";
    this.history = [];
    this.speechSegment = [];
    this.speakingFlag = false;
    this.speechFrameCount = 0;
    this.silenceFrameCount = 0;
  }
}
