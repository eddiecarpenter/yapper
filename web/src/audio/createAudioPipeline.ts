/**
 * `createAudioPipeline` — factory that composes a `MicrophoneCapture`
 * with a `VAD` into a single `AudioPipeline` object the SPA wires to
 * `useDialogue`'s `vad` option, and exposes an observable
 * `pipelineState` so the UI shell can render permission-denied /
 * error / running states without crashing on a `getUserMedia`
 * rejection.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.1 + the Feature
 * #16 Design Plan (KD-5 — permission-denied as observable state).
 *
 * ## Why this layer exists
 *
 * `MicrophoneCapture` and `SileroVAD` are deliberately independent:
 * the former delivers 16 kHz frames, the latter consumes them. The
 * SPA needs to wire them together AND it needs to turn the
 * `MicPermissionDeniedError` that `microphone.start()` can throw
 * into a renderable state — without that turn, a denied permission
 * would surface as an unhandled Promise rejection in React, which
 * crashes the dev overlay and would crash the production app too.
 *
 * KD-5 (Design Plan): treat permission denial as an observable
 * `"permission-denied"` state, not a thrown error. This layer is
 * where that translation lives.
 *
 * ## Lifecycle separation — pipeline ≠ VAD
 *
 * The pipeline does NOT own the VAD's lifecycle. `useDialogue`
 * receives the `VAD` instance as a constructor option and may keep
 * it across pipeline restarts (e.g. permission re-granted after
 * initial denial — the dialogue hook holds the loaded model, the
 * SPA shell builds a new pipeline). Consequently `dispose()` here
 * disposes the microphone (which owns the audio graph + Web Audio
 * context) but only DETACHES the VAD wiring — `useDialogue` calls
 * `vad.dispose()` on its own unmount.
 */
import { MicPermissionDeniedError, MicrophoneCapture } from "./MicrophoneCapture";

import type { VAD } from "../vad/types";

/**
 * Observable lifecycle states the pipeline transitions through. The
 * shape matches `LoadingState` on the sibling Whisper / Kokoro
 * modules in spirit (single-string enum + observable) but uses
 * pipeline-specific names because the meaning differs:
 *
 *   - `idle`               — constructed, not started.
 *   - `starting`           — `start()` is in flight.
 *   - `running`            — mic is delivering frames into the VAD.
 *   - `permission-denied`  — `start()` failed with
 *                            `MicPermissionDeniedError`. The pipeline
 *                            is recoverable: a fresh `start()` after
 *                            the user re-grants permission transitions
 *                            back to `running`.
 *   - `error`              — `start()` failed with a non-permission
 *                            error (hardware busy, worklet load
 *                            failure, etc.). `getError()` returns the
 *                            underlying message for UI display.
 */
export type PipelineState = "idle" | "starting" | "running" | "permission-denied" | "error";

/**
 * Public surface of the composed pipeline. The SPA's wiring code
 * uses this type alone — no need to reach for `MicrophoneCapture`
 * or `VAD` types directly.
 */
export interface AudioPipeline {
  /**
   * Open the mic, wire frames into the VAD, and transition the
   * observable state to `"running"`. If `getUserMedia` denies the
   * permission, transitions to `"permission-denied"` instead of
   * rejecting — the SPA must not crash on a denial (AC-4).
   *
   * Calling `start()` while already running is a safe no-op.
   * Calling `start()` after a previous denial does fully retry
   * (constructs a new `MicrophoneCapture` so the post-dispose
   * sticky-disposed flag does not lock the pipeline out).
   */
  start(): Promise<void>;
  /** Current pipeline state. */
  pipelineState: PipelineState;
  /** Subscribe to state transitions; returns an unsubscribe function. */
  subscribe(listener: (state: PipelineState) => void): () => void;
  /**
   * Last underlying error message for the `"error"` /
   * `"permission-denied"` states. `null` in all other states.
   */
  getError(): string | null;
  /**
   * Dispose the microphone and detach the VAD wiring. Does NOT
   * dispose the VAD itself (see header comment). Idempotent.
   */
  dispose(): void;
}

/**
 * Constructor options for the pipeline. The VAD is mandatory (no
 * sensible default — the dialogue hook supplies the loaded
 * SileroVAD). The microphone is optional and defaults to a freshly
 * constructed `MicrophoneCapture`; tests inject a stub.
 */
export interface CreateAudioPipelineOptions {
  /** The VAD instance — typically a `SileroVAD` constructed by the SPA. */
  vad: VAD;
  /**
   * Microphone capture instance. Defaults to `new MicrophoneCapture()`
   * so the common path is a one-liner. Tests inject a stub matching
   * the production class's public surface.
   */
  microphone?: MicrophoneCapture;
}

/**
 * Factory — async only because future variants may want to do
 * initialisation work that isn't sync (e.g. probe device
 * capabilities). For now, the body is synchronous and the returned
 * Promise resolves on the next microtask.
 *
 * Returns a fresh `AudioPipeline` whose `start()` must be called
 * separately — the factory is just construction, the side-effecting
 * `getUserMedia` call lives in `start()` so the SPA can choose when
 * to ask for the mic permission (after a user gesture, after the
 * UI is mounted, etc.).
 */
export async function createAudioPipeline(
  options: CreateAudioPipelineOptions,
): Promise<AudioPipeline> {
  const vad = options.vad;
  let microphone: MicrophoneCapture = options.microphone ?? new MicrophoneCapture();

  let state: PipelineState = "idle";
  let lastError: string | null = null;
  const listeners: Set<(state: PipelineState) => void> = new Set();
  let disposed = false;

  function setState(next: PipelineState): void {
    if (state === next) return;
    state = next;
    for (const listener of listeners) {
      try {
        listener(next);
      } catch {
        /* best-effort — never let one buggy listener block the rest */
      }
    }
  }

  function wireMicToVad(): void {
    // The pipeline OWNS the onFrame binding — that's the integration
    // point. Any prior onFrame on the mic is overwritten; on dispose
    // we set it back to null so the mic is left in a clean state.
    microphone.onFrame = (frame: Float32Array) => {
      // process() may throw (e.g. wrong frame size, post-dispose).
      // Swallowing keeps the audio graph alive even on transient
      // errors — the VAD's own state machine reports load failures
      // via its loadingState observable, and a frame-size mismatch
      // would be a programmer bug not a user-facing condition.
      try {
        vad.process(frame);
      } catch {
        /* defensive — see comment */
      }
    };
  }

  const pipeline: AudioPipeline = {
    get pipelineState(): PipelineState {
      return state;
    },
    set pipelineState(_v: PipelineState) {
      // Read-only from outside; setter present only so the property
      // appears on the public interface.
      throw new Error("AudioPipeline.pipelineState is read-only");
    },
    async start(): Promise<void> {
      if (disposed) {
        throw new Error("AudioPipeline: cannot start() after dispose()");
      }
      if (state === "running" || state === "starting") {
        // Idempotent — caller may legitimately re-invoke start() on
        // the same pipeline after a UI prompt; do not re-construct.
        return;
      }

      // If we previously errored or denied, the underlying mic
      // instance may be in a sticky state. Construct a fresh one
      // unless the caller injected their own (test path) — in which
      // case it is the caller's responsibility to reset between
      // start attempts.
      if (state === "permission-denied" || state === "error") {
        if (options.microphone === undefined) {
          microphone = new MicrophoneCapture();
        }
      }

      lastError = null;
      setState("starting");
      try {
        await microphone.start();
        // Wire AFTER start so a thrown denial does not leave a
        // dangling onFrame handler on a half-constructed instance.
        wireMicToVad();
        setState("running");
      } catch (err: unknown) {
        // KD-5: permission denial is observable state, not a
        // crashing rejection. Other errors fall through to "error".
        if (err instanceof MicPermissionDeniedError) {
          lastError = err.message;
          setState("permission-denied");
          return;
        }
        lastError = err instanceof Error ? err.message : String(err);
        setState("error");
        return;
      }
    },
    subscribe(listener: (state: PipelineState) => void): () => void {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getError(): string | null {
      return lastError;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      // Detach the VAD wiring — but do NOT dispose the VAD. The
      // dialogue hook owns the VAD lifecycle (see header comment).
      try {
        // The microphone's own dispose() clears its onFrame; we set
        // null explicitly first as a belt-and-braces measure so a
        // stub microphone (test path) without that clearing behaviour
        // still sees the field cleared.
        microphone.onFrame = null;
      } catch {
        /* defensive — stubs may make this property read-only */
      }
      try {
        microphone.dispose();
      } catch {
        /* defensive — never throw out of dispose */
      }
      listeners.clear();
    },
  };

  return pipeline;
}
