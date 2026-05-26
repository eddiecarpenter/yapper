/**
 * `MicrophoneCapture` — browser-side microphone capture that delivers
 * 16 kHz mono Float32 frames to a downstream `VAD`/`Transcriber`.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.1 + AD-1 / AD-2.
 *
 * ## Why this module exists
 *
 * The voice loop needs a clean 16 kHz frame stream because both Silero
 * VAD and Whisper STT expect 16 kHz mono input. Browsers do not let us
 * open the microphone at a specific sample rate — `getUserMedia` plus
 * `AudioContext` ends up at the device native rate (typically 48 kHz
 * on desktops, 44.1 kHz on some Bluetooth headsets). This module
 * bridges that gap:
 *
 *   1. `getUserMedia({ audio: true })` opens the mic.
 *   2. An `AudioContext` wraps the device output rate.
 *   3. An `AudioWorkletProcessor` (see `decimator-worklet.ts`) decimates
 *      the device rate down to 16 kHz inside the worklet — keeping the
 *      per-128-sample DSP off the main thread (Design Plan KD-2).
 *   4. The worklet posts 512-sample frames at 16 kHz to the main thread
 *      via `port.postMessage`; this class forwards each frame to the
 *      consumer-supplied `onFrame` callback.
 *
 * ## Module shape
 *
 * Mirrors the established pattern of `WhisperTranscriber.ts` and
 * `SupertonicSpeaker.ts` (small focused public surface, pinned constants,
 * barrel re-export) but is NOT model-backed — so there is no
 * `LoadingState` observable here. The pipeline-level
 * `createAudioPipeline` (Task 4) is the appropriate place to surface
 * "starting / running / permission-denied / error" state to the UI.
 */
import { DECIMATOR_WORKLET_NAME, DECIMATOR_WORKLET_SOURCE } from "./decimator-worklet";

/**
 * Sample count per delivered frame. 512 samples at 16 kHz = 32 ms —
 * the standard Silero VAD frame length (per Silero v5 documentation
 * and the Design Plan). Module-local constant so consumers can import
 * a single source of truth rather than hard-coding "512" everywhere.
 */
export const FRAME_SAMPLE_COUNT = 512;

/**
 * Target output rate the worklet decimates to. Matches
 * `REQUIRED_SAMPLE_RATE_HZ` in `web/src/stt/WhisperTranscriber.ts`
 * and `STT_SAMPLE_RATE_HZ` in `web/src/dialogue/wire.ts` — keeping
 * the three in sync is enforced by a typecheck-time `as const` plus
 * a unit-test pin in the sibling test file.
 */
export const TARGET_SAMPLE_RATE_HZ = 16000;

/**
 * Typed error thrown by `MicrophoneCapture.start()` when the browser
 * surfaces the user denying microphone permission. Carries an
 * actionable message the UI shell can render verbatim — the AC-4
 * requirement is that the SPA does not crash on a denial, which means
 * callers can `instanceof`-check this class and translate it into an
 * observable "permission-denied" state instead of an unhandled
 * rejection (Design Plan KD-5; the `createAudioPipeline` factory in
 * Task 4 does exactly that).
 *
 * The constructor accepts an optional message override to make the
 * error message customisable from a future i18n layer; if omitted,
 * the canonical English actionable message is used.
 */
export class MicPermissionDeniedError extends Error {
  constructor(message?: string) {
    super(
      message ??
        "Microphone access was denied. To use Yapper, click the lock icon in the address bar and allow microphone access, then reload.",
    );
    this.name = "MicPermissionDeniedError";
    // Restore the prototype chain — required for `instanceof` to work
    // reliably when this class is transpiled down to ES5/CJS through
    // toolchains that drop the `extends Error` prototype link.
    Object.setPrototypeOf(this, MicPermissionDeniedError.prototype);
  }
}

/**
 * Shape of the message posted from the worklet to the main thread.
 * Currently only one type is carried — `"frame"` — but typing the
 * envelope explicitly leaves room for future control messages (e.g.
 * `"underrun"` / `"overrun"` telemetry) without breaking the parser
 * on the main thread.
 */
type WorkletMessage = { type: "frame"; frame: Float32Array };

/**
 * Public surface for the microphone capture module.
 *
 *   - `start()` resolves once the worklet is registered and the mic
 *     track is connected (frames may not have flowed yet — the first
 *     frame arrives after the worklet has accumulated 512 output
 *     samples, ~32 ms after start).
 *   - `onFrame` is set by the consumer before or after `start()`. The
 *     class invokes it for every delivered frame; setting it to null
 *     pauses frame delivery without tearing down the capture pipeline.
 *   - `dispose()` is idempotent — calling it twice is a safe no-op,
 *     calling it before `start()` is a safe no-op.
 */
export class MicrophoneCapture {
  /**
   * Consumer-supplied per-frame callback. Public field (not a method)
   * so the test suite can simply write `mic.onFrame = handler` without
   * the class having to expose `subscribe()`/`unsubscribe()`. Set to
   * `null` to suspend delivery; `dispose()` clears it.
   */
  onFrame: ((frame: Float32Array) => void) | null = null;

  /**
   * The owned `AudioContext`. Lazily constructed during `start()`
   * because some browsers require a user gesture before the
   * constructor can be invoked without throwing. Cleared on
   * `dispose()`.
   */
  private audioContext: AudioContext | null = null;

  /**
   * The `MediaStream` returned by `getUserMedia`. Held so `dispose()`
   * can iterate its tracks and call `track.stop()` on each — without
   * which the browser keeps the mic LED lit and the underlying
   * hardware open.
   */
  private mediaStream: MediaStream | null = null;

  /**
   * The Web Audio node that wraps the mic `MediaStream` and routes it
   * into the worklet. Holding the reference lets `dispose()` call
   * `disconnect()` so the audio graph is torn down cleanly.
   */
  private sourceNode: MediaStreamAudioSourceNode | null = null;

  /**
   * The worklet node itself — holds the `port` over which the
   * decimated frames flow back to the main thread. `dispose()` clears
   * the `port.onmessage` handler and disconnects the node.
   */
  private workletNode: AudioWorkletNode | null = null;

  /**
   * The Blob URL the worklet source was loaded from. We retain it so
   * `dispose()` can call `URL.revokeObjectURL()` — without that, the
   * Blob leaks for the lifetime of the document.
   */
  private workletUrl: string | null = null;

  /**
   * Sticky disposal flag. Once `dispose()` runs, subsequent
   * `start()` calls reject so a torn-down instance cannot accidentally
   * be reused — that would lead to a half-constructed audio graph
   * with the old (closed) context still referenced.
   */
  private disposed = false;

  /**
   * Open the microphone, build the audio graph, and begin delivering
   * 16 kHz / 512-sample frames to `onFrame`.
   *
   * Behaviour:
   *
   *   - First call: opens the mic via `getUserMedia`, constructs the
   *     owned `AudioContext`, loads the decimator worklet via a Blob
   *     URL (bundler-agnostic — see `decimator-worklet.ts` for the
   *     rationale), constructs the `AudioWorkletNode`, and wires
   *     mic → worklet. Resolves once the graph is live.
   *   - Permission denial (`NotAllowedError` thrown by
   *     `getUserMedia`): re-thrown as `MicPermissionDeniedError` so
   *     the caller (typically `createAudioPipeline`) can branch on
   *     the typed class rather than parsing the underlying browser
   *     message — which differs across Chromium / Firefox / Safari.
   *   - Post-`dispose()`: rejects with a clear error rather than
   *     attempting to re-construct on a torn-down instance.
   *
   * Idempotency: `start()` is NOT idempotent — calling it twice
   * without a `dispose()` between will try to open a second mic
   * stream and is a programmer error. The pipeline layer (Task 4)
   * is the right place to guard against that.
   */
  async start(): Promise<void> {
    if (this.disposed) {
      throw new Error(
        "MicrophoneCapture: cannot start() after dispose() — construct a new instance instead",
      );
    }

    let stream: MediaStream;
    try {
      // `getUserMedia` is the single mic-open entry point in browsers.
      // We do NOT pass a `sampleRate` constraint — browsers ignore it
      // for the AudioContext anyway, and the worklet handles whatever
      // device rate we end up at.
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err: unknown) {
      // Translate the browser's `NotAllowedError` into our typed
      // class. `NotAllowedError` is the standardised `DOMException`
      // name for permission-denied (W3C MediaCapture spec) — covers
      // both "user clicked deny" and "the page is on http: and the
      // browser blocked it pre-prompt".
      if (err instanceof Error && err.name === "NotAllowedError") {
        throw new MicPermissionDeniedError();
      }
      // Any other error (NotFoundError when no mic is plugged in,
      // NotReadableError for a hardware failure, OverconstrainedError
      // for a bad constraint) is re-thrown unchanged — the pipeline
      // layer translates them into the generic "error" state.
      throw err;
    }

    this.mediaStream = stream;

    // Construct the owned context. We do not pin a sample rate — the
    // browser picks the device native rate, which the worklet then
    // decimates to 16 kHz.
    this.audioContext = new AudioContext();

    // Load the worklet via a Blob URL — see `decimator-worklet.ts`
    // for why this is bundler-agnostic. The Blob is held in
    // `workletUrl` so `dispose()` can revoke it.
    const blob = new Blob([DECIMATOR_WORKLET_SOURCE], {
      type: "application/javascript",
    });
    this.workletUrl = URL.createObjectURL(blob);
    await this.audioContext.audioWorklet.addModule(this.workletUrl);

    // Construct the worklet node. We pass the device's actual sample
    // rate as a processor option so the worklet's resampler math is
    // correct even on devices where the rate is not 48 kHz.
    this.workletNode = new AudioWorkletNode(this.audioContext, DECIMATOR_WORKLET_NAME, {
      processorOptions: {
        deviceSampleRate: this.audioContext.sampleRate,
        targetSampleRate: TARGET_SAMPLE_RATE_HZ,
        frameSampleCount: FRAME_SAMPLE_COUNT,
      },
    });
    // Fan messages from the worklet's port into `onFrame`. We accept
    // late-bound `onFrame` assignment by reading from `this.onFrame`
    // at message-fire time rather than capturing the reference now.
    this.workletNode.port.onmessage = (ev: MessageEvent) => {
      const data = ev.data as WorkletMessage | undefined;
      if (data && data.type === "frame" && data.frame instanceof Float32Array) {
        const cb = this.onFrame;
        if (cb !== null) {
          try {
            cb(data.frame);
          } catch {
            // Best-effort — never let a buggy frame handler tear down
            // the capture pipeline. Matches the resilience pattern
            // applied to subscriber callbacks in `WhisperTranscriber`
            // / `SupertonicSpeaker`.
          }
        }
      }
    };

    // Wire mic → worklet. We deliberately do NOT connect the worklet
    // to `audioContext.destination` because doing so would route the
    // mic input to the speakers (microphone loopback / feedback).
    // The worklet still runs — the Web Audio runtime drives any
    // `AudioWorkletNode` whose input has a live source even if its
    // output is not connected to the destination.
    this.sourceNode = this.audioContext.createMediaStreamSource(stream);
    this.sourceNode.connect(this.workletNode);
  }

  /**
   * Tear down the capture pipeline.
   *
   *   - Stops every `MediaStreamTrack` on the held `MediaStream` so
   *     the browser releases the mic hardware and turns off the
   *     mic-indicator LED.
   *   - Disconnects the source node and the worklet node so the audio
   *     graph is detached.
   *   - Calls `audioContext.close()` (fire-and-forget — `close()`
   *     returns a Promise but `dispose()` is synchronous-by-interface
   *     in this codebase).
   *   - Revokes the Blob URL the worklet was loaded from.
   *   - Clears `onFrame` and sets the sticky `disposed` flag.
   *
   * Idempotent: a second call is a safe no-op (every owned field is
   * already null and the disposed flag short-circuits the body).
   * Disposing without ever calling `start()` is also a safe no-op
   * (all fields are null at construction).
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    if (this.workletNode !== null) {
      const node = this.workletNode;
      this.workletNode = null;
      try {
        node.port.onmessage = null;
      } catch {
        /* defensive — closing port */
      }
      try {
        node.disconnect();
      } catch {
        /* defensive — already disconnected */
      }
    }

    if (this.sourceNode !== null) {
      const node = this.sourceNode;
      this.sourceNode = null;
      try {
        node.disconnect();
      } catch {
        /* defensive — already disconnected */
      }
    }

    if (this.mediaStream !== null) {
      const stream = this.mediaStream;
      this.mediaStream = null;
      for (const track of stream.getTracks()) {
        try {
          track.stop();
        } catch {
          /* defensive — already stopped */
        }
      }
    }

    if (this.audioContext !== null) {
      const ctx = this.audioContext;
      this.audioContext = null;
      try {
        const closeResult = ctx.close();
        // `close()` returns a Promise in the spec; jsdom mocks may
        // return undefined. Guard before chaining so the runtime
        // does not throw on `.catch` of a non-thenable.
        if (closeResult && typeof closeResult.catch === "function") {
          closeResult.catch(() => undefined);
        }
      } catch {
        /* defensive — context already closed */
      }
    }

    if (this.workletUrl !== null) {
      const url = this.workletUrl;
      this.workletUrl = null;
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* defensive — already revoked */
      }
    }

    this.onFrame = null;
  }
}
