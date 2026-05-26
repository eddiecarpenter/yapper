/**
 * `AudioPlayer` — Web Audio API playback for Float32 PCM buffers.
 *
 * Canonical specification: `docs/ARCHITECTURE.md` §6.4 + Design Plan KD-3.
 *
 * ## Why this module exists
 *
 * Supertonic TTS (Feature #14, `SupertonicSpeaker`) synthesises 24 kHz Float32
 * audio. The browser's preferred output rate is typically 48 kHz
 * (macOS) or 44.1 kHz (some other platforms). The Web Audio API's
 * built-in resampler handles the rate gap when an `AudioBufferSourceNode`
 * is connected to an `AudioContext` of a different rate — so the
 * "raw PCM → device output" path is mostly graph plumbing.
 *
 * `AudioPlayer` is a primitive: it does NOT implement the `Speaker`
 * contract (`SupertonicSpeaker` already owns that surface for the dialogue
 * loop). It exists as a lower-layer building block:
 *
 *   - The Phase 2 barge-in seam in `docs/ARCHITECTURE.md` §10 uses
 *     `cancel()` mid-utterance — modelled here so it can be exercised
 *     in isolation and reused if `SupertonicSpeaker` later refactors to
 *     delegate raw-PCM playback rather than constructing its own
 *     audio graph.
 *   - A future SPA shell wiring custom WAV / .ogg playback can use
 *     this without reaching for the Web Audio API directly.
 *
 * ## Independent AudioContext (KD-3)
 *
 * The Design Plan calls for separate `AudioContext` instances for
 * capture (`MicrophoneCapture` at 16 kHz) and playback (this module,
 * at the device output rate). A single shared context forces one rate
 * to dominate — either downsampling the playback path or upsampling
 * the capture path, both worse than letting each end pick the right
 * rate. The per-NFR cost (two contexts ≈ a few MB of audio buffers)
 * is negligible.
 *
 * ## Concurrent play() — explicit reject (Design Plan)
 *
 * A `play()` call while another is in flight rejects synchronously
 * with a documented error. This module is a primitive — sequencing
 * utterances is `SupertonicSpeaker`'s responsibility (it already
 * serialises via its own activeSource pointer + cancel chain).
 * Queueing here would create two competing serialisation
 * machineries and a future debug nightmare.
 */

/**
 * Public-error subclass thrown by `play()` when called while another
 * `play()` is still in flight. A named subclass (rather than a plain
 * `Error`) so the caller can `instanceof`-check rather than parsing
 * the message string.
 */
export class AudioPlayerBusyError extends Error {
  constructor(message?: string) {
    super(
      message ??
        "AudioPlayer.play() is already in flight — await or cancel() the previous play() before starting another.",
    );
    this.name = "AudioPlayerBusyError";
    Object.setPrototypeOf(this, AudioPlayerBusyError.prototype);
  }
}

export class AudioPlayer {
  /**
   * The owned `AudioContext`. Constructed lazily on first `play()`
   * call — keeps the constructor side-effect free so a module that
   * instantiates `AudioPlayer` ahead of any actual playback (e.g. a
   * React component holding a ref) does not trip the
   * user-gesture-required guard some browsers apply.
   *
   * Closed on `dispose()` and cleared so a subsequent `dispose()` is
   * a safe no-op.
   */
  private audioContext: AudioContext | null = null;

  /**
   * The `AudioBufferSourceNode` currently scheduled for playback, or
   * `null` if no playback is in flight. Used by `cancel()` to call
   * `source.stop()` and cleared in the `onended` handler so a cancel
   * after natural completion is a safe no-op.
   *
   * Pinned by the play() Promise closure so that cancel() / dispose()
   * can access it without going through async handoff.
   */
  private activeSource: AudioBufferSourceNode | null = null;

  /**
   * Resolver for the in-flight `play()` Promise. Pinned by the
   * Promise constructor closure so `cancel()` can resolve the awaiter
   * without going through the `source.onended` path. Resolving via
   * the field directly guarantees the caller is unblocked even if
   * `source.stop()` later throws.
   *
   * Mirrors the `activeResolver` field on `SupertonicSpeaker` so the two
   * implementations stay aligned.
   */
  private activeResolver: (() => void) | null = null;

  /**
   * Sticky disposal flag. Once `dispose()` runs, subsequent `play()`
   * calls reject so a torn-down instance cannot accidentally be
   * reused — that would lead to an attempt to use a closed context.
   */
  private disposed = false;

  /**
   * Lazily construct the owned `AudioContext`. We defer until the
   * first non-empty `play()` call because the `AudioContext`
   * constructor can throw before a user gesture in some browsers
   * (Chrome's autoplay policy). Once constructed, the context lives
   * for the lifetime of the `AudioPlayer` — `dispose()` closes it.
   */
  private ensureAudioContext(): AudioContext {
    if (this.audioContext === null) {
      // `AudioContext` is provided by the Web Audio API in browsers.
      // jsdom does not implement it; tests stub the global
      // constructor before instantiating the AudioPlayer.
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  /**
   * Play `samples` at the supplied `sampleRate` through the browser's
   * default output device. Resolves only when playback fully
   * completes — either naturally (the `ended` event fires) or via
   * `cancel()` (which also fires `ended` after calling
   * `source.stop()`).
   *
   * Empty input (zero samples) resolves immediately without
   * constructing an `AudioContext`, mirroring `SupertonicSpeaker.speak()`'s
   * empty-text contract — a no-content call is a successful no-op.
   *
   * A concurrent `play()` while another is in flight rejects
   * synchronously with `AudioPlayerBusyError` (see the header
   * comment for the design rationale).
   *
   * Post-`dispose()` calls reject with a clear error.
   */
  async play(samples: Float32Array, sampleRate: number): Promise<void> {
    if (this.disposed) {
      throw new Error("AudioPlayer: cannot play() after dispose()");
    }
    if (this.activeSource !== null || this.activeResolver !== null) {
      // Concurrent play — primitive does not queue.
      throw new AudioPlayerBusyError();
    }

    // Empty samples → resolve immediately without touching Web Audio.
    // No AudioContext is constructed; matches SupertonicSpeaker's
    // empty-text contract.
    if (samples.length === 0) {
      return;
    }

    if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
      throw new Error(
        `AudioPlayer: sampleRate must be a positive finite number, got ${sampleRate}`,
      );
    }

    const ctx = this.ensureAudioContext();

    // Mono buffer at the supplied sample rate. The Web Audio API
    // automatically resamples to the context's output rate
    // (typically 48 kHz on macOS) — no explicit resampling needed
    // here, matching the pattern in `SupertonicSpeaker.speak()`.
    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    // `copyToChannel` is the canonical Web Audio path — it handles
    // the detached-ArrayBuffer / SAB-cloning edge cases the
    // underlying implementation cares about. The cast narrows from
    // `Float32Array<ArrayBufferLike>` (the input type, which is
    // wide enough to admit SharedArrayBuffer-backed views) to
    // `Float32Array<ArrayBuffer>` (what `copyToChannel` accepts
    // under TS 5.7+). Callers in this codebase never produce
    // SAB-backed audio, so the cast is safe at runtime.
    buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    this.activeSource = source;

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
   * Cancel the in-flight `play()`. Stops the active source and
   * resolves (not rejects) the outstanding Promise — the semantics
   * match `SupertonicSpeaker.cancel()` so a caller can swap the two
   * Speakers without rewriting the cancel path.
   *
   * Resolver is invoked BEFORE `source.stop()` so the caller's
   * awaiter is unblocked even if `.stop()` throws (defence-in-depth
   * — `.stop()` should not throw, but Web Audio implementations
   * have surprised us before).
   *
   * With no active `play()`: safe no-op.
   *
   * Reusable: after a cancel, a follow-up `play()` works — the
   * owned `AudioContext` is not released here (`dispose()` does
   * that).
   */
  cancel(): void {
    const resolver = this.activeResolver;
    if (resolver !== null) {
      this.activeResolver = null;
      resolver();
    }

    const source = this.activeSource;
    if (source !== null) {
      this.activeSource = null;
      try {
        source.stop();
      } catch {
        // .stop() can throw `InvalidStateError` if the source was
        // never started or has already been stopped — both safe to
        // ignore in this code path.
      }
    }
  }

  /**
   * Tear down the player.
   *
   *   - Cancel any in-flight playback (same code path as `cancel()`).
   *   - Close the owned `AudioContext` (fire-and-forget — the
   *     returned `Promise<void>` is not awaited; rejections are
   *     swallowed so a teardown race in jsdom mocks does not
   *     surface as a noisy console error).
   *
   * Idempotent: a second call finds every field already null /
   * disposed and exits cleanly. Dispose before any `play()` is also
   * safe (no audio context to close, no active source to cancel).
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    // First cancel any in-flight playback so the caller's Promise is
    // unblocked and the source is stopped. Matches the cancel-first
    // pattern in `SupertonicSpeaker.dispose()`.
    this.cancel();

    if (this.audioContext !== null) {
      const ctx = this.audioContext;
      this.audioContext = null;
      try {
        const closeResult = ctx.close();
        // `close()` returns a Promise per the spec; jsdom mocks may
        // return undefined. Guard before chaining so the runtime
        // does not throw on `.catch` of a non-thenable.
        if (closeResult && typeof closeResult.catch === "function") {
          closeResult.catch(() => undefined);
        }
      } catch {
        /* defensive — context already closed */
      }
    }
  }
}
