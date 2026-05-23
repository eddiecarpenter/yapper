/**
 * Browser-side Voice Activity Detection contract.
 *
 * Canonical definition lives in `docs/ARCHITECTURE.md` §5.3.
 * Implemented by Feature #15 (audio capture + VAD module). This stub
 * exists so that the `useDialogue` hook (Feature #12) can be authored
 * against the interface without depending on the implementation.
 *
 * ## Two complementary surfaces
 *
 * The VAD has two surfaces, and the dialogue hook uses the higher-level one:
 *
 *   1. `process(frame): boolean`
 *      Per-frame polling (architecture §5.3) — pass in a 512-sample frame at
 *      16 kHz; receive `true` while speech is present. Useful for inline UI
 *      indicators ("speaking now…") and for non-React callers that prefer to
 *      drive the pump themselves.
 *
 *   2. `onSpeechEnd?: (segment: Float32Array) => void`
 *      Aggregated-segment callback (Feature #12 design plan, Risk R1). Fires
 *      once when the VAD detects the end of an utterance, passing the
 *      complete Float32 PCM segment. The dialogue hook subscribes to this so
 *      it can immediately invoke `transcribe()` without re-implementing
 *      silence-window aggregation itself.
 *
 * ## Adapter pattern
 *
 * If the VAD implementation only exposes `process(frame)`, callers can wrap
 * it in a thin adapter that buffers frames until `process()` returns
 * `false` for the configured silence window and then fires `onSpeechEnd`
 * with the accumulated buffer. The adapter lives in the VAD module
 * (Feature #15), not in the dialogue hook — the hook only consumes the
 * higher-level callback.
 */
export interface VAD {
  /**
   * Per-frame speech-detection probe. Returns `true` while speech is
   * present in the supplied 512-sample frame at 16 kHz.
   */
  process(frame: Float32Array): boolean;

  /**
   * Callback invoked once per detected utterance, after the VAD's silence
   * window has elapsed. The implementation either owns this callback
   * directly or it is wired up by an adapter as described above.
   */
  onSpeechEnd?: (segment: Float32Array) => void;

  /** Release any AudioWorkletNode, ports, or worker resources. */
  dispose(): void;
}
