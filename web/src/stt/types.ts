/**
 * Browser-side Speech-To-Text contract.
 *
 * Canonical definition lives in `docs/ARCHITECTURE.md` §5.3.
 * Implemented by Feature #16 (Whisper STT module) — this file is a stub
 * that Task 1 of #12 introduces so the `useDialogue` hook can be authored
 * against the interface without depending on the implementation.
 */
export interface Transcriber {
  /**
   * Transcribe a mono PCM audio segment to text.
   *
   * @param audio       Float32 PCM samples — single channel.
   * @param sampleRate  Sample rate in Hz (e.g. 16000 for the resampled VAD
   *                    output). Implementations may resample internally.
   * @returns           The decoded transcript. Promise resolves with `""`
   *                    when the segment contained no recognised speech.
   */
  transcribe(audio: Float32Array, sampleRate: number): Promise<string>;

  /** Release the underlying model, AudioContext, and any mic resources. */
  dispose(): void;
}
