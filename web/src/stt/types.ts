/**
 * Browser-side Speech-To-Text contract.
 *
 * Canonical definition lives in `docs/ARCHITECTURE.md` §5.3.
 * Implemented by Feature #13 (Whisper STT module — `WhisperTranscriber`).
 * The interface stub was authored in Feature #12 Task 1 so the
 * `useDialogue` hook could be written against the contract before the
 * implementation existed; Feature #13 fills the implementation in
 * behind the same interface.
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
