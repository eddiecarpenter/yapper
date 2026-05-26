/**
 * Browser-side Text-To-Speech contract.
 *
 * Canonical definition lives in `docs/ARCHITECTURE.md` §5.3.
 * Implemented by Feature #14 (Supertonic TTS module) — this file is a stub
 * that Task 1 of #12 introduces so the `useDialogue` hook can be authored
 * against the interface without depending on the implementation.
 */
export interface Speaker {
  /**
   * Synthesise `text` to speech and play it through the Web Audio API.
   *
   * Resolves only when playback has fully completed — this is what allows
   * the dialogue hook to sequence turns cleanly (no overlap between
   * `speak()` finishing and the next VAD trigger firing).
   */
  speak(text: string): Promise<void>;

  /**
   * Cancel any in-flight playback. Used in Phase 2 for barge-in and during
   * `useEffect` cleanup if the component unmounts mid-utterance (AC-4).
   */
  cancel(): void;

  /** Release the underlying model and audio resources. */
  dispose(): void;
}
