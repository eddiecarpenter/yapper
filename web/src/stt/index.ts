/**
 * Browser-side STT module barrel.
 *
 * Re-exports the canonical `Transcriber` interface and the concrete
 * `WhisperTranscriber` implementation so consumers (currently
 * `useDialogue` in `web/src/dialogue/`, a future SPA shell, and tests)
 * import from a single path. The Transformers.js dependency is hidden
 * behind this boundary — nothing outside `web/src/stt/` should import
 * `@huggingface/transformers` directly.
 */
export type { Transcriber } from "./types";
export { WhisperTranscriber } from "./WhisperTranscriber";
export type { Provider } from "./WhisperTranscriber";
