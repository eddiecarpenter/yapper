/**
 * Browser-side TTS module barrel.
 *
 * Re-exports the canonical `Speaker` interface and the concrete
 * `KokoroSpeaker` implementation so consumers (currently
 * `useDialogue` in `web/src/dialogue/`, a future SPA shell, and tests)
 * import from a single path. The Transformers.js dependency is hidden
 * behind this boundary — nothing outside `web/src/tts/` should import
 * `@huggingface/transformers` directly. Mirrors the layout of
 * `web/src/stt/index.ts`.
 */
export type { Speaker } from "./types";
export { KokoroSpeaker } from "./KokoroSpeaker";
export type { Provider, SupertonicVoiceId } from "./KokoroSpeaker";
