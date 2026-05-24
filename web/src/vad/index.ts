/**
 * Browser-side VAD module barrel.
 *
 * Re-exports the canonical `VAD` interface (authored as a stub in
 * Feature #12 so `useDialogue` could be written against the contract)
 * and the concrete `SileroVAD` implementation from Feature #16.
 * Mirrors `web/src/stt/index.ts` / `web/src/tts/index.ts` so the
 * four browser-side modules share one import convention.
 */
export type { VAD } from "./types";
export {
  DEFAULT_MIN_SILENCE_FRAMES,
  DEFAULT_MIN_SPEECH_FRAMES,
  DEFAULT_MODEL_URL,
  DEFAULT_PRE_ROLL_FRAMES,
  DEFAULT_SILENCE_THRESHOLD,
  DEFAULT_SPEECH_THRESHOLD,
  FRAME_SAMPLE_COUNT,
  SILERO_MODEL_ID,
  SileroVAD,
  TARGET_SAMPLE_RATE_HZ,
} from "./SileroVAD";
export type {
  LoadErrorCause,
  LoadingState,
  Provider,
  SileroVADOptions,
} from "./SileroVAD";
