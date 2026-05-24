/**
 * Browser-side audio module barrel.
 *
 * Re-exports the public surface of the audio capture / playback /
 * pipeline-composition layer. Mirrors the shape of
 * `web/src/stt/index.ts` and `web/src/tts/index.ts` so the four
 * browser-side modules share an import convention.
 */
export {
  FRAME_SAMPLE_COUNT,
  MicPermissionDeniedError,
  MicrophoneCapture,
  TARGET_SAMPLE_RATE_HZ,
} from "./MicrophoneCapture";
export { DECIMATOR_WORKLET_NAME } from "./decimator-worklet";
export { AudioPlayer, AudioPlayerBusyError } from "./AudioPlayer";
export { createAudioPipeline } from "./createAudioPipeline";
export type {
  AudioPipeline,
  CreateAudioPipelineOptions,
  PipelineState,
} from "./createAudioPipeline";
