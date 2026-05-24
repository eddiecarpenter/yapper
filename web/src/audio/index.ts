/**
 * Browser-side audio module barrel.
 *
 * Re-exports the public surface of the audio capture / playback layer.
 * Mirrors the shape of `web/src/stt/index.ts` and
 * `web/src/tts/index.ts` so the four browser-side modules share an
 * import convention.
 *
 * Task 4 of Feature #16 extends this barrel with `createAudioPipeline`
 * and the `AudioPipeline` interface.
 */
export {
  FRAME_SAMPLE_COUNT,
  MicPermissionDeniedError,
  MicrophoneCapture,
  TARGET_SAMPLE_RATE_HZ,
} from "./MicrophoneCapture";
export { DECIMATOR_WORKLET_NAME } from "./decimator-worklet";
export { AudioPlayer, AudioPlayerBusyError } from "./AudioPlayer";
