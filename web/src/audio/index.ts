/**
 * Browser-side audio module barrel.
 *
 * Re-exports the public surface of the audio capture / playback layer.
 * Mirrors the shape of `web/src/stt/index.ts` and
 * `web/src/tts/index.ts` so the four browser-side modules share an
 * import convention.
 *
 * Tasks 3 and 4 of Feature #16 extend this barrel with `AudioPlayer`
 * and `createAudioPipeline` / `AudioPipeline` respectively.
 */
export {
  FRAME_SAMPLE_COUNT,
  MicPermissionDeniedError,
  MicrophoneCapture,
  TARGET_SAMPLE_RATE_HZ,
} from "./MicrophoneCapture";
export { DECIMATOR_WORKLET_NAME } from "./decimator-worklet";
